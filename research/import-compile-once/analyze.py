"""Apply #162's validity guard and classification to the step records harness.py wrote.

    python analyze.py --mode screen OUT_DIR
    python analyze.py --mode full OUT_DIR [--json OUT_DIR/analysis.json]

The thresholds below are the preregistered ones, verbatim; they are not tuned to any data.
Standard library only. A row is one `harness.py step` record (`<label>.json`).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

# Preregistered (#162). Not edited after the first row is written.
PROBE_MAX_RATIO = 0.8  # laya_apple.backends.coreml_ane.PROBE_MAX_RATIO, unchanged
TIMING_BUCKET = 128
SAVES_T_RATIO = 0.80  # T <= 0.80 x B
SAVES_MIN_SAVING_S = 15.0  # B - T >= 15 s
SAVES_E_RATIO = 0.6  # E <= 0.6 x Eb
SAVES_F_SLACK_S = 5.0  # reader F <= baseline-median F + 5 s
NO_EFFECT_T_RATIO = 0.90  # T >= 0.90 x B
NO_EFFECT_E_RATIO = 0.9  # or E >= 0.9 x Eb
REUSE_SLACK_S = 10.0  # reported, no criterion: F <= W' + 10 s
FULL_REPEATS = 3
SCREEN_INJECTIONS = ("parity-fail",)
FULL_INJECTIONS = ("parity-fail", "probe-fail", "force-replace-probe-fail", "sigkill")


def load_rows(out_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(Path(out_dir).glob("*.json")):
        try:
            rec = json.loads(p.read_text())
        except ValueError:
            continue
        if isinstance(rec, dict) and "label" in rec and "arm" in rec and "inject" in rec:
            rows.append(rec)
    return rows


def strip_timing(summary: dict | None) -> dict | None:
    """The parity summary without timing fields (keys ending in _s / _ms or naming a time)."""
    if summary is None:
        return None
    return {k: v for k, v in summary.items() if not (k.endswith(("_s", "_ms")) or "time" in k)}


def _ratio_ok(probe: dict | None) -> bool:
    return (
        isinstance(probe, dict) and isinstance(probe.get("ratio"), (int, float)) and probe["ratio"] <= PROBE_MAX_RATIO
    )


def _imp(row) -> dict:
    return row.get("import") or {}


def _post(row, key="post") -> dict:
    return row.get(key) or {}


def _poll_attempts(row) -> list[dict]:
    return [r for r in (row.get("poller") or []) if r.get("event") == "attempt"]


def _publishing_t(row) -> float | None:
    ts = [e["t"] for e in _imp(row).get("events", []) if str(e.get("msg", "")).endswith(": publishing")]
    return ts[0] if ts else None


# ----------------------------------------------------------------------------- guard


def guard_import(row: dict, base_parity: dict | None) -> list[str]:
    """Guard 1: parity summary identical to baseline's (minus timing), passed, 0 hard
    mismatches; every probe ratio <= 0.8."""
    fails = []
    imp, post = _imp(row), _post(row)
    if not imp.get("ok"):
        return [f"import failed: {imp.get('error')}"]
    parity = post.get("parity")
    if not isinstance(parity, dict):
        return ["no parity summary in the registered manifest"]
    if parity.get("passed") is not True:
        fails.append("parity passed is not True")
    if parity.get("hard_mismatches") != 0:
        fails.append(f"hard_mismatches = {parity.get('hard_mismatches')}")
    if base_parity is not None and strip_timing(parity) != strip_timing(base_parity):
        fails.append("parity summary differs from baseline's")
    probes = imp.get("probes") or {}
    if not probes:
        fails.append("no probe recorded")
    for k, p in sorted(probes.items()):
        if not _ratio_ok(p):
            fails.append(f"probe {k} ratio {(p or {}).get('ratio')} > {PROBE_MAX_RATIO}")
    return fails


def guard_reader(row: dict, base_predictions: dict | None) -> list[str]:
    """Guard 2: the reader loads on CPU_AND_NE, probe <= 0.8, predictions bit-identical to
    baseline's reader."""
    r = row.get("reader") or {}
    if not r.get("ok"):
        return [f"reader failed: {r.get('error')}"]
    fails = []
    if "CPU_AND_NE" not in str(r.get("compute_unit")):
        fails.append(f"reader compute unit {r.get('compute_unit')!r}")
    if not _ratio_ok(r.get("probe")):
        fails.append(f"reader probe {r.get('probe') or r.get('probe_error')}")
    if base_predictions is not None and r.get("predictions") != base_predictions:
        fails.append("reader predictions differ from baseline's reader")
    return fails


def guard_poller(row: dict) -> list[str]:
    """Guard 3: before manifest.json appears the poller only gets ArtifactMissingError, never a
    model; nothing is moved into quarantine/."""
    attempts = _poll_attempts(row)
    if row.get("poller") is None:
        return ["no poller record"]
    if not attempts:
        return ["the poller made no attempt during the import"]
    fails = []
    pub = _publishing_t(row)
    for i, at in enumerate(attempts):
        if at.get("quarantine"):
            fails.append(f"attempt {i}: quarantine/ holds {at['quarantine']}")
        if at.get("outcome") == "ArtifactMissingError":
            continue
        if at.get("outcome") == "model" and pub is not None and at.get("t1", 0) >= pub:
            continue  # loaded after the manifest was published
        fails.append(f"attempt {i}: {at.get('outcome')} ({at.get('message')})")
    if _post(row).get("quarantine"):
        fails.append(f"quarantine/ holds {_post(row)['quarantine']} after the import")
    return fails


def guard_injection(row: dict) -> list[str]:
    """Guard 4, per injection: (i)/(ii) the call raises, nothing is registered, the final path is
    absent, the poller never sees a manifest, quarantine/ is empty; (iii) the old artifact is
    restored with byte-identical manifest.json and identical tree_sha256; (iv) the final path has
    no manifest, readers get ArtifactMissingError, the re-import succeeds with no leftover."""
    inj, imp, post = row.get("inject"), _imp(row), _post(row)
    fails = []
    if inj in ("parity-fail", "probe-fail"):
        if inj not in imp.get("inject_fired", []):
            fails.append("the injection did not fire")
        if imp.get("ok") is not False:
            fails.append(f"the call did not raise (ok={imp.get('ok')})")
        if post.get("manifest_exists"):
            fails.append("a manifest is registered")
        if post.get("final_exists") is not False:
            fails.append("the final path exists")
        if row.get("poller") is None or not _poll_attempts(row):
            fails.append("no poller attempt")
        for r in row.get("poller") or []:
            if r.get("event") == "manifest_seen" or r.get("outcome") == "model":
                fails.append(f"the poller saw a manifest ({r.get('event')}, {r.get('outcome')})")
            if r.get("quarantine"):
                fails.append(f"quarantine/ holds {r['quarantine']} during the import")
        if post.get("quarantine"):
            fails.append(f"quarantine/ holds {post['quarantine']}")
    elif inj == "force-replace-probe-fail":
        before = _post(row, "before")
        if not (row.get("setup") or {}).get("ok") or not before.get("manifest_exists"):
            fails.append("the L64 artifact to replace was not registered")
        if inj not in imp.get("inject_fired", []):
            fails.append("the injection did not fire")
        if imp.get("ok") is not False:
            fails.append(f"the call did not raise (ok={imp.get('ok')})")
        if not post.get("manifest_sha256") or post.get("manifest_sha256") != before.get("manifest_sha256"):
            fails.append("manifest.json is not byte-identical to the old one")
        if not post.get("tree_sha256") or post.get("tree_sha256") != before.get("tree_sha256"):
            fails.append("tree_sha256 differs from the old artifact's")
    elif inj == "sigkill":
        if not imp.get("killed"):
            fails.append("the import was not killed during the gate")
        if post.get("manifest_exists"):
            fails.append("the final path has a manifest after the kill")
        if ((row.get("reader") or {}).get("error") or {}).get("type") != "ArtifactMissingError":
            fails.append(f"reader after the kill got {(row.get('reader') or {}).get('error')}")
        if not (row.get("reimport") or {}).get("ok"):
            fails.append(f"re-import failed: {(row.get('reimport') or {}).get('error')}")
        rp = _post(row, "reimport_post")
        if not rp.get("manifest_exists"):
            fails.append("nothing registered after the re-import")
        for k in ("leftovers", "pending_files", "staging", "quarantine"):
            if rp.get(k) != []:
                fails.append(f"leftover after the re-import: {k} = {rp.get(k)}")
    else:
        fails.append(f"unknown injection {inj!r}")
    return fails


# ----------------------------------------------------------------------------- classification


def metrics(row: dict) -> dict:
    r = row.get("reader") or {}
    return {
        "T": _imp(row).get("T"),
        "E": (row.get("e5rt_import") or {}).get("bytes"),
        "E_count": (row.get("e5rt_import") or {}).get("count"),
        "F": r.get("F"),
        "W2": r.get("W2"),
        "phases": _imp(row).get("phases") or {},
        "reader_E": (row.get("e5rt_reader") or {}).get("bytes"),
    }


def classify(base: list[dict], final: list[dict]) -> dict:
    """SAVES / NO EFFECT / INCONCLUSIVE from metrics() of the baseline and final-path rows."""
    B = statistics.median(m["T"] for m in base)
    Eb = statistics.median(m["E"] for m in base)
    Fb = statistics.median(m["F"] for m in base)
    per = []
    for m in final:
        saves = (
            m["T"] <= SAVES_T_RATIO * B
            and B - m["T"] >= SAVES_MIN_SAVING_S
            and m["E"] <= SAVES_E_RATIO * Eb
            and m["F"] <= Fb + SAVES_F_SLACK_S
        )
        no_effect = m["T"] >= NO_EFFECT_T_RATIO * B or m["E"] >= NO_EFFECT_E_RATIO * Eb
        per.append({"T": m["T"], "E": m["E"], "F": m["F"], "saves": saves, "no_effect": no_effect})
    all_saves, all_none = all(p["saves"] for p in per), all(p["no_effect"] for p in per)
    if all_saves and all_none:
        label = "UNDEFINED"  # both rules hold; only possible with Eb = 0 and E = 0. Not resolved here.
    elif all_saves:
        label = "SAVES"
    elif all_none:
        label = "NO EFFECT"
    else:
        label = "INCONCLUSIVE"
    return {"B": B, "Eb": Eb, "Fb": Fb, "per_repeat": per, "label": label}


def analyze(rows: list[dict], mode: str) -> dict:
    timing = [r for r in rows if r.get("inject") == "none" and r.get("bucket") == TIMING_BUCKET]
    base = [r for r in timing if r["arm"] == "baseline"]
    final = [r for r in timing if r["arm"] == "final-path"]
    wanted = SCREEN_INJECTIONS if mode == "screen" else FULL_INJECTIONS
    inj = {k: [r for r in rows if r.get("inject") == k] for k in wanted}
    n = 1 if mode == "screen" else FULL_REPEATS
    out: dict = {"mode": mode, "n_baseline": len(base), "n_final": len(final), "guard": {}, "notes": []}

    missing = []
    if len(base) != n or len(final) != n:
        missing.append(
            f"expected {n} baseline and {n} final-path L{TIMING_BUCKET} rows, got {len(base)} and {len(final)}"
        )
    missing += [f"no injection row for {k}" for k, v in inj.items() if len(v) != 1]

    base_parity = _post(base[0]).get("parity") if base else None
    base_preds = (base[0].get("reader") or {}).get("predictions") if base else None
    for r in base:
        out["guard"][r["label"]] = guard_import(r, base_parity) + guard_reader(r, base_preds)
    for r in final:
        out["guard"][r["label"]] = guard_import(r, base_parity) + guard_reader(r, base_preds) + guard_poller(r)
    for k, v in inj.items():
        for r in v:
            out["guard"][r["label"]] = guard_injection(r)
    invalid = {k: v for k, v in out["guard"].items() if v}
    base_invalid = any(out["guard"].get(r["label"]) for r in base)

    out["metrics"] = {r["label"]: metrics(r) for r in base + final}
    out["reuse"] = {
        r["label"]: (
            None
            if out["metrics"][r["label"]]["F"] is None or out["metrics"][r["label"]]["W2"] is None
            else out["metrics"][r["label"]]["F"] <= out["metrics"][r["label"]]["W2"] + REUSE_SLACK_S
        )
        for r in base + final
    }
    if invalid:
        label = "INVALID (baseline arm)" if base_invalid else "INVALID"
        out["invalid"] = invalid
    elif missing:
        label = "INCOMPLETE"
        out["notes"] += missing
    else:
        out["classification"] = classify(
            [out["metrics"][r["label"]] for r in base], [out["metrics"][r["label"]] for r in final]
        )
        label = out["classification"]["label"]
    out["label"] = ("screen-" + label.replace("NO EFFECT", "NO-EFFECT")) if mode == "screen" else label
    if mode == "screen":
        out["notes"].append("a screen never declares SAVES; screen-SAVES means: full run in a separate slot")
    return out


def _fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def report(res: dict) -> str:
    lines = [f"mode: {res['mode']}  baseline rows: {res['n_baseline']}  final-path rows: {res['n_final']}", ""]
    lines.append("| row | T s | S | P | Q | R | E count | E GB | reader F s | W' s | reuse F<=W'+10 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for label, m in res.get("metrics", {}).items():
        ph = m["phases"]
        lines.append(
            f"| {label} | {_fmt(m['T'])} | {_fmt(ph.get('S'))} | {_fmt(ph.get('P'))} | {_fmt(ph.get('Q'))} | "
            f"{_fmt(ph.get('R'))} | {m['E_count']} | {_fmt(m['E'] / 1e9 if m['E'] is not None else None, 3)} | "
            f"{_fmt(m['F'])} | {_fmt(m['W2'])} | {res['reuse'].get(label)} |"
        )
    lines.append("")
    for label, fails in res["guard"].items():
        lines.append(f"guard {label}: {'pass' if not fails else 'FAIL: ' + '; '.join(fails)}")
    c = res.get("classification")
    if c:
        lines.append("")
        lines.append(f"B = {c['B']:.2f} s, Eb = {c['Eb'] / 1e9:.3f} GB, baseline F = {c['Fb']:.2f} s")
        for i, p in enumerate(c["per_repeat"]):
            lines.append(
                f"final-path {i + 1}: T {p['T']:.2f} s, E {p['E'] / 1e9:.3f} GB, F {p['F']:.2f} s, "
                f"SAVES rule {p['saves']}, NO EFFECT rule {p['no_effect']}"
            )
    for n in res["notes"]:
        lines.append(f"note: {n}")
    lines.append("")
    lines.append(f"LABEL: {res['label']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("out_dir")
    p.add_argument("--mode", choices=["screen", "full"], required=True)
    p.add_argument("--json", help="also write the analysis here")
    a = p.parse_args(argv)
    res = analyze(load_rows(Path(a.out_dir)), a.mode)
    print(report(res))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
