"""Derive results.json and tables.md from benchmarks/compare-v1.4/raw/.

    uv run python benchmarks/compare-v1.4/analyze.py            # write results.json, tables.md
    uv run python benchmarks/compare-v1.4/analyze.py --check    # fail if they are out of date

Latency, throughput, parity and energy are separate tables and are never merged into one
verdict. Parity uses the FP16 gate of laya_apple/parity (docs/correctness.md) applied to the
public answers every runtime returns:

- probability error: max |delta| over a question's option probabilities (noul: [1-p, p]);
- action-probability error: |delta| of act_probability;
- hard mismatch: the selected option differs and the reference top-1/top-2 margin >= 2 x tol;
- near-tie flip: the selected option differs inside that band; listed row by row;
- verdict: prob error <= tol, action error <= tol, 0 hard mismatches, 0 label-set
  mismatches, repeated calls identical. Cases a runtime refuses (for example a capacity
  error) are counted and listed separately; they are not answered rows.

Public answers are rounded to 4 decimals by every runtime (and by upstream), so errors
below 1e-4 are not resolved. This is an answer-level comparison; the v1.0 parity tables
compare unrounded logits and are a different measurement.

References: the committed goldens (laya_apple/parity/goldens, upstream Laya 0.3.20, PyTorch
CPU FP32) for the golden fixture sets; for every other fixture set, the answers of the
`upstream` runtime (the same upstream version, device and dtype) recorded in the same run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = HERE / "raw"
RESULTS = HERE / "results.json"
TABLES = HERE / "tables.md"
TOL = 0.02  # laya_apple.parity.TOLERANCE["float16"]; asserted in tests
P99_MIN_SAMPLES = 1000
REFERENCE_RUNTIME = "upstream"


# ------------------------------------------------------------------------------ parity
def compare_answers(ref: dict, got: dict, tol: float = TOL) -> list[dict]:
    """Per-question rows for one case. `ref`/`got` are normalised answers keyed by qid."""
    rows = []
    for qid, r in ref.items():
        g = got.get(qid)
        row = {"qid": qid, "type": r["type"]}
        rp = r["probs"]
        srt = sorted(rp)
        row["ref_margin"] = srt[-1] - srt[-2] if len(srt) > 1 else 1.0
        if g is None:
            row.update(missing=True, label_mismatch=True, mismatch=True, prob_max_abs=math.inf, act_abs=math.inf)
            rows.append(row)
            continue
        if list(g["labels"]) != list(r["labels"]):
            row.update(label_mismatch=True, mismatch=True, prob_max_abs=math.inf)
        else:
            gp = g["probs"]
            finite = all(math.isfinite(x) for x in gp)
            row["label_mismatch"] = False
            row["prob_max_abs"] = max(abs(a - b) for a, b in zip(gp, rp)) if finite else math.inf
            row["mismatch"] = max(range(len(gp)), key=gp.__getitem__) != max(range(len(rp)), key=rp.__getitem__)
        ra, ga = r.get("act_probability"), g.get("act_probability")
        row["act_abs"] = abs(ga - ra) if ra is not None and ga is not None else math.inf
        rows.append(row)
    return rows


def gate(rows: list[dict], tol: float = TOL, repeat_identical: bool | None = True) -> dict:
    hard = [r for r in rows if r["mismatch"] and not r.get("label_mismatch") and r["ref_margin"] >= 2 * tol]
    flips = [r for r in rows if r["mismatch"] and not r.get("label_mismatch") and r["ref_margin"] < 2 * tol]
    labels = [r for r in rows if r.get("label_mismatch")]
    prob = max((r["prob_max_abs"] for r in rows), default=0.0)
    act = max((r["act_abs"] for r in rows), default=0.0)
    return {
        "rows": len(rows),
        "prob_max_abs": prob,
        "act_prob_max_abs": act,
        "hard_mismatches": len(hard),
        "near_tie_flips": [{k: r[k] for k in ("case", "qid", "ref_margin") if k in r} for r in flips],
        "label_mismatches": [{k: r[k] for k in ("case", "qid") if k in r} for r in labels],
        "repeat_identical": repeat_identical,
        "passed": bool(
            rows and not hard and not labels and prob <= tol and act <= tol and repeat_identical is not False
        ),
    }


def golden_reference(model: str) -> dict:
    """{case name: normalised answers} from the committed goldens."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(HERE))
    from adapter import normalize_result

    from laya_apple.parity import load_goldens

    return {c["name"]: normalize_result(c["answers"]) for c in load_goldens(model)["cases"]}


def case_group(fixture_set: str, name: str) -> str:
    """Golden sets are split: cases whose prompt behaviour upstream changed after 0.3.5 apart."""
    if fixture_set.startswith("goldens"):
        return "goldens: upstream changes after 0.3.5" if name.startswith("drift-") else "goldens: 0.3.5-identical"
    return fixture_set


def weights_table(run: Path, manifest: dict) -> list[dict]:
    """What each configuration loaded. laya-coreml bundles are checked against the source
    weight hash their own manifest records; a bundle whose hash differs is flagged."""
    sys.path.insert(0, str(ROOT))
    from laya_apple.parity import load_goldens

    seen = {}
    for f in sorted((run / "parity").glob("*.json")) if (run / "parity").exists() else []:
        d = json.loads(f.read_text())
        seen[f"{d['runtime']}/{d['variant']}/{d['model']}"] = d.get("runtime_info", {})
    rows = []
    for key, w in sorted((manifest.get("weights") or {}).items()):
        rt, var, m = key.split("/")
        if key not in seen:
            continue
        same = w.get("same_as_pin")
        if rt == "laya-coreml":
            src = (seen[key].get("bundle_source") or {}).get("source_weights_sha256")
            same = None if src is None else src == load_goldens(m)["source_weights_sha256"]
        rows.append({"runtime": rt, "variant": var, "model": m, "source": w.get("source"), "same_as_pin": same})
    return rows


def parity_table(run: Path) -> list[dict]:
    pdir = run / "parity"
    if not pdir.exists():
        return []
    files = sorted(pdir.glob("*.json"))
    recs = [json.loads(f.read_text()) for f in files]
    refs: dict = {}
    for r in recs:
        if r["runtime"] == REFERENCE_RUNTIME:
            refs[(r["model"], r["fixture_set"])] = {c["name"]: c.get("answers") for c in r["cases"]}
    out = []
    for r in recs:
        m, fs = r["model"], r["fixture_set"]
        ref = golden_reference(m) if fs.startswith("goldens") else refs.get((m, fs))
        if ref is None:
            continue
        groups: dict = {}
        for c in r["cases"]:
            g = case_group(fs, c["name"])
            grp = groups.setdefault(g, {"rows": [], "refused": [], "no_reference": []})
            want = ref.get(c["name"])
            if want is None:
                grp["no_reference"].append(c["name"])
                continue
            if "answers" not in c:
                grp["refused"].append({"case": c["name"], "error": c.get("error", "")[:200]})
                continue
            for row in compare_answers(want, c["answers"]):
                row["case"] = c["name"]
                grp["rows"].append(row)
        for g, grp in sorted(groups.items()):
            res = gate(grp["rows"], repeat_identical=r.get("repeat_identical"))
            res.update(
                runtime=r["runtime"],
                variant=r["variant"],
                model=m,
                fixture_set=fs,
                group=g,
                refused=grp["refused"],
                no_reference=grp["no_reference"],
            )
            out.append(res)
    return out


# ------------------------------------------------------------------------------ latency
def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return math.nan
    k = q * (len(s) - 1)
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def window_power(e: dict, w: dict) -> tuple[float | None, str]:
    """Mean SoC power over the window itself, when the sampler's timeline allows.

    The sampler (research/energy-sampler) records cumulative per-rail joules since it started,
    against CLOCK_UPTIME_RAW; the adapter stamps each window with the same clock. The window's
    energy is soc(t1) - soc(t0), with soc = the sum of the rails, linearly interpolated at both
    window edges (the method of the sampler's own `window_energy`). Without a timeline that
    brackets the window, the sampler's whole-life mean is returned and labelled as such: it
    includes the 1 s lead-in and the tail up to SIGINT, so it is not the window mean.
    """
    pts = sorted((s[0], sum(s[2].values())) for s in e.get("samples") or [] if isinstance(s, list) and len(s) >= 3)
    t0, t1 = w.get("t_start_uptime_ns"), w.get("t_end_uptime_ns")

    def soc(t):
        for (ta, ja), (tb, jb) in zip(pts, pts[1:]):
            if ta <= t <= tb and tb > ta:
                return ja + (jb - ja) * (t - ta) / (tb - ta)
        return None

    if t0 is not None and t1 is not None and t1 > t0:
        j0, j1 = soc(t0), soc(t1)
        if j0 is not None and j1 is not None:
            return (j1 - j0) / ((t1 - t0) / 1e9), "window (interpolated)"
    return e.get("mean_power_w"), "sampler whole-life mean"


def latency_tables(run: Path) -> tuple[list[dict], list[dict], list[dict]]:
    ldir = run / "latency"
    if not ldir.exists():
        return [], [], []
    idle = {}
    for f in sorted(ldir.glob("idle-*.json")):
        d = json.loads(f.read_text())
        if d.get("mean_power_w") is not None:
            idle[d["round"]] = d["mean_power_w"]
    pooled: dict = {}
    for f in sorted(ldir.glob("r*.json")):
        d = json.loads(f.read_text())
        rnd = int(f.name[1:].split("-", 1)[0])
        for w in d.get("windows", []):
            key = (d["runtime"], d["variant"], d["model"], w["shape"])
            p = pooled.setdefault(
                key,
                {"lat": [], "win_p50": [], "calls": 0, "q": 0, "wall": 0.0, "devices": {}, "energy": [], "errors": []},
            )
            if "error" in w:
                p["errors"].append(w["error"])
                continue
            p["lat"] += w["latency_ms"]
            p["win_p50"].append(pct(w["latency_ms"], 0.5))
            p["calls"] += w["calls"]
            p["q"] += w["calls"] * w["questions_per_call"]
            p["wall"] += w["wall_s"]
            for k, v in w.get("devices", {}).items():
                p["devices"][k] = p["devices"].get(k, 0) + v
            e = w.get("energy")
            power, how = window_power(e, w) if e else (None, "")
            if power is not None:
                base = idle.get(rnd)
                net_w = power - base if base is not None else None
                p["energy"].append(
                    {
                        "mean_power_w": power,
                        "method": how,
                        "idle_w": base,
                        "net_w": net_w,
                        "wall_s": w["wall_s"],
                        "calls": w["calls"],
                        "q": w["calls"] * w["questions_per_call"],
                    }
                )
    lat, thr, en = [], [], []
    for (rt, var, m, shape), p in sorted(pooled.items()):
        base = {"runtime": rt, "variant": var, "model": m, "shape": shape}
        if not p["lat"]:
            lat.append({**base, "unsupported": p["errors"][:1]})
            continue
        n = len(p["lat"])
        lat.append(
            {
                **base,
                "samples": n,
                "windows": len(p["win_p50"]),
                "p50_ms": pct(p["lat"], 0.5),
                "p99_ms": pct(p["lat"], 0.99),
                "p99_low_samples": n < P99_MIN_SAMPLES,
                "window_p50_min_ms": min(p["win_p50"]),
                "window_p50_max_ms": max(p["win_p50"]),
                "devices": p["devices"],
            }
        )
        thr.append({**base, "calls_per_s": p["calls"] / p["wall"], "questions_per_s": p["q"] / p["wall"]})
        if p["energy"]:
            wall = sum(e["wall_s"] for e in p["energy"])
            calls = sum(e["calls"] for e in p["energy"])
            qs = sum(e["q"] for e in p["energy"])
            mean_w = sum(e["mean_power_w"] * e["wall_s"] for e in p["energy"]) / wall
            net = [e for e in p["energy"] if e["net_w"] is not None]
            net_j = sum(e["net_w"] * e["wall_s"] for e in net) if len(net) == len(p["energy"]) else None
            en.append(
                {
                    **base,
                    "mean_power_w": mean_w,
                    "net_j_per_call": None if net_j is None else net_j / calls,
                    "net_j_per_question": None if net_j is None else net_j / qs,
                    "windows": len(p["energy"]),
                }
            )
        else:
            en.append({**base, "mean_power_w": None, "net_j_per_call": None, "net_j_per_question": None, "windows": 0})
    return lat, thr, en


# ------------------------------------------------------------------------------ output
def analyze(raw: Path = RAW) -> dict:
    runs = []
    for run in sorted(p for p in raw.glob("*") if p.is_dir()) if raw.exists() else []:
        manifest = json.loads((run / "manifest.json").read_text()) if (run / "manifest.json").exists() else {}
        lat, thr, en = latency_tables(run)
        runs.append(
            {
                "run": run.name,
                "manifest": {
                    k: manifest.get(k) for k in ("method", "platform", "machine_state", "started", "finished")
                },
                "weights": weights_table(run, manifest),
                "parity": parity_table(run),
                "latency": lat,
                "throughput": thr,
                "energy": en,
            }
        )
    return {"tolerance": TOL, "p99_min_samples": P99_MIN_SAMPLES, "runs": runs}


def _f(x, nd=2):
    if x is None:
        return "—"
    if isinstance(x, float) and not math.isfinite(x):
        return "inf"
    if isinstance(x, float) and x != 0 and abs(x) < 0.01:
        return f"{x:.2e}"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def tables_md(res: dict) -> str:
    out = ["# Runtime comparison: derived tables", "", "Generated by `analyze.py` from `raw/`. Do not edit.", ""]
    if not res["runs"]:
        out.append("No campaign run is recorded yet.")
        return "\n".join(out) + "\n"
    for run in res["runs"]:
        out += [f"## Run `{run['run']}`", ""]
        if run["weights"]:
            out += [
                "### Weights",
                "",
                "Same = the weights are, or were exported from, the checkpoint laya-apple pins. A row whose",
                "weights differ is not a same-weights comparison.",
                "",
                "| Runtime | Variant | Model | Weights | Same as laya-apple pin |",
                "|---|---|---|---|---|",
            ]
            for w in run["weights"]:
                same = {True: "yes", False: "**no: different weights**", None: "unknown"}[w["same_as_pin"]]
                out.append(f"| {w['runtime']} | {w['variant']} | {w['model']} | {w['source']} | {same} |")
            out.append("")
        out += [
            "### Parity against upstream (answer level)",
            "",
            f"FP16 gate, tolerance {res['tolerance']}. Refused = cases the runtime declined (listed in results.json).",
            "",
            "| Runtime | Variant | Model | Fixtures | Rows | Refused | Hard mismatches | Near-tie flips | Label mismatches | Max prob error | Max act error | Repeat identical | Verdict |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
        for p in run["parity"]:
            out.append(
                f"| {p['runtime']} | {p['variant']} | {p['model']} | {p['group']} | {p['rows']} | {len(p['refused'])} | "
                f"{p['hard_mismatches']} | {len(p['near_tie_flips'])} | {len(p['label_mismatches'])} | {_f(p['prob_max_abs'], 4)} | "
                f"{_f(p['act_prob_max_abs'], 4)} | {p['repeat_identical']} | {'pass' if p['passed'] else 'fail'} |"
            )
        flips = [(p, f) for p in run["parity"] for f in p["near_tie_flips"]]
        if flips:
            out += ["", "Near-tie flips:", ""]
            for p, f in flips:
                out.append(
                    f"- {p['runtime']}/{p['variant']}/{p['model']} {f.get('case')}/{f.get('qid')}: reference margin {f['ref_margin']:.4f}"
                )
        out += [
            "",
            "### Latency (end to end, one process, sequential calls)",
            "",
            f"P99 marked * is computed from fewer than {res['p99_min_samples']} samples.",
            "",
            "| Runtime | Variant | Model | Shape | Samples | Windows | P50 ms | P99 ms | Window P50 range ms | Devices |",
            "|---|---|---|---|---:|---:|---:|---:|---|---|",
        ]
        for r in run["latency"]:
            if "unsupported" in r:
                out.append(
                    f"| {r['runtime']} | {r['variant']} | {r['model']} | {r['shape']} | — | — | — | — | unsupported | |"
                )
                continue
            dev = ", ".join(f"{k} {v}" for k, v in sorted(r["devices"].items()))
            star = "*" if r["p99_low_samples"] else ""
            out.append(
                f"| {r['runtime']} | {r['variant']} | {r['model']} | {r['shape']} | {r['samples']} | {r['windows']} | "
                f"{_f(r['p50_ms'])} | {_f(r['p99_ms'])}{star} | {_f(r['window_p50_min_ms'])}–{_f(r['window_p50_max_ms'])} | {dev} |"
            )
        out += [
            "",
            "### Throughput (closed loop, one client)",
            "",
            "| Runtime | Variant | Model | Shape | Calls/s | Questions/s |",
            "|---|---|---|---|---:|---:|",
        ]
        for r in run["throughput"]:
            out.append(
                f"| {r['runtime']} | {r['variant']} | {r['model']} | {r['shape']} | {_f(r['calls_per_s'], 1)} | {_f(r['questions_per_s'], 1)} |"
            )
        out += [
            "",
            "### Energy",
            "",
            "Net = SoC power minus the idle window of the same round. — = not measured.",
            "",
            "| Runtime | Variant | Model | Shape | Mean SoC power W | Net J / call | Net J / question |",
            "|---|---|---|---|---:|---:|---:|",
        ]
        for r in run["energy"]:
            out.append(
                f"| {r['runtime']} | {r['variant']} | {r['model']} | {r['shape']} | {_f(r['mean_power_w'])} | "
                f"{_f(r['net_j_per_call'], 4)} | {_f(r['net_j_per_question'], 4)} |"
            )
        out.append("")
    return "\n".join(out) + "\n"


def _json(x):
    def fix(o):
        if isinstance(o, float) and not math.isfinite(o):
            return str(o)
        if isinstance(o, dict):
            return {k: fix(v) for k, v in o.items()}
        if isinstance(o, list):
            return [fix(v) for v in o]
        return o

    return json.dumps(fix(x), indent=1, ensure_ascii=False) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if results.json/tables.md differ from raw/")
    ap.add_argument("--raw", type=Path, default=RAW, help="analyse another raw root (a dry run); writes next to it")
    a = ap.parse_args(argv)
    res = analyze(a.raw)
    global RESULTS, TABLES
    if a.raw != RAW:
        RESULTS, TABLES = a.raw / "results.json", a.raw / "tables.md"
    want = {RESULTS: _json(res), TABLES: tables_md(res)} if res["runs"] else {}
    if a.check:
        stale = [p.name for p, t in want.items() if not p.exists() or p.read_text() != t]
        stale += [p.name for p in (RESULTS, TABLES) if p not in want and p.exists()]
        if stale:
            print("out of date: " + ", ".join(stale) + " (run analyze.py)")
            return 1
        print(f"up to date ({len(res['runs'])} run(s) in raw/)")
        return 0
    if not want:
        print("no runs in raw/; nothing to write")
        return 0
    for p, t in want.items():
        p.write_text(t)
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
