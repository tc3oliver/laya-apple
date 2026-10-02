"""The slow-state trigger test: results.json and tables.md from raw/.

    uv run python research/coreml-slow-state-trigger/scripts/analyze.py [--check]

Inputs: raw/laya-<cell>-r<round>.json.gz (run_all.sh, run_config.py), raw/check.json
(check_cells.py) and raw/failed/*.log (run_all.sh's crash rule). The rules are in
../criteria.md; this file implements them (`decide`, pure and unit-tested).
  - A hetero window is slow if its short P99 >= 13.0 ms, normal below (design.is_slow).
  - A run of PB-R is slow if >= 2 of its 3 hetero windows are; a cell is normal in a run if
    0 of its 3 are slow.
  - Round 1: A, PB-R, PB-H, PB-W, PB-P. If A has any slow window: negative control slow, no
    causal conclusion. If PB-R is not slow: one round-2 PB-R rerun; then either "positive control
    not reproduced" or, if it is slow now, stop for a human decision. If PB-R is slow: every
    candidate (PB-H, PB-W, PB-P) with 3/3 normal windows goes to round 2 with A and PB-R; a
    candidate is confirmed with 0 of 6 slow while PB-R has >= 4 of 6 and A 0 of 6. With no
    candidate and each of PB-H, PB-W and PB-P slow (>= 2 of 3), outcome 4; any other mix is
    partial. Never a third round.
  - A second crash of any run (run_all.sh stops on it too) invalidates the campaign.
Every other record is descriptive: per window short P99, aggregate req/s, client-short and
client-long thread CPU, ANE and GPU thread CPU per forward, ANE stages, and the probe lateness
(PB-P); per run the pooled GPU return P50 and the warm-up counts (PB-W). #83's helpers
(research/coreml-prebind-predict/scripts/analyze.py and derive.py, loaded by path) compute them.

--raw / --out exist for checking the harness on runs kept outside the repository. The campaign
uses the defaults.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RAW = EXP / "raw"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


design = _load("slow_state_design", HERE / "design.py")  # by path: R1's scripts have a design.py too

pb83 = _load("pb83_analyze", EXP.parent / "coreml-prebind-predict" / "scripts" / "analyze.py")
gate = pb83.gate
derive = pb83.derive
gate57 = pb83.mix77.gate57

CELL_RE = "|".join(re.escape(c) for c in design.CELLS)
RUN_NAME = re.compile(rf"^{design.MODEL}-(?P<cell>{CELL_RE})-r(?P<round>\d+)\.json\.gz$")
FAILED_NAME = re.compile(rf"^(?P<run>{design.MODEL}-(?:{CELL_RE})-r\d+)\.(?P<attempt>\d+)\.log$")

OUTCOMES = {  # outcome id: text (criteria.md)
    1: "GIL release is one necessary condition for the slow-state transition; next: who takes CPU/GIL during "
    "the GIL-released period",
    2: "hetero transition / scheduler residency state is key; next: minimal conditioning/state-retention mechanism",
    3: "the 1 ms probe changes scheduler/CPU residency; next: minimal wake mechanism and its overhead; the probe is "
    "not a production fix",
    4: "stop tuning the Python thread path; next: native/Swift worker architecture research",
    5: "positive control not reproduced: no causal conclusion",
}
CONFIRMS = {"PB-H": 1, "PB-W": 2, "PB-P": 3}
NEGATIVE_CONTROL_SLOW = "negative control slow: no causal conclusion"
RERUN_SLOW = (
    "positive control not slow in round 1 but slow in its round-2 rerun: stopped and reported; any further round "
    "needs a human decision"
)
PARTIAL = "partial: reported, no interpretation; next step by human decision"


# ------------------------------------------------------------------ the rule


def decide(slow: dict[tuple[str, int], list[bool]]) -> dict:
    """slow: {(cell, round): per-window slow flags} of the runs on disk. Returns the outcome
    (a list of confirmed outcome ids 1-3, "not_confirmed", id 4 or 5, a named stop, or None while
    running), its text, the round-2 plan (cells), the candidates and the
    confirmed ones, and the next round's runs [(cell, round)] or None."""

    def n(cell, rnd):
        return sum(bool(x) for x in slow[(cell, rnd)])

    def res(outcome, text, nxt=None, **kw):
        return {"outcome": outcome, "text": text, "next": nxt, **kw}

    round1 = [(c, 1) for c in design.ROUND1]
    if any(k not in slow for k in round1):
        return res(None, "running: round 1", round1)
    counts = {c: n(c, 1) for c in design.CELLS}
    if counts["A"] > 0:
        return res("negative_control_slow", NEGATIVE_CONTROL_SLOW, round1_slow=counts)
    if counts["PB-R"] < 2:
        rerun = ("PB-R", 2)
        kw = {"round1_slow": counts, "round2": ["PB-R"]}
        if rerun not in slow:
            return res(None, "running: round 2, positive-control rerun (PB-R)", [rerun], **kw)
        if n(*rerun) >= 2:
            return res("positive_control_rerun_slow", RERUN_SLOW, **kw)
        return res(5, OUTCOMES[5], **kw)
    candidates = [c for c in design.CANDIDATES if counts[c] == 0]
    if candidates:
        round2 = [(c, 2) for c in ("A", "PB-R", *candidates)]
        if any(k not in slow for k in round2):
            return res(
                None,
                f"running: round 2 ({', '.join(c for c, _ in round2)})",
                round2,
                round1_slow=counts,
                candidates=candidates,
                round2=[c for c, _ in round2],
            )
        pbr, a = counts["PB-R"] + n("PB-R", 2), counts["A"] + n("A", 2)
        confirmed = [c for c in candidates if counts[c] + n(c, 2) == 0 and pbr >= 4 and a == 0]
        parts = [f"{c} {'confirmed' if c in confirmed else 'not confirmed'}" for c in candidates]
        return res(
            [CONFIRMS[c] for c in confirmed] or "not_confirmed",
            f"round 2: {'; '.join(parts)} (PB-R {pbr} of 6 slow, A {a} of 6 slow)"
            + ("" if confirmed else f"; {PARTIAL}"),
            round1_slow=counts,
            candidates=candidates,
            round2=[c for c, _ in round2],
            confirmed=confirmed,
            interpretations={c: OUTCOMES[CONFIRMS[c]] for c in confirmed},
        )
    if all(counts[c] >= 2 for c in design.CANDIDATES):
        return res(4, OUTCOMES[4], round1_slow=counts)
    return res("partial", PARTIAL, round1_slow=counts)


# ------------------------------------------------------------------ records


def load(path: Path) -> dict:
    run = pb83.load(path)
    for w in run["part_a"]["windows"]:  # per-request latencies: not used here, and the bulk of a run
        for s in w["streams"].values():
            s.pop("latency_ms", None)
    return run


def check_run(run: dict, cell: str, path: Path) -> None:
    res = run["research"]
    assert res["experiment"] == "coreml-slow-state-trigger" and res["cell"] == cell, path
    assert run["args"]["short"] == design.SHORT and run["args"]["long"] == design.LONG, path
    assert run["args"]["seconds"] == design.SECONDS and run["args"]["cycles"] == design.CYCLES, path
    assert len(gate.hetero_by_cycle(run)) == design.CYCLES, path


def _p50(x):
    x = [v for v in x if v is not None]
    return float(np.median(x)) if x else None


def _forward_rows(run: dict, dev: str, lo: int, hi: int) -> np.ndarray | None:
    x = run["research"]["forwards"].get(dev)
    if x is None:
        return None
    a = np.asarray(x, dtype=np.int64).reshape(-1, 3)
    return a[derive.in_windows(a[:, 0], [(lo, hi)])]


def window_stages(run: dict, lo: int, hi: int) -> dict:
    """P50 ms of each ANE stage (#83's derive.stages) over the forwards starting in [lo, hi)."""
    fw = _forward_rows(run, "ane", lo, hi)
    pr = np.asarray(run["research"]["predicts"] or [], dtype=np.int64).reshape(-1, 6)
    st: dict[str, list] = {k: [] for k in (*derive.STAGES, "predict")}
    if fw is not None and len(fw):
        idx = derive.join_predicts(fw[:, :2], pr)
        for f, i in zip(fw[idx >= 0], idx[idx >= 0]):
            for k, v in derive.stages(f[:2], pr[i]).items():
                if v is not None:
                    st[k].append(v / 1e6)
    return {k: _p50(v) for k, v in st.items()}


def window_records(run: dict) -> list[dict]:
    """One record per hetero window, in cycle order."""
    res = run["research"]
    hw = gate.hetero_by_cycle(run)
    bounds = gate57.hetero_bounds(run)
    rw = sorted(res["windows"], key=lambda w: w["start_ns"])
    out = []
    for i, k in enumerate(sorted(hw)):
        w = hw[k]
        lo, hi = bounds[i]
        r = next((x for x in rw if x["start_ns"] == lo and x["end_ns"] == hi), None)
        cpu = {}
        for s in ("short", "long"):  # each client's own closing snapshot, else R1's last one
            after = ((r or {}).get("after_by_stream") or {}).get(s) or (r or {}).get("after")
            ns = derive.thread_cpu_delta(r["before"], after).get(f"client-{s}") if after else None
            cpu[s] = None if ns is None else ns / 1e6
        cpu_fwd = {}
        for dev in ("ane", "gpu"):
            a = _forward_rows(run, dev, lo, hi)
            cpu_fwd[dev] = float(a[:, 2].mean() / 1e6) if a is not None and len(a) else None
        p99 = w["streams"]["short"]["p99_ms"]
        out.append(
            {
                "cycle": k,
                "short_p99_ms": p99,
                "slow": design.is_slow(p99),
                "long_p99_ms": w["streams"]["long"]["p99_ms"],
                "aggregate_req_s": gate.aggregate_req_s(w),
                "mismatches": sum(s["mismatches"] for s in w["streams"].values()),
                "devices": {s: w["streams"][s]["devices"] for s in ("short", "long")},
                "client_cpu_ms": cpu,
                "thread_cpu_per_forward_ms": cpu_fwd,
                "ane_stages_p50_ms": window_stages(run, lo, hi),
                "gil_probe": r.get("gil_probe") if r else None,
            }
        )
    return out


def run_record(run: dict) -> dict:
    res = run["research"]
    ws = window_records(run)
    summ = gate57.placement_summary([run])
    routing = [
        {"cycle": w["cycle"], "stream": st, "devices": w["devices"][st]}
        for w in ws
        for st, dev in (("short", "ane"), ("long", "gpu"))
        if set(w["devices"][st]) != {dev}
    ]
    pr = np.asarray(res["predicts"] or [], dtype=np.int64).reshape(-1, 6)
    pr = pr[derive.in_windows(pr[:, 0], gate57.hetero_bounds(run))]
    b = pb83.short_bucket({"crossings": res["crossings"]}, design.SHORT)
    return {
        "windows": ws,
        "slow_windows": sum(w["slow"] for w in ws),
        "gpu_return_p50_ms": summ["gpu_return_ms"]["p50"],
        "mismatches_all_windows": summ["mismatches_all_windows"],
        "routing_failures": routing,
        "ane_stages": pb83.in_process_records([run]),
        "forwards": pb83.forward_cpu([run]),
        "hetero_predicts": int(len(pr)),
        "hetero_predicts_with_native_stamps": int((pr[:, 2] > 0).sum()) if len(pr) else 0,
        "shim_flags": res.get("shim_flags"),
        "crossings_short_bucket": (res["crossings"] or {}).get(str(b)) if b else None,
        "warmups": res.get("warmups"),
        "gil_probe_ticks": None if res.get("gil_probe") is None else res["gil_probe"]["ticks"],
        "run_wall_s": res["run_wall_s"],
        "versions": {k: res[k] for k in ("laya_apple", "pyobjc", "coremltools", "mlx")},
    }


def crashes(raw: Path) -> dict[str, int]:
    """{run name: failure logs} from raw/failed (run_all.sh's crash rule)."""
    out: dict[str, int] = {}
    for p in sorted((raw / "failed").glob("*.log")):
        if x := FAILED_NAME.match(p.name):
            out[x["run"]] = out.get(x["run"], 0) + 1
    return out


def summarise(raw: Path) -> dict:
    present = {}
    for p in sorted(raw.glob(f"{design.MODEL}-*-r*.json.gz")):
        if x := RUN_NAME.match(p.name):
            present[(x["cell"], int(x["round"]))] = p
    loaded, runs = {}, {}
    for (cell, rnd), p in present.items():
        loaded[(cell, rnd)] = run = load(p)
        check_run(run, cell, p)
        runs[(cell, rnd)] = run_record(run)
    d = decide({k: [w["slow"] for w in r["windows"]] for k, r in runs.items()})
    crash = crashes(raw)
    second = sorted(r for r, k in crash.items() if k >= 2)
    if second:
        d = {"outcome": "invalid", "text": f"invalid: second crash of {', '.join(second)}; no verdict", "next": None}
    designed = {(c, 1) for c in design.ROUND1} | {(c, 2) for c in d.get("round2", [])}
    cells: dict[str, dict] = {}
    for c in design.CELLS:
        rs = [loaded[k] for k in sorted(present) if k[0] == c]
        if not rs:
            continue
        cells[c] = {
            "rounds": [k[1] for k in sorted(present) if k[0] == c],
            "slow_windows": sum(runs[k]["slow_windows"] for k in runs if k[0] == c),
            "windows": sum(len(runs[k]["windows"]) for k in runs if k[0] == c),
            "gpu_return_p50_ms_pooled": gate57.placement_summary(rs)["gpu_return_ms"]["p50"],
            "ane_stages_pooled": pb83.in_process_records(rs),
            "forwards_pooled": pb83.forward_cpu(rs),
        }
    check = raw / "check.json"
    return {
        "design": {
            "model": design.MODEL,
            "short": design.SHORT,
            "long": design.LONG,
            "round1": list(design.ROUND1),
            "slow_short_p99_ms": design.SLOW_P99_MS,
            "cycles": design.CYCLES,
            "seconds": design.SECONDS,
        },
        "check": json.loads(check.read_text()) if check.exists() else None,
        "failed_runs": sorted(p.name for p in (raw / "failed").glob("*.log")),
        "runs": {
            design.run_file(c, r)[: -len(".json.gz")]: {"cell": c, "round": r, **x}
            for (c, r), x in sorted(runs.items())
        },
        "unused_runs": sorted(design.run_file(*k) for k in present if k not in designed),
        "cells": cells,
        "decision": {k: v for k, v in d.items() if k != "next"},
        "outcome": d["text"],
        "next": [list(x) for x in d["next"]] if d["next"] else None,
    }


# ------------------------------------------------------------------ tables


f = pb83.f


def tables(res: dict) -> str:
    L = ["# Slow-state trigger test (laya, L128 / L512, R1's full protocol)\n"]
    chk = res.get("check")
    if not chk:
        L.append("Pre-campaign check (`raw/check.json`): not run yet\n")
    else:
        g = chk["gil_test"]
        L.append(
            f"Pre-campaign check (`raw/check.json`): PB-R bit-identical {chk['PB_R_bit_identical']}, PB-H "
            f"bit-identical {chk['PB_H_bit_identical']}, PB-R releases the GIL {chk['PB_R_releases_gil']} (probe "
            f"lateness P50 {f(g['PB-R']['lateness_p50_ms'], 3)} ms), PB-H holds the GIL {chk['PB_H_holds_gil']} "
            f"(lateness P50 {f(g['PB-H']['lateness_p50_ms'], 3)} ms, native predict P50 "
            f"{f(g['PB-H']['native_predict_p50_ms'], 3)} ms); all ok {chk['all_ok']}\n"
        )
    L.append(f"Outcome: {res['outcome']}\n")
    for c, t in (res["decision"].get("interpretations") or {}).items():
        L.append(f"- {c}: {t}")
    nxt = res["next"]
    L.append(f"\nNext: {', '.join(f'{c} r{r}' for c, r in nxt) if nxt else 'none'}\n")
    L.append(f"Crashed runs and re-runs (`raw/failed/`): {', '.join(res['failed_runs']) or 'none'}\n")
    L.append(f"Runs not used by the rule: {', '.join(res['unused_runs']) or 'none'}\n")
    L += [
        "\n## Hetero windows\n",
        f"Slow: short P99 ≥ {res['design']['slow_short_p99_ms']} ms. CPU: ms per window (client threads), "
        "ms per forward (ANE and GPU executing threads). Stages: P50 ms.\n",
        "| run | cycle | short P99 | slow | aggregate req/s | client-short CPU | client-long CPU "
        "| ANE CPU/fwd | GPU CPU/fwd | features | native | re-acquire | tail | predict | probe P50 / P99 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in res["runs"].items():
        for w in r["windows"]:
            st, c, t = w["ane_stages_p50_ms"], w["client_cpu_ms"], w["thread_cpu_per_forward_ms"]
            g = w["gil_probe"]
            L.append(
                f"| {name} | {w['cycle']} | {f(w['short_p99_ms'])} | {'**slow**' if w['slow'] else 'normal'} "
                f"| {f(w['aggregate_req_s'], 1)} | {f(c['short'], 0)} | {f(c['long'], 0)} "
                f"| {f(t['ane'], 3)} | {f(t['gpu'], 3)} | {f(st['features'], 3)} | {f(st['native'], 3)} "
                f"| {f(st['reacquire'], 3)} | {f(st['tail'], 3)} | {f(st['predict'], 3)} "
                f"| {'–' if not g else f(g['p50_ms'], 3) + ' / ' + f(g['p99_ms'], 3)} |"
            )
    L += [
        "\n## Runs\n",
        "| run | slow windows | GPU return P50 ms | mismatches | routing failures | predicts with native stamps "
        "| warm-ups (short / long requests, mismatches) | wall s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in res["runs"].items():
        wu = r["warmups"]
        wtext = (
            "–"
            if not wu
            else "; ".join(
                f"c{x['cycle']}: {x['streams']['short']['n']} / {x['streams']['long']['n']}, "
                f"{sum(s['mismatches'] for s in x['streams'].values())}"
                for x in wu
            )
        )
        L.append(
            f"| {name} | {r['slow_windows']} of {len(r['windows'])} | {f(r['gpu_return_p50_ms'], 3)} "
            f"| {r['mismatches_all_windows']} | {len(r['routing_failures'])} "
            f"| {r['hetero_predicts_with_native_stamps']} of {r['hetero_predicts']} | {wtext} | {f(r['run_wall_s'], 0)} |"
        )
    L += [
        "\n## Cells, pooled over their runs\n",
        "| cell | rounds | slow windows | GPU return P50 ms | features | pre | native | re-acquire | post | tail "
        "| ANE CPU/fwd | GPU CPU/fwd |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c, x in res["cells"].items():
        st, fw = x["ane_stages_pooled"], x["forwards_pooled"]

        def p50(k, st=st):
            return f(st[k]["p50_ms"], 3) if k in st else "–"

        L.append(
            f"| {c} | {', '.join(map(str, x['rounds']))} | {x['slow_windows']} of {x['windows']} "
            f"| {f(x['gpu_return_p50_ms_pooled'], 3)} | {p50('features')} | {p50('pre')} | {p50('native')} "
            f"| {p50('reacquire')} | {p50('post')} | {p50('tail')} "
            f"| {f(None if fw['ane'] is None else fw['ane']['thread_cpu_ms_mean'], 3)} "
            f"| {f(None if fw['gpu'] is None else fw['gpu']['thread_cpu_ms_mean'], 3)} |"
        )
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if results.json / tables.md are stale")
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=EXP)
    a = ap.parse_args()
    res = summarise(a.raw)
    js = json.dumps(res, indent=1, sort_keys=True) + "\n"
    md = tables(res)
    targets = ((a.out / "results.json", js), (a.out / "tables.md", md))
    if a.check:
        stale = [p.name for p, s in targets if not p.exists() or p.read_text() != s]
        if stale:
            sys.exit(f"stale: {stale}")
        print("outputs are up to date")
        return
    for p, s in targets:
        p.write_text(s)
    print(md)


if __name__ == "__main__":
    main()
