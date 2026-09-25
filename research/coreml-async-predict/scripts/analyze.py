"""The async-predict screen: results.json and tables.md from raw/.

    uv run python research/coreml-async-predict/scripts/analyze.py [--check]

Inputs: raw/laya-<cell>-r<round>.json.gz (run_all.sh, run_config.py), raw/phase0.json
(phase0.py) and raw/failed/*.log (run_all.sh's crash rule). The rules are in ../criteria.md;
this file implements them (`decide`, pure and unit-tested).
  - A hetero window is slow if its short P99 >= 13.0 ms (design.is_slow). A run is slow with
    >= 2 of its 3 hetero windows slow, normal with 0 of 3.
  - The pattern, per round (`pattern`): A normal (0 of 3 slow), PB-SYNC slow (>= 2 of 3),
    PB-ASYNC normal (0 of 3), 0 mismatches in every window of the three runs, and PB-ASYNC's GPU
    completion isolation in that run (its hetero GPU-return P50 <= 1 ms, and A's at least 5x
    PB-ASYNC's; #57's limits).
  - Round 1 (A, PB-SYNC, PB-ASYNC) is a screen only and never yields "async solved". With the
    pattern, round 2 replicates in the reversed order (PB-ASYNC, PB-SYNC, A); round 2 alone must
    show the same pattern for "strong causal evidence", else INCONCLUSIVE. Without it, no round
    2: A with any slow window -> INCONCLUSIVE (negative control); PB-SYNC <= 1 of 3 ->
    INCONCLUSIVE (positive control); PB-SYNC and PB-ASYNC both slow -> async does not solve the
    slow state; anything else -> mixed. Never a third round.
  - A second crash of any run (run_all.sh stops on it too) invalidates the campaign.
Every other record is descriptive, per cell, run and hetero window: short P99, aggregate req/s,
GPU return P50, the ANE stages (features, pre, submit, completion to Python, waiter handoff,
post, tail), ANE and GPU thread CPU per forward, client and Core ML callback thread CPU, callback
latency and mismatches. #83's helpers (research/coreml-prebind-predict/scripts/, by path) do
the window bookkeeping.

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


design = _load("async_predict_design", HERE / "design.py")  # by path: other experiments have a design.py
pb83 = _load("pb83_analyze", EXP.parent / "coreml-prebind-predict" / "scripts" / "analyze.py")
gate = pb83.gate
derive = pb83.derive
gate57 = pb83.mix77.gate57
LIMITS = gate57.LIMITS  # return_p50_max_ms 1.0, return_min_gain 5.0

CELL_RE = "|".join(re.escape(c) for c in sorted(design.CELLS, key=len, reverse=True))
RUN_NAME = re.compile(rf"^{design.MODEL}-(?P<cell>{CELL_RE})-r(?P<round>\d+)\.json\.gz$")
FAILED_NAME = re.compile(rf"^(?P<run>{design.MODEL}-(?:{CELL_RE})-r\d+)\.(?P<attempt>\d+)\.log$")

PATTERN_ROUND1 = "round 1 shows the pattern (a screen only, not a result); next: round 2 replication"
STRONG = "strong causal evidence: PB-ASYNC is the 1.5 leading candidate"
ROUND2_NOT_REPRODUCED = (
    "INCONCLUSIVE: round 2 did not reproduce the pattern; async is not claimed to solve the slow state; "
    "no productionization"
)
NEGATIVE_CONTROL_SLOW = "INCONCLUSIVE: negative control slow, invalid for causal interpretation"
POSITIVE_NOT_REPRODUCED = "INCONCLUSIVE: positive control not reproduced"
ASYNC_SLOW = "async does not solve the slow state: resume #89"
MIXED = "mixed: not interpreted; human decision"
STAGES = ("features", "pre", "submit", "completion", "handoff", "post", "tail", "predict")


# ------------------------------------------------------------------ the rule


def isolated(a_p50: float | None, async_p50: float | None) -> bool:
    """PB-ASYNC's GPU completion isolation in one run: its hetero GPU-return P50 <= 1 ms and A's
    P50 at least 5x PB-ASYNC's (A of the same round)."""
    if a_p50 is None or async_p50 is None:
        return False
    return async_p50 <= LIMITS["return_p50_max_ms"] and (
        async_p50 == 0 or a_p50 / async_p50 >= LIMITS["return_min_gain"]
    )


def pattern(rnd: int, slow: dict, mismatches: dict, gpu_p50: dict) -> dict:
    """The five conditions on round `rnd`'s three runs."""

    def n(cell):
        return sum(bool(x) for x in slow[(cell, rnd)])

    conds = {
        "A_normal": n("A") == 0,
        "PB_SYNC_slow": n("PB-SYNC") >= 2,
        "PB_ASYNC_normal": n("PB-ASYNC") == 0,
        "no_mismatches": all(mismatches.get((c, rnd), 0) == 0 for c in design.CELLS),
        "PB_ASYNC_isolation": isolated(gpu_p50.get(("A", rnd)), gpu_p50.get(("PB-ASYNC", rnd))),
    }
    return {"slow": {c: n(c) for c in design.CELLS}, "conditions": conds, "holds": all(conds.values())}


def decide(slow: dict, mismatches: dict, gpu_p50: dict) -> dict:
    """slow: {(cell, round): per-window slow flags}; mismatches: {(cell, round): all-window
    mismatches}; gpu_p50: {(cell, round): the run's hetero GPU-return P50 ms} for A and PB-ASYNC.
    Returns the outcome key, its text, the per-round pattern and the next round's runs
    [(cell, round)] or None."""

    def res(outcome, text, nxt=None, **kw):
        return {"outcome": outcome, "text": text, "next": nxt, **kw}

    round1 = [(c, 1) for c in design.ROUND1]
    if any(k not in slow for k in round1):
        return res(None, "running: round 1", round1)
    p1 = pattern(1, slow, mismatches, gpu_p50)
    if p1["holds"]:
        round2 = [(c, 2) for c in design.ROUND2]
        if any(k not in slow for k in round2):
            return res("pattern_round1", PATTERN_ROUND1, round2, round2=list(design.ROUND2), round1=p1)
        p2 = pattern(2, slow, mismatches, gpu_p50)
        if p2["holds"]:
            return res("strong", STRONG, round2=list(design.ROUND2), round1=p1, round2_pattern=p2)
        return res(
            "round2_not_reproduced", ROUND2_NOT_REPRODUCED, round2=list(design.ROUND2), round1=p1, round2_pattern=p2
        )
    counts = p1["slow"]
    if counts["A"] > 0:
        return res("negative_control_slow", NEGATIVE_CONTROL_SLOW, round1=p1)
    if counts["PB-SYNC"] < 2:
        return res("positive_control_not_reproduced", POSITIVE_NOT_REPRODUCED, round1=p1)
    if counts["PB-ASYNC"] >= 2:
        return res("async_slow", ASYNC_SLOW, round1=p1)
    return res("mixed", MIXED, round1=p1)


# ------------------------------------------------------------------ records


def load(path: Path) -> dict:
    run = pb83.load(path)
    for w in run["part_a"]["windows"]:  # per-request latencies: not used here, and the bulk of a run
        for s in w["streams"].values():
            s.pop("latency_ms", None)
    return run


def check_run(run: dict, cell: str, path: Path) -> None:
    res = run["research"]
    assert res["experiment"] == "coreml-async-predict" and res["cell"] == cell, path
    assert run["args"]["short"] == design.SHORT and run["args"]["long"] == design.LONG, path
    assert run["args"]["seconds"] == design.SECONDS and run["args"]["cycles"] == design.CYCLES, path
    assert len(gate.hetero_by_cycle(run)) == design.CYCLES, path


def _med(x):
    x = [v for v in x if v is not None]
    return float(np.median(x)) if x else None


def _forward_rows(run: dict, dev: str, bounds) -> np.ndarray | None:
    x = run["research"]["forwards"].get(dev)
    if x is None:
        return None
    a = np.asarray(x, dtype=np.int64).reshape(-1, 3)
    return a[derive.in_windows(a[:, 0], bounds)]


def stages(binding: str, forward, predict) -> dict:
    """Stage durations (ns) of one ANE forward. predict: the Stamped row (entry, four binding
    stamps, exit). PB-SYNC: pre = python_before - entry, completion = native (native_after -
    native_before), handoff = GIL re-acquire (python_after - native_after), post = exit -
    python_after. PB-ASYNC: pre = submit_before - entry, submit = submit_after - submit_before,
    completion = callback_entry - submit_before, handoff = wake - callback_entry, post = exit -
    wake. coremltools: only features, tail and the whole predict."""
    s0, s1 = int(forward[0]), int(forward[1])
    entry, p1, p2, p3, p4, exit_ = (int(x) for x in predict)
    out = dict.fromkeys(STAGES)
    out.update(features=entry - s0, tail=s1 - exit_, predict=exit_ - entry)
    if binding == "prebind" and p2:
        out.update(pre=p1 - entry, completion=p3 - p2, handoff=p4 - p3, post=exit_ - p4)
    elif binding == "prebind_async" and p1:
        out.update(pre=p1 - entry, submit=p2 - p1, completion=p3 - p1, handoff=p4 - p3, post=exit_ - p4)
    return out


def stage_values(run: dict, bounds) -> dict[str, list]:
    """Every ANE forward starting inside bounds: its stages in ms."""
    res = run["research"]
    fw = _forward_rows(run, "ane", bounds)
    pr = np.asarray(res["predicts"] or [], dtype=np.int64).reshape(-1, 6)
    st: dict[str, list] = {k: [] for k in STAGES}
    if fw is not None and len(fw):
        idx = derive.join_predicts(fw[:, :2], pr)
        for f, i in zip(fw[idx >= 0], idx[idx >= 0]):
            for k, v in stages(res["ane_predict"], f[:2], pr[i]).items():
                if v is not None:
                    st[k].append(v / 1e6)
    return st


def stage_p50(values: dict[str, list]) -> dict:
    return {k: _med(v) for k, v in values.items()}


def window_records(run: dict) -> list[dict]:
    """One record per hetero window, in cycle order."""
    res = run["research"]
    hw = gate.hetero_by_cycle(run)
    bounds = gate57.hetero_bounds(run)
    rw = sorted(res["windows"], key=lambda w: w["start_ns"])
    cbs = np.asarray((res.get("callbacks") or {}).get("hetero") or [], np.int64).reshape(-1, 3)
    out = []
    for i, k in enumerate(sorted(hw)):
        w = hw[k]
        lo, hi = bounds[i]
        r = next((x for x in rw if x["start_ns"] == lo and x["end_ns"] == hi), None)
        cpu = {}
        for s in ("short", "long"):  # each client's own closing snapshot, else R1's last one
            after = ((r or {}).get("after_by_stream") or {}).get(s) or (r or {}).get("after")
            ns = derive.thread_cpu_delta(r["before"], after).get(f"client-{s}") if after else None
            cpu[f"client-{s}"] = None if ns is None else ns / 1e6
        if r and r.get("after"):
            d = derive.thread_cpu_delta(r["before"], r["after"])
            cpu["coreml-callback"] = (
                d.get("coreml-callback", 0) / 1e6 if res["ane_predict"] == "prebind_async" else None
            )
            cpu["other"] = d.get("other", 0) / 1e6
        cpu_fwd = {}
        for dev in ("ane", "gpu"):
            a = _forward_rows(run, dev, [(lo, hi)])
            cpu_fwd[dev] = float(a[:, 2].mean() / 1e6) if a is not None and len(a) else None
        sv = stage_values(run, [(lo, hi)])
        inw = cbs[(cbs[:, 0] >= lo) & (cbs[:, 0] < hi)] if len(cbs) else cbs
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
                "gpu_return_p50_ms": _gpu_return_p50(run, [(lo, hi)]),
                "thread_cpu_ms": cpu,
                "thread_cpu_per_forward_ms": cpu_fwd,
                "ane_stages_p50_ms": stage_p50(sv),
                "callbacks": None
                if res["ane_predict"] != "prebind_async"
                else {
                    "n": int(len(inw)),
                    "errors": int(inw[:, 2].sum()) if len(inw) else 0,
                    "threads": int(len(set(inw[:, 1].tolist()))) if len(inw) else 0,
                    "completion_p99_ms": float(np.percentile(sv["completion"], 99)) if sv["completion"] else None,
                },
            }
        )
    return out


def _gpu_return_p50(run: dict, bounds) -> float | None:
    g = run["gpu_return"]
    x = [u / 1e3 for t, u in zip(g["received_ns"], g["return_us"], strict=True) if any(a <= t < b for a, b in bounds)]
    return float(np.median(x)) if x else None


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
    bounds = gate57.hetero_bounds(run)
    pr = np.asarray(res["predicts"] or [], dtype=np.int64).reshape(-1, 6)
    pr = pr[derive.in_windows(pr[:, 0], bounds)]
    b = pb83.short_bucket({"crossings": res["crossings"]}, design.SHORT)
    cb = res.get("callbacks")
    return {
        "windows": ws,
        "slow_windows": sum(w["slow"] for w in ws),
        "gpu_return_p50_ms": summ["gpu_return_ms"]["p50"],
        "mismatches_all_windows": summ["mismatches_all_windows"],
        "routing_failures": routing,
        "ane_stages_p50_ms": stage_p50(stage_values(run, bounds)),
        "forwards": pb83.forward_cpu([run]),
        "hetero_predicts": int(len(pr)),
        "hetero_predicts_with_binding_stamps": int((pr[:, 1] > 0).sum()) if len(pr) else 0,
        "crossings_short_bucket": (res["crossings"] or {}).get(str(b)) if b else None,
        "callbacks": None
        if cb is None
        else {
            "submits": cb["submits"],
            "callbacks": cb["total"],
            "errors": cb["errors"],
            "anomalies": len(cb["anomalies"]),
            "thread_ids": len(cb["thread_ids"]),
            "queue_labels": cb["queue_labels"],
            "per_forward_at_load": res.get("callbacks_per_forward_at_load"),
        },
        "backing_modes": None
        if res.get("backings") is None
        else sorted({m for x in res["backings"].values() for m in x["modes"].values()}),
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
    gpu_p50 = {k: r["gpu_return_p50_ms"] for k, r in runs.items() if k[0] in ("A", "PB-ASYNC")}
    d = decide(
        {k: [w["slow"] for w in r["windows"]] for k, r in runs.items()},
        {k: r["mismatches_all_windows"] for k, r in runs.items()},
        gpu_p50,
    )
    crash = crashes(raw)
    second = sorted(r for r, k in crash.items() if k >= 2)
    if second:
        d = {"outcome": "invalid", "text": f"invalid: second crash of {', '.join(second)}; no verdict", "next": None}
    designed = {(c, 1) for c in design.ROUND1} | {(c, 2) for c in d.get("round2", [])}
    cells: dict[str, dict] = {}
    for c in design.CELLS:
        keys = [k for k in sorted(present) if k[0] == c]
        if not keys:
            continue
        rs = [loaded[k] for k in keys]
        sv: dict[str, list] = {s: [] for s in STAGES}
        for r in rs:
            for s, v in stage_values(r, gate57.hetero_bounds(r)).items():
                sv[s] += v
        cells[c] = {
            "rounds": [k[1] for k in keys],
            "slow_windows": sum(runs[k]["slow_windows"] for k in keys),
            "windows": sum(len(runs[k]["windows"]) for k in keys),
            "mismatches": sum(runs[k]["mismatches_all_windows"] for k in keys),
            "gpu_return_p50_ms_pooled": gate57.placement_summary(rs)["gpu_return_ms"]["p50"],
            "ane_stages_p50_ms_pooled": stage_p50(sv),
            "forwards_pooled": pb83.forward_cpu(rs),
        }
    phase0 = raw / "phase0.json"
    return {
        "design": {
            "model": design.MODEL,
            "short": design.SHORT,
            "long": design.LONG,
            "round1": list(design.ROUND1),
            "round2": list(design.ROUND2),
            "slow_short_p99_ms": design.SLOW_P99_MS,
            "cycles": design.CYCLES,
            "seconds": design.SECONDS,
            "isolation": {k: LIMITS[k] for k in ("return_p50_max_ms", "return_min_gain")},
        },
        "phase0": json.loads(phase0.read_text()) if phase0.exists() else None,
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
    L = ["# Async prebound predict screen (laya, L128 / L512, R1's full protocol)\n"]
    p0 = res.get("phase0")
    if not p0:
        L.append("Phase 0 (`raw/phase0.json`): not run yet\n")
    else:
        L.append(
            "Phase 0 (`raw/phase0.json`): "
            + ", ".join(f"{k} {'PASS' if v else '**FAIL**'}" for k, v in p0["gates"].items())
            + "\n"
        )
        st = p0.get("stamped") or {}
        if st.get("built"):
            d = st["native_completion_to_callback_entry"]
            L.append(
                f"PB-ASYNC-STAMPED native completion → Python callback entry ({st['label']}): P50 "
                f"{f(d.get('p50_ms'), 3)} ms, P99 {f(d.get('p99_ms'), 3)} ms, n = {d['n']}\n"
            )
        elif st:
            L.append(f"PB-ASYNC-STAMPED: not built ({st.get('reason')})\n")
    L.append(f"Outcome: {res['outcome']}\n")
    nxt = res["next"]
    L.append(f"Next: {', '.join(f'{c} r{r}' for c, r in nxt) if nxt else 'none'}\n")
    L.append(f"Crashed runs and re-runs (`raw/failed/`): {', '.join(res['failed_runs']) or 'none'}\n")
    L.append(f"Runs not used by the rule: {', '.join(res['unused_runs']) or 'none'}\n")
    L += [
        "\n## Hetero windows\n",
        f"Slow: short P99 ≥ {res['design']['slow_short_p99_ms']} ms. Thread CPU: ms per window; ANE and GPU: ms "
        "per forward. Stages: P50 ms; completion is callback entry − submit (PB-ASYNC) or the native predict "
        "(PB-SYNC); handoff is wake − callback entry (PB-ASYNC) or the GIL re-acquire (PB-SYNC).\n",
        "| run | cycle | short P99 | slow | aggregate req/s | GPU return P50 | mismatches | client-short CPU "
        "| client-long CPU | callback CPU | ANE CPU/fwd | GPU CPU/fwd | features | pre | submit | completion "
        "| handoff | post | tail | callbacks |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in res["runs"].items():
        for w in r["windows"]:
            st, c, t, cb = w["ane_stages_p50_ms"], w["thread_cpu_ms"], w["thread_cpu_per_forward_ms"], w["callbacks"]
            L.append(
                f"| {name} | {w['cycle']} | {f(w['short_p99_ms'])} | {'**slow**' if w['slow'] else 'normal'} "
                f"| {f(w['aggregate_req_s'], 1)} | {f(w['gpu_return_p50_ms'], 3)} | {w['mismatches']} "
                f"| {f(c.get('client-short'), 0)} | {f(c.get('client-long'), 0)} | {f(c.get('coreml-callback'), 0)} "
                f"| {f(t['ane'], 3)} | {f(t['gpu'], 3)} | "
                + " | ".join(
                    f(st[k], 3) for k in ("features", "pre", "submit", "completion", "handoff", "post", "tail")
                )
                + f" | {'–' if cb is None else str(cb['n']) + ' (' + str(cb['errors']) + ' errors)'} |"
            )
    L += [
        "\n## Runs\n",
        "| run | slow windows | GPU return P50 ms | mismatches | routing failures | predicts with binding stamps "
        "| submits / callbacks / errors / anomalies | callback threads | backings | wall s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in res["runs"].items():
        cb = r["callbacks"]
        counts = "–" if cb is None else f"{cb['submits']} / {cb['callbacks']} / {cb['errors']} / {cb['anomalies']}"
        L.append(
            f"| {name} | {r['slow_windows']} of {len(r['windows'])} | {f(r['gpu_return_p50_ms'], 3)} "
            f"| {r['mismatches_all_windows']} | {len(r['routing_failures'])} "
            f"| {r['hetero_predicts_with_binding_stamps']} of {r['hetero_predicts']} "
            f"| {counts} "
            f"| {'–' if cb is None else cb['thread_ids']} | {', '.join(r['backing_modes'] or []) or '–'} "
            f"| {f(r['run_wall_s'], 0)} |"
        )
    L += [
        "\n## Cells, pooled over their runs\n",
        "| cell | rounds | slow windows | mismatches | GPU return P50 ms | features | pre | submit | completion "
        "| handoff | post | tail | ANE CPU/fwd | GPU CPU/fwd |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c, x in res["cells"].items():
        st, fw = x["ane_stages_p50_ms_pooled"], x["forwards_pooled"]
        L.append(
            f"| {c} | {', '.join(map(str, x['rounds']))} | {x['slow_windows']} of {x['windows']} | {x['mismatches']} "
            f"| {f(x['gpu_return_p50_ms_pooled'], 3)} | "
            + " | ".join(f(st[k], 3) for k in ("features", "pre", "submit", "completion", "handoff", "post", "tail"))
            + f" | {f(None if fw['ane'] is None else fw['ane']['thread_cpu_ms_mean'], 3)} "
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
