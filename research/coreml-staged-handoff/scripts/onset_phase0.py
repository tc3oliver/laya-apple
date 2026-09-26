"""Phase 0: is the window-onset E-core residency hetero-specific? (existing #96 / #99 raw data only)

    uv run python research/coreml-staged-handoff/scripts/onset_phase0.py [--check]

Post-hoc closure analysis of raw data that already exists; it runs nothing and changes no earlier
verdict. Inputs (read-only):
  research/coreml-async-transient/raw/laya-{A-r1,A-r2,PB-ASYNC-r1,PB-ASYNC-r2}.json.gz   (#96)
  research/coreml-dependency-qos/raw/laya-{B,O,Q}-r1.json.gz                            (#99)
Cells: A = the production synchronous coremltools path (A-r1, A-r2); PB-ASYNC family = the async
prebound path (PB-ASYNC-r1, PB-ASYNC-r2, and #99's B, O, Q, which are PB-ASYNC and PB-ASYNC plus
QoS variants). Outputs: phase0.{md,json} next to scripts/.

Everything below was fixed before this script's output was read.

Data: the recount series (cumulative PROC_PIDTHREADCOUNTS per thread and perf level, sampled every
100 ms); the delta between two consecutive rows of a thread is assigned to the interval midpoint.
The counter helpers (intervals, derive, switch_time, group names) are imported from
research/coreml-dependency-qos/scripts/placement_posthoc.py so the definitions are identical.

Windows: taken from the record's own `windows_t` (exact start_ns per stream and instance), grouped
by (start_ns, end_ns, instance) in time order. An auto window with only the short stream is
solo_short, only the long stream solo_long, both hetero; a gpu-instance window is gpu_only. The
sequence must equal `research.window_order`, or the script fails. t0 = the window's start_ns.
gpu_only runs on the separate GPU-only instance, whose threads are not sampled: listed, not
analysed.

Groups: chain (laya-ane-dispatch, client-short, coreml-callback; A has no coreml-callback),
parent-other (MainThread, client-long, laya-gpu-dispatch), worker (the auto instance's GPU worker
process, all threads named gpu-worker). parent-active = chain + parent-other. A thread is active in
a window if it used >= 20 ms of CPU in [t0, t0 + 4 s); only active threads enter an aggregate,
CPU-weighted.

Per window and bucket (s from t0: 0-0.5, 0.5-1, 1-2, 2-4, 4-8, 10-20), for parent-active, chain,
parent-other and worker: CPU ms, E share (E CPU / all CPU), and the relative effective cycle rate
cycles / CPU ns on P and on E separately (a relative figure only, not a frequency).

E->P switch time, per window for the parent-active and the worker aggregate: the placement_posthoc
rule on 0.5 s bins over [t0, t0 + 20): the start of the first bin from which every later bin with
>= 2 ms of CPU has E share < 0.5; "never" if none; reported as "P from onset" if that is bin 0.

Reading rules (parent-active aggregate only):
  readable       parent-active CPU >= 20 ms in both 0-0.5 and 0.5-1.
  E-onset        readable, and E share >= 0.5 in 0-0.5 or in 0.5-1.
  P-onset        readable, and E share <= 0.25 in both 0-0.5 and 0.5-1.
  returns <= 1 s switch time is a number <= 1.0 (bin start 0, 0.5 or 1.0).

Classification per cell (all runs of the cell, both cycles; solo = solo_short and solo_long):
  hetero windows must all be readable, else G3.
  G2 hetero-specific: every readable solo window is P-onset, every hetero window is E-onset, and
     every run has at least one readable solo window.
  G1 general activation: every run has at least one E-onset solo window; every E-onset solo window
     returns <= 1 s; every hetero window is E-onset and does not return <= 1 s (switch > 1.0 or
     never).
  G3 insufficient / mixed: anything else.
  Unreadable solo windows are listed and excluded from the solo evidence.
Overall: the common class if both cells agree, otherwise G3 (mixed), with both cells shown.
Separately reported: whether each of A's hetero windows is E-onset, with its bucket E shares and
switch time (A had no host-slow transient in #96).

Limits: only the sampled threads (the listed parent threads and the auto GPU worker); not
system-wide; no cause is observable in these counters; post-hoc.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RESEARCH = EXP.parent
_spec = importlib.util.spec_from_file_location(
    "placement_posthoc", RESEARCH / "coreml-dependency-qos" / "scripts" / "placement_posthoc.py"
)
pp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pp)

RUNS = {
    "A-r1": ("A", RESEARCH / "coreml-async-transient" / "raw" / "laya-A-r1.json.gz"),
    "A-r2": ("A", RESEARCH / "coreml-async-transient" / "raw" / "laya-A-r2.json.gz"),
    "PB-ASYNC-r1": ("PB-ASYNC", RESEARCH / "coreml-async-transient" / "raw" / "laya-PB-ASYNC-r1.json.gz"),
    "PB-ASYNC-r2": ("PB-ASYNC", RESEARCH / "coreml-async-transient" / "raw" / "laya-PB-ASYNC-r2.json.gz"),
    "B-r1": ("PB-ASYNC", RESEARCH / "coreml-dependency-qos" / "raw" / "laya-B-r1.json.gz"),
    "O-r1": ("PB-ASYNC", RESEARCH / "coreml-dependency-qos" / "raw" / "laya-O-r1.json.gz"),
    "Q-r1": ("PB-ASYNC", RESEARCH / "coreml-dependency-qos" / "raw" / "laya-Q-r1.json.gz"),
}
CELLS = ("A", "PB-ASYNC")
S = 1_000_000_000
BUCKETS = (("0-0.5", 0.0, 0.5), ("0.5-1", 0.5, 1.0), ("1-2", 1.0, 2.0), ("2-4", 2.0, 4.0),
           ("4-8", 4.0, 8.0), ("10-20", 10.0, 20.0))  # fmt: skip
ACTIVE_MS = 20.0
READ_MS = 20.0
WIN_S, BIN_S = 20.0, 0.5
E_DOM, P_DOM = 0.5, 0.25
AGGS = ("parent-active", "chain", "parent-other", "worker")


def windows(rec: dict) -> list[dict]:
    """Windows in time order, named from their streams and checked against research.window_order."""
    seen: dict[tuple, set] = {}
    for w in rec["windows_t"]:
        seen.setdefault((w["start_ns"], w["end_ns"], w["instance"]), set()).add(w["stream"])
    out = []
    for (lo, hi, inst), streams in sorted(seen.items()):
        if inst == "gpu":
            name = "gpu_only"
        else:
            name = {frozenset({"short"}): "solo_short", frozenset({"long"}): "solo_long"}.get(
                frozenset(streams), "hetero"
            )
        out.append({"start_ns": lo, "end_ns": hi, "instance": inst, "condition": name})
    order = rec["research"]["window_order"]
    if [w["condition"] for w in out] != [n for _, n in order]:
        raise SystemExit(f"windows_t does not match window_order: {[w['condition'] for w in out]}")
    for w, (cyc, _) in zip(out, order):
        w["cycle"] = cyc
    return out


def sums(rc: dict, s: dict, t0: int) -> tuple[dict, dict, list]:
    b = {n: pp.acc() for n, _, _ in BUCKETS}
    onset = pp.acc()
    bins = [pp.acc() for _ in range(int(WIN_S / BIN_S))]
    for mid, p, e in pp.intervals(rc, s):
        x = (mid - t0) / S
        for n, lo, hi in BUCKETS:
            if lo <= x < hi:
                pp.add(b[n], p, e)
        if 0.0 <= x < 4.0:
            pp.add(onset, p, e)
        if 0.0 <= x < WIN_S:
            pp.add(bins[int(x // BIN_S)], p, e)
    return b, onset, bins


def merge(xs: list[dict]) -> dict:
    t = pp.acc()
    for x in xs:
        pp.add(t, (x["p_instr"], x["p_cyc"], x["p_cpu"]), (x["e_instr"], x["e_cyc"], x["e_cpu"]))
    return t


def switch(bins: list[dict]):
    s = pp.switch_time([(i * BIN_S, pp.derive(v)) for i, v in enumerate(bins)])
    return "P from onset" if s == 0.0 else s


def slim(d: dict) -> dict:
    return {k: d[k] for k in ("cpu_ms", "e_share", "rate_p", "rate_e")}


def window(rc: dict, parent: int, w: dict) -> dict:
    t0 = w["start_ns"]
    th = []
    for s in rc["series"].values():
        g = pp.group_of(s["name"], s["pid"], parent)
        b, onset, bins = sums(rc, s, t0)
        on = pp.derive(onset)
        th.append({"name": s["name"], "tid": s["tid"], "group": g, "onset_ms": on["cpu_ms"], "b": b, "bins": bins})
    act = [t for t in th if t["onset_ms"] >= ACTIVE_MS]
    members = {
        "parent-active": [t for t in act if t["group"] in ("chain", "parent-other")],
        "chain": [t for t in act if t["group"] == "chain"],
        "parent-other": [t for t in act if t["group"] == "parent-other"],
        "worker": [t for t in act if t["group"] == "worker"],
    }
    aggs = {}
    for g, m in members.items():
        bk = {n: slim(pp.derive(merge([t["b"][n] for t in m]))) for n, _, _ in BUCKETS}
        sw = switch([merge([t["bins"][i] for t in m]) for i in range(int(WIN_S / BIN_S))]) if m else None
        aggs[g] = {"buckets": bk, "switch_s": sw}
    pa = aggs["parent-active"]["buckets"]
    first = (pa["0-0.5"], pa["0.5-1"])
    readable = all(x["cpu_ms"] >= READ_MS for x in first)
    e_onset = readable and any(x["e_share"] >= E_DOM for x in first)
    p_onset = readable and all(x["e_share"] <= P_DOM for x in first)
    sw = aggs["parent-active"]["switch_s"]
    returns = sw == "P from onset" or (isinstance(sw, float) and sw <= 1.0)
    return {
        "cycle": w["cycle"],
        "condition": w["condition"],
        "start_ns": t0,
        "sampled": True,
        "active_threads": [
            f"{t['group']}:{t['name']}:{t['tid']}({t['onset_ms']:.0f} ms)"
            for t in sorted(act, key=lambda t: (t["group"], t["name"], t["tid"]))
        ],
        "aggregates": aggs,
        "readable": readable,
        "e_onset": e_onset,
        "p_onset": p_onset,
        "returns_within_1s": returns,
    }


def classify(runs: dict) -> tuple[str, list[str]]:
    why = []
    het = [(k, w) for k, r in runs.items() for w in r["windows"] if w["condition"] == "hetero"]
    solo = {k: [w for w in r["windows"] if w["condition"] in ("solo_short", "solo_long")] for k, r in runs.items()}
    if not all(w["readable"] for _, w in het):
        return "G3", ["a hetero window is unreadable"]
    readable_solo = {k: [w for w in v if w["readable"]] for k, v in solo.items()}
    if (
        all(readable_solo.values())
        and all(w["p_onset"] for v in readable_solo.values() for w in v)
        and all(w["e_onset"] for _, w in het)
    ):
        return "G2", ["every readable solo window P-onset; every hetero window E-onset"]
    e_solo = {k: [w for w in v if w["e_onset"]] for k, v in solo.items()}
    if (
        all(e_solo.values())
        and all(w["returns_within_1s"] for v in e_solo.values() for w in v)
        and all(w["e_onset"] and not w["returns_within_1s"] for _, w in het)
    ):
        return "G1", ["every run has an E-onset solo window returning <= 1 s; hetero stays E longer"]
    for k, v in solo.items():
        for w in v:
            if not w["readable"]:
                why.append(f"{k} c{w['cycle']} {w['condition']} unreadable (excluded)")
            elif w["e_onset"]:
                why.append(
                    f"{k} c{w['cycle']} {w['condition']} E-onset, switch {w['aggregates']['parent-active']['switch_s']}"
                )
            elif not w["p_onset"]:
                why.append(f"{k} c{w['cycle']} {w['condition']} neither E- nor P-onset")
    for k, w in het:
        if not w["e_onset"]:
            why.append(f"{k} c{w['cycle']} hetero not E-onset")
        elif w["returns_within_1s"]:
            why.append(f"{k} c{w['cycle']} hetero returns to P within 1 s")
    return "G3", why


def analyse() -> dict:
    out = {"runs": {}, "cells": {}}
    for run, (cell, path) in RUNS.items():
        rec = json.load(gzip.open(path))
        rc = rec["research"]["recount"]
        parent = next(s["pid"] for s in rc["series"].values() if s["name"] == "laya-ane-dispatch")
        ws = []
        for w in windows(rec):
            if w["instance"] != "auto":
                ws.append(
                    {"cycle": w["cycle"], "condition": w["condition"], "start_ns": w["start_ns"], "sampled": False}
                )
            else:
                ws.append(window(rc, parent, w))
        out["runs"][run] = {"cell": cell, "parent_pid": parent, "windows": [w for w in ws if w["sampled"]],
                            "not_sampled": [f"c{w['cycle']} {w['condition']}" for w in ws if not w["sampled"]]}  # fmt: skip
    for cell in CELLS:
        cls, why = classify({k: v for k, v in out["runs"].items() if v["cell"] == cell})
        out["cells"][cell] = {"class": cls, "reasons": why}
    cs = {c["class"] for c in out["cells"].values()}
    out["overall"] = cs.pop() if len(cs) == 1 else "G3"
    out["a_hetero"] = [
        {
            "run": run,
            "cycle": w["cycle"],
            "e_onset": w["e_onset"],
            "e_share": {n: w["aggregates"]["parent-active"]["buckets"][n]["e_share"] for n, _, _ in BUCKETS},
            "worker_e_share": {n: w["aggregates"]["worker"]["buckets"][n]["e_share"] for n, _, _ in BUCKETS},
            "switch_s": w["aggregates"]["parent-active"]["switch_s"],
            "worker_switch_s": w["aggregates"]["worker"]["switch_s"],
        }
        for run, r in out["runs"].items()
        if r["cell"] == "A"
        for w in r["windows"]
        if w["condition"] == "hetero"
    ]
    return out


f = pp.f


def render(res: dict) -> str:
    bn = [n for n, _, _ in BUCKETS]
    L = [
        "# Phase 0: is the window-onset E-core residency hetero-specific? (#96 / #99 raw data only)\n",
        "Post-hoc closure analysis of existing raw data; it changes no earlier verdict. Definitions, reading "
        "rules and the G1/G2/G3 classification are in `scripts/onset_phase0.py`, fixed before its output "
        "was read.\n",
        f"Overall: **{res['overall']}**. "
        + "; ".join(f"{c}: **{x['class']}**" for c, x in res["cells"].items())
        + ".\n",
    ]
    for c, x in res["cells"].items():
        L.append(f"- {c}: " + ("; ".join(x["reasons"]) or "–"))
    L += [
        "\n## Production A: hetero windows at onset\n",
        "A had no host-slow transient in #96. Parent-active E share by bucket (worker in parentheses), "
        "and the E->P switch time.\n",
        "| run | cycle | E-onset | " + " | ".join(bn) + " | switch parent / worker |",
        "|---|---|---|" + "---|" * len(bn) + "---|",
    ]
    for a in res["a_hetero"]:
        L.append(
            f"| {a['run']} | {a['cycle']} | {'yes' if a['e_onset'] else 'no'} | "
            + " | ".join(f"{f(a['e_share'][n])} ({f(a['worker_e_share'][n])})" for n in bn)
            + f" | {f(a['switch_s'], 1)} / {f(a['worker_switch_s'], 1)} |"
        )
    L += [
        "\n## Every sampled window: reading summary (parent-active)\n",
        "| run | cell | cycle | condition | 0-0.5 E (ms) | 0.5-1 E (ms) | readable | onset | switch parent / worker |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for run, r in res["runs"].items():
        for w in r["windows"]:
            pa = w["aggregates"]["parent-active"]
            b0, b1 = pa["buckets"]["0-0.5"], pa["buckets"]["0.5-1"]
            on = "E" if w["e_onset"] else "P" if w["p_onset"] else "mixed" if w["readable"] else "–"
            L.append(
                f"| {run} | {r['cell']} | {w['cycle']} | {w['condition']} | {f(b0['e_share'])} ({f(b0['cpu_ms'], 1)}) "
                f"| {f(b1['e_share'])} ({f(b1['cpu_ms'], 1)}) | {'yes' if w['readable'] else 'no'} | {on} "
                f"| {f(pa['switch_s'], 1)} / {f(w['aggregates']['worker']['switch_s'], 1)} |"
            )
    L.append("\nNot sampled (separate GPU-only instance): " + "; ".join(
        f"{run}: {', '.join(r['not_sampled'])}" for run, r in res["runs"].items()) + ".\n")  # fmt: skip
    L += [
        "## Active threads per window (>= 20 ms CPU in [t0, t0 + 4 s))\n",
        "| run | cycle | condition | active threads (group:name:tid, onset CPU) |",
        "|---|---|---|---|",
    ]
    for run, r in res["runs"].items():
        for w in r["windows"]:
            L.append(f"| {run} | {w['cycle']} | {w['condition']} | {', '.join(w['active_threads']) or '–'} |")
    L += [
        "\n## Per bucket: E share, CPU ms, relative cycle rate P / E (cycles per CPU ns; relative, not a frequency)\n",
        "| run | cycle | condition | aggregate | " + " | ".join(bn) + " |",
        "|---|---|---|---|" + "---|" * len(bn),
    ]
    for run, r in res["runs"].items():
        for w in r["windows"]:
            for g in AGGS:
                bk = w["aggregates"][g]["buckets"]
                if not any(bk[n]["cpu_ms"] for n in bn):
                    continue
                L.append(
                    f"| {run} | {w['cycle']} | {w['condition']} | {g} | "
                    + " | ".join(
                        f"{f(bk[n]['e_share'])} · {f(bk[n]['cpu_ms'], 0)} · {f(bk[n]['rate_p'])}/{f(bk[n]['rate_e'])}"
                        for n in bn
                    )
                    + " |"
                )
    L += [
        "\n## Limits\n",
        "- Only the sampled threads: the listed parent threads (MainThread, client-short, client-long, "
        "laya-ane-dispatch, laya-gpu-dispatch, coreml-callback where it exists) and the auto instance's GPU "
        "worker. Other parent threads and the GPU-only instance are not in the data.",
        "- Not system-wide: other processes and core occupancy are not observed.",
        "- No cause is observable: these counters show where threads ran, not why they were placed there.",
        "- Post-hoc on #96 and #99 raw data; it changes no earlier verdict.",
    ]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="verify the outputs are up to date")
    a = ap.parse_args()
    res = analyse()
    md, js = render(res), json.dumps(res, indent=1, sort_keys=True) + "\n"
    pm, pj = EXP / "phase0.md", EXP / "phase0.json"
    if a.check:
        stale = [p.name for p, c in ((pm, md), (pj, js)) if not p.exists() or p.read_text() != c]
        if stale:
            raise SystemExit(f"stale: {stale}")
        print("outputs are up to date")
        return
    pm.write_text(md)
    pj.write_text(js)
    print(f"overall: {res['overall']}; " + ", ".join(f"{c} {x['class']}" for c, x in res["cells"].items()))


if __name__ == "__main__":
    main()
