"""The staged-handoff screen: results.json and tables.md from raw/ (../criteria.md).

    uv run python research/coreml-staged-handoff/scripts/analyze.py [--check] [--raw DIR --out DIR]

Inputs: raw/laya-<cell>-r<rep>.json.gz (run_config.py; design.py: cells A, B, H32, H64; reps 1-2
round 1, 3-5 round 2) and raw/failed/<run>.<attempt>.log (run_all.sh's crash rule). The thresholds
are criteria.md's; the constants below carry them. #94's analyze.py
(research/coreml-async-transient/scripts/, loaded by path, unchanged) supplies the requests joined
to part_a's client latencies (requests(), by order), transient(), gpu_returns(), hetero(),
_window() and the transitions; #99's analyze.py supplies the routing rule (ROUTES, every window);
#99's placement_posthoc.py supplies the counter rules (intervals, derive, switch_time, the parent
thread groups), as phase 0 does. A partial raw directory is reported, with what is still pending.

Per transition (t0 = the hetero window's research start_ns, end = its end_ns):
- decisions (research.handoff.decisions: decision_ns, path 0 sync / 1 async, state, count). An
  episode start is a decision with count 1, path sync and a state other than "armed" (handoff.py
  resets the count only when an episode starts). H cells: t_h = the decision time of the first
  async decision after the episode start, inside [t0, end). Structurally valid iff exactly one
  episode start in [t0, end), t_h exists, and the decisions from the episode start up to t_h are
  exactly N, all sync and none "armed". B: t_h = t0. A: no t_h.
- short P99 over [t0, t0 + 4 s) (#94's requests(): part_a's client latencies matched by order to
  the traced ANE requests, by submit time) and over the whole window (part_a's short p99_ms);
- native Core ML mean of the async forwards: predicts rows with a native stamp and entry in
  [t_h, end), native_completion - submit_after, one value per predict call; the predict mean
  (exit - entry) of every predict with entry in [t0, end) is A's Core ML duration;
- #94's transient from t0 (A validity, B phenotype) and from t_h on the requests submitted in
  [t_h, end); host-slow share over [t0 + 10 s, t0 + 20 s) (#94's _window);
- GPU return P50 / P95 / P99 over GPU requests received in [t_h + 1 s, end) (A: the whole window,
  reported only); #92's aggregate req/s (the sum of the streams' req/s);
- per run: mismatches and routing failures in every window, crash (raw/failed logs).
Mechanism (report only, never a gate): E share of the parent's active threads (phase 0's
parent-active: chain + parent-other) and of the auto GPU worker over [t_h, t_h + 0.5),
[t_h + 0.5, t_h + 4), [t_h + 4, end) (A: t0 as origin); relative cycle rate P / E; the E->P
switch time per aggregate (placement_posthoc's rule, 0.5 s bins over [t0, min(t0 + 20, end)));
"P-dominant at the handoff and stays" iff both aggregates have E share <= 0.25 in all three spans.

Gates, per H transition, each with its value, limit and pass flag (criteria.md, "User-facing
gates"); references are medians over the A transitions of the rounds in use (round 1: A r1-r2;
replication: A r1-r4); allowed(x) = max(1.05 x, x + 1.0 ms). A candidate passes a round only if
every one of its transitions passes every gate; nothing is averaged.

Interpretation choices where criteria.md leaves room (the stricter reading each time):
- Thresholds written "<=" / ">=" are inclusive with #94's EPS; "<" (steady host-slow share) is
  strict: the share must be < 0.10 - EPS. A gate whose value cannot be measured fails.
- Whole-window short P99 is part_a's recorded p99_ms of the short stream (all its requests); the
  onset P99 needs times, so it uses #94's matched requests.
- Native Core ML and A's predict mean are per predict call (per forward), not per request.
- Structural validity also requires every decision between the episode start and t_h to belong
  to the episode (sync, not "armed"); an episode starting before t0 does not count as in-window.
- Crash: any raw/failed log of an H run fails that run's correctness; a log cannot prove that
  the crash came before the auto instance served its first request.
- A validity: mismatches and routing failures are checked in every window of each A run, not only
  its hetero windows. The replication's A validity covers A r1-r4 (every A the references use).
- A reference that cannot be computed (an A window with no short request or no predict) is a
  validity failure (INCONCLUSIVE), never a candidate failure.
- A run whose protocol differs (cycles != 2, seconds != 20, or a guard other than the cell's) is
  analysed and shown but never judged: its round stays pending.
- Round 2's fallback (H64 after H32 fails) uses design.round2("H64"): H64 r3-r5, and A r3-r4 as
  run_all.sh leaves them (it skips existing files).
- Mechanism: parent-active and worker are CPU-weighted aggregates of the threads active (>= 20 ms
  CPU) in [t0, t0 + 4 s), as phase 0; the switch time is per aggregate, from t0.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RAW = EXP / "raw"
ROOT = HERE.parents[2]
ASYNC94 = ROOT / "research" / "coreml-async-transient" / "scripts"
QOS99 = ROOT / "research" / "coreml-dependency-qos" / "scripts"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


design = _load("staged_handoff_design", HERE / "design.py")
a94 = _load("async_transient_analyze", ASYNC94 / "analyze.py")  # #94's, unchanged
q99 = _load("dependency_qos_analyze", QOS99 / "analyze.py")  # #99's, unchanged (ROUTES)
pp = _load("placement_posthoc", QOS99 / "placement_posthoc.py")  # #99's counter rules

S = a94.S
EPS = a94.EPS
GUARDS = {"H32": 32, "H64": 64}
ONSET = (0.0, 4.0)
STEADY = a94.STEADY  # (10, 20) s from t0
ALLOWED = {"ratio": 1.05, "plus_ms": 1.0}
NATIVE = {"ratio": 1.05, "plus_ms": 0.3}
TRANSIENT_S = 1.0
STEADY_HOST_SLOW = 0.10
GPU_RETURN_FROM_S = 1.0
GPU_RETURN_P50_MS = 1.0
THROUGHPUT_RATIO = 0.95
A_P99_MS = 13.0
A_TRANSIENT_S = 1.0
A_STEADY_HOST_SLOW = 0.10
A_AGGREGATE_REQ_S = (109.9, 135.0)
B_TRANSIENT_S = 2.0
B_MIN = 2
MECH_SPANS = (("0-0.5", 0.0, 0.5), ("0.5-4", 0.5, 4.0), ("4-end", 4.0, None))
MECH_ACTIVE_MS = 20.0
P_DOMINANT = 0.25
R1_A = (("A", 1), ("A", 2))
R2_A = (("A", 1), ("A", 2), ("A", 3), ("A", 4))
TRANSITIONS = {1: 2 * design.CYCLES, 2: 3 * design.CYCLES}  # per candidate
GATES = (
    "onset_p99",
    "window_p99",
    "native_ratio",
    "native_plus",
    "transient_from_th",
    "steady_host_slow",
    "gpu_return_p50",
    "throughput",
    "mismatches",
    "routing",
    "crash",
    "structure",
)
RUN_NAME = re.compile(rf"^{design.MODEL}-(?P<cell>A|B|H32|H64)-r(?P<rep>[1-5])\.json\.gz$")

INCONCLUSIVE_A = "INCONCLUSIVE: an A window failed the validity guard; no candidate is judged in this round"
INCONCLUSIVE_B = "INCONCLUSIVE: B did not reproduce the phenotype (fewer than 2 of 4 transitions >= 2.0 s)"
CLOSED = "RESEARCH-CLOSED / NO PRODUCT CHANGE"
STOP = "STOP the staged-handoff route -> " + CLOSED + ": neither H32 nor H64 passed round 1; no other N or variant"
REPLICATES = (
    "{c} replicates: Phase 4 (a production prototype in laya_apple/) is next, under an addendum committed "
    "before its first run"
)


def _mean(x):
    return float(np.mean(x)) if len(x) else None


def _pct(x, q):
    return float(np.percentile(x, q)) if len(x) else None


def _median(xs):
    return None if not xs or any(x is None for x in xs) else float(np.median(xs))


def allowed(x: float) -> float:
    return max(ALLOWED["ratio"] * x, x + ALLOWED["plus_ms"])


# ------------------------------------------------------------------ per transition


def decisions(run: dict) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """decision_ns, path, state and episode-start flag per decision, in decision order."""
    rows = (run["research"].get("handoff") or {}).get("decisions") or []
    ns = np.asarray([r[0] for r in rows], np.int64)
    path = np.asarray([r[1] for r in rows], np.int64)
    counts = np.asarray([r[3] for r in rows], np.int64)
    states = [r[2] for r in rows]
    starts = np.asarray([p == 0 and c == 1 and s != "armed" for p, c, s in zip(path, counts, states)], bool)
    return ns, path, states, starts


def structure(run: dict, cell: str, t0: int, end: int) -> dict:
    """t_h and the structural validity of one transition (criteria.md, "t_h")."""
    ns, path, states, starts = decisions(run)
    inw = (ns >= t0) & (ns < end)
    out = {
        "forwards_sync": int((inw & (path == 0)).sum()),
        "forwards_async": int((inw & (path == 1)).sum()),
    }
    if cell == "A":
        return {**out, "t_h_ns": None, "t_h_s": None, "valid": None}
    if cell == "B":
        return {**out, "t_h_ns": t0, "t_h_s": 0.0, "valid": None}
    guard = GUARDS[cell]
    idx = np.flatnonzero(inw & starts)
    out["episodes"] = len(idx)
    th = None
    before = None
    if len(idx):
        s = int(idx[0])
        later = np.flatnonzero((path == 1) & (np.arange(len(ns)) > s) & (ns < end))
        if len(later):
            h = int(later[0])
            th = int(ns[h])
            seg = range(s, h)
            before = len(seg)
            out["before_all_sync_in_episode"] = all(path[i] == 0 and states[i] != "armed" for i in seg)
            out["sync_after_t_h"] = int(((ns >= th) & (ns < end) & (path == 0)).sum())
    out["t_h_ns"] = th
    out["t_h_s"] = None if th is None else (th - t0) / S
    out["sync_before_t_h"] = before
    out["valid"] = bool(
        len(idx) == 1 and th is not None and before == guard and out.get("before_all_sync_in_episode", False)
    )
    return out


def predicts(run: dict) -> np.ndarray:
    pr = np.asarray(run["research"].get("predicts") or [], np.int64).reshape(-1, 7)
    return pr[np.argsort(pr[:, 0], kind="stable")] if len(pr) else pr


def mechanism(run: dict, t0: int, end: int, origin: int) -> dict:
    """E share and relative cycle rate of parent-active and the auto GPU worker (report only)."""
    rc = run["research"].get("recount") or {}
    series = rc.get("series") or {}
    parent = next((s["pid"] for s in series.values() if s["name"] == "laya-ane-dispatch"), None)
    if parent is None or not rc.get("t_ns"):
        return {"spans": {}, "switch_s": {}, "p_dominant_stays": False}
    spans = [(n, origin + int(a * S), end if b is None else origin + int(b * S)) for n, a, b in MECH_SPANS]
    horizon = min(t0 + int(20.0 * S), end)
    nb = max(0, int(np.ceil((horizon - t0) / (pp.BIN_S * S))))
    agg = {
        g: {"spans": {n: pp.acc() for n, _, _ in spans}, "bins": [pp.acc() for _ in range(nb)]}
        for g in ("parent", "worker")
    }
    for s in series.values():
        g = pp.group_of(s["name"], s["pid"], parent)
        g = {"chain": "parent", "parent-other": "parent", "worker": "worker"}.get(g)
        if g is None:
            continue
        onset = pp.acc()
        mine = {n: pp.acc() for n, _, _ in spans}
        bins = [pp.acc() for _ in range(nb)]
        for mid, p, e in pp.intervals(rc, s):
            if t0 <= mid < t0 + int(ONSET[1] * S):
                pp.add(onset, p, e)
            for n, lo, hi in spans:
                if lo <= mid < hi:
                    pp.add(mine[n], p, e)
            if t0 <= mid < horizon:
                k = int((mid - t0) // int(pp.BIN_S * S))
                if k < nb:
                    pp.add(bins[k], p, e)
        if pp.derive(onset)["cpu_ms"] < MECH_ACTIVE_MS:
            continue
        for n in mine:
            _merge(agg[g]["spans"][n], mine[n])
        for a, b in zip(agg[g]["bins"], bins):
            _merge(a, b)
    out_spans = {
        g: {
            n: {k: d[k] for k in ("cpu_ms", "e_share", "rate_p", "rate_e")}
            for n, d in ((n, pp.derive(x)) for n, x in agg[g]["spans"].items())
        }
        for g in agg
    }
    switch = {g: pp.switch_time([(i * pp.BIN_S, pp.derive(b)) for i, b in enumerate(agg[g]["bins"])]) for g in agg}
    stays = all(
        out_spans[g][n]["e_share"] is not None and out_spans[g][n]["e_share"] <= P_DOMINANT + EPS
        for g in out_spans
        for n in out_spans[g]
    )
    return {"spans": out_spans, "switch_s": switch, "p_dominant_stays": stays}


def _merge(a: dict, b: dict) -> None:
    for k in a:
        a[k] += b[k]


def transitions(run: dict, name: str, cell: str) -> list[dict]:
    """#94's transitions with this experiment's measures added."""
    base = a94.transitions(run, name, cell)
    rws = {k: (pw, rw) for k, pw, rw in a94.hetero(run)}
    pr = predicts(run)
    recv, ret = a94.gpu_returns(run)
    out = []
    for t in base:
        pw, rw = rws[t["cycle"]]
        t0, end = rw["start_ns"], rw["end_ns"]
        rq = t["_rq"]
        st = structure(run, cell, t0, end)
        th = st["t_h_ns"]
        sub = rq["submit_ns"]
        m_on = (sub >= t0 + int(ONSET[0] * S)) & (sub < t0 + int(ONSET[1] * S))
        wpr = pr[(pr[:, 0] >= t0) & (pr[:, 0] < end)] if len(pr) else pr
        rec = {
            **{k: v for k, v in t.items() if k not in ("buckets", "periods")},
            "end_ns": end,
            "structure": st,
            "t_h_s": st["t_h_s"],
            "onset_p99_ms": _pct(rq["client"][m_on], 99),
            "window_p99_ms": t["window"]["short_p99_ms"],
            "predict_mean_ms": _mean((wpr[:, 6] - wpr[:, 0]) / 1e6) if len(wpr) else None,
            "steady_host_slow_share": t["window"]["steady_host_slow_share"],
            "aggregate_req_s": t["window"]["aggregate_req_s"],
        }
        g_lo = t0 if cell == "A" else (th + int(GPU_RETURN_FROM_S * S) if th is not None else None)
        if th is not None:
            apr = wpr[(wpr[:, 0] >= th) & (wpr[:, 3] > 0)] if len(wpr) else wpr
            rec["native_mean_ms"] = _mean((apr[:, 3] - apr[:, 2]) / 1e6) if len(apr) else None
            rec["native_n"] = len(apr)
            m = sub >= th
            rec["transient_from_th"] = a94.transient(sub[m], rq["prepare"][m], rq["client"][m], th, end)
        else:
            rec["native_mean_ms"], rec["native_n"], rec["transient_from_th"] = None, 0, None
        gr = ret[(recv >= g_lo) & (recv < end)] if g_lo is not None else np.zeros(0)
        rec["gpu_return_ms"] = {"from_s": None if g_lo is None else (g_lo - t0) / S, "n": len(gr)}
        rec["gpu_return_ms"].update({f"p{q}": _pct(gr, q) for q in (50, 95, 99)})
        rec["mechanism"] = mechanism(run, t0, end, th if th is not None else t0)
        out.append(rec)
    return out


# ------------------------------------------------------------------ per run


def run_stats(run: dict, name: str, cell: str, crashed: bool = False) -> dict:
    windows = run["part_a"]["windows"]
    mismatches = sum(s["mismatches"] for w in windows for s in w["streams"].values())
    routing = [f"{w['cycle']}:{w['condition']}:{s}" for w in windows for s in q99._routing_failures(w)]
    res = run["research"]
    h = res.get("handoff") or {}
    protocol = {
        "cycles": run["args"]["cycles"] == design.CYCLES,
        "seconds": run["args"]["seconds"] == design.SECONDS,
        "guard": h.get("guard") == GUARDS.get(cell),
        "installed": bool(h.get("installed")) and not h.get("errors"),
    }
    return {
        "run": name,
        "cell": cell,
        "transitions": transitions(run, name, cell),
        "mismatches": mismatches,
        "routing_failures": routing,
        "crashed": crashed,
        "protocol": protocol,
        "protocol_ok": all(protocol.values()),
        "handoff": {
            k: h.get(k) for k in ("guard", "episodes", "forwards", "forwards_sync", "forwards_async", "errors")
        },
    }


# ------------------------------------------------------------------ gates and validity


def references(a_runs: list[dict]) -> dict:
    trs = [t for r in a_runs for t in r["transitions"]]
    return {
        "transitions": [(t["run"], t["cycle"]) for t in trs],
        "A_onset_p99_ms": _median([t["onset_p99_ms"] for t in trs]),
        "A_window_p99_ms": _median([t["window_p99_ms"] for t in trs]),
        "A_predict_mean_ms": _median([t["predict_mean_ms"] for t in trs]),
        "A_agg_req_s": _median([t["aggregate_req_s"] for t in trs]),
    }


def _gate(value, limit, op: str) -> dict:
    if limit is None:
        ok = None  # pending: the reference is not available yet
    elif value is None:
        ok = False
    elif op == "<=":
        ok = value <= limit + EPS
    elif op == "<":
        ok = value < limit - EPS
    elif op == ">=":
        ok = value >= limit - EPS
    else:  # "=="
        ok = value == limit
    return {"value": value, "op": op, "limit": limit, "pass": ok}


def gates(t: dict, run: dict, ref: dict | None) -> dict:
    """Every user-facing gate of one H transition (criteria.md, gates 1-5)."""
    ref = ref or {}
    on, win = ref.get("A_onset_p99_ms"), ref.get("A_window_p99_ms")
    pm, agg = ref.get("A_predict_mean_ms"), ref.get("A_agg_req_s")
    tr = t["transient_from_th"]
    g = {
        "onset_p99": _gate(t["onset_p99_ms"], None if on is None else allowed(on), "<="),
        "window_p99": _gate(t["window_p99_ms"], None if win is None else allowed(win), "<="),
        "native_ratio": _gate(t["native_mean_ms"], None if pm is None else NATIVE["ratio"] * pm, "<="),
        "native_plus": _gate(t["native_mean_ms"], None if pm is None else pm + NATIVE["plus_ms"], "<="),
        "transient_from_th": _gate(None if tr is None else tr["duration_s"], TRANSIENT_S, "<="),
        "steady_host_slow": _gate(t["steady_host_slow_share"], STEADY_HOST_SLOW, "<"),
        "gpu_return_p50": _gate(t["gpu_return_ms"]["p50"], GPU_RETURN_P50_MS, "<="),
        "throughput": _gate(t["aggregate_req_s"], None if agg is None else THROUGHPUT_RATIO * agg, ">="),
        "mismatches": _gate(run["mismatches"], 0, "=="),
        "routing": _gate(len(run["routing_failures"]), 0, "=="),
        "crash": _gate(run["crashed"], False, "=="),
        "structure": _gate(t["structure"]["valid"], True, "=="),
    }
    flags = [x["pass"] for x in g.values()]
    return {
        "gates": g,
        "failed": [k for k, x in g.items() if x["pass"] is False],
        "pending": [k for k, x in g.items() if x["pass"] is None],
        "pass": None if None in flags else all(flags),
    }


def a_validity(a_runs: list[dict]) -> dict:
    """#94's guard on every A window (criteria.md, "A validity"), plus measurable references."""
    lo, hi = A_AGGREGATE_REQ_S
    per = []
    for r in a_runs:
        for t in r["transitions"]:
            p99, share, agg = t["window_p99_ms"], t["steady_host_slow_share"], t["aggregate_req_s"]
            checks = {
                "short_p99": p99 is not None and p99 < A_P99_MS - EPS,
                "transient": t["transient"]["duration_s"] <= A_TRANSIENT_S + EPS,
                "steady_host_slow": share is not None and share < A_STEADY_HOST_SLOW - EPS,
                "aggregate": agg is not None and lo - EPS <= agg <= hi + EPS,
                "no_mismatch": r["mismatches"] == 0,
                "no_routing_failure": not r["routing_failures"],
                "references_measurable": t["onset_p99_ms"] is not None and t["predict_mean_ms"] is not None,
            }
            per.append(
                {
                    "run": t["run"],
                    "cycle": t["cycle"],
                    "short_p99_ms": p99,
                    "transient_s": t["transient"]["duration_s"],
                    "steady_host_slow_share": share,
                    "aggregate_req_s": agg,
                    "failed": [k for k, v in checks.items() if not v],
                }
            )
    return {"per_transition": per, "valid": bool(per) and not any(x["failed"] for x in per)}


def b_phenotype(b_runs: list[dict]) -> dict:
    ds = [t["transient"]["duration_s"] for r in b_runs for t in r["transitions"]]
    n = sum(1 for d in ds if d >= B_TRANSIENT_S - EPS)
    return {"transient_s": ds, "n_ge_2s": n, "reproduced": len(ds) == 2 * design.CYCLES and n >= B_MIN}


def judge_candidate(runs: list[dict], ref: dict, n_expected: int) -> dict:
    per = []
    for r in runs:
        for t in r["transitions"]:
            g = gates(t, r, ref)
            per.append({"run": t["run"], "cycle": t["cycle"], "preceded_by": t["preceded_by"], **g})
    flags = [x["pass"] for x in per]
    ok = len(per) == n_expected and all(f is True for f in flags)
    return {
        "transitions": per,
        "n": len(per),
        "pass": ok,
        "failing": [f"{x['run']} c{x['cycle']}: {', '.join(x['failed'])}" for x in per if x["failed"]],
        "mechanism_reading_holds": None,
    }


# ------------------------------------------------------------------ rounds


def load(path: Path) -> dict:
    with gzip.open(path, "rt") as fh:
        return json.load(fh)


def _present(raw: Path) -> dict[tuple[str, int], Path]:
    out = {}
    for p in sorted(raw.glob(f"{design.MODEL}-*-r*.json.gz")):
        if x := RUN_NAME.match(p.name):
            out[(x["cell"], int(x["rep"]))] = p
    return out


def _failed(raw: Path) -> list[str]:
    return sorted(p.name for p in (raw / "failed").glob("*.log"))


def _stats(p: Path, cell: str, failed: list[str]) -> dict:
    run = load(p)
    res = run["research"]
    assert res["experiment"] == "coreml-staged-handoff" and res["cell"] == cell, p
    assert res["base"]["cell"] == "PB-ASYNC", p
    name = p.name[: -len(".json.gz")]
    crashed = cell in GUARDS and any(f.startswith(name + ".") for f in failed)
    st = run_stats(run, name, cell, crashed)
    st["run_wall_s"] = res.get("run_wall_total_s", res.get("run_wall_s"))
    st["seconds"] = run["args"]["seconds"]
    return st


def _mech_holds(runs: list[dict]) -> bool:
    return all(t["mechanism"]["p_dominant_stays"] for r in runs for t in r["transitions"])


def round1(stats: dict, present) -> dict:
    missing = [f"{c} r{r}" for c, r in design.ROUND1 if (c, r) not in present]
    deviating = [f"{c} r{r}" for c, r in design.ROUND1 if (c, r) in stats and not stats[(c, r)]["protocol_ok"]]
    a_runs = [stats[k] for k in R1_A if k in stats]
    ref = references(a_runs) if len(a_runs) == len(R1_A) else None
    cands = {}
    for c in design.CANDIDATES:
        runs = [stats[(c, r)] for r in (1, 2) if (c, r) in stats]
        cands[c] = judge_candidate(runs, ref, TRANSITIONS[1])
        cands[c]["mechanism_reading_holds"] = _mech_holds(runs) if runs else None
    out = {"references": ref, "candidates": cands, "missing": missing, "deviating": deviating}
    if missing or deviating:
        out.update(complete=False, decision=None, leader=None)
        return out
    av = a_validity(a_runs)
    bp = b_phenotype([stats[("B", r)] for r in (1, 2)])
    out.update(complete=True, A_validity=av, B_phenotype=bp)
    if not av["valid"]:
        out.update(decision=INCONCLUSIVE_A, leader=None)
    elif not bp["reproduced"]:
        out.update(decision=INCONCLUSIVE_B, leader=None)
    else:
        leader = next((c for c in design.CANDIDATES if cands[c]["pass"]), None)
        out["leader"] = leader
        out["decision"] = (
            f"leader {leader}; round 2 runs: " + ", ".join(f"{c} r{r}" for c, r in design.round2(leader))
            if leader
            else STOP
        )
    return out


def round2(stats: dict, present, cand: str) -> dict:
    runs2 = design.round2(cand)
    missing = [f"{c} r{r}" for c, r in runs2 if (c, r) not in present]
    deviating = [f"{c} r{r}" for c, r in runs2 if (c, r) in stats and not stats[(c, r)]["protocol_ok"]]
    a_runs = [stats[k] for k in R2_A if k in stats]
    ref = references(a_runs) if len(a_runs) == len(R2_A) else None
    runs = [stats[(cand, r)] for r in (3, 4, 5) if (cand, r) in stats]
    j = judge_candidate(runs, ref, TRANSITIONS[2])
    j["mechanism_reading_holds"] = _mech_holds(runs) if runs else None
    out = {"candidate": cand, "runs": [list(x) for x in runs2], "references": ref, "judgement": j}
    out.update(missing=missing, deviating=deviating)
    if missing or deviating:
        out.update(complete=False, valid=None, passed=None)
        return out
    av = a_validity(a_runs)
    out.update(complete=True, A_validity=av, valid=av["valid"], passed=av["valid"] and j["pass"])
    return out


def summarise(raw: Path) -> dict:
    present = _present(raw)
    failed = _failed(raw)
    stats = {k: _stats(p, k[0], failed) for k, p in present.items()}
    r1 = round1(stats, present)
    r2s: list[dict] = []
    if not r1["complete"]:
        parts = []
        if r1["missing"]:
            parts.append("pending: " + ", ".join(r1["missing"]))
        if r1["deviating"]:
            parts.append("protocol deviation (not judged): " + ", ".join(r1["deviating"]))
        outcome = "round 1 " + "; ".join(parts)
    elif r1["leader"] is None:
        outcome = r1["decision"]
    else:
        outcome = None
        order = [r1["leader"]]
        if r1["leader"] == "H32" and r1["candidates"]["H64"]["pass"]:
            order.append("H64")  # the fallback, used only if H32 fails round 2
        for c in order:
            r2 = round2(stats, present, c)
            r2s.append(r2)
            if not r2["complete"]:
                outcome = f"round 2 ({c}) pending: " + ", ".join(r2["missing"] + r2["deviating"])
                break
            if not r2["valid"]:
                outcome = f"round 2 ({c}): " + INCONCLUSIVE_A
                break
            if r2["passed"]:
                outcome = REPLICATES.format(c=c)
                if not r2["judgement"]["mechanism_reading_holds"]:
                    outcome += "; the gates pass but the 'P-dominant at the handoff and stays' reading does not hold"
                break
        if outcome is None:
            outcome = f"no candidate replicates: {CLOSED}"
    planned = set(design.ROUND1)
    for r2 in r2s:
        planned |= {tuple(x) for x in r2["runs"]}
    unexpected = sorted(present[k].name for k in present if k not in planned)

    def public(st):
        return {
            **{k: v for k, v in st.items() if k != "transitions"},
            "transitions": [{k: v for k, v in t.items() if not k.startswith("_")} for t in st["transitions"]],
        }

    return {
        "design": {
            "round1": [list(x) for x in design.ROUND1],
            "guards": GUARDS,
            "onset_s": list(ONSET),
            "steady_s": list(STEADY),
            "allowed": ALLOWED,
            "native": NATIVE,
            "transient_s": TRANSIENT_S,
            "steady_host_slow": STEADY_HOST_SLOW,
            "gpu_return_from_s": GPU_RETURN_FROM_S,
            "gpu_return_p50_ms": GPU_RETURN_P50_MS,
            "throughput_ratio": THROUGHPUT_RATIO,
            "a_validity": {
                "short_p99_ms": A_P99_MS,
                "transient_s": A_TRANSIENT_S,
                "steady_host_slow": A_STEADY_HOST_SLOW,
                "aggregate_req_s": list(A_AGGREGATE_REQ_S),
            },
            "b_phenotype": {"transient_s": B_TRANSIENT_S, "min_transitions": B_MIN},
            "p_dominant": P_DOMINANT,
        },
        "failed_runs": failed,
        "unexpected_files": unexpected,
        "runs": {st["run"]: public(st) for st in stats.values()},
        "round1": r1,
        "round2": r2s,
        "outcome": outcome,
    }


# ------------------------------------------------------------------ tables


def f(x, d=2):
    if x is None:
        return "–"
    if isinstance(x, (bool, str, int, np.integer)):
        return str(x)
    return f"{x:.{d}f}"


def _cell(g: dict) -> str:
    flag = {True: "pass", False: "FAIL", None: "pending"}[g["pass"]]
    v, lim = g["value"], g["limit"]
    d = 3 if isinstance(v, float) and abs(v) < 1 else 2
    return f"{f(v, d)} {g['op']} {f(lim, d)} {flag}"


def _gate_table(L: list, j: dict) -> None:
    L += [
        "| transition | after | " + " | ".join(GATES) + " | failed |",
        "|---|---|" + "---|" * (len(GATES) + 1),
    ]
    for x in j["transitions"]:
        L.append(
            f"| {x['run']} c{x['cycle']} | {x['preceded_by']} | "
            + " | ".join(_cell(x["gates"][g]) for g in GATES)
            + f" | {', '.join(x['failed']) or ('pending: ' + ', '.join(x['pending']) if x['pending'] else '–')} |"
        )
    L.append(f"\nCandidate passes: **{j['pass']}** ({j['n']} transitions)")
    for s in j["failing"]:
        L.append(f"- failing: {s}")
    L.append(f"- 'P-dominant at the handoff and stays' holds in every transition: {j['mechanism_reading_holds']}\n")


def _refs(L: list, ref: dict | None) -> None:
    if not ref:
        L.append("A references: pending\n")
        return
    L.append(
        f"A references (median over {len(ref['transitions'])} A transitions): onset P99 {f(ref['A_onset_p99_ms'])} ms, "
        f"window P99 {f(ref['A_window_p99_ms'])} ms, predict mean {f(ref['A_predict_mean_ms'], 3)} ms, "
        f"aggregate {f(ref['A_agg_req_s'], 1)} req/s\n"
    )


def _a_validity(L: list, av: dict) -> None:
    L += [
        f"A validity: **{av['valid']}**\n",
        "| run | cycle | short P99 ms | transient s | steady host-slow | aggregate req/s | failed |",
        "|---|---|---|---|---|---|---|",
    ]
    for x in av["per_transition"]:
        L.append(
            f"| {x['run']} | {x['cycle']} | {f(x['short_p99_ms'])} | {f(x['transient_s'], 1)} "
            f"| {f(x['steady_host_slow_share'], 3)} | {f(x['aggregate_req_s'], 1)} | {', '.join(x['failed']) or '–'} |"
        )
    L.append("")


def tables(res: dict) -> str:
    L = ["# Staged handoff screen: A, B, H32, H64 at the hetero transition (laya, L128 / L512)\n"]
    L.append(f"Outcome: {res['outcome']}\n")
    L.append(f"Crashed runs and re-runs (`raw/failed/`): {', '.join(res['failed_runs']) or 'none'}\n")
    if res["unexpected_files"]:
        L.append(f"Files outside the plan (not judged): {', '.join(res['unexpected_files'])}\n")
    L += [
        "\n## Runs\n",
        "| run | seconds | protocol ok | mismatches | routing failures | crashed | episodes | forwards sync / async | wall s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in res["runs"].values():
        h = r["handoff"]
        L.append(
            f"| {r['run']} | {f(r['seconds'], 0)} | {r['protocol_ok']} | {r['mismatches']} "
            f"| {', '.join(r['routing_failures']) or '–'} | {r['crashed']} | {h['episodes']} "
            f"| {h['forwards_sync']} / {h['forwards_async']} | {f(r['run_wall_s'], 0)} |"
        )
    L += [
        "\n## Transitions\n",
        "t_h in s from t0. Transient: #94's, from t0 and from t_h. GPU return over [t_h + 1 s, end) (A: whole window).\n",
        "| run | cycle | after | t_h s | structure (episodes / sync before t_h / valid) | onset P99 | window P99 "
        "| native mean (n) | predict mean | transient t0 / t_h s | steady host-slow | GPU return P50 / P95 / P99 | agg req/s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in res["runs"].values():
        for t in r["transitions"]:
            st, g, th = t["structure"], t["gpu_return_ms"], t["transient_from_th"]
            L.append(
                f"| {t['run']} | {t['cycle']} | {t['preceded_by']} | {f(t['t_h_s'], 3)} "
                f"| {st.get('episodes', '–')} / {f(st.get('sync_before_t_h'))} / {f(st['valid'])} "
                f"| {f(t['onset_p99_ms'])} | {f(t['window_p99_ms'])} | {f(t['native_mean_ms'], 3)} ({t['native_n']}) "
                f"| {f(t['predict_mean_ms'], 3)} | {f(t['transient']['duration_s'], 1)} / "
                f"{f(None if th is None else th['duration_s'], 1)} | {f(t['steady_host_slow_share'], 3)} "
                f"| {f(g['p50'], 3)} / {f(g['p95'], 3)} / {f(g['p99'], 3)} | {f(t['aggregate_req_s'], 1)} |"
            )
    L += [
        "\n## Mechanism (report only, never a gate)\n",
        "E share (relative cycle rate P / E) of parent-active and the auto GPU worker; spans from t_h (A: t0). "
        "Switch: E->P switch time from t0 (placement_posthoc's rule).\n",
        "| run | cycle | group | " + " | ".join(n for n, _, _ in MECH_SPANS) + " | switch s | P-dominant and stays |",
        "|---|---|---|" + "---|" * (len(MECH_SPANS) + 2),
    ]
    for r in res["runs"].values():
        for t in r["transitions"]:
            m = t["mechanism"]
            for g, sp in m["spans"].items():
                L.append(
                    f"| {t['run']} | {t['cycle']} | {g} | "
                    + " | ".join(f"{f(x['e_share'])} ({f(x['rate_p'])} / {f(x['rate_e'])})" for x in sp.values())
                    + f" | {f(m['switch_s'][g], 1)} | {m['p_dominant_stays']} |"
                )
    r1 = res["round1"]
    L.append("\n## Round 1\n")
    _refs(L, r1["references"])
    if r1["complete"]:
        _a_validity(L, r1["A_validity"])
        bp = r1["B_phenotype"]
        L.append(
            f"B phenotype: **{bp['reproduced']}** ({bp['n_ge_2s']} of {len(bp['transient_s'])} transitions with a "
            f"transient >= {B_TRANSIENT_S} s: {', '.join(f(x, 1) for x in bp['transient_s'])})\n"
        )
    for c, j in r1["candidates"].items():
        L.append(f"\n### {c}\n")
        if j["transitions"]:
            _gate_table(L, j)
        else:
            L.append("pending\n")
    L.append(f"After round 1: {r1['decision'] or 'pending'}\n")
    for r2 in res["round2"]:
        L.append(f"\n## Round 2: {r2['candidate']}\n")
        _refs(L, r2["references"])
        if r2["complete"]:
            _a_validity(L, r2["A_validity"])
        if r2["judgement"]["transitions"]:
            _gate_table(L, r2["judgement"])
        L.append(f"Round 2 passed: {f(r2['passed'])}\n")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if results.json / tables.md are stale")
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=EXP)
    a = ap.parse_args()
    res = summarise(a.raw)
    js = json.dumps(res, indent=1, sort_keys=True, default=a94._json) + "\n"
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
