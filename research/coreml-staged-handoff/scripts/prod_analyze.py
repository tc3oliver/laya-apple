"""The production-path phases of the staged handoff: prod_results.json and prod_tables.md from
raw-prod/ (../criteria.md, "Addendum 2: protocols of Phases 4-7").

    uv run python research/coreml-staged-handoff/scripts/prod_analyze.py [--check] [--raw DIR --out DIR]

Inputs: raw-prod/<schedule>-<model>-<cell>-r<rep>.json.gz (prod_run.py; cells A and P) and
raw-prod/failed/<run>.<attempt>.log (the crash rule). Phases: 4 = mix-laya, 5 =
mix-laya-typed-decisions, 5b = mix-laya-multilingual (P r1, 5 s windows), 6 = product-laya,
7 = soak-laya. The screen's analyze.py (loaded by path, unchanged) supplies the gate rule
(_gate, allowed, EPS and its thresholds), #99's routing rule and #94's transient().

Per hetero window (t0 = start_ns, end = end_ns):
- short requests: the trace's ANE rows submitted in [t0, end), in submit order, prepare =
  prepared - submit; their latency is the short stream's client latency matched by order (closed
  loop, one short client, as #94's requests()) when the counts agree, else response - submit
  from the trace ("latency source", reported per transition);
- onset P99 over the requests submitted in [t0, t0 + 4 s); window P99 over every client latency
  of the short stream;
- P: t_h = the first async decision after the episode start in [t0, end); structure as the
  screen's (exactly one episode start in the window, exactly N sync, non-"armed" decisions from
  it to t_h). An episode start is a sync decision with count 1 and a state other than "armed";
- ANE forward duration = service_end - service_start: mean over the ANE rows submitted in
  [t_h, end) (P) or [t0, end) (A);
- #94's transient() from t0 and (P) from t_h over the requests submitted in [t_h, end);
- steady host-slow share: [t0 + 10 s, t0 + 20 s) in Phases 4-6; Phase 7: [t_h + 1 s, end) for
  P and [t0 + 1 s, end) for A (A has no t_h);
- GPU return = received - service_end of the trace's GPU rows received in [t_h + 1 s, end) (P)
  or [t0, end) (A, reported only), P50 / P95 / P99;
- aggregate req/s = the sum of the window's streams' req/s.

Per run: mismatches and routing failures in every window (#99's rule: hetero short -> ANE and
long -> GPU, solo_short -> ANE, solo_long -> GPU, gpu_only -> GPU), a crash log, both workers
alive at the end, and the handoff not disabled at the end (handoff_snapshot is a dict with no
truthy "disabled", a null "disabled_reason" and no "enabled": false).

Interpretation choices where the addendum leaves room (the stricter reading each time):
- N is the replicated leader's, GUARD (32: H32 led round 1); a run whose handoff_snapshot carries
  a different "guard" is a protocol deviation. A run whose args, window sequence or schedule differ
  from the phase's is shown but not judged: its phase stays pending.
- A references are medians over the phase's A hetero transitions; Phase 6 per window index (the
  schedule is fixed, so an index is a position) and, for non-hetero windows, per index and stream;
  Phase 7 over all 60 A soak episodes. A reference that cannot be computed is an A validity failure.
- A validity (every phase with A): 0 mismatches and 0 routing failures in every A window; per A
  hetero window a transient from t0 <= 1 s and a steady host-slow share < 0.10; Phases 4 and 6
  also whole-window short P99 < 13 ms and aggregate req/s in [109.9, 135.0].
- Phase 6 non-hetero gates are per stream (gpu_only has two).
- Phase 7 re-arm: the first decision inside each hetero window has state "sync_guard" and count 1.
  Degradation: median window P99 of the last 15 hetero windows <= 1.05 x that of the first 15.
- Phase 5b: the handoff is not enabled iff no decision was logged, the snapshot is absent or
  disabled, and info()'s ane_placement (when recorded) is "process".
- Crash: any failed log of a P run fails that run's correctness. A run missing after two failed
  attempts ends the phase: FAIL for P (a candidate failure), INCONCLUSIVE for A (harness).
- Order: 4, then 5 and 5b, then 6, then 7. A phase whose predecessors did not all PASS is shown
  with status "not reached" (its data and its would-be status are still reported).
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
RAW = EXP / "raw-prod"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sh = _load("staged_handoff_analyze", HERE / "analyze.py")  # the screen's, unchanged
prod_run = _load("staged_handoff_prod_run", HERE / "prod_run.py")  # schedule() and the columns
a94 = sh.a94

S = sh.S
EPS = sh.EPS
GUARD = 32
ONSET_S = 4.0
STEADY = sh.STEADY  # (10, 20) s from t0, Phases 4-6
SOAK_STEADY_FROM_S = 1.0  # Phase 7: [t_h + 1 s, end)
SOAK_EPISODES = 60
SOAK_SPLIT = 15
SOAK_DEGRADATION = 1.05
A_P99_PHASES = ("4", "6")
NON_HETERO = ("solo_short", "solo_long", "gpu_only")
PHASES = {
    "4": {"schedule": "mix", "model": "laya", "short": 128, "long": 512, "seconds": 20.0},
    "5": {"schedule": "mix", "model": "laya-typed-decisions", "short": 128, "long": 1024, "seconds": 20.0},
    "5b": {"schedule": "mix", "model": "laya-multilingual", "short": 128, "long": 512, "seconds": 5.0},
    "6": {"schedule": "product", "model": "laya", "short": 128, "long": 512, "seconds": 20.0},
    "7": {"schedule": "soak", "model": "laya", "short": 128, "long": 512, "seconds": 20.0},
}
RUNS = {
    "4": (("P", 1), ("A", 1), ("A", 2), ("P", 2)),
    "5": (("P", 1), ("A", 1), ("A", 2), ("P", 2)),
    "5b": (("P", 1),),
    "6": (("P", 1), ("A", 1), ("A", 2), ("P", 2)),
    "7": (("P", 1), ("A", 1)),
}
ORDER = (("4",), ("5", "5b"), ("6",), ("7",))
GATES = (
    "onset_p99",
    "window_p99",
    "forward_ratio",
    "forward_plus",
    "transient_from_th",
    "steady_host_slow",
    "gpu_return_p50",
    "throughput",
    "mismatches",
    "routing",
    "crash",
    "workers_alive",
    "handoff_enabled",
    "structure",
)
RUN_NAME = re.compile(
    r"^(?P<schedule>mix|product|soak)-(?P<model>laya-typed-decisions|laya-multilingual|laya)"
    r"-(?P<cell>A|P)-r(?P<rep>\d+)\.json\.gz$"
)
CLOSED = "RESEARCH-CLOSED / NO PRODUCT CHANGE"
CANDIDATE = "PRODUCT-CANDIDATE: laya, typed, product mix, soak and correctness all pass"


def run_name(phase: str, cell: str, rep: int) -> str:
    c = PHASES[phase]
    return f"{c['schedule']}-{c['model']}-{cell}-r{rep}"


def phase_of(schedule: str, model: str) -> str | None:
    return next((p for p, c in PHASES.items() if c["schedule"] == schedule and c["model"] == model), None)


def _mean(x):
    return float(np.mean(x)) if len(x) else None


def _pct(x, q):
    return float(np.percentile(x, q)) if len(x) else None


def _disabled(snap) -> bool:
    if not isinstance(snap, dict):
        return True
    return bool(snap.get("disabled")) or snap.get("disabled_reason") is not None or snap.get("enabled") is False


# ------------------------------------------------------------------ records


def trace(run: dict) -> dict[str, np.ndarray]:
    """The auto instance's RequestTrace rows as columns, in submit order."""
    cols = run.get("trace_columns") or prod_run.TRACE_COLUMNS
    rows = run.get("trace") or []
    out = {"target": np.asarray([str(r[cols.index("target")]) for r in rows], dtype=object)}
    for c in cols:
        if c != "target":
            out[c] = np.asarray([r[cols.index(c)] for r in rows], np.int64)
    order = np.argsort(out["submit_ns"], kind="stable")
    return {k: v[order] for k, v in out.items()}


def decisions(run: dict) -> dict:
    rows = run.get("decisions") or []
    ns = np.asarray([r[0] for r in rows], np.int64)
    path = np.asarray([r[1] for r in rows], np.int64)
    count = np.asarray([r[3] for r in rows], np.int64)
    states = [str(r[2]) for r in rows]
    starts = np.asarray([p == 0 and c == 1 and s != "armed" for p, c, s in zip(path, count, states)], bool)
    return {"ns": ns, "path": path, "count": count, "states": states, "starts": starts}


def structure(dec: dict, t0: int, end: int, guard: int) -> dict:
    """t_h and the structural validity of one P hetero window (the screen's rule)."""
    ns, path, states, starts = dec["ns"], dec["path"], dec["states"], dec["starts"]
    inw = (ns >= t0) & (ns < end)
    out = {"forwards_sync": int((inw & (path == 0)).sum()), "forwards_async": int((inw & (path == 1)).sum())}
    first = np.flatnonzero(inw)
    out["first_decision"] = (
        None if not len(first) else {"state": states[first[0]], "count": int(dec["count"][first[0]])}
    )
    idx = np.flatnonzero(inw & starts)
    out["episodes"] = len(idx)
    th, before, all_sync = None, None, False
    if len(idx):
        s = int(idx[0])
        later = np.flatnonzero((path == 1) & (np.arange(len(ns)) > s) & (ns < end))
        if len(later):
            h = int(later[0])
            th = int(ns[h])
            before = h - s
            all_sync = all(path[i] == 0 and states[i] != "armed" for i in range(s, h))
    out.update(t_h_ns=th, t_h_s=None if th is None else (th - t0) / S, sync_before_t_h=before)
    out["valid"] = bool(len(idx) == 1 and th is not None and before == guard and all_sync)
    return out


def transition(run_id: str, tr: dict, dec: dict, w: dict, prev: dict | None, cell: str, phase: str) -> dict:
    t0, end = w["start_ns"], w["end_ns"]
    sub = tr["submit_ns"]
    ane = (tr["target"] == "ane") & (sub >= t0) & (sub < end)
    s_sub = sub[ane]
    prep = (tr["prepared_ns"][ane] - s_sub) / 1e6
    client = np.asarray((w["streams"].get("short") or {}).get("latency_ms") or [], float)
    if len(client) == len(s_sub):
        lat, source = client, "client"
    else:
        lat, source = (tr["response_ns"][ane] - s_sub) / 1e6, "trace"
    fwd = (tr["service_end_ns"][ane] - tr["service_start_ns"][ane]) / 1e6
    if cell == "P":
        st = structure(dec, t0, end, GUARD)
    else:
        st = {"t_h_ns": None, "t_h_s": None, "valid": None, "first_decision": None}
    th = st["t_h_ns"]
    rec = {
        "run": run_id,
        "index": w["index"],
        "preceded_by": None if prev is None else prev["condition"],
        "start_ns": t0,
        "end_ns": end,
        "structure": st,
        "t_h_s": st["t_h_s"],
        "n_short": int(len(s_sub)),
        "n_client": int(len(client)),
        "latency_source": source,
        "onset_p99_ms": _pct(lat[s_sub < t0 + int(ONSET_S * S)], 99),
        "window_p99_ms": _pct(client, 99),
        "aggregate_req_s": sum(x.get("req_s") or 0.0 for x in w["streams"].values()),
        "transient": a94.transient(s_sub, prep, lat, t0, end),
    }
    if cell == "P":
        m = s_sub >= th if th is not None else np.zeros(len(s_sub), bool)
        rec["forward_mean_ms"] = _mean(fwd[m])
        rec["forward_n"] = int(m.sum())
        rec["transient_from_th"] = None if th is None else a94.transient(s_sub[m], prep[m], lat[m], th, end)
        g_lo = None if th is None else th + int(sh.GPU_RETURN_FROM_S * S)
    else:
        rec["forward_mean_ms"], rec["forward_n"] = _mean(fwd), int(len(fwd))
        rec["transient_from_th"] = None
        g_lo = t0
    if phase == "7":
        origin = th if cell == "P" else t0
        lo, hi = (None, None) if origin is None else (origin + int(SOAK_STEADY_FROM_S * S), end)
    else:
        lo, hi = t0 + int(STEADY[0] * S), min(t0 + int(STEADY[1] * S), end)
    ms = (s_sub >= lo) & (s_sub < hi) if lo is not None else np.zeros(len(s_sub), bool)
    rec["steady_span_s"] = None if lo is None else [(lo - t0) / S, (hi - t0) / S]
    rec["steady_host_slow_share"] = float((prep[ms] > a94.HOST_SLOW_PREPARE_MS).mean()) if ms.any() else None
    if g_lo is None:
        ret = np.zeros(0)
    else:
        g = (tr["target"] == "gpu") & (tr["received_ns"] >= g_lo) & (tr["received_ns"] < end)
        ret = (tr["received_ns"][g] - tr["service_end_ns"][g]) / 1e6
    rec["gpu_return_ms"] = {"from_s": None if g_lo is None else (g_lo - t0) / S, "n": int(len(ret))}
    rec["gpu_return_ms"].update({f"p{q}": _pct(ret, q) for q in (50, 95, 99)})
    return rec


def run_stats(run: dict, name: str, phase: str, cell: str, crashed: bool = False) -> dict:
    cfg = PHASES[phase]
    windows = run.get("windows") or []
    tr, dec = trace(run), decisions(run)
    mismatches = sum(s.get("mismatches", 0) for w in windows for s in w["streams"].values())
    routing = [f"{w['index']}:{w['condition']}:{s}" for w in windows for s in sh.q99._routing_failures(w)]
    alive = run.get("workers_alive_at_end")
    snap = run.get("handoff_snapshot")
    args = run.get("args") or {}
    expected = [x["condition"] for x in prod_run.schedule(cfg["schedule"], cfg["seconds"])]
    protocol = {
        "cell": args.get("cell") == cell,
        "model": args.get("model") == cfg["model"],
        "lengths": args.get("short") == cfg["short"] and args.get("long") == cfg["long"],
        "schedule": args.get("schedule") == cfg["schedule"],
        "seconds": cfg["schedule"] != "mix" or args.get("seconds") == cfg["seconds"],
        "windows": [w["condition"] for w in windows] == expected,
        "guard": not isinstance(snap, dict) or snap.get("guard") in (None, GUARD),
    }
    info = run.get("info_end") or run.get("info_start") or {}
    trs = []
    for i, w in enumerate(windows):
        if w["condition"] == "hetero":
            trs.append(transition(name, tr, dec, w, windows[i - 1] if i else None, cell, phase))
    return {
        "run": name,
        "phase": phase,
        "cell": cell,
        "transitions": trs,
        "windows": [
            {
                "index": w["index"],
                "condition": w["condition"],
                "streams": {
                    s: {"p99_ms": _pct(x.get("latency_ms") or [], 99), "req_s": x.get("req_s")}
                    for s, x in w["streams"].items()
                },
            }
            for w in windows
        ],
        "mismatches": mismatches,
        "routing_failures": routing,
        "crashed": crashed,
        "workers_alive": isinstance(alive, dict) and all(bool(alive.get(k)) for k in ("ane", "gpu")),
        "workers_alive_at_end": alive,
        "handoff_enabled_at_end": not _disabled(snap),
        "handoff_snapshot": snap,
        "ane_placement": info.get("ane_placement") if isinstance(info, dict) else None,
        "decisions": len(dec["ns"]),
        "protocol": protocol,
        "protocol_ok": all(protocol.values()),
        "run_wall_s": run.get("run_wall_s"),
    }


# ------------------------------------------------------------------ references, gates, validity


def _ref(trs: list[dict]) -> dict:
    return {
        "transitions": [(t["run"], t["index"]) for t in trs],
        "A_onset_p99_ms": sh._median([t["onset_p99_ms"] for t in trs]),
        "A_window_p99_ms": sh._median([t["window_p99_ms"] for t in trs]),
        "A_forward_mean_ms": sh._median([t["forward_mean_ms"] for t in trs]),
        "A_agg_req_s": sh._median([t["aggregate_req_s"] for t in trs]),
    }


def references(a_runs: list[dict], phase: str) -> dict:
    trs = [t for r in a_runs for t in r["transitions"]]
    out = {"all": _ref(trs) if trs else None}
    if phase == "6":
        out["by_index"] = {str(i): _ref([t for t in trs if t["index"] == i]) for i in sorted({t["index"] for t in trs})}
        streams: dict[str, dict] = {}
        for r in a_runs:
            for w in r["windows"]:
                if w["condition"] in NON_HETERO:
                    for s, x in w["streams"].items():
                        streams.setdefault(f"{w['index']}:{s}", []).append(x)
        out["non_hetero"] = {
            k: {"p99_ms": sh._median([x["p99_ms"] for x in v]), "req_s": sh._median([x["req_s"] for x in v])}
            for k, v in streams.items()
        }
    return out


def _ref_for(t: dict, refs: dict | None, phase: str) -> dict | None:
    if refs is None:
        return None
    return refs["by_index"].get(str(t["index"])) if phase == "6" else refs["all"]


def gates(t: dict, run: dict, ref: dict | None) -> dict:
    """Gates 1-5 of one P hetero transition, with the addendum's adaptations."""
    ref = ref or {}
    on, win = ref.get("A_onset_p99_ms"), ref.get("A_window_p99_ms")
    fm, agg = ref.get("A_forward_mean_ms"), ref.get("A_agg_req_s")
    tr = t["transient_from_th"]
    G = sh._gate
    g = {
        "onset_p99": G(t["onset_p99_ms"], None if on is None else sh.allowed(on), "<="),
        "window_p99": G(t["window_p99_ms"], None if win is None else sh.allowed(win), "<="),
        "forward_ratio": G(t["forward_mean_ms"], None if fm is None else sh.NATIVE["ratio"] * fm, "<="),
        "forward_plus": G(t["forward_mean_ms"], None if fm is None else fm + sh.NATIVE["plus_ms"], "<="),
        "transient_from_th": G(None if tr is None else tr["duration_s"], sh.TRANSIENT_S, "<="),
        "steady_host_slow": G(t["steady_host_slow_share"], sh.STEADY_HOST_SLOW, "<"),
        "gpu_return_p50": G(t["gpu_return_ms"]["p50"], sh.GPU_RETURN_P50_MS, "<="),
        "throughput": G(t["aggregate_req_s"], None if agg is None else sh.THROUGHPUT_RATIO * agg, ">="),
        "mismatches": G(run["mismatches"], 0, "=="),
        "routing": G(len(run["routing_failures"]), 0, "=="),
        "crash": G(run["crashed"], False, "=="),
        "workers_alive": G(run["workers_alive"], True, "=="),
        "handoff_enabled": G(run["handoff_enabled_at_end"], True, "=="),
        "structure": G(t["structure"]["valid"], True, "=="),
    }
    return _verdict(g)


def _verdict(g: dict) -> dict:
    flags = [x["pass"] for x in g.values()]
    return {
        "gates": g,
        "failed": [k for k, x in g.items() if x["pass"] is False],
        "pending": [k for k, x in g.items() if x["pass"] is None],
        "pass": None if None in flags else all(flags),
    }


def a_validity(a_runs: list[dict], phase: str) -> dict:
    lo, hi = sh.A_AGGREGATE_REQ_S
    per = []
    for r in a_runs:
        run_checks = {"no_mismatch": r["mismatches"] == 0, "no_routing_failure": not r["routing_failures"]}
        for t in r["transitions"]:
            p99, share, agg = t["window_p99_ms"], t["steady_host_slow_share"], t["aggregate_req_s"]
            checks = {
                **run_checks,
                "transient": t["transient"]["duration_s"] <= sh.A_TRANSIENT_S + EPS,
                "steady_host_slow": share is not None and share < sh.A_STEADY_HOST_SLOW - EPS,
                "references_measurable": all(
                    t[k] is not None for k in ("onset_p99_ms", "window_p99_ms", "forward_mean_ms", "aggregate_req_s")
                ),
            }
            if phase in A_P99_PHASES:
                checks["short_p99"] = p99 is not None and p99 < sh.A_P99_MS - EPS
                checks["aggregate"] = agg is not None and lo - EPS <= agg <= hi + EPS
            per.append(
                {
                    "run": t["run"],
                    "index": t["index"],
                    "short_p99_ms": p99,
                    "transient_s": t["transient"]["duration_s"],
                    "steady_host_slow_share": share,
                    "aggregate_req_s": agg,
                    "failed": [k for k, v in checks.items() if not v],
                }
            )
        if not r["transitions"]:
            per.append({"run": r["run"], "index": None, "failed": ["no_hetero_window"]})
    return {"per_transition": per, "valid": bool(per) and not any(x["failed"] for x in per)}


def non_hetero(p_runs: list[dict], refs: dict | None) -> dict:
    """Phase 6: every solo_short, solo_long and gpu_only stream of P against A at that position."""
    per = []
    nh = (refs or {}).get("non_hetero") or {}
    for r in p_runs:
        for w in r["windows"]:
            if w["condition"] not in NON_HETERO:
                continue
            for s, x in w["streams"].items():
                ref = nh.get(f"{w['index']}:{s}") if refs is not None else None
                rp, rq = (ref or {}).get("p99_ms"), (ref or {}).get("req_s")
                missing = refs is not None and (rp is None or rq is None)
                g = {
                    "p99": sh._gate(x["p99_ms"], None if rp is None else sh.allowed(rp), "<="),
                    "req_s": sh._gate(x["req_s"], None if rq is None else sh.THROUGHPUT_RATIO * rq, ">="),
                }
                if missing:  # A has no reference at this position: cannot pass
                    g = {k: {**v, "pass": False} for k, v in g.items()}
                per.append(
                    {"run": r["run"], "index": w["index"], "condition": w["condition"], "stream": s, **_verdict(g)}
                )
    flags = [x["pass"] for x in per]
    return {
        "windows": per,
        "failing": [f"{x['run']} w{x['index']} {x['stream']}: {', '.join(x['failed'])}" for x in per if x["failed"]],
        "pass": None if None in flags else bool(per) and all(flags),
    }


def soak(p_run: dict) -> dict:
    """Phase 7's extra gates on P's soak run (the per-episode gates are in the transitions)."""
    trs = p_run["transitions"]
    rearm_fail = [
        t["index"]
        for t in trs
        if not (fd := t["structure"].get("first_decision")) or fd["state"] != "sync_guard" or fd["count"] != 1
    ]
    p99 = [t["window_p99_ms"] for t in trs]
    first = sh._median(p99[:SOAK_SPLIT]) if len(p99) >= 2 * SOAK_SPLIT else None
    last = sh._median(p99[-SOAK_SPLIT:]) if len(p99) >= 2 * SOAK_SPLIT else None
    g = {
        "episodes": sh._gate(len(trs), SOAK_EPISODES, "=="),
        "rearm": sh._gate(len(rearm_fail), 0, "=="),
        "degradation": sh._gate(last, None if first is None else SOAK_DEGRADATION * first, "<="),
        "mismatches": sh._gate(p_run["mismatches"], 0, "=="),
    }
    if first is None:
        g["degradation"]["pass"] = False
    return {**_verdict(g), "rearm_failures": rearm_fail, "first15_p99_ms": first, "last15_p99_ms": last}


def multilingual(r: dict) -> dict:
    """Phase 5b: the handoff is not enabled (process placement), 0 mismatches / routing failures,
    no crash."""
    enabled = r["decisions"] > 0 or r["handoff_enabled_at_end"] or r["ane_placement"] not in (None, "process")
    g = {
        "handoff_not_enabled": sh._gate(not enabled, True, "=="),
        "mismatches": sh._gate(r["mismatches"], 0, "=="),
        "routing": sh._gate(len(r["routing_failures"]), 0, "=="),
        "crash": sh._gate(r["crashed"], False, "=="),
    }
    return _verdict(g)


# ------------------------------------------------------------------ phases


def load(path: Path) -> dict:
    with gzip.open(path, "rt") as fh:
        return json.load(fh)


def _present(raw: Path) -> dict[tuple[str, str, int], Path]:
    out = {}
    for p in sorted(raw.glob("*.json.gz")):
        if (x := RUN_NAME.match(p.name)) and (ph := phase_of(x["schedule"], x["model"])):
            out[(ph, x["cell"], int(x["rep"]))] = p
    return out


def _failed(raw: Path) -> list[str]:
    return sorted(p.name for p in (raw / "failed").glob("*.log"))


def _stats(p: Path, phase: str, cell: str, failed: list[str]) -> dict:
    run = load(p)
    assert run.get("experiment") == "coreml-staged-handoff production", p
    name = p.name[: -len(".json.gz")]
    crashed = cell == "P" and any(f.startswith(name + ".") for f in failed)
    return run_stats(run, name, phase, cell, crashed)


def judge_phase(phase: str, stats: dict, failed: list[str]) -> dict:
    keys = RUNS[phase]
    missing = [k for k in keys if (phase, *k) not in stats]
    deviating = [k for k in keys if (phase, *k) in stats and not stats[(phase, *k)]["protocol_ok"]]
    exhausted = [
        k for k in missing if sum(f.startswith(run_name(phase, *k) + ".") for f in failed) >= 2
    ]  # the crash rule: a second failure stops the campaign
    p_runs = [stats[(phase, *k)] for k in keys if k[0] == "P" and (phase, *k) in stats]
    a_keys = [k for k in keys if k[0] == "A"]
    a_runs = [stats[(phase, *k)] for k in a_keys if (phase, *k) in stats]
    refs = references(a_runs, phase) if a_keys and len(a_runs) == len(a_keys) else None
    out = {
        "runs": [run_name(phase, *k) for k in keys],
        "missing": [run_name(phase, *k) for k in missing],
        "deviating": [run_name(phase, *k) for k in deviating],
        "exhausted": [run_name(phase, *k) for k in exhausted],
        "references": refs,
    }
    if phase == "5b":
        out["checks"] = multilingual(p_runs[0]) if p_runs else None
    else:
        per = []
        for r in p_runs:
            for t in r["transitions"]:
                g = gates(t, r, _ref_for(t, refs, phase))
                per.append({"run": r["run"], "index": t["index"], "preceded_by": t["preceded_by"], **g})
        n_exp = len([k for k in keys if k[0] == "P"]) * sum(
            x["condition"] == "hetero" for x in prod_run.schedule(PHASES[phase]["schedule"])
        )
        flags = [x["pass"] for x in per]
        out["judgement"] = {
            "transitions": per,
            "n": len(per),
            "n_expected": n_exp,
            "pass": None if None in flags else len(per) == n_exp and all(flags),
            "failing": [f"{x['run']} w{x['index']}: {', '.join(x['failed'])}" for x in per if x["failed"]],
        }
        if phase == "6":
            out["non_hetero"] = non_hetero(p_runs, refs)
        if phase == "7":
            out["soak"] = soak(p_runs[0]) if p_runs else None
    if any(k[0] == "P" for k in exhausted):
        out.update(status="FAIL", reason="a P run failed twice (crash rule)")
    elif exhausted:
        out.update(status="INCONCLUSIVE", reason="an A run failed twice (harness)")
    elif missing or deviating:
        parts = []
        if missing:
            parts.append("pending: " + ", ".join(out["missing"]))
        if deviating:
            parts.append("protocol deviation (not judged): " + ", ".join(out["deviating"]))
        out.update(status="pending", reason="; ".join(parts))
    else:
        if a_keys:
            out["A_validity"] = av = a_validity(a_runs, phase)
            if not av["valid"]:
                out.update(status="INCONCLUSIVE", reason="an A window failed the validity guard")
                return out
        if phase == "5b":
            ok = out["checks"]["pass"]
        else:
            extras = [out.get("non_hetero"), out.get("soak")]
            ok = out["judgement"]["pass"] is True and all(e["pass"] is True for e in extras if e is not None)
        out.update(status="PASS" if ok else "FAIL", reason=None if ok else "a P gate failed")
    return out


def summarise(raw: Path) -> dict:
    present = _present(raw)
    failed = _failed(raw) if (raw / "failed").exists() else []
    stats = {k: _stats(p, k[0], k[1], failed) for k, p in present.items()}
    phases: dict[str, dict] = {}
    blocker = None
    outcome = None
    for group in ORDER:
        for ph in group:
            j = judge_phase(ph, stats, failed)
            if blocker:
                j.update(judged_status=j["status"], status="not reached", reason=blocker)
            phases[ph] = j
        if blocker:
            continue
        st = {ph: phases[ph]["status"] for ph in group}
        if all(s == "PASS" for s in st.values()):
            continue
        blocker = "; ".join(f"phase {ph} {s}" for ph, s in st.items() if s != "PASS")
        bad = [ph for ph, s in st.items() if s == "FAIL"]
        inc = [ph for ph, s in st.items() if s == "INCONCLUSIVE"]
        if bad:
            outcome = f"phase {', '.join(bad)} FAIL: {CLOSED}; the production change is not proposed"
        elif inc:
            outcome = "INCONCLUSIVE: " + "; ".join(f"phase {ph}: {phases[ph]['reason']}" for ph in inc)
        else:
            outcome = "; ".join(f"phase {ph} {phases[ph]['reason']}" for ph, s in st.items() if s == "pending")
    if outcome is None:
        outcome = CANDIDATE
    planned = {(ph, *k) for ph in RUNS for k in RUNS[ph]}
    unexpected = sorted(present[k].name for k in present if k not in planned)

    return {
        "design": {
            "phases": PHASES,
            "runs": {ph: [list(k) for k in v] for ph, v in RUNS.items()},
            "order": [list(g) for g in ORDER],
            "guard": GUARD,
            "onset_s": [0.0, ONSET_S],
            "steady_s": list(STEADY),
            "soak_steady_from_t_h_s": SOAK_STEADY_FROM_S,
            "allowed": sh.ALLOWED,
            "forward": sh.NATIVE,
            "transient_s": sh.TRANSIENT_S,
            "steady_host_slow": sh.STEADY_HOST_SLOW,
            "gpu_return_from_s": sh.GPU_RETURN_FROM_S,
            "gpu_return_p50_ms": sh.GPU_RETURN_P50_MS,
            "throughput_ratio": sh.THROUGHPUT_RATIO,
            "a_validity": {
                "short_p99_ms": sh.A_P99_MS,
                "transient_s": sh.A_TRANSIENT_S,
                "steady_host_slow": sh.A_STEADY_HOST_SLOW,
                "aggregate_req_s": list(sh.A_AGGREGATE_REQ_S),
                "p99_and_aggregate_phases": list(A_P99_PHASES),
            },
            "soak": {"episodes": SOAK_EPISODES, "split": SOAK_SPLIT, "degradation": SOAK_DEGRADATION},
        },
        "failed_runs": failed,
        "unexpected_files": unexpected,
        "runs": {st["run"]: st for st in stats.values()},
        "phases": phases,
        "outcome": outcome,
    }


# ------------------------------------------------------------------ tables

f = sh.f
_cell = sh._cell


def _gate_row(label: str, v: dict, names) -> str:
    extra = ", ".join(v["failed"]) or ("pending: " + ", ".join(v["pending"]) if v["pending"] else "–")
    return f"| {label} | " + " | ".join(_cell(v["gates"][g]) for g in names) + f" | {extra} |"


def tables(res: dict) -> str:
    L = ["# Staged handoff, production path: Phases 4-7 (criteria.md, addendum 2)\n"]
    L.append(f"Outcome: {res['outcome']}\n")
    L.append(f"Crashed runs and re-runs (`raw-prod/failed/`): {', '.join(res['failed_runs']) or 'none'}\n")
    if res["unexpected_files"]:
        L.append(f"Files outside the plan (not judged): {', '.join(res['unexpected_files'])}\n")
    L += [
        "| phase | status | reason |",
        "|---|---|---|",
    ]
    for ph, j in res["phases"].items():
        extra = f" (would be {j['judged_status']})" if "judged_status" in j else ""
        L.append(f"| {ph} | {j['status']}{extra} | {j.get('reason') or '–'} |")
    L += [
        "\n## Runs\n",
        "| run | protocol ok | mismatches | routing failures | crashed | workers alive | handoff enabled at end "
        "| ANE placement | decisions | wall s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in res["runs"].values():
        L.append(
            f"| {r['run']} | {r['protocol_ok']} | {r['mismatches']} | {', '.join(r['routing_failures']) or '–'} "
            f"| {r['crashed']} | {r['workers_alive']} | {r['handoff_enabled_at_end']} | {r['ane_placement'] or '–'} "
            f"| {r['decisions']} | {f(r['run_wall_s'], 0)} |"
        )
    for ph, j in res["phases"].items():
        c = PHASES[ph]
        L.append(f"\n## Phase {ph}: {c['model']}, L{c['short']} / L{c['long']}, {c['schedule']}\n")
        L.append(f"Status: **{j['status']}**" + (f" ({j['reason']})" if j.get("reason") else "") + "\n")
        trs = [t for r in j["runs"] if r in res["runs"] for t in res["runs"][r]["transitions"]]
        if trs:
            L += [
                "t_h in s from t0. Forward: ANE service_end - service_start (P from t_h, A whole window). "
                "GPU return over [t_h + 1 s, end) (A: whole window).\n",
                "| run | window | after | t_h s | episodes / sync before t_h / valid | latency source | onset P99 "
                "| window P99 | forward mean (n) | transient t0 / t_h s | steady host-slow | GPU return P50 / P95 / P99 "
                "| agg req/s |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
            ]
            for t in trs:
                st, g, th = t["structure"], t["gpu_return_ms"], t["transient_from_th"]
                L.append(
                    f"| {t['run']} | {t['index']} | {t['preceded_by']} | {f(t['t_h_s'], 3)} "
                    f"| {st.get('episodes', '–')} / {f(st.get('sync_before_t_h'))} / {f(st['valid'])} "
                    f"| {t['latency_source']} | {f(t['onset_p99_ms'])} | {f(t['window_p99_ms'])} "
                    f"| {f(t['forward_mean_ms'], 3)} ({t['forward_n']}) | {f(t['transient']['duration_s'], 1)} / "
                    f"{f(None if th is None else th['duration_s'], 1)} | {f(t['steady_host_slow_share'], 3)} "
                    f"| {f(g['p50'], 3)} / {f(g['p95'], 3)} / {f(g['p99'], 3)} | {f(t['aggregate_req_s'], 1)} |"
                )
            L.append("")
        refs = j["references"]
        if refs and refs.get("all"):
            a = refs["all"]
            L.append(
                f"A references (median over {len(a['transitions'])} A hetero windows"
                f"{'; gates use the per-position medians' if ph == '6' else ''}): onset P99 "
                f"{f(a['A_onset_p99_ms'])} ms, window P99 {f(a['A_window_p99_ms'])} ms, forward mean "
                f"{f(a['A_forward_mean_ms'], 3)} ms, aggregate {f(a['A_agg_req_s'], 1)} req/s\n"
            )
        if "A_validity" in j:
            av = j["A_validity"]
            bad = [x for x in av["per_transition"] if x["failed"]]
            L.append(f"A validity: **{av['valid']}**" + ("" if not bad else " – failing:") + "\n")
            for x in bad:
                L.append(f"- {x['run']} w{x['index']}: {', '.join(x['failed'])}")
            if bad:
                L.append("")
        if j.get("checks"):
            names = list(j["checks"]["gates"])
            L += ["| run | " + " | ".join(names) + " | failed |", "|---|" + "---|" * (len(names) + 1)]
            L.append(_gate_row(j["runs"][0], j["checks"], names))
            L.append(f"\nPhase 5b checks pass: **{j['checks']['pass']}**\n")
        jd = j.get("judgement")
        if jd and jd["transitions"]:
            L += ["| transition | " + " | ".join(GATES) + " | failed |", "|---|" + "---|" * (len(GATES) + 1)]
            for x in jd["transitions"]:
                L.append(_gate_row(f"{x['run']} w{x['index']} (after {x['preceded_by']})", x, GATES))
            L.append(f"\nEvery P transition passes: **{jd['pass']}** ({jd['n']} of {jd['n_expected']} transitions)\n")
        nh = j.get("non_hetero")
        if nh and nh["windows"]:
            L += [
                "Non-hetero windows (P against A at the same position):\n",
                "| window | p99 | req_s | failed |",
                "|---|---|---|---|",
            ]
            for x in nh["windows"]:
                L.append(_gate_row(f"{x['run']} w{x['index']} {x['condition']} {x['stream']}", x, ("p99", "req_s")))
            L.append(f"\nNon-hetero gates pass: **{nh['pass']}**\n")
        sk = j.get("soak")
        if sk:
            names = list(sk["gates"])
            L += ["Soak gates:\n", "| run | " + " | ".join(names) + " | failed |", "|---|" + "---|" * (len(names) + 1)]
            L.append(_gate_row(j["runs"][0], sk, names))
            L.append(
                f"\nFirst 15 / last 15 median window P99: {f(sk['first15_p99_ms'])} / {f(sk['last15_p99_ms'])} ms; "
                f"re-arm failures at windows: {', '.join(map(str, sk['rearm_failures'])) or 'none'}\n"
            )
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if prod_results.json / prod_tables.md are stale")
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=EXP)
    a = ap.parse_args()
    res = summarise(a.raw)
    js = json.dumps(res, indent=1, sort_keys=True, default=a94._json) + "\n"
    md = tables(res)
    targets = ((a.out / "prod_results.json", js), (a.out / "prod_tables.md", md))
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
