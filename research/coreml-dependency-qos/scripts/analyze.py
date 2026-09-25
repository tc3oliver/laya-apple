"""The dependency QoS screen: results.json and tables.md from raw/ (../criteria.md).

    uv run python research/coreml-dependency-qos/scripts/analyze.py [--check] [--raw DIR --out DIR]

Inputs: raw/laya-<cell>-r<round>.json.gz (run_config.py; design.py) and raw/failed/*.log. The
thresholds are criteria.md's; the constants below carry them. #94's analyze.py
(research/coreml-async-transient/scripts/, loaded by path, unchanged) supplies every derived measure:
the requests joined with the native stamps, the per-thread counter deltas and metrics, the
transient metrics and #94's buckets.

Per transition (t0 = the hetero window's start):
- #94's transient metrics (host-slow request = prepare > 0.3 ms, 0.5 s bins, recovery, duration,
  present >= 0.5 s, peak short P99 over 1 s bins in [t0, t0 + 5 s)) and #94's buckets;
- onset measures over [t0, t0 + 4 s): E share of the ANE dispatcher, client-short and the callback
  threads, the ANE dispatcher's CPU per forward, the transient duration;
- the deciding measure: the ANE dispatcher's E share over W = [t0 + 0.5 s, t0 + 4 s), classified
  E-resident (>= 0.50), avoided (<= 0.10) or partial (between; counts as not avoided); no counter
  data is "unassessable" (not avoided, not E-resident);
- client-short's and the callback threads' E share over W (CHAIN-E).

Per run: native Core ML mean and GPU-return P50 pooled over both hetero windows, mismatches and
routing in every window, aggregate req/s per hetero window (#92's: the sum of the streams' req/s)
and pooled over both (requests / seconds), the dispatcher capture, the requested QoS records, and in
O the override's structural validity.

Per round: B's validity, each candidate's guards against that round's B, MECH / LAT / CHAIN-E, its
outcome row and its placement change; round 1's stop rule and round-2 trigger; after round 2, the
leading candidate (O preferred over Q).
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


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


design = _load("dependency_qos_design", HERE / "design.py")
a94 = _load("async_transient_analyze", ASYNC94 / "analyze.py")  # #94's, unchanged

S = a94.S
EPS = a94.EPS
ONSET = (0.0, 4.0)
W = (0.5, 4.0)
E_RESIDENT = 0.50
AVOIDED = 0.10
CHAIN_E = 0.50
LAT_S = 1.0
GUARD_NATIVE = {"ratio": 1.05, "plus_ms": 0.3}
GPU_RETURN_P50_MS = 1.0
THROUGHPUT_RATIO = 0.95
B_AGGREGATE_REQ_S = (96.5, 141.7)  # 0.9 x 107.3 to 1.1 x 128.8, as preregistered
USER_INITIATED = 0x19
CHAIN_THREADS = ("client-short", "callback")
ONSET_THREADS = ("ane-dispatch", "client-short", "callback")
# routing per condition: the auto instance sends short to the ANE and long to the GPU
ROUTES = {
    "solo_short": {"short": {"ane"}},
    "solo_long": {"long": {"gpu"}},
    "hetero": {"short": {"ane"}, "long": {"gpu"}},
    "gpu_only": {"short": {"gpu"}, "long": {"gpu"}},
}
RUN_NAME = re.compile(rf"^{design.MODEL}-(?P<cell>B|O|Q)-r(?P<round>[12])\.json\.gz$")

OUTCOMES = {
    "PASS": "PASS: it replicates",
    "LAT_ONLY": "latency improved, but #96's mechanism did not change: QoS is not claimed to fix it; no replication",
    "MECH_CHAIN": "dispatcher-only QoS changes placement but is insufficient. The rest of the dependency chain "
    "stays on E. This is not a disproof of the QoS route; no replication",
    "MECH_ONLY": "QoS affects placement but is insufficient; not productionized; no replication",
    "NEITHER": "the dispatcher's placement did not change enough; no replication",
    "GUARD": "the failed guard is reported; no replication",
}
B_INVALID = (
    "INCONCLUSIVE: B is not valid in this round; no candidate is judged, nothing replicates automatically, "
    "the next step is a human decision"
)
STOP = (
    "STOP this USER_INITIATED dispatcher-QoS route: no further QoS variants and no USER_INTERACTIVE "
    "trial-and-error; a USER_INTERACTIVE ceiling test may be opened later only as a separately "
    "preregistered diagnostic, by human decision, never as a mitigation candidate from this screen; "
    "the next step is re-evaluating the architecture or scheduler-state priming"
)
CHAIN_FOLLOW_UP = (
    "one follow-up is allowed, an architecture decision: whether a clean, public, production-safe way "
    "exists to propagate QoS along the whole dependency chain (client -> dispatcher -> Core ML callback); "
    "it does not try USER_INTERACTIVE, a warm-up, a probe or any other parameter"
)
HUMAN = "reported as observed; nothing replicates; the next step is a human decision (no parameter trials)"
R2_INCONCLUSIVE = "INCONCLUSIVE: not productionized; the next step is a human decision"
NOT_PRODUCTION = (
    "a leading candidate is not a production change: a separate change has to cover the override lifetime "
    "with several concurrent requests, cancellation, exception paths and shutdown, no override leak, energy, "
    "typed-decisions and the full product-mix gate"
)
CHANGE = {0: "changed nothing", 1: "changed placement in some transitions", 2: "changed placement in both"}


def _mean(x):
    return float(np.mean(x)) if len(x) else None


def _pct(x, q):
    return float(np.percentile(x, q)) if len(x) else None


# ------------------------------------------------------------------ per transition


def classify(e_share: float | None) -> str:
    if e_share is None:
        return "unassessable"
    if e_share >= E_RESIDENT - EPS:
        return "E-resident"
    if e_share <= AVOIDED + EPS:
        return "avoided"
    return "partial"


def _e_shares(run: dict, rw: dict, t0: int, span: tuple[float, float]) -> dict:
    rc = run["research"].get("recount") or {}
    levels = rc.get("levels", 2)
    groups = a94.counter_groups(run, rw)
    lo, hi = t0 + int(span[0] * S), t0 + int(span[1] * S)
    out = {}
    for g in ONSET_THREADS:
        m = a94.counter_metrics(a94.counter_delta(rc.get("t_ns", []), groups[g], levels, lo, hi), levels)
        out[g] = None if m is None else m["e_share"]
    return out


def transitions(run: dict, name: str, cell: str) -> list[dict]:
    """#94's transitions with the onset measures, W and the classification added."""
    base = a94.transitions(run, name, cell)
    rws = {k: rw for k, _, rw in a94.hetero(run)}
    ane = a94.forwards(run, "ane")
    out = []
    for t in base:
        rw, t0 = rws[t["cycle"]], t["t0_ns"]
        lo, hi = t0 + int(ONSET[0] * S), t0 + int(ONSET[1] * S)
        fa = ane[(ane[:, 0] >= lo) & (ane[:, 0] < hi)]
        onset = {f"{g}_e_share": v for g, v in _e_shares(run, rw, t0, ONSET).items()}
        onset["ane_cpu_per_forward_ms"] = _mean(fa[:, 2] / 1e6) if len(fa) else None
        onset["transient_s"] = t["transient"]["duration_s"]
        w = _e_shares(run, rw, t0, W)
        t["onset"] = onset
        t["W"] = {f"{g}_e_share": v for g, v in w.items()}
        t["class"] = classify(w["ane-dispatch"])
        t["chain_e"] = any(w[g] is not None and w[g] >= CHAIN_E - EPS for g in CHAIN_THREADS)
        out.append(t)
    return out


# ------------------------------------------------------------------ per run


def _routing_failures(pw: dict) -> list[str]:
    want = ROUTES.get(pw["condition"], {})
    return [s for s, dev in want.items() if s in pw["streams"] and set(pw["streams"][s].get("devices") or {}) != dev]


def override_validity(ov: dict | None, dispatcher: dict | None, bounds: list[tuple[int, int]]) -> dict:
    """O's structural validity (criteria.md, "O only"). Returns the checks and ok."""
    if not ov:
        return {"checks": {"recorded": False}, "ok": False}
    o = ov["overrides"]
    n = len(o["start_ns"])
    cap = (dispatcher or {}).get("captured_ns")
    exit_ns = (dispatcher or {}).get("exit_ns")
    ended = [e for e in o["end_ns"] if e]
    checks = {
        "recorded": True,
        "starts": ov["starts"] > 0 and ov["starts"] == n,
        "no_null_start": ov["null_starts"] == 0,
        "one_end_each": ov["ends"] == ov["starts"] and ov["double_ends"] == 0 and len(ended) == n,
        "end_rc_zero": ov["end_errors"] == 0 and all(r == 0 for r in o["end_rc"]),
        "none_outstanding": ov["outstanding_at_end"] == 0 and ov["outstanding"] == 0,
        "lifetime": cap is not None
        and all(s >= cap for s in o["start_call_ns"])
        and (exit_ns is None or all(e and e <= exit_ns for e in o["end_ret_ns"])),
        "active_each_hetero_window": bool(bounds)
        and all(any(s < hi and e >= lo for s, e in zip(o["start_ns"], o["end_ns"])) for lo, hi in bounds),
    }
    return {"checks": checks, "ok": all(checks.values())}


def override_cost_us(ov: dict | None) -> dict | None:
    if not ov or not ov["overrides"]["start_ns"]:
        return None
    o = ov["overrides"]
    st = (np.asarray(o["start_ns"]) - np.asarray(o["start_call_ns"])) / 1e3
    en = np.asarray([r - e for e, r in zip(o["end_ns"], o["end_ret_ns"]) if e], float) / 1e3
    return {
        "start_p50": _pct(st, 50),
        "start_p99": _pct(st, 99),
        "end_p50": _pct(en, 50),
        "end_p99": _pct(en, 99),
    }


def dispatcher_alignment(run: dict) -> dict:
    """Was the pthread_t captured on the laya-ane-dispatch thread the counters sample?"""
    d = (run["research"].get("qos") or {}).get("dispatcher") or {}
    series = ((run["research"].get("recount") or {}).get("series") or {}).values()
    tids = sorted({s["tid"] for s in series if s["name"] == "laya-ane-dispatch"})
    return {
        "thread_name": d.get("thread_name"),
        "native_id": d.get("native_id"),
        "pthread_threadid": d.get("pthread_threadid"),
        "recount_dispatch_tids": tids,
        "ok": bool(
            d
            and d.get("thread_name") == "laya-ane-dispatch"
            and d.get("native_id") == d.get("pthread_threadid")
            and tids == [d.get("native_id")]
        ),
    }


def run_stats(run: dict, name: str, cell: str) -> dict:
    trs = transitions(run, name, cell)
    hw = {k: pw for k, pw, _ in a94.hetero(run)}
    bounds = [(t["t0_ns"], t["t0_ns"] + int(t["window_s"] * S)) for t in trs]
    native = np.concatenate([t["_rq"]["native"] for t in trs]) if trs else np.zeros(0)
    recv, ret = a94.gpu_returns(run)
    gm = np.zeros(len(recv), bool)
    for lo, hi in bounds:
        gm |= (recv >= lo) & (recv < hi)
    per_window, n_req, n_s = [], 0, 0.0
    for t in trs:
        st = hw[t["cycle"]]["streams"]
        agg = sum(x["req_s"] for x in st.values())
        per_window.append(agg)
        n = {s: len(x.get("latency_ms") or []) or x.get("n") or 0 for s, x in st.items()}
        n_req += sum(n.values())
        n_s += max((n[s] / x["req_s"] for s, x in st.items() if x["req_s"]), default=0.0)
    windows = run["part_a"]["windows"]
    mismatches = sum(s["mismatches"] for w in windows for s in w["streams"].values())
    routing = [f"{w['cycle']}:{w['condition']}:{s}" for w in windows for s in _routing_failures(w)]
    q = run["research"].get("qos") or {}
    ov = q.get("override")
    reqs = (ov or {}).get("short_requests") or {}
    return {
        "run": name,
        "cell": cell,
        "transitions": trs,
        "native_mean_ms": a94._nanmean(native),
        "gpu_return_p50_ms": _pct(ret[gm], 50),
        "mismatches": mismatches,
        "routing_failures": routing,
        "aggregate_req_s_per_window": per_window,
        "aggregate_req_s_pooled": n_req / n_s if n_s else None,
        "qos": {
            "at_load": q.get("at_load"),
            "set": q.get("set"),
            "at_end": q.get("at_end"),
            "errors": q.get("errors"),
        },
        "dispatcher": dispatcher_alignment(run),
        "override": None
        if ov is None
        else {
            **{k: v for k, v in ov.items() if k not in ("overrides", "short_requests")},
            "cost_us": override_cost_us(ov),
            "short_requests": len(reqs.get("device") or []),
            "short_off_ane": sum(1 for d in reqs.get("device") or [] if d != "ane"),
        },
        "override_validity": override_validity(ov, q.get("dispatcher"), bounds) if cell == "O" else None,
        "q_readback": q_readback(q) if cell == "Q" else None,
    }


def q_readback(q: dict) -> dict:
    after = ((q.get("set") or {}).get("readback") or {}).get("qos")
    end = (q.get("at_end") or {}).get("qos")
    return {"after_call": after, "at_end": end, "ok": after == USER_INITIATED and end == USER_INITIATED}


# ------------------------------------------------------------------ per round


def b_validity(b: dict) -> dict:
    lo, hi = B_AGGREGATE_REQ_S
    checks = {
        "e_resident_transition": any(t["class"] == "E-resident" for t in b["transitions"]),
        "no_mismatch": b["mismatches"] == 0,
        "gpu_return_p50": b["gpu_return_p50_ms"] is not None and b["gpu_return_p50_ms"] <= GPU_RETURN_P50_MS + EPS,
        "aggregate_in_range": len(b["aggregate_req_s_per_window"]) == design.CYCLES
        and all(lo - EPS <= x <= hi + EPS for x in b["aggregate_req_s_per_window"]),
    }
    return {"checks": checks, "valid": all(checks.values()), "failed": [k for k, v in checks.items() if not v]}


def guards(c: dict, b: dict) -> dict:
    """A candidate's guards against B of the same round."""
    cn, bn = c["native_mean_ms"], b["native_mean_ms"]
    ca, ba = c["aggregate_req_s_pooled"], b["aggregate_req_s_pooled"]
    checks = {
        "native": cn is not None
        and bn is not None
        and not (cn > bn * GUARD_NATIVE["ratio"] + EPS and cn > bn + GUARD_NATIVE["plus_ms"] + EPS),
        "gpu_return_p50": c["gpu_return_p50_ms"] is not None and c["gpu_return_p50_ms"] <= GPU_RETURN_P50_MS + EPS,
        "correctness": c["mismatches"] == 0 and not c["routing_failures"],
        "throughput": ca is not None and ba is not None and ca >= ba * THROUGHPUT_RATIO - EPS,
    }
    if c["cell"] == "O":
        checks["override_validity"] = bool((c.get("override_validity") or {}).get("ok"))
    if c["cell"] == "Q":
        checks["q_readback"] = bool((c.get("q_readback") or {}).get("ok"))
    return {"checks": checks, "pass": all(checks.values()), "failed": [k for k, v in checks.items() if not v]}


def outcome(mech: bool, lat: bool, chain_e: bool, guards_pass: bool, failed=()) -> str:
    if mech and lat:
        if guards_pass:
            return OUTCOMES["PASS"]
        return OUTCOMES["GUARD"] + (f" (failed: {', '.join(failed)})" if failed else "")
    if lat:
        return OUTCOMES["LAT_ONLY"]
    if mech:
        return OUTCOMES["MECH_CHAIN"] if chain_e else OUTCOMES["MECH_ONLY"]
    return OUTCOMES["NEITHER"]


def placement_change(classes: list[str]) -> str:
    return CHANGE[min(2, sum(1 for c in classes if c == "avoided"))]


def judge(c: dict, b: dict) -> dict:
    trs = c["transitions"]
    classes = [t["class"] for t in trs]
    mech = len(trs) == design.CYCLES and all(x == "avoided" for x in classes)
    lat = len(trs) == design.CYCLES and all(t["transient"]["duration_s"] <= LAT_S + EPS for t in trs)
    chain = any(t["chain_e"] for t in trs)
    g = guards(c, b)
    text = outcome(mech, lat, chain, g["pass"], g["failed"])
    return {
        "classes": classes,
        "MECH": mech,
        "LAT": lat,
        "CHAIN_E": chain,
        "guards": g,
        "placement_change": placement_change(classes),
        "pass": text == OUTCOMES["PASS"],
        "outcome": text,
    }


def judge_round(stats: dict[str, dict]) -> dict:
    """One round alone: B's validity, then each candidate present in the round."""
    b = stats.get("B")
    if b is None:
        return {"complete": False}
    bv = b_validity(b)
    cands = {}
    for c in design.CANDIDATES:
        if c in stats:
            cands[c] = judge(stats[c], b) if bv["valid"] else {"outcome": B_INVALID, "pass": False}
    return {"complete": True, "B": bv, "candidates": cands, "passing": [c for c, j in cands.items() if j["pass"]]}


def round1_decision(r1: dict) -> dict:
    """The stop rule, the chain follow-up and the round-2 trigger, from round 1."""
    if not r1.get("complete") or not r1["B"]["valid"]:
        return {"stop": False, "round2": [], "next": B_INVALID if r1.get("complete") else None}
    cands = r1["candidates"]
    passing = r1["passing"]
    stop = all(cands[c]["placement_change"] == CHANGE[0] for c in design.CANDIDATES if c in cands) and set(
        cands
    ) == set(design.CANDIDATES)
    if passing:
        nxt = "round 2 runs: " + ", ".join(f"{c} r{r}" for c, r in design.round2(passing))
    elif stop:
        nxt = STOP
    elif any(j["outcome"] == OUTCOMES["MECH_CHAIN"] for j in cands.values()):
        nxt = CHAIN_FOLLOW_UP
    else:
        nxt = HUMAN
    return {"stop": bool(stop and not passing), "round2": passing, "next": nxt}


def leading(r1: dict, r2: dict | None) -> dict:
    """Candidates that passed both rounds; O preferred, Q the fallback."""
    passed1 = r1.get("passing") or []
    passed2 = (r2 or {}).get("passing") or []
    both = [c for c in ("O", "Q") if c in passed1 and c in passed2]
    per = {c: ("leads" if c in both else R2_INCONCLUSIVE) for c in passed1}
    lead = both[0] if both else None
    return {
        "per_candidate": per,
        "leading": lead,
        "fallback": "Q" if lead == "O" and "Q" in both else None,
        "text": (
            f"leading candidate: {lead}" + (" (Q is the fallback)" if lead == "O" and "Q" in both else "")
            if lead
            else "no leading candidate"
        ),
    }


# ------------------------------------------------------------------ summary


def load(path: Path) -> dict:
    with gzip.open(path, "rt") as fh:
        return json.load(fh)


def _present(raw: Path) -> dict[tuple[str, int], Path]:
    out = {}
    for p in sorted(raw.glob(f"{design.MODEL}-*-r*.json.gz")):
        if x := RUN_NAME.match(p.name):
            out[(x["cell"], int(x["round"]))] = p
    return out


def _stats(p: Path, cell: str) -> dict:
    run = load(p)
    res = run["research"]
    assert res["experiment"] == "coreml-dependency-qos" and res["cell"] == cell, p
    assert res["base"]["cell"] == "PB-ASYNC", p
    assert run["args"]["cycles"] == design.CYCLES and run["args"]["seconds"] == design.SECONDS, p
    st = run_stats(run, p.name[: -len(".json.gz")], cell)
    st["run_wall_s"] = res.get("run_wall_total_s", res.get("run_wall_s"))
    rc = res.get("recount") or {}
    st["recount"] = {
        "samples": len(rc.get("t_ns", [])),
        "series": len(rc.get("series", {})),
        "errors": rc.get("errors"),
        "sampler_cpu_fraction_of_core": rc.get("sampler_cpu_fraction_of_core"),
    }
    cb = res.get("callbacks") or {}
    st["callbacks"] = {k: cb.get(k) for k in ("submits", "total", "errors")} if cb else None
    st["predicts_native"] = sum(1 for r in res.get("predicts") or [] if r[3] > 0)
    return st


def _round(present, rnd: int, runs) -> tuple[dict, dict]:
    stats = {c: _stats(present[(c, r)], c) for c, r in runs if (c, r) in present}
    if not all((c, r) in present for c, r in runs):
        return stats, {"complete": False, "missing": [f"{c} r{r}" for c, r in runs if (c, r) not in present]}
    return stats, judge_round(stats)


def round2_trigger(raw: Path) -> list[str]:
    """The candidates round 2 runs for (design.round2), judged from round 1's raw files."""
    stats, r1 = _round(_present(raw), 1, design.ROUND1)
    return round1_decision(r1)["round2"]


def summarise(raw: Path) -> dict:
    present = _present(raw)
    s1, r1 = _round(present, 1, design.ROUND1)
    d1 = round1_decision(r1)
    s2, r2, lead = {}, None, None
    runs2 = design.round2(d1["round2"])
    if not r1["complete"]:
        outcome_text = "running: " + ", ".join(r1["missing"])
    elif not runs2:
        outcome_text = d1["next"]
    else:
        s2, r2 = _round(present, 2, runs2)
        if not r2["complete"]:
            outcome_text = "round 1 passed " + ", ".join(d1["round2"]) + "; running: " + ", ".join(r2["missing"])
        else:
            if not r2["B"]["valid"]:
                r2["note"] = "B is not valid in round 2: every round-1 pass is INCONCLUSIVE"
            lead = leading(r1, r2)
            outcome_text = lead["text"] + (f"; {NOT_PRODUCTION}" if lead["leading"] else "")
    extra = [p for (c, r), p in present.items() if (r == 2 and (c, r) not in runs2)]

    def public(stats):
        return {
            c: {
                **{k: v for k, v in st.items() if k != "transitions"},
                "transitions": [
                    {k: v for k, v in t.items() if not k.startswith("_") and k != "periods"} for t in st["transitions"]
                ],
            }
            for c, st in stats.items()
        }

    return {
        "design": {
            "round1": [list(x) for x in design.ROUND1],
            "round2": [list(x) for x in runs2],
            "onset_s": list(ONSET),
            "W_s": list(W),
            "e_resident": E_RESIDENT,
            "avoided": AVOIDED,
            "chain_e": CHAIN_E,
            "lat_s": LAT_S,
            "b_aggregate_req_s": list(B_AGGREGATE_REQ_S),
            "buckets": [list(b) for b in a94.BUCKETS],
        },
        "failed_runs": sorted(p.name for p in (raw / "failed").glob("*.log")),
        "unexpected_round2_files": sorted(p.name for p in extra),
        "rounds": {
            "1": {"runs": public(s1), "judgement": r1, "decision": d1},
            "2": None if r2 is None else {"runs": public(s2), "judgement": r2},
        },
        "leading": lead,
        "outcome": outcome_text,
    }


# ------------------------------------------------------------------ tables


def f(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def _hex(q):
    return "–" if not q else f"0x{q['qos']:02x}"


def tables(res: dict) -> str:
    L = ["# Dependency QoS screen: PB-ASYNC's hetero onset under B, O and Q (laya, L128 / L512)\n"]
    L.append(f"Outcome: {res['outcome']}\n")
    L.append(f"Crashed runs and re-runs (`raw/failed/`): {', '.join(res['failed_runs']) or 'none'}\n")
    for rnd in ("1", "2"):
        R = res["rounds"][rnd]
        if R is None:
            continue
        L += [
            f"\n## Round {rnd}: runs\n",
            "| run | mismatches | routing failures | native mean ms | GPU return P50 ms | agg req/s per window "
            "| agg req/s pooled | dispatcher tid ok | QoS load / after set / end | overrides | wall s |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for c, r in R["runs"].items():
            q, ov = r["qos"], r["override"]
            ovs = "–"
            if ov:
                cost = ov["cost_us"] or {}
                ovs = (
                    f"{ov['starts']} starts / {ov['ends']} ends / {ov['null_starts']} NULL / {ov['end_errors']} "
                    f"end errors / {ov['outstanding_at_end']} outstanding; start {f(cost.get('start_p50'), 1)} / "
                    f"{f(cost.get('start_p99'), 1)} µs, end {f(cost.get('end_p50'), 1)} / "
                    f"{f(cost.get('end_p99'), 1)} µs (P50 / P99); off-ANE {ov['short_off_ane']}"
                )
            L.append(
                f"| {r['run']} | {r['mismatches']} | {', '.join(r['routing_failures']) or '–'} "
                f"| {f(r['native_mean_ms'], 3)} | {f(r['gpu_return_p50_ms'], 3)} "
                f"| {' / '.join(f(x, 1) for x in r['aggregate_req_s_per_window'])} | {f(r['aggregate_req_s_pooled'], 1)} "
                f"| {r['dispatcher']['ok']} | {_hex(q['at_load'])} / {_hex((q['set'] or {}).get('readback'))} / "
                f"{_hex(q['at_end'])} | {ovs} | {f(r['run_wall_s'], 0)} |"
            )
        L += [
            f"\n## Round {rnd}: transitions\n",
            "Onset [t0, t0 + 4 s); W = [t0 + 0.5 s, t0 + 4 s). E-resident: dispatcher E share over W >= 0.50; "
            "avoided: <= 0.10; between: partial (not avoided).\n",
            "| run | cycle | after | onset E share dispatcher / client-short / callback | onset ANE CPU/fwd ms "
            "| W E share dispatcher / client-short / callback | class | CHAIN-E | transient s | peak short P99 ms |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in R["runs"].values():
            for t in r["transitions"]:
                o, w = t["onset"], t["W"]
                L.append(
                    f"| {t['run']} | {t['cycle']} | {t['preceded_by']} "
                    f"| {' / '.join(f(o[f'{g}_e_share']) for g in ONSET_THREADS)} | {f(o['ane_cpu_per_forward_ms'], 3)} "
                    f"| {' / '.join(f(w[f'{g}_e_share']) for g in ONSET_THREADS)} | {t['class']} | {t['chain_e']} "
                    f"| {f(t['transient']['duration_s'], 1)} | {f(t['transient']['peak_short_p99_ms'])} |"
                )
        L += [
            f"\n## Round {rnd}: onset buckets (seconds from t0, #94's buckets)\n",
            "| run | cycle | bucket | n | short P99 ms | host-slow | E share dispatcher / client-short / callback "
            "| ANE CPU/fwd ms | native ms |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in R["runs"].values():
            for t in r["transitions"]:
                for b, x in t["buckets"].items():
                    cn = x["counters"]
                    es = " / ".join(f((cn.get(g) or {}).get("e_share")) for g in ONSET_THREADS)
                    L.append(
                        f"| {t['run']} | {t['cycle']} | {b} | {x['short_n']} | {f(x['short_p99_ms'])} "
                        f"| {f(x['host_slow_share'])} | {es} | {f(x['ane_cpu_per_forward_ms'], 3)} "
                        f"| {f(x['native_mean_ms'], 3)} |"
                    )
        j = R["judgement"]
        if not j.get("complete"):
            continue
        L += [f"\n## Round {rnd}: B validity\n"]
        L.append(
            f"Valid: **{j['B']['valid']}**"
            + (f" (failed: {', '.join(j['B']['failed'])})" if j["B"]["failed"] else "")
            + f". Aggregate req/s range per hetero window: [{B_AGGREGATE_REQ_S[0]}, {B_AGGREGATE_REQ_S[1]}].\n"
        )
        L += [
            f"\n## Round {rnd}: candidates\n",
            "| candidate | classes | MECH | LAT | CHAIN-E | placement change | failed guards | outcome |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for c, x in j["candidates"].items():
            if "classes" not in x:
                L.append(f"| {c} | – | – | – | – | – | – | {x['outcome']} |")
                continue
            L.append(
                f"| {c} | {', '.join(x['classes'])} | {x['MECH']} | {x['LAT']} | {x['CHAIN_E']} "
                f"| {x['placement_change']} | {', '.join(x['guards']['failed']) or '–'} | {x['outcome']} |"
            )
        if rnd == "1":
            L.append(f"\nAfter round 1: {R['decision']['next']}\n")
        elif j.get("note"):
            L.append(f"\n{j['note']}\n")
    if res.get("leading"):
        L += ["\n## Leading candidate\n", res["leading"]["text"]]
        for c, v in res["leading"]["per_candidate"].items():
            L.append(f"- {c}: {v}")
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
