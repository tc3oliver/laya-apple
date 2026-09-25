"""The async transient experiment: results.json and tables.md from raw/.

    uv run python research/coreml-async-transient/scripts/analyze.py [--check]

Inputs: raw/laya-<cell>-r<n>.json.gz (run_config.py; design.RUNS) and raw/failed/*.log. The
definitions and thresholds are ../criteria.md's; the constants below carry them.

Per transition (each hetero window; t0 = its start_at) and per bucket (BUCKETS, seconds from t0,
requests by submit time, forwards by service start, counters interpolated at the bucket bounds):
short-client P50/P99, prepare wall and host-slow share, ANE stages (features, native Core ML =
native_completion - submit_after, native completion -> callback entry, callback -> waiter wake,
post, action head), ANE dispatcher and GPU worker CPU per forward (forwards records), and from
the per-thread perf-level counters (recount.py) per thread group: CPU on P and E cores, E share,
relative effective cycle rate per level (cycles / CPU time: relative only, not a clock
frequency), IPC, CPU per request or forward.

Transient metrics (`transient`): host-slow request = prepare > 0.3 ms; 0.5 s bins from t0 to
t0 + 20 s (or the window end); recovery = start of the first bin from which every later
non-empty bin has host-slow share < 10%; duration = recovery - t0 (the window length if never);
present = duration >= 0.5 s; peak short P99 = max over 1 s bins in [t0, t0 + 5 s) of the client
P99. Transient period = [t0, recovery); steady = [t0 + 10 s, t0 + 20 s).

Evaluators (PB-ASYNC, all 4 of its transitions; transient periods pooled over its transitions
with a transient, steady pooled over all of them):
  native_rises    mean native > steady x 1.10 and > steady + 0.3 ms
  callback_rises  P50 (callback entry - native completion) > 3 x steady and > steady + 0.2 ms
  cpu_rises       mean ANE dispatcher CPU/forward >= 2 x steady
  sched_changes   per-thread counters, per transition with a transient, for the ANE dispatcher,
                  client-short and callback threads, transient vs reference ([t0+4, t0+5) if
                  recovered by t0 + 4 s, else [t0-2, t0)): E-level CPU share +25 points or more, or
                  P-level relative effective cycle rate <= 0.8 x reference. ipc_drop (IPC <= 0.8 x
                  reference on one of those threads) is reported with it.
  transient_resolves  a transient in >= 1 PB-ASYNC transition, every one recovered before
                  t0 + 5 s, and PB-ASYNC steady: GPU return P50 <= 1 ms, aggregate req/s >= A's
                  steady x 0.95, short P99 <= A's steady x 1.05.
Readings (all that apply, no single verdict): A native_rises; B callback_rises and not
native_rises; C cpu_rises and sched_changes; D cpu_rises and not sched_changes (with the IPC
clause if ipc_drop); E transient_resolves. No PB-ASYNC transient at all: "phenotype not
reproduced: no causal reading".

Validity guard on A (internal validity control, per A hetero window; A_VALIDITY_*): short client
P99 over the whole window >= 13.0 ms, transient duration > 1.0 s, host-slow share over steady
[t0+10, t0+20) >= 10%, aggregate req/s outside [0.9 x 122.1, 1.1 x 122.7], or any mismatch or
failed routing. Any of them in any A window replaces the readings with VALIDITY_CONCERN; every
number is still reported.

--raw / --out exist for checking the harness on runs kept outside the repository.
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


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


design = _load("async_transient_design", HERE / "design.py")

S = 1_000_000_000
BUCKETS = (("-2-0", -2.0, 0.0), ("0-0.5", 0.0, 0.5), ("0.5-1", 0.5, 1.0), ("1-2", 1.0, 2.0))
BUCKETS += (("2-4", 2.0, 4.0), ("4-5", 4.0, 5.0), ("steady 10-20", 10.0, 20.0))
HOST_SLOW_PREPARE_MS = 0.3
BIN_S = 0.5
HOST_SLOW_SHARE = 0.10
HORIZON_S = 20.0
PRESENT_S = 0.5
PEAK_SPAN_S = 5.0
STEADY = (10.0, 20.0)
RESOLVE_BY_S = 5.0
REFERENCE_RECOVERED_BY_S = 4.0
NATIVE = {"ratio": 1.10, "plus_ms": 0.3}
CALLBACK = {"ratio": 3.0, "plus_ms": 0.2}
CPU_RATIO = 2.0
SCHED = {"e_share_points": 0.25, "cycle_rate_ratio": 0.8, "ipc_ratio": 0.8}
EPS = 1e-9  # threshold comparisons are inclusive; guards against float rounding at the edge
RESOLVE = {"gpu_return_p50_ms": 1.0, "aggregate_ratio": 0.95, "short_p99_ratio": 1.05}
SCHED_THREADS = ("ane-dispatch", "client-short", "callback")
GROUPS = ("ane-dispatch", "client-short", "client-long", "gpu-dispatch", "callback", "gpu-worker", "main")
# Validity guard on A: A must keep its historical phenotype. #92's A hetero windows had short P99
# 11.0-12.9 ms (R1's slow-window classifier: >= 13.0 ms) and 122.1-122.7 aggregate req/s; #93
# located A's host-slow transient at <= ~0.3 s. A sustained host-slow state in A's steady span
# (>= 10%, the transient's own share threshold), a mismatch or a failed routing is never A's phenotype.
A_VALIDITY_SHORT_P99_MS = 13.0
A_VALIDITY_TRANSIENT_S = 1.0
A_VALIDITY_STEADY_HOST_SLOW = 0.10
A_VALIDITY_AGGREGATE_REQ_S = (0.9 * 122.1, 1.1 * 122.7)
ROUTES = (("short", "ane"), ("long", "gpu"))
IPC_TEXT = "; supports further investigation of shared-resource / memory-hierarchy effects"
PHENOTYPE_NOT_REPRODUCED = "phenotype not reproduced: no causal reading"
VALIDITY_CONCERN = "validity concern: A deviates from its historical phenotype (#92); no causal interpretation"
READINGS = {
    "A": "A: Core ML / ANE runtime transition becomes the leading hypothesis",
    "B": "B: callback / GIL handoff becomes the leading hypothesis",
    "C": "C: supports a perf-level placement / performance-state hypothesis",
    "D": "D: the slowdown is not explained by the measured placement / cycle-rate factors",
    "E": "E: a production-safe priming / residency strategy is studied next; "
    "the existing 2 s benchmark warm-up is not a fix",
}
RUN_NAME = re.compile(rf"^{design.MODEL}-(?P<cell>A|PB-ASYNC)-r(?P<round>\d+)\.json\.gz$")


def _pct(x, q):
    return float(np.percentile(x, q)) if len(x) else None


def _mean(x):
    return float(np.mean(x)) if len(x) else None


def _ratio(a, b):
    return None if a is None or b in (None, 0) else a / b


# ------------------------------------------------------------------ requests and forwards


def requests(run: dict, lo: int, hi: int, window: dict) -> dict[str, np.ndarray]:
    """Short (ANE) requests of one hetero window, matched to the client latencies by order
    (#93's tail_decomposition), joined with the predict stamps on service_start <= entry <
    service_end. Arrays, ms except submit_ns."""
    t = run["research"]["trace"]
    idx = [i for i in range(len(t["request_id"])) if t["target"][i] == "ane" and lo <= t["submit_ns"][i] < hi]
    idx.sort(key=lambda i: t["submit_ns"][i])
    client = window["streams"]["short"].get("latency_ms") or []
    m = min(len(idx), len(client))
    idx = idx[:m]
    col = {k: np.asarray([t[k][i] for i in idx], np.int64) for k in t if k != "target"}
    pr = np.asarray(run["research"]["predicts"] or [], np.int64).reshape(-1, 7)
    pr = pr[np.argsort(pr[:, 0])] if len(pr) else pr
    out = {
        "submit_ns": col.get("submit_ns", np.zeros(0, np.int64)),
        "client": np.asarray(client[:m], float),
        "prepare": (col["prepared_ns"] - col["submit_ns"]) / 1e6 if m else np.zeros(0),
    }
    keys = ("features", "native", "callback_delay", "handoff", "post", "action_head", "predict")
    for k in keys:
        out[k] = np.full(m, np.nan)
    if m and len(pr):
        j = np.searchsorted(pr[:, 0], col["service_start_ns"])
        ok = (j < len(pr)) & (pr[np.minimum(j, len(pr) - 1), 0] < col["service_end_ns"])
        p = pr[np.minimum(j, len(pr) - 1)]
        f = np.where(ok, 1.0, np.nan)
        out["features"] = f * (p[:, 0] - col["service_start_ns"]) / 1e6
        out["predict"] = f * (p[:, 6] - p[:, 0]) / 1e6
        out["action_head"] = f * (col["service_end_ns"] - p[:, 6]) / 1e6
        has = ok & (p[:, 3] > 0)
        g = np.where(has, 1.0, np.nan)
        out["native"] = g * (p[:, 3] - p[:, 2]) / 1e6
        out["callback_delay"] = g * (p[:, 4] - p[:, 3]) / 1e6
        out["handoff"] = g * (p[:, 5] - p[:, 4]) / 1e6
        out["post"] = g * (p[:, 6] - p[:, 5]) / 1e6
    return out


def all_requests(run: dict, target: str, lo: int, hi: int) -> np.ndarray:
    t = run["research"]["trace"]
    return np.asarray(
        [
            t["submit_ns"][i]
            for i in range(len(t["request_id"]))
            if t["target"][i] == target and lo <= t["submit_ns"][i] < hi
        ],
        np.int64,
    )


def forwards(run: dict, dev: str) -> np.ndarray:
    x = run["research"]["forwards"].get(dev)
    return np.asarray(x if x is not None else [], np.int64).reshape(-1, 3)


def gpu_returns(run: dict) -> tuple[np.ndarray, np.ndarray]:
    g = run["gpu_return"]
    return np.asarray(g["received_ns"], np.int64), np.asarray(g["return_us"], float) / 1e3


# ------------------------------------------------------------------ counters


def counter_groups(run: dict, window: dict) -> dict[str, list[dict]]:
    """recount series per thread group; the client groups are that window's own client threads."""
    rc = run["research"].get("recount") or {}
    tids = (window or {}).get("tids") or {}
    out: dict[str, list] = {g: [] for g in GROUPS}
    for s in (rc.get("series") or {}).values():
        n = s["name"]
        if n == "laya-ane-dispatch":
            out["ane-dispatch"].append(s)
        elif n == "laya-gpu-dispatch":
            out["gpu-dispatch"].append(s)
        elif n == "coreml-callback":
            out["callback"].append(s)
        elif n == "gpu-worker":
            out["gpu-worker"].append(s)
        elif n == "MainThread":
            out["main"].append(s)
        elif n == "client-short" and s["tid"] == tids.get("short"):
            out["client-short"].append(s)
        elif n == "client-long" and s["tid"] == tids.get("long"):
            out["client-long"].append(s)
    return out


def counter_delta(t_ns: list[int], series: list[dict], levels: int, lo: int, hi: int) -> np.ndarray | None:
    """Summed [instructions, cycles, cpu_ns, energy_nj] per level over [lo, hi), each thread's
    cumulative counters linearly interpolated at the bounds (clamped outside its samples)."""
    if not series:
        return None
    t = np.asarray(t_ns, np.int64)
    total = np.zeros(4 * levels)
    for s in series:
        rows = np.asarray(s["rows"], np.int64)
        if not len(rows):
            continue
        ts = t[rows[:, 0]]
        for j in range(4 * levels):
            v = rows[:, 1 + j].astype(float)
            total[j] += np.interp(hi, ts, v) - np.interp(lo, ts, v)
    return total


def counter_metrics(d: np.ndarray | None, levels: int, per: int | None = None) -> dict | None:
    """E share, relative effective cycle rate (cycles per CPU ns; relative only, not a clock
    frequency) and IPC per level from one delta; level 0 = P, 1 = E."""
    if d is None:
        return None
    lv = [d[4 * i : 4 * i + 4] for i in range(levels)]
    cpu = [x[2] for x in lv]
    tot = sum(cpu)
    names = ["P", "E"][:levels] + [f"L{i}" for i in range(2, levels)]
    out = {"cpu_ms": tot / 1e6, "e_share": (cpu[1] / tot if tot > 0 and levels > 1 else None)}
    for n, x in zip(names, lv):
        out[f"cpu_ms_{n}"] = x[2] / 1e6
        out[f"cycle_rate_rel_{n}"] = x[1] / x[2] if x[2] > 0 else None
        out[f"ipc_{n}"] = x[0] / x[1] if x[1] > 0 else None
    ins, cyc = sum(x[0] for x in lv), sum(x[1] for x in lv)
    out["ipc"] = ins / cyc if cyc > 0 else None
    out["energy_mj"] = sum(x[3] for x in lv) / 1e6
    out["cpu_ms_per_unit"] = tot / 1e6 / per if per else None
    return out


# ------------------------------------------------------------------ transient


def transient(submit_ns: np.ndarray, prepare_ms: np.ndarray, client_ms: np.ndarray, t0: int, end: int) -> dict:
    """The transient metrics of one transition (criteria.md)."""
    horizon = min(t0 + int(HORIZON_S * S), end)
    nbins = max(0, int(np.ceil((horizon - t0) / (BIN_S * S))))
    rel = (submit_ns - t0) / S
    slow = prepare_ms > HOST_SLOW_PREPARE_MS
    share = []
    for b in range(nbins):
        m = (rel >= b * BIN_S) & (rel < (b + 1) * BIN_S)
        share.append(float(slow[m].mean()) if m.any() else None)
    rec = None
    for i in range(nbins):
        if all(x is None or x < HOST_SLOW_SHARE for x in share[i:]):
            rec = i
            break
    duration = rec * BIN_S if rec is not None else (horizon - t0) / S
    peaks = []
    for k in range(int(PEAK_SPAN_S)):
        m = (rel >= k) & (rel < k + 1)
        if m.any():
            peaks.append(_pct(client_ms[m], 99))
    return {
        "bins_share": share,
        "recovered": rec is not None,
        "recovery_s": None if rec is None else rec * BIN_S,
        "duration_s": duration,
        "present": duration >= PRESENT_S,
        "peak_short_p99_ms": max(peaks) if peaks else None,
    }


def reference_interval(tr: dict) -> tuple[float, float]:
    rec = tr["recovery_s"]
    return (4.0, 5.0) if rec is not None and rec <= REFERENCE_RECOVERED_BY_S else (-2.0, 0.0)


# ------------------------------------------------------------------ per run


def load(path: Path) -> dict:
    with gzip.open(path, "rt") as fh:
        return json.load(fh)


def hetero(run: dict) -> list[tuple[int, dict, dict]]:
    """(cycle, part_a window, research window) of each hetero window, in cycle order."""
    pw = {w["cycle"]: w for w in run["part_a"]["windows"] if w["condition"] == "hetero"}
    rw = sorted(run["research"]["windows"], key=lambda w: w["start_ns"])
    return [(k, pw[k], rw[i] if i < len(rw) else None) for i, k in enumerate(sorted(pw))]


def transitions(run: dict, name: str, cell: str) -> list[dict]:
    res = run["research"]
    rc = res.get("recount") or {}
    levels = rc.get("levels", 2)
    order = [tuple(x) for x in res["window_order"]]
    ane, gpu = forwards(run, "ane"), forwards(run, "gpu")
    recv, ret = gpu_returns(run)
    out = []
    for k, pw, rw in hetero(run):
        t0, end = rw["start_ns"], rw["end_ns"]
        rq = requests(run, t0, end, pw)
        tr = transient(rq["submit_ns"], rq["prepare"], rq["client"], t0, end)
        groups = counter_groups(run, rw)
        rel = (rq["submit_ns"] - t0) / S
        buckets = {}
        for bname, a, b in BUCKETS:
            lo, hi = t0 + int(a * S), t0 + int(b * S)
            m = (rel >= a) & (rel < b)
            fa = ane[(ane[:, 0] >= lo) & (ane[:, 0] < hi)]
            fg = gpu[(gpu[:, 0] >= lo) & (gpu[:, 0] < hi)]
            n_long = len(all_requests(run, "gpu", lo, hi))
            per = {"client-short": int(m.sum()), "client-long": n_long, "ane-dispatch": len(fa), "gpu-worker": len(fg)}
            per["callback"] = len(fa)
            cnt = {
                g: counter_metrics(counter_delta(rc.get("t_ns", []), groups[g], levels, lo, hi), levels, per.get(g))
                for g in GROUPS
            }
            buckets[bname] = {
                "short_n": int(m.sum()),
                "short_p50_ms": _pct(rq["client"][m], 50),
                "short_p99_ms": _pct(rq["client"][m], 99),
                "prepare_mean_ms": _mean(rq["prepare"][m]),
                "host_slow_share": float((rq["prepare"][m] > HOST_SLOW_PREPARE_MS).mean()) if m.any() else None,
                **{
                    f"{s}_mean_ms": _nanmean(rq[s][m])
                    for s in ("features", "native", "handoff", "post", "action_head", "predict")
                },
                "callback_delay_p50_ms": _nanpct(rq["callback_delay"][m], 50),
                "ane_cpu_per_forward_ms": _mean(fa[:, 2] / 1e6) if len(fa) else None,
                "gpu_cpu_per_forward_ms": _mean(fg[:, 2] / 1e6) if len(fg) else None,
                "gpu_return_p50_ms": _pct(ret[(recv >= lo) & (recv < hi)], 50),
                "counters": cnt,
            }
        rec = {
            "run": name,
            "cell": cell,
            "cycle": k,
            "preceded_by": order[order.index((k, "hetero")) - 1][1],
            "t0_ns": t0,
            "window_s": (end - t0) / S,
            "transient": tr,
            "buckets": buckets,
            "window_client_cpu_ms_per_request": _client_cpu(rw, pw),
            "window": _window(pw, rq, t0, end),
            "_rq": rq,
            "_steady": _steady(run, rq, t0, end),
        }
        rec["periods"] = _periods(run, rq, tr, t0, end, groups, rc, levels, ane)
        out.append(rec)
    return out


def _window(pw: dict, rq: dict, t0: int, end: int) -> dict:
    """The A validity guard's inputs for one hetero window: part_a's whole-window short P99,
    aggregate req/s (#92's gate.aggregate_req_s), mismatches and routing, and the host-slow share
    over steady [t0 + 10, t0 + 20)."""
    st = pw["streams"]
    lo, hi = t0 + int(STEADY[0] * S), min(t0 + int(STEADY[1] * S), end)
    m = (rq["submit_ns"] >= lo) & (rq["submit_ns"] < hi)
    return {
        "short_p99_ms": st["short"].get("p99_ms"),
        "aggregate_req_s": sum(x["req_s"] for x in st.values()),
        "mismatches": sum(x["mismatches"] for x in st.values()),
        "routing_failures": [s for s, dev in ROUTES if set(st[s].get("devices") or {}) != {dev}],
        "steady_host_slow_share": float((rq["prepare"][m] > HOST_SLOW_PREPARE_MS).mean()) if m.any() else None,
    }


def _nanmean(x):
    x = x[~np.isnan(x)]
    return float(x.mean()) if len(x) else None


def _nanpct(x, q):
    x = x[~np.isnan(x)]
    return float(np.percentile(x, q)) if len(x) else None


def _client_cpu(rw: dict, pw: dict) -> dict:
    out = {}
    for s in ("short", "long"):
        after = ((rw or {}).get("after_by_stream") or {}).get(s) or (rw or {}).get("after")
        n = len(pw["streams"][s].get("latency_ms") or []) or pw["streams"][s].get("n")
        if after and rw.get("before") and n:
            d = after["threads"].get(f"client-{s}", 0) - rw["before"]["threads"].get(f"client-{s}", 0)
            out[s] = d / 1e6 / n
        else:
            out[s] = None
    return out


def _steady(run: dict, rq: dict, t0: int, end: int) -> dict:
    lo, hi = t0 + int(STEADY[0] * S), min(t0 + int(STEADY[1] * S), end)
    m = (rq["submit_ns"] >= lo) & (rq["submit_ns"] < hi)
    recv, ret = gpu_returns(run)
    n_all = len(all_requests(run, "ane", lo, hi)) + len(all_requests(run, "gpu", lo, hi))
    return {
        "seconds": max(0, hi - lo) / S,
        "requests": n_all,
        "short_client": rq["client"][m].tolist(),
        "gpu_returns": ret[(recv >= lo) & (recv < hi)].tolist(),
    }


def _periods(run, rq, tr, t0, end, groups, rc, levels, ane) -> dict:
    """Transient period [t0, recovery), steady [t0+10, t0+20) and the counter reference."""
    rec_s = tr["duration_s"]
    spans = {"transient": (0.0, rec_s), "steady": STEADY, "reference": reference_interval(tr)}
    out = {}
    rel = (rq["submit_ns"] - t0) / S
    for name, (a, b) in spans.items():
        lo, hi = t0 + int(a * S), min(t0 + int(b * S), end) if name != "reference" else t0 + int(b * S)
        m = (rel >= a) & (rel < (hi - t0) / S)
        fa = ane[(ane[:, 0] >= lo) & (ane[:, 0] < hi)]
        out[name] = {
            "span_s": [a, (hi - t0) / S],
            "native": rq["native"][m][~np.isnan(rq["native"][m])].tolist(),
            "callback_delay": rq["callback_delay"][m][~np.isnan(rq["callback_delay"][m])].tolist(),
            "ane_cpu_per_forward": (fa[:, 2] / 1e6).tolist(),
            "features": rq["features"][m][~np.isnan(rq["features"][m])].tolist(),
            "counters": {
                g: counter_metrics(counter_delta(rc.get("t_ns", []), groups[g], levels, lo, hi), levels)
                for g in SCHED_THREADS
            }
            if hi > lo
            else None,
        }
    return out


# ------------------------------------------------------------------ evaluators


def validity(a_trs: list[dict]) -> dict:
    """The A validity guard (criteria.md), per A hetero window; a concern if any window trips any check."""
    lo, hi = A_VALIDITY_AGGREGATE_REQ_S
    per = []
    for t in a_trs:
        w = t["window"]
        p99, agg, share = w["short_p99_ms"], w["aggregate_req_s"], w["steady_host_slow_share"]
        checks = {
            "short_p99": p99 is not None and p99 >= A_VALIDITY_SHORT_P99_MS - EPS,
            "transient": t["transient"]["duration_s"] > A_VALIDITY_TRANSIENT_S + EPS,
            "steady_host_slow": share is not None and share >= A_VALIDITY_STEADY_HOST_SLOW - EPS,
            "aggregate": agg is None or not (lo <= agg <= hi),
            "mismatch_or_routing": bool(w["mismatches"] or w["routing_failures"]),
        }
        per.append(
            {
                "run": t["run"],
                "cycle": t["cycle"],
                **w,
                "transient_s": t["transient"]["duration_s"],
                "flags": [k for k, v in checks.items() if v],
            }
        )
    return {"per_transition": per, "concern": any(x["flags"] for x in per)}


def _pool(trs, period, key):
    return [x for t in trs for x in t["periods"][period][key]]


def sched_changes(trs: list[dict]) -> dict:
    """Counters, per transition with a transient: transient vs reference, for SCHED_THREADS."""
    per = []
    for t in trs:
        if not t["transient"]["present"]:
            continue
        tr, ref = t["periods"]["transient"]["counters"], t["periods"]["reference"]["counters"]
        for g in SCHED_THREADS:
            a, b = (tr or {}).get(g), (ref or {}).get(g)
            if not a or not b:
                per.append({"run": t["run"], "cycle": t["cycle"], "thread": g, "assessable": False})
                continue
            e = (
                a["e_share"] is not None
                and b["e_share"] is not None
                and a["e_share"] - b["e_share"] >= SCHED["e_share_points"] - EPS
            )
            fr = _ratio(a.get("cycle_rate_rel_P"), b.get("cycle_rate_rel_P"))
            ip = _ratio(a.get("ipc"), b.get("ipc"))
            per.append(
                {
                    "run": t["run"],
                    "cycle": t["cycle"],
                    "thread": g,
                    "assessable": True,
                    "e_share": [b["e_share"], a["e_share"]],
                    "e_share_up": bool(e),
                    "p_cycle_rate_ratio": fr,
                    "p_cycle_rate_down": fr is not None and fr <= SCHED["cycle_rate_ratio"] + EPS,
                    "ipc_ratio": ip,
                    "ipc_drop": ip is not None and ip <= SCHED["ipc_ratio"] + EPS,
                }
            )
    ok = [x for x in per if x["assessable"]]
    return {
        "per_transition_thread": per,
        "placement_or_cycle_rate": any(x["e_share_up"] or x["p_cycle_rate_down"] for x in ok),
        "ipc_drop": any(x["ipc_drop"] for x in ok),
        "assessable": bool(ok),
    }


def steady_stats(trs: list[dict]) -> dict:
    sec = sum(t["_steady"]["seconds"] for t in trs)
    short = [x for t in trs for x in t["_steady"]["short_client"]]
    ret = [x for t in trs for x in t["_steady"]["gpu_returns"]]
    return {
        "seconds": sec,
        "aggregate_req_s": sum(t["_steady"]["requests"] for t in trs) / sec if sec else None,
        "short_p99_ms": _pct(short, 99),
        "gpu_return_p50_ms": _pct(ret, 50),
    }


def evaluate(a_trs: list[dict], pb_trs: list[dict]) -> dict:
    """The readings on all of PB-ASYNC's transitions, gated by A's validity guard."""
    val = validity(a_trs)
    present = [t for t in pb_trs if t["transient"]["present"]]
    nat_t, nat_s = _mean(_pool(present, "transient", "native")), _mean(_pool(pb_trs, "steady", "native"))
    cb_t, cb_s = (
        _pct(_pool(present, "transient", "callback_delay"), 50),
        _pct(_pool(pb_trs, "steady", "callback_delay"), 50),
    )
    cpu_t, cpu_s = (
        _mean(_pool(present, "transient", "ane_cpu_per_forward")),
        _mean(_pool(pb_trs, "steady", "ane_cpu_per_forward")),
    )
    native_rises = bool(
        present
        and nat_t is not None
        and nat_s is not None
        and nat_t > nat_s * NATIVE["ratio"]
        and nat_t > nat_s + NATIVE["plus_ms"]
    )
    callback_rises = bool(
        present
        and cb_t is not None
        and cb_s is not None
        and cb_t > cb_s * CALLBACK["ratio"]
        and cb_t > cb_s + CALLBACK["plus_ms"]
    )
    cpu_rises = bool(present and cpu_t is not None and cpu_s and cpu_t >= CPU_RATIO * cpu_s)
    sc = sched_changes(pb_trs)
    sa, sp = steady_stats(a_trs), steady_stats(pb_trs)
    resolves = bool(
        present
        and all(t["transient"]["recovered"] and t["transient"]["duration_s"] < RESOLVE_BY_S for t in pb_trs)
        and sp["gpu_return_p50_ms"] is not None
        and sp["gpu_return_p50_ms"] <= RESOLVE["gpu_return_p50_ms"]
        and sa["aggregate_req_s"]
        and sp["aggregate_req_s"] is not None
        and sp["aggregate_req_s"] >= sa["aggregate_req_s"] * RESOLVE["aggregate_ratio"]
        and sa["short_p99_ms"] is not None
        and sp["short_p99_ms"] is not None
        and sp["short_p99_ms"] <= sa["short_p99_ms"] * RESOLVE["short_p99_ratio"]
    )
    readings = []
    if native_rises:
        readings.append(READINGS["A"])
    if callback_rises and not native_rises:
        readings.append(READINGS["B"])
    if cpu_rises and sc["placement_or_cycle_rate"]:
        readings.append(READINGS["C"])
    if cpu_rises and not sc["placement_or_cycle_rate"]:
        readings.append(READINGS["D"] + (IPC_TEXT if sc["ipc_drop"] else ""))
    if resolves:
        readings.append(READINGS["E"])
    if not present:
        readings = [PHENOTYPE_NOT_REPRODUCED]
    if val["concern"]:
        readings = [VALIDITY_CONCERN]
    return {
        "validity": val,
        "phenotype": bool(present),
        "transitions_used": [(t["run"], t["cycle"]) for t in pb_trs],
        "native": {"transient_mean_ms": nat_t, "steady_mean_ms": nat_s, "rises": native_rises},
        "callback": {"transient_p50_ms": cb_t, "steady_p50_ms": cb_s, "rises": callback_rises},
        "ane_cpu": {"transient_mean_ms": cpu_t, "steady_mean_ms": cpu_s, "rises": cpu_rises},
        "sched_changes": sc,
        "steady": {"A": sa, "PB-ASYNC": sp},
        "transient_resolves": resolves,
        "readings": readings or ["no reading applies"],
    }


# ------------------------------------------------------------------ summary


def summarise(raw: Path) -> dict:
    present = {}
    for p in sorted(raw.glob(f"{design.MODEL}-*-r*.json.gz")):
        if x := RUN_NAME.match(p.name):
            present[(x["cell"], int(x["round"]))] = p
    trs: dict[str, list] = {c: [] for c in design.CELLS}
    runs = {}
    for cell, rnd in design.RUNS:
        p = present.get((cell, rnd))
        if p is None:
            continue
        run = load(p)
        res = run["research"]
        assert res["experiment"] == "coreml-async-transient" and res["cell"] == cell, p
        assert run["args"]["cycles"] == design.CYCLES and run["args"]["seconds"] == design.SECONDS, p
        name = p.name[: -len(".json.gz")]
        t = transitions(run, name, cell)
        trs[cell] += t
        rc = res.get("recount") or {}
        cb = res.get("callbacks") or {}
        runs[name] = {
            "cell": cell,
            "round": rnd,
            "run_wall_s": res["run_wall_s"],
            "mismatches": sum(s["mismatches"] for w in run["part_a"]["windows"] for s in w["streams"].values()),
            "recount": {
                "samples": len(rc.get("t_ns", [])),
                "series": len(rc.get("series", {})),
                "errors": rc.get("errors"),
                "sampler_cpu_fraction_of_core": rc.get("sampler_cpu_fraction_of_core"),
                "interval_ns": rc.get("interval_ns"),
            },
            "callbacks": {k: cb.get(k) for k in ("submits", "total", "errors")} if cb else None,
            "ioreport": res.get("ioreport"),
        }
    complete = all((c, r) in present for c, r in design.RUNS)
    ev = evaluate(trs["A"], trs["PB-ASYNC"]) if complete else None
    public = {
        c: [{k: v for k, v in t.items() if not k.startswith("_") and k != "periods"} for t in ts]
        for c, ts in trs.items()
    }
    return {
        "design": {
            "runs": [list(x) for x in design.RUNS],
            "buckets": [list(b) for b in BUCKETS],
            "host_slow_prepare_ms": HOST_SLOW_PREPARE_MS,
            "bin_s": BIN_S,
            "host_slow_share": HOST_SLOW_SHARE,
        },
        "failed_runs": sorted(p.name for p in (raw / "failed").glob("*.log")),
        "runs": runs,
        "transitions": public,
        "evaluation": ev,
        "outcome": "running: " + ", ".join(f"{c} r{r}" for c, r in design.RUNS if (c, r) not in present)
        if not complete
        else "; ".join(ev["readings"]),
    }


# ------------------------------------------------------------------ tables


def f(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def tables(res: dict) -> str:
    L = ["# Async transient: what happens at a hetero transition (laya, L128 / L512)\n"]
    L.append(f"Outcome: {res['outcome']}\n")
    L.append(f"Crashed runs and re-runs (`raw/failed/`): {', '.join(res['failed_runs']) or 'none'}\n")
    L += [
        "\n## Runs\n",
        "| run | mismatches | recount samples / series / sampler CPU | wall s |",
        "|---|---|---|---|",
    ]
    for n, r in res["runs"].items():
        rc = r["recount"]
        L.append(
            f"| {n} | {r['mismatches']} "
            f"| {rc['samples']} / {rc['series']} / {f(None if rc['sampler_cpu_fraction_of_core'] is None else 100 * rc['sampler_cpu_fraction_of_core'], 3)}% "
            f"| {f(r['run_wall_s'], 0)} |"
        )
    L += [
        "\n## Transitions\n",
        "| cell | run | cycle | after | transient s | recovered | present | peak short P99 ms | client CPU/req short |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c, ts in res["transitions"].items():
        for t in ts:
            tr = t["transient"]
            L.append(
                f"| {c} | {t['run']} | {t['cycle']} | {t['preceded_by']} | {f(tr['duration_s'], 1)} "
                f"| {tr['recovered']} | {tr['present']} | {f(tr['peak_short_p99_ms'])} "
                f"| {f(t['window_client_cpu_ms_per_request']['short'], 3)} |"
            )
    L += [
        "\n## Buckets (seconds from t0)\n",
        "| run | cycle | bucket | n | short P50 / P99 | prepare | host-slow | features | native | cb delay P50 | handoff "
        "| post | action head | ANE CPU/fwd | GPU CPU/fwd | GPU return P50 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for ts in res["transitions"].values():
        for t in ts:
            for b, x in t["buckets"].items():
                L.append(
                    f"| {t['run']} | {t['cycle']} | {b} | {x['short_n']} | {f(x['short_p50_ms'])} / {f(x['short_p99_ms'])} "
                    f"| {f(x['prepare_mean_ms'], 3)} | {f(x['host_slow_share'], 2)} | {f(x['features_mean_ms'], 3)} "
                    f"| {f(x['native_mean_ms'], 3)} | {f(x['callback_delay_p50_ms'], 3)} | {f(x['handoff_mean_ms'], 3)} "
                    f"| {f(x['post_mean_ms'], 3)} | {f(x['action_head_mean_ms'], 3)} | {f(x['ane_cpu_per_forward_ms'], 3)} "
                    f"| {f(x['gpu_cpu_per_forward_ms'], 3)} | {f(x['gpu_return_p50_ms'], 3)} |"
                )
    L += [
        "\n## Per-thread counters by bucket\n",
        "E share: CPU time on E cores / all. Relative effective cycle rate: cycles / CPU ns per level (relative "
        "only, not a clock frequency). CPU/unit: per short request "
        "(client-short), long request (client-long), ANE forward (ANE dispatcher, callback), GPU forward (GPU worker).\n",
        "| run | cycle | bucket | thread | CPU ms | E share | relative effective cycle rate P / E | IPC | CPU/unit ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for ts in res["transitions"].values():
        for t in ts:
            for b, x in t["buckets"].items():
                for g, m in x["counters"].items():
                    if not m:
                        continue
                    L.append(
                        f"| {t['run']} | {t['cycle']} | {b} | {g} | {f(m['cpu_ms'], 1)} | {f(m['e_share'], 2)} "
                        f"| {f(m.get('cycle_rate_rel_P'))} / {f(m.get('cycle_rate_rel_E'))} | {f(m['ipc'])} "
                        f"| {f(m['cpu_ms_per_unit'], 3)} |"
                    )
    ev = res.get("evaluation")
    if ev:
        v = ev["validity"]
        lo, hi = A_VALIDITY_AGGREGATE_REQ_S
        L += [
            "\n## Validity guard (A vs its historical phenotype, #92 / #93)\n",
            f"Concern if any A hetero window has: short P99 >= {A_VALIDITY_SHORT_P99_MS} ms, transient > "
            f"{A_VALIDITY_TRANSIENT_S} s, steady host-slow share >= {A_VALIDITY_STEADY_HOST_SLOW:.0%}, aggregate "
            f"req/s outside [{lo:.1f}, {hi:.1f}], or a mismatch / failed routing.\n",
            "| run | cycle | short P99 ms | transient s | steady host-slow | aggregate req/s | mismatches "
            "| routing failures | flags |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for x in v["per_transition"]:
            L.append(
                f"| {x['run']} | {x['cycle']} | {f(x['short_p99_ms'])} | {f(x['transient_s'], 1)} "
                f"| {f(x['steady_host_slow_share'], 3)} | {f(x['aggregate_req_s'], 1)} | {x['mismatches']} "
                f"| {', '.join(x['routing_failures']) or '–'} | {', '.join(x['flags']) or '–'} |"
            )
        L.append(f"\nValidity concern: **{v['concern']}**\n")
        L.append("\n## Readings\n")
        for r in ev["readings"]:
            L.append(f"- {r}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if results.json / tables.md are stale")
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=EXP)
    a = ap.parse_args()
    res = summarise(a.raw)
    js = json.dumps(res, indent=1, sort_keys=True, default=_json) + "\n"
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


def _json(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    raise TypeError(type(x))


if __name__ == "__main__":
    main()
