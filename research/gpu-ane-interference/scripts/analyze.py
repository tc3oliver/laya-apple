"""Analyse interference.py raw runs: latency decomposition, service-time inflation, tails.

    uv run python research/gpu-ane-interference/scripts/analyze.py            # all raw/*.json.gz
    uv run python research/gpu-ane-interference/scripts/analyze.py --check    # results.json up to date?

Writes research/gpu-ane-interference/results.json (every number the README quotes) and
prints the README's tables. Definitions, per request (all from one monotonic clock):

    e2e          response - arrival            what the caller waits for
    pre          queue_enter - arrival         prompt build, eligibility check, routing
    queue        device_start - queue_enter    waiting in the device's FIFO queue
    service      device_end - device_start     the device's FIFO is occupied by this request
    post         response - device_end         wake the caller, format answers
    forward      backend.forward in the executing process (RuntimeInfo.device_ms)
    dispatch     service - forward             IPC round trip (process) or ~0 (thread)
    device_exec  mx.eval / Core ML predict inside forward
    host         forward - device_exec         NumPy features, action head, graph build

Only requests that arrived inside a window's measurement interval count. Pooled statistics
are over all cycles of a cell; `cycle_range` gives the per-cycle spread; P99 carries a
bootstrap 95% CI and `p99_tail_n` (samples above the P99) -- a P99 with fewer than 10 tail
samples is flagged `p99_reliable: false`.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
RAW = ROOT / "raw"
OUT = ROOT / "results.json"
RNG_SEED = 20260924
BOOT = 1000


def rows(cols: dict) -> list:
    keys = list(cols)
    return [dict(zip(keys, vals)) for vals in zip(*(cols[k] for k in keys))] if keys else []


# ----------------------------------------------------------------------------- statistics


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


def describe(values, boot: bool = True) -> dict:
    a = np.asarray([v for v in values if v is not None], np.float64)
    if a.size == 0:
        return {"n": 0}
    d = {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "p50": pct(a, 50),
        "p95": pct(a, 95),
        "p99": pct(a, 99),
        "max": float(a.max()),
    }
    d["p99_tail_n"] = int((a > d["p99"]).sum())
    d["p99_reliable"] = d["p99_tail_n"] >= 10
    if boot and a.size >= 20:
        rng = np.random.default_rng(RNG_SEED)
        idx = rng.integers(0, a.size, (BOOT, a.size))
        s = a[idx]
        d["p99_ci95"] = [float(x) for x in np.percentile(np.percentile(s, 99, axis=1), [2.5, 97.5])]
        d["mean_ci95"] = [float(x) for x in np.percentile(s.mean(axis=1), [2.5, 97.5])]
    return d


def ratio_ci(conc, solo, stat=np.mean) -> dict:
    """stat(conc)/stat(solo) with a bootstrap 95% CI (independent resamples of both)."""
    c, s = np.asarray(conc, np.float64), np.asarray(solo, np.float64)
    if c.size == 0 or s.size == 0:
        return {}
    r = float(stat(c) / stat(s))
    rng = np.random.default_rng(RNG_SEED)
    bc = stat(c[rng.integers(0, c.size, (BOOT, c.size))], axis=1)
    bs = stat(s[rng.integers(0, s.size, (BOOT, s.size))], axis=1)
    lo, hi = np.percentile(bc / bs, [2.5, 97.5])
    return {"ratio": r, "ci95": [float(lo), float(hi)]}


def median(a, axis=None):
    return np.median(a, axis=axis)


# ----------------------------------------------------------------------------- joining


class Phases:
    """Backend phase records (device spans, CPU time) looked up by device and time."""

    def __init__(self, cols: dict | None):
        self.by_kind = defaultdict(list)
        for p in rows(cols or {}):  # compact runs (compact.py) carry no phase list
            self.by_kind[p["kind"]].append(p)
        self.t0 = {k: [p["t0"] for p in v] for k, v in self.by_kind.items()}

    def within(self, kind: str, lo: float, hi: float):
        """The forward that started inside [lo, hi] (the dispatcher's device call)."""
        ts = self.t0.get(kind, [])
        i = bisect.bisect_left(ts, lo)
        if i < len(ts) and ts[i] <= hi:
            return self.by_kind[kind][i]
        return None


def busy_union(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def overlap_fraction(a, b, busy, starts):
    """Fraction of [a, b] covered by the (sorted, disjoint) busy intervals."""
    if b <= a:
        return 0.0
    i = max(0, bisect.bisect_right(starts, a) - 1)
    cov = 0.0
    while i < len(busy) and busy[i][0] < b:
        lo, hi = max(a, busy[i][0]), min(b, busy[i][1])
        if hi > lo:
            cov += hi - lo
        i += 1
    return cov / (b - a)


def enrich(recs: list, phases: Phases) -> None:
    for r in recs:
        r["e2e"] = r["response"] - r["arrival"]
        if "device_start" in r:
            r["pre"] = r["queue_enter"] - r["arrival"]
            r["queue"] = r["device_start"] - r["queue_enter"]
            r["service"] = r["device_end"] - r["device_start"]
            r["post"] = r["response"] - r["device_end"]
            if r.get("forward_ms") is not None:
                r["dispatch"] = r["service"] - r["forward_ms"]
            p = None if "device_exec" in r else phases.within(r["device"], r["device_start"], r["device_end"])
            if p is not None:
                r["device_exec"] = sum(y - x for x, y in p["dev"])
                r["host"] = (p["t1"] - p["t0"]) - r["device_exec"]
                r["cpu_ms"] = p["cpu_ms"]


METRICS = ("e2e", "pre", "queue", "service", "post", "forward_ms", "dispatch", "device_exec", "host", "cpu_ms")


# ----------------------------------------------------------------------------- per run


def decode(enc: dict) -> list:
    from compact import decode as _decode  # compact.py imports this module: import late

    return _decode(enc)


def load(path: Path) -> dict:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def analyse_run(d: dict) -> dict:
    phases = Phases(d["phases"])
    cells: dict = defaultdict(lambda: {"windows": []})
    for w in d["windows"]:
        lo, hi = (x * 1e3 for x in w["measure"])  # window bounds are in s, record times in ms
        span_s = (hi - lo) / 1e3
        per_stream = {}
        all_recs = {}
        for label, st in w["streams"].items():
            recs = decode(st["encoded"]) if "encoded" in st else rows(st["records"])
            enrich(recs, phases)
            for r in recs:
                r["cycle"] = w["cycle"]
            all_recs[label] = recs
        # overlap with the OTHER device's busy time (from its own stream's device intervals)
        busy = {}
        for label, recs in all_recs.items():
            dev = w["streams"][label]["device"]
            if dev is None:  # product streams: split by the device that served each request
                for r in recs:
                    if "device_start" in r:
                        busy.setdefault(r["device"], []).append((r["device_start"], r["device_end"]))
            else:
                busy.setdefault(dev, []).extend((r["device_start"], r["device_end"]) for r in recs if "device_start" in r)
        busy = {k: busy_union(v) for k, v in busy.items()}
        starts = {k: [x[0] for x in v] for k, v in busy.items()}
        util = {}
        for k, v in busy.items():
            cov = sum(max(0.0, min(b, hi) - max(a, lo)) for a, b in v)
            util[k] = cov / (hi - lo)
        for label, recs in all_recs.items():
            for r in recs:
                if "device_start" in r:
                    other = "ane" if r["device"] == "gpu" else "gpu"
                    r["overlap"] = (
                        overlap_fraction(r["device_start"], r["device_end"], busy[other], starts[other])
                        if other in busy
                        else 0.0
                    )
            ms = [r for r in recs if r["in_window"]]
            per_stream[label] = {
                "n": len(ms),
                "req_s": len(ms) / span_s,
                "mismatches": sum(not r["match"] for r in ms),
                "errors": len(w["streams"][label]["errors"]),
                "recs": ms,
            }
        cells[w["cell"]]["stream_specs"] = [
            (st["device"], st["shape"], st["mode"], st["rate"]) for st in w["streams"].values()
        ]
        cells[w["cell"]]["kind"] = w["kind"]
        cells[w["cell"]]["spec"] = w["spec"]
        cells[w["cell"]]["windows"].append(
            {
                "cycle": w["cycle"],
                "util": util,
                "streams": per_stream,
                "aggressors": w["aggressors"],
                "conditions": w["conditions_before"],
            }
        )
    out = {}
    for name, c in cells.items():
        streams = defaultdict(list)
        for w in c["windows"]:
            for label, s in w["streams"].items():
                streams[label].append(s)
        cell = {"kind": c["kind"], "spec": c["spec"], "stream_specs": c["stream_specs"], "streams": {}, "device_util": {}}
        for k in ("gpu", "ane"):
            us = [w["util"].get(k, 0.0) for w in c["windows"]]
            cell["device_util"][k] = {"mean": float(np.mean(us)), "range": [float(min(us)), float(max(us))]}
        cell["aggressors"] = c["windows"][0]["aggressors"]
        cell["thermal_or_perf_warning"] = any(
            w["conditions"]["thermal_warning"] or w["conditions"]["performance_warning"] for w in c["windows"]
        )
        for label, ws in streams.items():
            recs = [r for s in ws for r in s["recs"]]
            st = {
                "n": len(recs),
                "windows": len(ws),
                "req_s": float(np.median([s["req_s"] for s in ws])),
                "req_s_range": [float(min(s["req_s"] for s in ws)), float(max(s["req_s"] for s in ws))],
                "mismatches": int(sum(s["mismatches"] for s in ws)),
                "errors": int(sum(s["errors"] for s in ws)),
                "devices": dict(sorted(_count(r.get("device") for r in recs).items())),
                "overlap_mean": float(np.mean([r["overlap"] for r in recs if "overlap" in r])) if recs else None,
            }
            for m in METRICS:
                st[m] = describe([r.get(m) for r in recs], boot=m in ("e2e", "service", "queue"))
            st["cycle_range"] = {}
            for m in ("e2e", "service"):
                per = [[r[m] for r in s["recs"] if m in r] for s in ws]
                if all(per):
                    st["cycle_range"][m] = {
                        "p50": [min(pct(v, 50) for v in per), max(pct(v, 50) for v in per)],
                        "p99": [min(pct(v, 99) for v in per), max(pct(v, 99) for v in per)],
                    }
            st["_recs"] = recs  # dropped before writing
            cell["streams"][label] = st
        out[name] = cell
    return out


def _count(xs):
    c = defaultdict(int)
    for x in xs:
        c[x] += 1
    return c


# ----------------------------------------------------------------------------- comparisons


def solo_key(label: str) -> str:
    return f"solo:{label}"


def inflation(cells: dict) -> dict:
    """Every non-solo device cell: each stream's service/phase inflation against its solo cell."""
    res = {}
    for name, c in cells.items():
        if c["kind"] in ("solo",) or c["kind"].startswith("product"):
            continue
        res[name] = {}
        for label, st in c["streams"].items():
            solo = cells.get(solo_key(label))
            if solo is None:
                continue
            s = solo["streams"][label]
            recs, srecs = st["_recs"], s["_recs"]
            entry = {"role": _role(c, label)}
            for m in ("service", "forward_ms", "device_exec", "host", "dispatch", "cpu_ms"):
                a = [r[m] for r in recs if r.get(m) is not None]
                b = [r[m] for r in srecs if r.get(m) is not None]
                if a and b and m in ("service", "forward_ms", "device_exec", "host"):
                    entry[m] = {"mean": ratio_ci(a, b, np.mean), "median": ratio_ci(a, b, median)}
                if a and b:
                    entry[m + "_delta_ms"] = float(np.mean(a) - np.mean(b))
            entry["p99_e2e_ratio"] = st["e2e"]["p99"] / s["e2e"]["p99"]
            entry["p99_service_ratio"] = st["service"]["p99"] / s["service"]["p99"]
            entry["req_s_ratio"] = st["req_s"] / s["req_s"]
            res[name][label] = entry
    return res


def _role(cell, label):
    for dev, shp, mode, _rate in cell["stream_specs"]:
        if f"{dev}_{shp}" == label:
            return "victim" if mode == "closed" and cell["kind"] == "sweep" else ("aggressor" if mode == "poisson" else "stream")
    return "stream"


def overlap_curve(cells: dict, device: str, bins=10) -> dict:
    """Per-request service time vs the fraction of it the other device was busy, pooled over
    the matrix and sweep cells, per (device, shape). Normalised by the solo mean."""
    out = {}
    for label in sorted({lab for c in cells.values() for lab in c["streams"] if lab.startswith(device + "_")}):
        solo = cells.get(solo_key(label))
        if solo is None:
            continue
        base = np.mean([r["service"] for r in solo["streams"][label]["_recs"]])
        base_exec = np.mean([r["device_exec"] for r in solo["streams"][label]["_recs"] if "device_exec" in r] or [np.nan])
        pts = []
        for c in cells.values():
            if c["kind"] not in ("matrix", "sweep", "solo") or label not in c["streams"]:
                continue
            pts += [(r["overlap"], r["service"], r.get("device_exec")) for r in c["streams"][label]["_recs"] if "overlap" in r]
        if not pts:
            continue
        f = np.array([p[0] for p in pts])
        s = np.array([p[1] for p in pts]) / base
        e = np.array([p[2] if p[2] is not None else np.nan for p in pts]) / base_exec
        edges = np.linspace(0, 1, bins + 1)
        curve = []
        for i in range(bins):
            m = (f >= edges[i]) & ((f < edges[i + 1]) if i < bins - 1 else (f <= 1))
            if m.sum() >= 20:
                curve.append(
                    {
                        "overlap": [float(edges[i]), float(edges[i + 1])],
                        "n": int(m.sum()),
                        "service_ratio_mean": float(s[m].mean()),
                        "service_ratio_p50": float(np.median(s[m])),
                        "exec_ratio_mean": float(np.nanmean(e[m])),
                    }
                )
        fit = None
        ok = np.isfinite(f) & np.isfinite(s)
        if ok.sum() >= 20 and np.ptp(f[ok]) > 0.1:  # a fit needs a spread of overlaps
            slope, icpt = np.polyfit(f[ok], s[ok], 1)
            pred = slope * f[ok] + icpt
            r2 = 1 - ((s[ok] - pred) ** 2).sum() / ((s[ok] - s[ok].mean()) ** 2).sum()
            fit = {"intercept": float(icpt), "slope": float(slope), "r2": float(r2)}
        out[label] = {"solo_mean_service_ms": float(base), "n": int(len(f)), "linear_fit": fit, "bins": curve}
    return out


def sweep_curve(cells: dict) -> dict:
    """Victim inflation vs the aggressor device's realised utilisation (sweep + matrix + solo)."""
    out = defaultdict(list)
    for name, c in cells.items():
        if c["kind"] not in ("sweep", "matrix"):
            continue
        streams = c["stream_specs"]
        pairs = []
        if c["kind"] == "sweep":
            vic = next(s for s in streams if s[2] == "closed")
            agg = next(s for s in streams if s[2] == "poisson")
            pairs.append((vic, agg))
        else:
            pairs += [(streams[0], streams[1]), (streams[1], streams[0])]
        for vic, agg in pairs:
            vl, al = f"{vic[0]}_{vic[1]}", f"{agg[0]}_{agg[1]}"
            solo = cells.get(solo_key(vl))
            if solo is None:
                continue
            a = [r["service"] for r in c["streams"][vl]["_recs"]]
            b = [r["service"] for r in solo["streams"][vl]["_recs"]]
            out[f"{vl}|{al}"].append(
                {
                    "cell": name,
                    "aggressor_util": c["device_util"][agg[0]]["mean"],
                    "offered_util": c["spec"].get("util", 1.0) if c["kind"] == "sweep" else 1.0,
                    "service_ratio": ratio_ci(a, b, np.mean),
                    "victim_p99_service_ms": c["streams"][vl]["service"]["p99"],
                    "victim_n": len(a),
                }
            )
    for v in out.values():
        v.insert(0, {"cell": "solo", "aggressor_util": 0.0, "offered_util": 0.0, "service_ratio": {"ratio": 1.0, "ci95": [1.0, 1.0]}})
        v.sort(key=lambda x: x["aggressor_util"])
    return dict(out)


def sparse_curve(cells: dict) -> dict:
    """One device alone at low offered load: service time relative to its closed-loop solo."""
    out = defaultdict(list)
    for name, c in cells.items():
        if c["kind"] != "sparse":
            continue
        (label,) = c["streams"]
        solo = cells.get(solo_key(label))
        if solo is None:
            continue
        st = c["streams"][label]
        a = [r["service"] for r in st["_recs"]]
        b = [r["service"] for r in solo["streams"][label]["_recs"]]
        cpu_a = [r["cpu_ms"] for r in st["_recs"] if "cpu_ms" in r]
        cpu_b = [r["cpu_ms"] for r in solo["streams"][label]["_recs"] if "cpu_ms" in r]
        out[label].append(
            {
                "cell": name,
                "offered_util": c["spec"].get("util"),
                "realised_util": c["device_util"][label.split("_")[0]]["mean"],
                "aggressors": [g["kind"] for g in c["aggressors"]],
                "n": len(a),
                "service_ratio": ratio_ci(a, b, np.mean),
                "service_ratio_median": ratio_ci(a, b, median),
                "cpu_ms_ratio": ratio_ci(cpu_a, cpu_b, np.mean) if cpu_a and cpu_b else None,
                "exec_ratio": ratio_ci(
                    [r["device_exec"] for r in st["_recs"] if "device_exec" in r],
                    [r["device_exec"] for r in solo["streams"][label]["_recs"] if "device_exec" in r],
                    np.mean,
                ),
            }
        )
    for v in out.values():
        v.sort(key=lambda x: (x["aggressors"] != [], x["offered_util"]))
    return dict(out)


def tail_attribution(st: dict) -> dict:
    """Mean of each latency component over the requests at or above the e2e P99, and overall."""
    recs = [r for r in st["_recs"] if "service" in r]
    if not recs:
        return {}
    p99 = np.percentile([r["e2e"] for r in recs], 99)
    tail = [r for r in recs if r["e2e"] >= p99]
    comp = ("pre", "queue", "service", "post")
    return {
        "p99_e2e_ms": float(p99),
        "tail_n": len(tail),
        "tail_mean": {k: float(np.mean([r[k] for r in tail])) for k in comp},
        "all_mean": {k: float(np.mean([r[k] for r in recs])) for k in comp},
    }


def routing_eval(cells: dict) -> dict:
    """Product runs: how well backlog + static estimate predicted each request's completion."""
    out = {}
    for name, c in cells.items():
        if not c["kind"].startswith("product"):
            continue
        for label, st in c["streams"].items():
            recs = [r for r in st["_recs"] if "service" in r]
            by = {}
            for dev in ("gpu", "ane"):
                rs = [r for r in recs if r["device"] == dev]
                if not rs:
                    continue
                pred = np.array([r["backlog_at_enter_ms"] + r["estimate_ms"] for r in rs])
                act = np.array([r["queue"] + r["service"] for r in rs])
                svc_err = np.array([r["service"] / r["estimate_ms"] for r in rs])
                by[dev] = {
                    "n": len(rs),
                    "completion_error_ms": describe(act - pred, boot=False),
                    "service_over_estimate": describe(svc_err, boot=False),
                    "reasons": dict(_count(r.get("reason") for r in rs)),
                }
            out[f"{name}|{label}"] = by
    return out


def routing_hindsight(cells: dict, auto_shape: str = "A") -> dict:
    """Product open-loop runs: was each queue-aware choice for an ANE-eligible request right?

    For every ANE-eligible request (the Part B short 1-question class), at its decision time
    t (queue_enter) the other device's FIFO was free at F = the latest device_end of the jobs
    queued there before t (observed; jobs run one at a time in queue order). Its hindsight
    completion on the other device is (F - t) + S, where S is the median service time that
    device actually gave this class in the same window (so S already carries that window's
    interference). A choice is a miss when that counterfactual beats the observed completion
    by more than 1 ms. Queueing effects of the counterfactual on later requests are ignored."""
    out = {}
    for name, c in cells.items():
        if c["kind"] != "product_open":
            continue
        for label, st in c["streams"].items():
            recs = [r for r in st["_recs"] if "service" in r]
            svc = {
                dev: float(np.median([r["service"] for r in recs if r["device"] == dev and r.get("shape") == auto_shape]))
                for dev in ("gpu", "ane")
                if any(r["device"] == dev and r.get("shape") == auto_shape for r in recs)
            }
            if len(svc) < 2:
                out[f"{name}|{label}"] = {"note": "one device never served the short class", "service_median": svc}
                continue
            jobs = {dev: sorted((r["queue_enter"], r["device_end"]) for r in recs if r["device"] == dev) for dev in svc}
            enters = {dev: [q for q, _ in js] for dev, js in jobs.items()}
            prefix_end = {}
            for dev, js in jobs.items():
                m, acc = -np.inf, []
                for _, e in js:
                    m = max(m, e)
                    acc.append(m)
                prefix_end[dev] = acc
            res = defaultdict(lambda: {"n": 0, "miss": 0, "miss_cost_ms": [], "gain_ms": []})
            for r in recs:
                if r.get("shape") != auto_shape:
                    continue
                t, dev = r["queue_enter"], r["device"]
                other = "ane" if dev == "gpu" else "gpu"
                i = bisect.bisect_left(enters[other], t) - 1
                free = max(t, prefix_end[other][i]) if i >= 0 else t
                cf = (free - t) + svc[other]
                act = r["device_end"] - t
                busy = (r.get("ane_backlog_ms") or 0) > 0 or (r.get("gpu_backlog_ms") or 0) > 0
                key = f"{r['reason']}|{'busy' if busy else 'idle'}"
                e = res[key]
                e["n"] += 1
                if cf + 1.0 < act:
                    e["miss"] += 1
                    e["miss_cost_ms"].append(act - cf)
                else:
                    e["gain_ms"].append(cf - act)
            out[f"{name}|{label}"] = {
                "service_median": svc,
                "by_reason": {
                    k: {
                        "n": v["n"],
                        "miss": v["miss"],
                        "miss_rate": v["miss"] / v["n"],
                        "miss_cost_ms_mean": float(np.mean(v["miss_cost_ms"])) if v["miss_cost_ms"] else 0.0,
                        "miss_cost_ms_p95": float(np.percentile(v["miss_cost_ms"], 95)) if v["miss_cost_ms"] else 0.0,
                    }
                    for k, v in sorted(res.items())
                },
            }
    return out


# Written before the A/B was run (research/gpu-ane-interference/README.md, "Scheduler
# prototype"): what counts as an improvement, so the verdict cannot be chosen afterwards.
AB_CRITERIA = {
    "short_p99_min_reduction": 0.10,  # short-class P99 at least 10% lower ...
    "ci_excludes_one": True,  # ... with the bootstrap 95% CI of the P99 ratio below 1.0
    "min_load_levels": 2,  # in at least 2 of a workload's 3 load levels
    "throughput_tolerance": 0.02,  # aggregate completed req/s within 2%
    "other_class_max_regression": 0.10,  # no other class's P99 >10% worse with its CI above 1.0
    "mismatches": 0,
}


def p99_ratio_ci(new, old) -> dict:
    return ratio_ci(new, old, lambda a, axis=None: np.percentile(a, 99, axis=axis))


def scheduler_ab(cells: dict) -> dict:
    """Pairs each open-loop cell with its #contention twin (same arrivals, same windows)."""
    out = {}
    for name, c in cells.items():
        if c["kind"] != "product_open" or name.endswith("#contention") or f"{name}#contention" not in cells:
            continue
        twin = cells[f"{name}#contention"]
        ((label, base),) = c["streams"].items()
        new = twin["streams"][label]
        rb, rn = base["_recs"], new["_recs"]
        entry = {"n": [len(rb), len(rn)], "classes": {}, "devices": {}}
        for shape in sorted({r.get("shape") for r in rb}):
            eb = [r["e2e"] for r in rb if r.get("shape") == shape]
            en = [r["e2e"] for r in rn if r.get("shape") == shape]
            entry["classes"][shape] = {
                "baseline": describe(eb, boot=False),
                "contention": describe(en, boot=False),
                "p99_ratio": p99_ratio_ci(en, eb),
                "mean_ratio": ratio_ci(en, eb),
                "gpu_share": [
                    float(np.mean([r["device"] == "gpu" for r in rb if r.get("shape") == shape])),
                    float(np.mean([r["device"] == "gpu" for r in rn if r.get("shape") == shape])),
                ],
                "queue_mean": [float(np.mean([r["queue"] for r in rb if r.get("shape") == shape])),
                               float(np.mean([r["queue"] for r in rn if r.get("shape") == shape]))],
                "service_mean": [float(np.mean([r["service"] for r in rb if r.get("shape") == shape])),
                                 float(np.mean([r["service"] for r in rn if r.get("shape") == shape]))],
            }
        allb, alln = [r["e2e"] for r in rb], [r["e2e"] for r in rn]
        entry["all"] = {"baseline": describe(allb, boot=False), "contention": describe(alln, boot=False),
                        "p99_ratio": p99_ratio_ci(alln, allb)}
        entry["req_s"] = [base["req_s"], new["req_s"]]
        for dev in ("gpu", "ane"):
            entry["devices"][dev] = [sum(r["device"] == dev for r in rb), sum(r["device"] == dev for r in rn)]
        entry["reasons"] = [base["devices"], new["devices"]]
        entry["routing_reasons"] = [dict(_count(r.get("reason") for r in rb)), dict(_count(r.get("reason") for r in rn))]
        entry["mismatches"] = [base["mismatches"], new["mismatches"]]
        entry["short_p99_by_cycle"] = {
            str(cy): [
                pct([r["e2e"] for r in rb if r.get("shape") == "A" and r["cycle"] == cy], 99),
                pct([r["e2e"] for r in rn if r.get("shape") == "A" and r["cycle"] == cy], 99),
            ]
            for cy in sorted({r["cycle"] for r in rb})
        }
        a = entry["classes"].get("A", {})
        pr = a.get("p99_ratio", {})
        entry["short_improved"] = bool(
            pr and pr["ratio"] <= 1 - AB_CRITERIA["short_p99_min_reduction"] and pr["ci95"][1] < 1.0
        )
        entry["throughput_ok"] = abs(entry["req_s"][1] / entry["req_s"][0] - 1) <= AB_CRITERIA["throughput_tolerance"]
        entry["other_regressed"] = [
            k for k, v in entry["classes"].items()
            if k != "A" and v["p99_ratio"]["ratio"] > 1 + AB_CRITERIA["other_class_max_regression"]
            and v["p99_ratio"]["ci95"][0] > 1.0
        ]
        out[name] = entry
    workloads = defaultdict(list)
    for name, e in out.items():
        workloads[name.split("@")[0]].append(e)
    verdict = {
        w: {
            "levels": len(es),
            "short_improved_levels": sum(e["short_improved"] for e in es),
            "throughput_ok_all": all(e["throughput_ok"] for e in es),
            "other_regressions": sum(len(e["other_regressed"]) for e in es),
            "mismatches": sum(sum(e["mismatches"]) for e in es),
            "passes": sum(e["short_improved"] for e in es) >= AB_CRITERIA["min_load_levels"]
            and all(e["throughput_ok"] for e in es)
            and not any(e["other_regressed"] for e in es)
            and sum(sum(e["mismatches"]) for e in es) == AB_CRITERIA["mismatches"],
        }
        for w, es in workloads.items()
    }
    return {"criteria": AB_CRITERIA, "cells": out, "verdict": verdict}


# ----------------------------------------------------------------------------- main


def analyse_all(paths) -> dict:
    results = {"generated_by": "research/gpu-ane-interference/scripts/analyze.py", "inputs": {}, "runs": {}}
    for p in paths:
        d = load(p)
        results["inputs"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        key = f"{p.name.split('-', 1)[0]}:{d['args']['model']}:{d['args']['ane_placement']}"
        cells = analyse_run(d)
        run = {
            "environment": {k: d["environment"][k] for k in ("laya_apple", "git_commit", "platform", "python", "mlx",
                                                              "coremltools", "numpy", "cpu_perflevels", "clock")},
            "args": d["args"],
            "shapes": d["shapes"],
            "ane_probes": d["environment"]["laya_info"].get("ane_probes"),
            "ane_buckets": d["environment"]["laya_info"].get("ane_buckets"),
            "cells": cells,
        }
        if d["plan"] == "device":
            run["inflation"] = inflation(cells)
            run["overlap_curve"] = {dev: overlap_curve(cells, dev) for dev in ("gpu", "ane")}
            run["sweep_curve"] = sweep_curve(cells)
            run["sparse_curve"] = sparse_curve(cells)
        run["tail_attribution"] = {
            f"{n}|{lab}": tail_attribution(st) for n, c in cells.items() for lab, st in c["streams"].items()
        }
        if d["plan"] == "product":
            run["routing"] = routing_eval(cells)
            run["routing_hindsight"] = routing_hindsight(cells)
            if d["args"].get("scheduler", "baseline") == "both":
                run["scheduler_ab"] = scheduler_ab(cells)
                run["contention_params"] = d.get("contention_params")
        results["runs"][key] = run
    return _strip(results)


def _strip(x):
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if k != "_recs"}
    if isinstance(x, list):
        return [_strip(v) for v in x]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check", action="store_true", help="fail if --out differs from a fresh analysis")
    args = ap.parse_args()
    paths = args.paths or sorted(RAW.glob("*.json.gz"))
    res = analyse_all(paths)
    text = json.dumps(res, indent=1, sort_keys=True) + "\n"
    if args.check:
        if not args.out.exists() or args.out.read_text() != text:
            sys.exit(f"{args.out} is stale: re-run analyze.py")
        print(f"{args.out} is up to date")
        return
    args.out.write_text(text)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
