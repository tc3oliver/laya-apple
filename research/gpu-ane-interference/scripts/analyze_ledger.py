"""Analyse runtime-trace runs through the request ledger: join health, decomposition, prediction error.

    uv run python research/gpu-ane-interference/scripts/analyze_ledger.py            # all ledger/raw/*.json.gz
    uv run python research/gpu-ane-interference/scripts/analyze_ledger.py --check    # outputs up to date?

Reads research/gpu-ane-interference/ledger/raw/*.json.gz (interference.py, pipeline
"runtime-trace"), builds one ledger record per request with ledger.build() and
ledger.annotate_overlap() -- the only source of lifecycle timing here -- and writes
ledger/results.json, ledger/tables.md and ledger/prediction_error.csv. The timing taxonomy
(measured / derived / unattributed / generator) is ledger.py's `KIND`; it is copied into
results.json. Per run ("<plan>:<model>:<placement>"):

    join           ledger.build's join report verbatim, plus each pass condition
    validation     per device: queue / service / occupancy distributions, forward <= service
                   and backend span inside the service window (counts), unattributed time
    timeline       ~300 ms from the middle of the first matrix window, per request segments
    decomposition  device plan: every stream in a solo and a matrix cell, solo vs concurrent
                   mean of each component, delta with a bootstrap 95% CI, ratios, and the
                   historical harness's numbers (../results.json) side by side
and, over all runs, "prediction": the router's predicted completion (target backlog + service
estimate) against the measured completion (received - routed), split exactly into a queue
error and a service error, grouped by device, other-device state, shape, depth, overlap and
routing reason. Only requests that arrived inside a window's measurement interval count
(the timeline also shows the others). Statistics helpers and RNG_SEED come from analyze.py.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ledger  # noqa: E402
from analyze import BOOT, RNG_SEED, describe, ratio_ci  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "ledger"
RAW = LEDGER / "raw"
HISTORICAL = ROOT / "results.json"
ENV_KEYS = ("laya_apple", "git_commit", "platform", "python", "mlx", "coremltools", "numpy", "cpu_perflevels", "clock")
SLICE_MS = 300.0
# Timeline segments in time order. Inside the service window the three parts are durations
# stacked in this fixed order, not their order in time (device spans interleave with host work).
SEGMENTS = ("pre", "queue", "dispatch", "device_exec", "host", "unattributed", "return", "postprocess")
COMPONENTS = (
    "pre",
    "queue",
    "dispatch",
    "return",
    "dispatch_plus_return",
    "host",
    "device_exec",
    "unattributed",
    "postprocess",
    "forward",
    "service",
    "occupancy",
    "e2e",
    "cpu_ms",
)
RATIO_COMPONENTS = ("occupancy", "forward", "device_exec", "host")
SHARE_COMPONENTS = ("dispatch", "return", "device_exec", "host", "unattributed")
DIMENSIONS = (
    "all",
    "target",
    "other_device",
    "target|other_device",
    "tokens",
    "shape_class",
    "target_depth",
    "overlap",
    "reason",
)
IDENTITY_TOL_MS = 1e-6
MIN_SHARE_DELTA_MS = 1e-3  # shares of a smaller occupancy change (below the 1-us clock grain) are None


# ----------------------------------------------------------------------------- helpers


def _round(x, nd=6):
    if isinstance(x, float):
        return None if not np.isfinite(x) else round(x, nd)
    if isinstance(x, dict):
        return {k: _round(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_round(v, nd) for v in x]
    if isinstance(x, np.generic):
        return _round(x.item(), nd)
    return x


def component(r: dict, name: str):
    """One decomposition component of a ledger record (ms), or None when it is not known."""
    t = r["timing_ms"]
    if name == "pre":
        return t["prepare"] + t["route"] + t["enqueue"]
    if name == "dispatch_plus_return":
        return t["dispatch"] + t["return"]
    if name == "cpu_ms":
        return r["backend"]["cpu_ms"] if r["backend"] else None
    return t[name]


def delta_ci(conc, solo) -> dict:
    """mean(conc) - mean(solo) with a bootstrap 95% CI (independent resamples of both)."""
    c, s = np.asarray(conc, np.float64), np.asarray(solo, np.float64)
    if c.size == 0 or s.size == 0:
        return {}
    rng = np.random.default_rng(RNG_SEED)
    bc = c[rng.integers(0, c.size, (BOOT, c.size))].mean(axis=1)
    bs = s[rng.integers(0, s.size, (BOOT, s.size))].mean(axis=1)
    lo, hi = np.percentile(bc - bs, [2.5, 97.5])
    return {"delta_ms": float(c.mean() - s.mean()), "ci95": [float(lo), float(hi)]}


def _dist(values) -> dict:
    d = describe(values, boot=False)
    return {k: d[k] for k in ("n", "mean", "p50", "p99", "max") if k in d}


def _sort_key(v: str):
    return (0, int(v), "") if v.isdigit() else (1, 0, v)


# ----------------------------------------------------------------------------- per run


def join_section(report: dict) -> dict:
    checks = {
        "join_rate_is_1": report["join_rate"] == 1.0,
        "no_backend_event_outside_service_window": report["backend_event_outside_service_window"] == 0,
        "no_device_mismatch": report["backend_device_mismatch"] == 0,
        "no_trace_with_several_backend_events": report["traces_with_several_backend_events"] == 0,
        "no_backend_event_without_trace": report["backend_events_without_trace"] == 0,
    }
    return {"report": report, "checks": checks, "pass": all(checks.values())}


def validation_section(recs: list, run: dict) -> dict:
    """Per device, over in-window requests. `recs` are in-window ledger records."""
    events = defaultdict(list)
    for e in ledger.rows(run["backend"]):
        if e["request_id"] is not None:
            events[e["request_id"]].append(e)
    out = {}
    for dev in sorted({r["device"] for r in recs}):
        rs = [r for r in recs if r["device"] == dev]
        joined = [r for r in rs if r["timing_ms"]["forward"] is not None]
        inside = 0
        for r in joined:
            evs = events.get(r["request_id"], [])
            t = r["t_ms"]
            if (
                len(evs) == 1
                and t["service_start"] <= evs[0]["t0_us"] / 1e3 <= evs[0]["t1_us"] / 1e3 <= t["service_end"]
            ):
                inside += 1
        unat = [r["timing_ms"]["unattributed"] for r in rs]
        e2e = [r["timing_ms"]["e2e"] for r in rs]
        out[dev] = {
            "n": len(rs),
            "n_with_backend": len(joined),
            "queue_ms": _dist(r["timing_ms"]["queue"] for r in rs),
            "service_ms": _dist(r["timing_ms"]["service"] for r in rs),
            "occupancy_ms": _dist(r["timing_ms"]["occupancy"] for r in rs),
            "forward_le_service": sum(r["timing_ms"]["forward"] <= r["timing_ms"]["service"] for r in joined),
            "forward_gt_service": sum(r["timing_ms"]["forward"] > r["timing_ms"]["service"] for r in joined),
            "backend_span_inside_service": inside,
            "backend_span_not_inside_service": len(joined) - inside,
            "device_exec_gt_forward": sum(r["timing_ms"]["host"] < 0 for r in joined),  # overlapping spans
            "unattributed_ms": _dist(unat),
            "unattributed_ms_with_backend": _dist(r["timing_ms"]["unattributed"] for r in joined),
            "unattributed_share_of_e2e": {
                "of_total": float(sum(unat) / sum(e2e)) if rs and sum(e2e) > 0 else None,
                "mean_per_request": float(np.mean([u / e for u, e in zip(unat, e2e) if e > 0])) if rs else None,
            },
        }
    return out


def pick_timeline_window(run: dict, records: list) -> tuple[dict, str]:
    matrix = sorted((w for w in run["windows"] if w["kind"] == "matrix"), key=lambda w: (w["cycle"], w["t_start"]))
    if matrix:
        return matrix[0], "first matrix window"
    devs = defaultdict(set)
    for r in records:
        devs[(r["cell"], r["cycle"])].add(r["device"])
    for w in run["windows"]:
        if {"gpu", "ane"} <= devs[(w["cell"], w["cycle"])]:
            return w, "first window with both devices"
    return run["windows"][0], "first window"


def timeline_section(run: dict, records: list) -> dict:
    w, why = pick_timeline_window(run, records)
    a, b = (x * 1e3 for x in w["measure"])
    lo = (a + b) / 2 - SLICE_MS / 2
    hi = lo + SLICE_MS
    recs = sorted(
        (
            r
            for r in records
            if r["cell"] == w["cell"]
            and r["cycle"] == w["cycle"]
            and r["t_ms"]["submit"] < hi
            and r["t_ms"]["response"] > lo
        ),
        key=lambda r: (r["t_ms"]["submit"], r["request_id"]),
    )
    ends = defaultdict(list)  # per lane: end time of each sub-row, to stack overlapping requests
    lanes = defaultdict(list)
    for r in recs:
        segs = [[name, component(r, name) or 0.0] for name in SEGMENTS]  # no backend record: exec/host 0
        start, end = r["t_ms"]["submit"], r["t_ms"]["response"]
        rows_ = ends[r["device"]]
        row = next((i for i, e in enumerate(rows_) if e <= start), len(rows_))
        if row == len(rows_):
            rows_.append(end)
        else:
            rows_[row] = end
        lanes[r["device"]].append(
            {
                "request_id": r["request_id"],
                "stream": r["stream"],
                "in_window": r["in_window"],
                "row": row,
                "start_ms": start - lo,
                "segments": segs,
            }
        )
    return {
        "cell": w["cell"],
        "cycle": w["cycle"],
        "kind": w["kind"],
        "selection": why,
        "t0_ms": lo,
        "span_ms": SLICE_MS,
        "segment_order": list(SEGMENTS),
        "note": "start_ms is relative to t0_ms (ms after the run's t_ref); device_exec, host and unattributed "
        "are stacked durations inside the service window, not their order in time",
        "lanes": {dev: lanes.get(dev, []) for dev in ("gpu", "ane")},
    }


def _historical_entry(historical: dict | None, key: str, cell: str, label: str):
    try:
        h = historical["runs"][key]["inflation"][cell][label]
    except (KeyError, TypeError):
        return None
    return {
        "dispatch_delta_ms": h.get("dispatch_delta_ms"),
        "host_delta_ms": h.get("host_delta_ms"),
        "service_mean_ratio": h.get("service", {}).get("mean", {}).get("ratio"),
        "device_exec_mean_ratio": h.get("device_exec", {}).get("mean", {}).get("ratio"),
        "forward_ms_mean_ratio": h.get("forward_ms", {}).get("mean", {}).get("ratio"),
    }


def decomposition_section(recs: list, key: str, historical: dict | None) -> dict:
    """Device plan: each stream present in its solo cell and in a matrix cell. Only requests
    with a backend record count, so every component is over the same requests and the parts add
    up (occupancy = dispatch + device_exec + host + unattributed + return); the others are counted."""
    by, excluded = defaultdict(list), defaultdict(int)
    for r in recs:
        if r["backend"] is None:
            excluded[(r["cell"], r["stream"])] += 1
        else:
            by[(r["cell"], r["stream"])].append(r)
    out = {}
    for cell in sorted({c for c, _ in by if c and c.startswith("matrix:")}):
        for label in sorted({s for c, s in by if c == cell}):
            solo, conc = by.get((f"solo:{label}", label), []), by[(cell, label)]
            if not solo or not conc:
                continue
            comps = {}
            for name in COMPONENTS:
                s = [v for v in (component(r, name) for r in solo) if v is not None]
                c = [v for v in (component(r, name) for r in conc) if v is not None]
                e = {
                    "n_solo": len(s),
                    "n_concurrent": len(c),
                    "solo_mean": float(np.mean(s)) if s else None,
                    "concurrent_mean": float(np.mean(c)) if c else None,
                    **({"delta": delta_ci(c, s)} if s and c else {}),
                }
                if name in RATIO_COMPONENTS and s and c:
                    e["ratio"] = ratio_ci(c, s, np.mean)
                comps[name] = e

            def dm(n):
                return comps[n].get("delta", {}).get("delta_ms")

            occ = dm("occupancy")
            defined = occ is not None and abs(occ) >= MIN_SHARE_DELTA_MS
            share = {n: (dm(n) / occ if defined and dm(n) is not None else None) for n in SHARE_COMPONENTS}
            share["sum"] = sum(v for v in share.values() if v is not None) if defined else None
            identity = {}
            for side in ("solo_mean", "concurrent_mean"):
                parts = [comps[n][side] for n in SHARE_COMPONENTS]
                identity[side] = comps["occupancy"][side] - sum(parts) if all(p is not None for p in parts) else None
            hist = _historical_entry(historical, key, cell, label)
            new = {
                "dispatch_plus_return_delta_ms": dm("dispatch_plus_return"),
                "host_delta_ms": dm("host"),
                "occupancy_mean_ratio": comps["occupancy"].get("ratio", {}).get("ratio"),
                "device_exec_mean_ratio": comps["device_exec"].get("ratio", {}).get("ratio"),
                "forward_mean_ratio": comps["forward"].get("ratio", {}).get("ratio"),
            }
            out.setdefault(cell, {})[label] = {
                "excluded_without_backend": {
                    "solo": excluded[(f"solo:{label}", label)],
                    "concurrent": excluded[(cell, label)],
                },
                "components": comps,
                "share_of_occupancy_inflation": share,
                "occupancy_minus_parts_ms": identity,
                "historical": hist,
                "comparison": {
                    "dispatch": {
                        "historical_dispatch_delta_ms": hist and hist["dispatch_delta_ms"],
                        "new_dispatch_plus_return_delta_ms": new["dispatch_plus_return_delta_ms"],
                    },
                    "host_delta_ms": {"historical": hist and hist["host_delta_ms"], "new": new["host_delta_ms"]},
                    "service_ratio": {
                        "historical_service_mean_ratio": hist and hist["service_mean_ratio"],
                        "new_occupancy_mean_ratio": new["occupancy_mean_ratio"],
                    },
                    "device_exec_ratio": {
                        "historical": hist and hist["device_exec_mean_ratio"],
                        "new": new["device_exec_mean_ratio"],
                    },
                    "forward_ratio": {
                        "historical": hist and hist["forward_ms_mean_ratio"],
                        "new": new["forward_mean_ratio"],
                    },
                },
            }
    return out


# ----------------------------------------------------------------------------- prediction


def prediction_errors(r: dict) -> dict:
    """signed = completion - predicted = queue_error + service_error (exactly, up to rounding)."""
    t, ts, ro = r["timing_ms"], r["t_ms"], r["routing"]
    pred = ro["predicted_completion_ms"]
    signed = t["completion"] - pred
    q_err = (ts["dispatch"] - ts["routed"]) - ro["target_backlog_ms"]
    s_err = t["occupancy"] - ro["estimated_service_ms"]
    return {
        "predicted": pred,
        "actual": t["completion"],
        "signed": signed,
        "queue_error": q_err,
        "service_error": s_err,
        "relative": signed / pred if pred > 0 else None,
    }


def group_values(r: dict) -> dict:
    dev = r["device"]
    other = "ane" if dev == "gpu" else "gpu"
    q = r["queue"]
    if q[f"{other}_running"] is None and q[f"{other}_depth"] is None:
        state = "no_worker"
    else:
        state = "busy" if q[f"{other}_running"] or (q[f"{other}_depth"] or 0) > 0 else "idle"
    depth = q[f"{dev}_depth"]
    ov = r.get("overlap")
    return {
        "all": "all",
        "target": dev,
        "other_device": state,
        "target|other_device": f"{dev}|{state}",
        "tokens": str(r["shape"]["tokens"]),
        "shape_class": str(r["shape"]["class"]),
        "target_depth": "no_snapshot" if depth is None else ("3+" if depth >= 3 else str(depth)),
        "overlap": "none" if ov is None else ("0" if ov == 0 else ("(0,0.5]" if ov <= 0.5 else "(0.5,1]")),
        "reason": str(r["routing"]["reason"]),
    }


def error_stats(errs: list) -> dict:
    s = np.array([e["signed"] for e in errs])
    a = np.abs(s)
    rel = [e["relative"] for e in errs if e["relative"] is not None]
    return {
        "n": len(errs),
        "mean_predicted_ms": float(np.mean([e["predicted"] for e in errs])),
        "mean_actual_ms": float(np.mean([e["actual"] for e in errs])),
        "mean_signed_ms": float(s.mean()),
        "median_signed_ms": float(np.median(s)),
        "p10_signed_ms": float(np.percentile(s, 10)),
        "p90_signed_ms": float(np.percentile(s, 90)),
        "mean_abs_ms": float(a.mean()),
        "p90_abs_ms": float(np.percentile(a, 90)),
        "p99_abs_ms": float(np.percentile(a, 99)),
        "mean_relative": float(np.mean(rel)) if rel else None,
        "mean_queue_error_ms": float(np.mean([e["queue_error"] for e in errs])),
        "mean_service_error_ms": float(np.mean([e["service_error"] for e in errs])),
    }


def prediction_section(scopes: dict) -> dict:
    """scopes: name -> {"plan", "raw_files", "records"} (in-window ledger records)."""
    out, n, worst = {}, 0, 0.0
    for name, sc in scopes.items():
        groups = defaultdict(lambda: defaultdict(list))
        for r in sc["records"]:
            e = prediction_errors(r)
            worst = max(worst, abs(e["signed"] - (e["queue_error"] + e["service_error"])))
            n += 1
            for dim, val in group_values(r).items():
                groups[dim][val].append(e)
        out[name] = {
            "plan": sc["plan"],
            "raw_files": sc["raw_files"],
            "n": len(sc["records"]),
            "groups": {
                dim: {v: error_stats(groups[dim][v]) for v in sorted(groups[dim], key=_sort_key)}
                for dim in DIMENSIONS
                if dim in groups
            },
        }
    if worst > IDENTITY_TOL_MS:
        raise AssertionError(f"signed error != queue_error + service_error (max residual {worst} ms)")
    return {
        "definitions": {
            "predicted": "routing.predicted_completion_ms = target backlog at routing + service estimate",
            "actual": "timing_ms.completion = received - routed",
            "signed": "actual - predicted",
            "relative": "signed / predicted",
            "queue_error": "(dispatch - routed) - routing.target_backlog_ms",
            "service_error": "timing_ms.occupancy - routing.estimated_service_ms",
            "identity": "signed = queue_error + service_error",
            "other_device": "busy: the other device's snapshot at routing was running or had queued jobs; "
            "no_worker: no snapshot",
            "overlap": "ledger.annotate_overlap: fraction of this request's occupancy with the other device occupied",
        },
        "identity": {"n": n, "max_abs_residual_ms": worst, "tolerance_ms": IDENTITY_TOL_MS},
        "scopes": out,
    }


# ----------------------------------------------------------------------------- all runs


def return_alignment_section(run: dict, recs: list) -> dict:
    """Matrix cells: does a GPU result wait for the ANE's Core ML predict to end?

    For each in-window GPU request, find the ANE predict span (a backend device span) that
    was running when the GPU forward ended (service_end), if any. If the parent can take the
    reply only once predict releases the GIL, received lands just after that span's end and
    the return leg equals the predict time still left at service_end."""
    spans = sorted((a, b) for e in ledger.rows(run["backend"]) if e["device"] == "ane" for a, b in e["spans"])
    starts = [a for a, _ in spans]
    out = {}
    for cell in sorted({r["cell"] for r in recs if r["cell"] and r["cell"].startswith("matrix:")}):
        inside, outside = [], []
        for r in recs:
            if r["device"] != "gpu" or r["cell"] != cell:
                continue
            end_us, recv_us = r["t_ms"]["service_end"] * 1e3, r["t_ms"]["received"] * 1e3
            i = bisect.bisect_right(starts, end_us) - 1
            if i >= 0 and spans[i][0] <= end_us < spans[i][1]:
                inside.append((recv_us - spans[i][1], spans[i][1] - end_us, recv_us - end_us))
            else:
                outside.append(recv_us - end_us)
        entry = {"gpu_requests": len(inside) + len(outside), "ended_inside_ane_predict": len(inside)}
        if inside:
            a = np.asarray(inside, np.float64) / 1e3
            entry.update(
                received_minus_predict_end_ms={str(q): float(np.percentile(a[:, 0], q)) for q in (5, 50, 95)},
                return_mean_ms=float(a[:, 2].mean()),
                predict_remaining_mean_ms=float(a[:, 1].mean()),
                corr_return_vs_remaining=float(np.corrcoef(a[:, 2], a[:, 1])[0, 1]) if len(a) > 2 else None,
            )
        if outside:
            entry["return_mean_ms_when_no_predict_running"] = float(np.mean(outside) / 1e3)
        out[cell] = entry
    return out


def run_key(run: dict) -> str:
    return f"{run['plan']}:{run['args']['model']}:{run['args']['ane_placement']}"


def analyse_run(run: dict, historical: dict | None = None) -> tuple[dict, list]:
    """(results entry without prediction, in-window ledger records)."""
    records, report = ledger.build(run)
    ledger.annotate_overlap(records)
    inwin = [r for r in records if r["in_window"]]
    counts = defaultdict(int)
    for r in inwin:
        counts[f"{r['cell']}|{r['stream']}"] += 1
    entry = {
        "plan": run["plan"],
        "pipeline": run.get("pipeline"),
        "environment": {k: run["environment"][k] for k in ENV_KEYS if k in run.get("environment", {})},
        "args": run["args"],
        "shapes": run.get("shapes"),
        "in_window_requests": dict(sorted(counts.items())),
        "join": join_section(report),
        "validation": validation_section(inwin, run),
        "timeline": timeline_section(run, records),
    }
    if run["plan"] == "device":
        entry["decomposition"] = decomposition_section(inwin, run_key(run), historical)
        entry["return_alignment"] = return_alignment_section(run, inwin)
    return entry, inwin


def analyse_all(paths, historical: dict | None) -> dict:
    results = {
        "generated_by": "research/gpu-ane-interference/scripts/analyze_ledger.py",
        "taxonomy": ledger.KIND,
        "inputs": {},
        "runs": {},
    }
    scopes, pooled = {}, defaultdict(lambda: {"raw_files": [], "records": []})
    for p in paths:
        run = ledger.load(p)
        key = run_key(run)
        results["inputs"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        entry, inwin = analyse_run(run, historical)
        entry["raw_file"] = p.name
        results["runs"][key] = entry
        scopes[key] = {"plan": run["plan"], "raw_files": [p.name], "records": inwin}
        pooled[run["plan"]]["raw_files"].append(p.name)
        pooled[run["plan"]]["records"] += inwin
    for plan, sc in sorted(pooled.items()):
        if len(sc["raw_files"]) > 1:
            scopes[f"pooled:{plan}"] = {"plan": plan, **sc}
    results["prediction"] = prediction_section(scopes)
    return _round(results)


# ----------------------------------------------------------------------------- tables / csv


def _f(v, fmt="{:.2f}"):
    return "–" if v is None else fmt.format(v)


def _ci(d):
    return f"{d['ci95'][0]:.2f}–{d['ci95'][1]:.2f}" if d and "ci95" in d else "–"


def render_tables(res: dict) -> str:
    out = [
        "# Request-ledger analysis tables",
        "",
        "Generated by `scripts/analyze_ledger.py` from `ledger/raw/`; do not edit by hand. Times in ms.",
        "",
    ]
    out += [
        "## Join (ledger.build)",
        "",
        "| Run | Raw file | traces | backend events | joined 1:1 | no backend | several | outside service "
        "| device mismatch | backend without trace | backend without request_id | join rate | pass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for key, run in res["runs"].items():
        j = run["join"]["report"]
        noid = ", ".join(f"{k}: {v}" for k, v in sorted(j["backend_events_without_request_id"].items())) or "0"
        out.append(
            f"| {key} | {run['raw_file']} | {j['traces']} | {j['backend_events']} | {j['traces_joined_1to1']} "
            f"| {j['traces_without_backend_event']} | {j['traces_with_several_backend_events']} "
            f"| {j['backend_event_outside_service_window']} | {j['backend_device_mismatch']} "
            f"| {j['backend_events_without_trace']} | {noid} | {_f(j['join_rate'], '{:.4f}')} "
            f"| {'yes' if run['join']['pass'] else 'NO'} |"
        )
    out += [
        "",
        "## Validation (in-window requests)",
        "",
        "| Run | Raw file | Device | n | queue mean / P99 | service mean / P99 | occupancy mean / P99 "
        "| forward ≤ service | span inside service | unattributed mean / P50 / P99 / max | unattributed / e2e |",
        "|---|---|---|---:|---|---|---|---|---|---|---:|",
    ]
    for key, run in res["runs"].items():
        for dev, v in run["validation"].items():
            u = v["unattributed_ms"]
            out.append(
                f"| {key} | {run['raw_file']} | {dev} | {v['n']} "
                f"| {_f(v['queue_ms'].get('mean'))} / {_f(v['queue_ms'].get('p99'))} "
                f"| {_f(v['service_ms'].get('mean'))} / {_f(v['service_ms'].get('p99'))} "
                f"| {_f(v['occupancy_ms'].get('mean'))} / {_f(v['occupancy_ms'].get('p99'))} "
                f"| {v['forward_le_service']}/{v['n_with_backend']} "
                f"| {v['backend_span_inside_service']}/{v['n_with_backend']} "
                f"| {_f(u.get('mean'), '{:.3f}')} / {_f(u.get('p50'), '{:.3f}')} / {_f(u.get('p99'), '{:.3f}')} "
                f"/ {_f(u.get('max'), '{:.3f}')} | {_f(v['unattributed_share_of_e2e']['of_total'], '{:.2%}')} |"
            )
    out += ["", "## Decomposition: solo vs concurrent (mean ms, Δ with bootstrap 95% CI)", ""]
    for key, run in res["runs"].items():
        for cell, labels in run.get("decomposition", {}).items():
            for label, d in labels.items():
                out += [
                    f"### {key} — {cell} — {label} (raw: {run['raw_file']})",
                    "",
                    "| Component | solo | concurrent | Δ ms | Δ 95% CI | ratio | ratio 95% CI | share of Δ occupancy |",
                    "|---|---:|---:|---:|---|---:|---|---:|",
                ]
                for name, c in d["components"].items():
                    r = c.get("ratio", {})
                    share = d["share_of_occupancy_inflation"].get(name)
                    out.append(
                        f"| {name} | {_f(c['solo_mean'])} | {_f(c['concurrent_mean'])} "
                        f"| {_f(c.get('delta', {}).get('delta_ms'), '{:+.2f}')} | {_ci(c.get('delta'))} "
                        f"| {_f(r.get('ratio'), '×{:.3f}')} | {_ci(r)} | {_f(share, '{:.1%}')} |"
                    )
                cmp_ = d["comparison"]
                out += [
                    "",
                    "| Historical harness (../results.json) | historical | this ledger |",
                    "|---|---:|---:|",
                    f"| dispatch Δ ms (historical dispatch = service − forward; here dispatch + return) "
                    f"| {_f(cmp_['dispatch']['historical_dispatch_delta_ms'], '{:+.2f}')} "
                    f"| {_f(cmp_['dispatch']['new_dispatch_plus_return_delta_ms'], '{:+.2f}')} |",
                    f"| host Δ ms | {_f(cmp_['host_delta_ms']['historical'], '{:+.2f}')} "
                    f"| {_f(cmp_['host_delta_ms']['new'], '{:+.2f}')} |",
                    f"| service × (historical service; here occupancy) "
                    f"| {_f(cmp_['service_ratio']['historical_service_mean_ratio'], '×{:.3f}')} "
                    f"| {_f(cmp_['service_ratio']['new_occupancy_mean_ratio'], '×{:.3f}')} |",
                    f"| device exec × | {_f(cmp_['device_exec_ratio']['historical'], '×{:.3f}')} "
                    f"| {_f(cmp_['device_exec_ratio']['new'], '×{:.3f}')} |",
                    f"| forward × | {_f(cmp_['forward_ratio']['historical'], '×{:.3f}')} "
                    f"| {_f(cmp_['forward_ratio']['new'], '×{:.3f}')} |",
                    "",
                ]
    out += [
        "## GPU return leg vs the ANE's Core ML predict (matrix cells)",
        "",
        "For each GPU request: was an ANE predict running when the GPU forward ended, and when did the "
        "dispatcher get the result relative to that predict's end?",
        "",
        "| Run | Raw file | Cell | GPU requests | ended inside a predict | received − predict end P5 / P50 / P95 "
        "| return mean | predict left at forward end, mean | corr | return mean, no predict running |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for key, run in res["runs"].items():
        for cell, e in run.get("return_alignment", {}).items():
            q = e.get("received_minus_predict_end_ms", {})
            out.append(
                f"| {key} | {run['raw_file']} | {cell} | {e['gpu_requests']} | {e['ended_inside_ane_predict']} "
                f"| {_f(q.get('5'), '{:.3f}')} / {_f(q.get('50'), '{:.3f}')} / {_f(q.get('95'), '{:.3f}')} "
                f"| {_f(e.get('return_mean_ms'))} | {_f(e.get('predict_remaining_mean_ms'))} "
                f"| {_f(e.get('corr_return_vs_remaining'), '{:.3f}')} "
                f"| {_f(e.get('return_mean_ms_when_no_predict_running'), '{:.3f}')} |"
            )
    out.append("")
    p = res["prediction"]
    out += [
        "## Completion prediction error (actual − predicted, ms)",
        "",
        f"signed = queue error + service error; max residual over {p['identity']['n']} requests: "
        f"{p['identity']['max_abs_residual_ms']:.2e} ms.",
        "",
    ]
    for scope, sc in p["scopes"].items():
        out += [
            f"### {scope} (raw: {', '.join(sc['raw_files'])})",
            "",
            "| Group | Value | n | mean signed | median | mean abs | P90 abs | P99 abs | mean rel. "
            "| mean queue err | mean service err |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for dim, vals in sc["groups"].items():
            for v, s in vals.items():
                out.append(
                    f"| {dim} | {v} | {s['n']} | {s['mean_signed_ms']:+.2f} | {s['median_signed_ms']:+.2f} "
                    f"| {s['mean_abs_ms']:.2f} | {s['p90_abs_ms']:.2f} | {s['p99_abs_ms']:.2f} "
                    f"| {_f(s['mean_relative'], '{:+.1%}')} | {s['mean_queue_error_ms']:+.2f} "
                    f"| {s['mean_service_error_ms']:+.2f} |"
                )
        out.append("")
    return "\n".join(out).rstrip() + "\n"


CSV_FIELDS = (
    "scope",
    "plan",
    "raw_files",
    "dimension",
    "value",
    "n",
    "mean_predicted_ms",
    "mean_actual_ms",
    "mean_signed_ms",
    "median_signed_ms",
    "p10_signed_ms",
    "p90_signed_ms",
    "mean_abs_ms",
    "p90_abs_ms",
    "p99_abs_ms",
    "mean_relative",
    "mean_queue_error_ms",
    "mean_service_error_ms",
)


def render_csv(res: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_FIELDS)
    for scope, sc in res["prediction"]["scopes"].items():
        for dim, vals in sc["groups"].items():
            for v, s in vals.items():
                row = {
                    "scope": scope,
                    "plan": sc["plan"],
                    "raw_files": ";".join(sc["raw_files"]),
                    "dimension": dim,
                    "value": v,
                    **s,
                }
                w.writerow(["" if row[k] is None else row[k] for k in CSV_FIELDS])
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out-dir", type=Path, default=LEDGER)
    ap.add_argument("--historical", type=Path, default=HISTORICAL)
    ap.add_argument("--check", action="store_true", help="fail if any output differs from a fresh analysis")
    args = ap.parse_args()
    paths = args.paths or sorted(args.raw.glob("*.json.gz"))
    if not paths:
        sys.exit(f"no raw runs in {args.raw}")
    historical = json.loads(args.historical.read_text()) if args.historical.exists() else None
    res = analyse_all(paths, historical)
    outputs = {
        args.out_dir / "results.json": json.dumps(res, indent=1, sort_keys=True) + "\n",
        args.out_dir / "tables.md": render_tables(res),
        args.out_dir / "prediction_error.csv": render_csv(res),
    }
    if args.check:
        stale = [p.name for p, text in outputs.items() if not p.exists() or p.read_text() != text]
        if stale:
            sys.exit(f"stale: {stale}; re-run analyze_ledger.py")
        print("ledger outputs are up to date")
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for p, text in outputs.items():
        p.write_text(text)
        print("wrote", p)


if __name__ == "__main__":
    main()
