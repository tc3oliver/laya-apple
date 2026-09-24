"""The canonical request ledger: runtime RequestTrace + research backend phases, joined on request_id.

    uv run python research/gpu-ane-interference/scripts/ledger.py RUN.json.gz [--jsonl OUT.jsonl]

A run written by interference.py carries three tables (columns, times in integer µs relative
to the run's reference `t_ref_ns`, all from time.monotonic_ns):

    traces    one row per completed request: laya_apple.RequestTrace as the runtime emitted it
    backend   one row per backend forward: jobtrace.py's phase record (device spans, CPU)
    streams   per window and stream, the workload generator's own fields (arrival, seed,
              shape class, answer match, in_window)

`build()` joins them into one record per request (`RequestLedger`):

    {"request_id": 18421, "cell": "matrix:gpu_M+ane_B", "cycle": 0, "stream": "gpu_M",
     "device": "gpu", "shape": {"class": "M", "tokens": 128, "questions": 1},
     "routing": {"reason": "gpu_requested", "gpu_backlog_ms": 3.2, "ane_backlog_ms": 14.7,
                 "target_backlog_ms": 3.2, "estimated_service_ms": 12.2,
                 "predicted_completion_ms": 15.4},
     "queue": {"gpu_depth": 1, "ane_depth": 3, "gpu_running": true, "ane_running": true},
     "t_ms": {"arrival": ..., "submit": ..., ..., "response": ...},
     "timing_ms": {"prepare": 0.8, "route": 0.04, "enqueue": 0.01, "queue": 2.1,
                   "dispatch": 3.4, "service": 10.6, "return": 3.5, "occupancy": 17.5,
                   "forward": 10.55, "device_exec": 9.8, "host": 0.75, "unattributed": 0.05,
                   "postprocess": 0.3, "e2e": 20.7, "completion": 19.1, "generator_lag": 0.0},
     "backend": {"pid": 812, "cpu_ms": 1.4, "spans": 1}, ...}

What each duration is, and how it is known (`KIND`):

  measured   (runtime trace)  prepare, route, enqueue, queue, dispatch, service, return,
                              postprocess, e2e, occupancy, completion
  measured   (backend hooks)  forward (the hook's span around backend.forward),
                              device_exec (sum of mx.eval / Core ML predict spans), cpu_ms
  derived                     host = forward - device_exec: everything else inside the forward
                              (graph construction, NumPy features, heads)
  unattributed                service - forward: the runtime's service window that no hook span
                              covers (the two clock reads and the hook's own wrapper); the whole
                              service when a request has no backend record
  generator                   generator_lag = submit - arrival: open loop only, how late the
                              workload generator submitted a scheduled arrival (not runtime time)

Identities that hold by construction: occupancy = dispatch + service + return (the device's
FIFO is held from dispatch to received); e2e = prepare + route + enqueue + queue + occupancy +
postprocess; service = forward + unattributed = device_exec + host + unattributed. Nothing
unknown is folded into a known phase.

`completion` = received - routed is what the router's `predicted_completion_ms` (= backlog of
the target at routing + the service estimate charged to it) predicts: scheduling.decide_queued
compares exactly `backlog + service estimate` per device, and the backlog is the sum of these
estimates over the target's queue.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

STAMPS = (
    "submit",
    "prepared",
    "routed",
    "queue_enter",
    "dispatch",
    "service_start",
    "service_end",
    "received",
    "response",
)

KIND = {
    "prepare": "measured",
    "route": "measured",
    "enqueue": "measured",
    "queue": "measured",
    "dispatch": "measured",
    "service": "measured",
    "return": "measured",
    "occupancy": "measured",
    "postprocess": "measured",
    "e2e": "measured",
    "completion": "measured",
    "forward": "measured",
    "device_exec": "measured",
    "host": "derived",
    "unattributed": "unattributed",
    "generator_lag": "generator",
}


def rows(cols: dict) -> list:
    keys = list(cols)
    return [dict(zip(keys, vals)) for vals in zip(*(cols[k] for k in keys))] if keys else []


def load(path: Path) -> dict:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- one request


def _ms(us):
    return None if us is None else us / 1e3


def _snapshot(t: dict, dev: str):
    if t.get(f"{dev}_backlog_ms") is None:
        return None
    return {"backlog_ms": t[f"{dev}_backlog_ms"], "depth": t[f"{dev}_queued_jobs"], "running": bool(t[f"{dev}_running"])}


def ledger_record(t: dict, backend: dict | None, gen: dict | None) -> dict:
    """One request: t = a traces row, backend = its phase row (or None), gen = its stream row."""
    ts = {s: _ms(t[s + "_us"]) for s in STAMPS}
    d = lambda a, b: ts[b] - ts[a]  # noqa: E731
    gpu, ane = _snapshot(t, "gpu"), _snapshot(t, "ane")
    target = gpu if t["target"] == "gpu" else ane
    target_backlog = target["backlog_ms"] if target else 0.0
    timing = {
        "prepare": d("submit", "prepared"),
        "route": d("prepared", "routed"),
        "enqueue": d("routed", "queue_enter"),
        "queue": d("queue_enter", "dispatch"),
        "dispatch": d("dispatch", "service_start"),
        "service": d("service_start", "service_end"),
        "return": d("service_end", "received"),
        "occupancy": d("dispatch", "received"),
        "postprocess": d("received", "response"),
        "e2e": d("submit", "response"),
        "completion": d("routed", "received"),
    }
    if backend is not None:
        fwd = (backend["t1_us"] - backend["t0_us"]) / 1e3
        exec_ms = sum(b - a for a, b in backend["spans"]) / 1e3
        timing.update(forward=fwd, device_exec=exec_ms, host=fwd - exec_ms, unattributed=timing["service"] - fwd)
        be = {"pid": backend["pid"], "cpu_ms": backend["cpu_us"] / 1e3, "spans": len(backend["spans"])}
    else:
        timing.update(forward=None, device_exec=None, host=None, unattributed=timing["service"])
        be = None
    arrival = _ms(gen["arrival_us"]) if gen else None
    timing["generator_lag"] = ts["submit"] - arrival if arrival is not None else None
    shape_class = gen.get("shape") if gen else None
    return {
        "request_id": t["request_id"],
        "cell": gen.get("cell") if gen else None,
        "cycle": gen.get("cycle") if gen else None,
        "stream": gen.get("stream") if gen else None,
        "in_window": bool(gen["in_window"]) if gen else False,
        "match": bool(gen["match"]) if gen else None,
        "seed": gen.get("seed") if gen else None,
        "device": t["target"],
        "shape": {"class": shape_class, "tokens": t["sequence_length"], "questions": t["question_count"]},
        "routing": {
            "reason": t["routing_reason"],
            "gpu_backlog_ms": gpu["backlog_ms"] if gpu else 0.0,
            "ane_backlog_ms": ane["backlog_ms"] if ane else 0.0,
            "target_backlog_ms": target_backlog,
            "estimated_service_ms": t["service_estimate_ms"],
            "predicted_completion_ms": target_backlog + t["service_estimate_ms"],
        },
        "queue": {
            "gpu_depth": gpu["depth"] if gpu else None,
            "ane_depth": ane["depth"] if ane else None,
            "gpu_running": gpu["running"] if gpu else None,
            "ane_running": ane["running"] if ane else None,
        },
        "t_ms": {"arrival": arrival, **ts},
        "timing_ms": timing,
        "backend": be,
    }


# ----------------------------------------------------------------------------- the join


def build(run: dict) -> tuple[list, dict]:
    """(ledger records, join report). Every trace yields a record; nothing is dropped. The
    report counts every trace and backend event by how it joined, and names what did not."""
    traces = rows(run["traces"])
    events = rows(run["backend"])
    gen = {}
    for w in run["windows"]:
        for label, st in w["streams"].items():
            for r in rows(st["records"]):
                if r.get("request_id") is not None:
                    gen[r["request_id"]] = {**r, "cell": w["cell"], "cycle": w["cycle"], "stream": label}
    by_id = defaultdict(list)
    no_id = []
    for e in events:
        (by_id[e["request_id"]] if e["request_id"] is not None else no_id).append(e)
    trace_ids = {t["request_id"] for t in traces}
    first_submit = min((t["submit_us"] for t in traces), default=0)
    out, outside, device_mismatch, multi = [], [], [], []
    for t in traces:
        evs = by_id.get(t["request_id"], [])
        if len(evs) > 1:
            multi.append(t["request_id"])
        e = evs[0] if len(evs) == 1 else None
        if e is not None:
            if e["device"] != t["target"]:
                device_mismatch.append(t["request_id"])
            if not (t["service_start_us"] <= e["t0_us"] <= e["t1_us"] <= t["service_end_us"]):
                outside.append(t["request_id"])
        out.append(ledger_record(t, e, gen.get(t["request_id"])))
    no_id_kind = Counter("before first request" if e["t0_us"] < first_submit else "during the run" for e in no_id)
    orphans = sorted(i for i in by_id if i not in trace_ids)
    report = {
        "traces": len(traces),
        "backend_events": len(events),
        "traces_joined_1to1": sum(1 for t in traces if len(by_id.get(t["request_id"], [])) == 1),
        "traces_without_backend_event": sum(1 for t in traces if not by_id.get(t["request_id"])),
        "traces_with_several_backend_events": len(multi),
        "backend_events_without_request_id": dict(no_id_kind),
        "backend_events_without_trace": len(orphans),
        "backend_event_outside_service_window": len(outside),
        "backend_device_mismatch": len(device_mismatch),
        "generator_rows_without_trace": sum(1 for i in gen if i not in trace_ids),
        "traces_without_generator_row": sum(1 for t in traces if t["request_id"] not in gen),
        "examples": {"outside": outside[:5], "orphans": orphans[:5], "multi": multi[:5]},
    }
    report["join_rate"] = report["traces_joined_1to1"] / report["traces"] if traces else None
    return out, report


# ----------------------------------------------------------------------------- overlap


def busy_union(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def overlap_fraction(a, b, busy, starts):
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


def annotate_overlap(records: list) -> None:
    """Per (cell, cycle): the fraction of each request's device occupancy (dispatch to
    received) during which the OTHER device was occupied by some request."""
    groups = defaultdict(list)
    for r in records:
        groups[(r["cell"], r["cycle"])].append(r)
    for recs in groups.values():
        busy = defaultdict(list)
        for r in recs:
            busy[r["device"]].append((r["t_ms"]["dispatch"], r["t_ms"]["received"]))
        busy = {k: busy_union(v) for k, v in busy.items()}
        starts = {k: [x[0] for x in v] for k, v in busy.items()}
        for r in recs:
            other = "ane" if r["device"] == "gpu" else "gpu"
            r["overlap"] = (
                overlap_fraction(r["t_ms"]["dispatch"], r["t_ms"]["received"], busy[other], starts[other])
                if other in busy
                else 0.0
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--jsonl", type=Path, help="write the ledger, one request per line")
    args = ap.parse_args()
    records, report = build(load(args.run))
    print(json.dumps(report, indent=2))
    if args.jsonl:
        annotate_overlap(records)
        with args.jsonl.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
