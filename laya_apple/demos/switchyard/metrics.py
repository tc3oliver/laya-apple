"""Round metrics from trace.jsonl records (pure; no model, no clock).

Per request (all `time.monotonic_ns()`, RequestTrace fields plus `arrival_ns`):
  decision latency = response_ns - arrival_ns
  queue wait       = dispatch_ns - queue_enter_ns     (RequestTrace.queue_ms)
  inference        = service_end_ns - service_start_ns (RequestTrace.service_ms)
  submit lag       = submit_ns - arrival_ns
  overhead         = decision latency - queue wait - inference

A train is late when its decision latency exceeds the deadline, misrouted when on time with
an answer other than the oracle, delivered otherwise. Percentiles are
laya_apple.benchmark.stats. Background classes are reported separately and never mixed
into the train metrics. Throughput, latency and correctness stay separate numbers.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict

from ...benchmark import stats
from . import world

DELIVERED, LATE, MISROUTED = "delivered", "late", "misrouted"
OUTCOMES = (DELIVERED, LATE, MISROUTED)


def latency_ms(rec: dict) -> float:
    return (rec["response_ns"] - rec["arrival_ns"]) / 1e6


def queue_ms(rec: dict) -> float:
    return (rec["dispatch_ns"] - rec["queue_enter_ns"]) / 1e6


def service_ms(rec: dict) -> float:
    return (rec["service_end_ns"] - rec["service_start_ns"]) / 1e6


def submit_lag_ms(rec: dict) -> float:
    return (rec["submit_ns"] - rec["arrival_ns"]) / 1e6


def overhead_ms(rec: dict) -> float:
    return latency_ms(rec) - queue_ms(rec) - service_ms(rec)


def outcome(latency: float, answer: str, oracle: str, deadline_ms: float) -> str:
    if latency > deadline_ms:
        return LATE
    return DELIVERED if answer == oracle else MISROUTED


def _stats(values) -> dict:
    return stats(values) if len(values) else {"n": 0}


def miss_rates(latencies, thresholds=world.THRESHOLDS_MS) -> dict:
    """Fraction of decisions slower than each threshold, keyed by the threshold in ms."""
    n = len(latencies)
    return {str(int(t)): (sum(x > t for x in latencies) / n if n else 0.0) for t in thresholds}


def summarize(windows, deadline_ms: float = world.DEADLINE_MS, thresholds=world.THRESHOLDS_MS) -> dict:
    """{game, systems} for one round, or pooled over several: `windows` is a list of
    (records, round_start_ns). Pooling concatenates samples; it never averages percentiles."""
    records = [r for recs, _ in windows for r in recs]
    makespan_s = sum((max(r["response_ns"] for r in recs) - start) / 1e9 for recs, start in windows if recs)
    trains = [r for r in records if r["class"] == world.TRAIN]
    lat = [latency_ms(r) for r in trains]
    outcomes = Counter(outcome(x, r["answer"], r["oracle"], deadline_ms) for x, r in zip(lat, trains))
    n = len(trains)
    background = {}
    for cls in world.BACKGROUND:
        rs = [r for r in records if r["class"] == cls]
        background[cls] = {
            "decision_latency": _stats([latency_ms(r) for r in rs]),
            "queue_wait": _stats([queue_ms(r) for r in rs]),
            "devices": dict(Counter(r["target"] for r in rs)),
        }
    return {
        "game": {
            "trains": n,
            DELIVERED: outcomes[DELIVERED],
            LATE: outcomes[LATE],
            MISROUTED: outcomes[MISROUTED],
            "route_accuracy": (sum(r["answer"] == r["oracle"] for r in trains) / n) if n else 0.0,
        },
        "systems": {
            "late": {"count": outcomes[LATE], "rate": outcomes[LATE] / n if n else 0.0, "deadline_ms": deadline_ms},
            "decision_latency": _stats(lat),
            "queue_wait": _stats([queue_ms(r) for r in trains]),
            "miss_rate_at_ms": miss_rates(lat, thresholds),
            "secondary": {
                "decisions_per_s": n / makespan_s if makespan_s else 0.0,
                "completed_req_s": len(records) / makespan_s if makespan_s else 0.0,
                "makespan_s": makespan_s,
                "submit_lag": _stats([submit_lag_ms(r) for r in trains]),
                # every request, background included: evidence the open loop kept to schedule
                "submit_lag_all_requests": _stats([submit_lag_ms(r) for r in records]),
                "inference": _stats([service_ms(r) for r in trains]),
                "overhead": _stats([overhead_ms(r) for r in trains]),
                "devices": dict(Counter(r["target"] for r in trains)),
                "routing_reasons": dict(Counter(r["routing_reason"] for r in trains)),
                "background": background,
            },
        },
    }


def timetable_sha256(records: list[dict], round_start_ns: int) -> str:
    """Fingerprint of what a round actually submitted: schedule index, class, scheduled
    offset (ns) and train contents. Equal across rounds means the same timetable ran."""
    rows = [
        [r["index"], r["class"], r["arrival_ns"] - round_start_ns, r.get("train_id"), r.get("pattern"), r.get("line")]
        for r in sorted(records, key=lambda r: r["index"])
    ]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def decision_disagreements(rounds) -> int:
    """Trains (by schedule index) whose answer differs between any two of `rounds`, a list of
    record lists. Latency never enters: this compares decisions only."""
    answers = defaultdict(set)
    for records in rounds:
        for r in records:
            if r["class"] == world.TRAIN:
                answers[r["index"]].add(r["answer"])
    return sum(len(a) > 1 for a in answers.values())
