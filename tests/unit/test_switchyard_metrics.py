"""Switchyard metrics on synthetic trace records (no model, no clock)."""

from __future__ import annotations

import pytest

from laya_apple.benchmark import stats
from laya_apple.demos.switchyard import metrics

MS = 1_000_000


def rec(
    index,
    latency_ms,
    *,
    cls="train",
    answer="A",
    oracle="A",
    queue_ms=1.0,
    service_ms=5.0,
    lag_ms=0.1,
    target="ane",
    start_ns=0,
    offset_ms=None,
    reason="validated_short_single_question_path",
):
    """A trace.jsonl record whose derived intervals are exactly the given ones."""
    arrival = start_ns + int((offset_ms if offset_ms is not None else index * 10) * MS)
    submit = arrival + int(lag_ms * MS)
    enter = submit + 1 * MS // 10
    dispatch = enter + int(queue_ms * MS)
    start = dispatch + MS // 100
    end = start + int(service_ms * MS)
    response = arrival + int(latency_ms * MS)
    assert response >= end, "latency too small for the given intervals"
    r = {
        "request_id": 1000 + index,
        "sequence_length": 64,
        "question_count": 1,
        "target": target,
        "routing_reason": reason,
        "service_estimate_ms": 9.0,
        "gpu": None,
        "ane": None,
        "submit_ns": submit,
        "prepared_ns": submit,
        "routed_ns": enter,
        "queue_enter_ns": enter,
        "dispatch_ns": dispatch,
        "service_start_ns": start,
        "service_end_ns": end,
        "received_ns": end,
        "response_ns": response,
        "round": 0,
        "index": index,
        "class": cls,
        "arrival_ns": arrival,
    }
    if cls == "train":
        r |= {
            "line": 1,
            "train_id": 4000 + index,
            "pattern": 0,
            "answer": answer,
            "probabilities": {"A": 0.9, "B": 0.05, "C": 0.05},
            "oracle": oracle,
            "outcome": metrics.outcome(latency_ms, answer, oracle, 100.0),
        }
    return r


def test_outcome_classification():
    assert metrics.outcome(99.9, "A", "A", 100) == "delivered"
    assert metrics.outcome(100.0, "A", "A", 100) == "delivered"  # late means strictly over
    assert metrics.outcome(100.001, "A", "A", 100) == "late"
    assert metrics.outcome(150, "B", "A", 100) == "late"  # late wins over misrouted
    assert metrics.outcome(20, "B", "A", 100) == "misrouted"


def test_intervals_come_from_raw_timestamps():
    r = rec(0, 42.0, queue_ms=7.0, service_ms=9.0, lag_ms=0.3)
    assert metrics.latency_ms(r) == pytest.approx(42.0)
    assert metrics.queue_ms(r) == pytest.approx(7.0)
    assert metrics.service_ms(r) == pytest.approx(9.0)
    assert metrics.submit_lag_ms(r) == pytest.approx(0.3)
    assert metrics.overhead_ms(r) == pytest.approx(42.0 - 7.0 - 9.0)


def test_round_summary_counts_and_percentiles():
    lat = [10, 20, 30, 60, 120, 260, 600, 15, 22, 40]
    recs = [rec(i, x) for i, x in enumerate(lat)]
    recs[1] = rec(1, 20, answer="B")  # misrouted, on time
    recs[4] = rec(4, 120, answer="C")  # late and wrong: counted late
    recs.append(rec(10, 500, cls="long_1q", target="gpu", reason="sequence_exceeds_ane_auto_range"))
    s = metrics.summarize([(recs, 0)])
    g, sy = s["game"], s["systems"]
    assert g == {"trains": 10, "delivered": 6, "late": 3, "misrouted": 1, "route_accuracy": 0.8}
    assert sy["late"] == {"count": 3, "rate": 0.3, "deadline_ms": 100.0}
    expected = stats(lat)
    for key in ("n", "p50_ms", "p95_ms", "p99_ms", "mean_ms", "max_ms"):
        assert sy["decision_latency"][key] == pytest.approx(expected[key])
    assert sy["miss_rate_at_ms"] == {"25": 0.6, "50": 0.4, "100": 0.3, "250": 0.2, "500": 0.1}
    sec = sy["secondary"]
    assert sec["devices"] == {"ane": 10} and sec["routing_reasons"] == {"validated_short_single_question_path": 10}
    assert sec["background"]["long_1q"]["decision_latency"]["n"] == 1  # background never in the train metrics
    assert sec["background"]["medium_1q"]["decision_latency"] == {"n": 0}
    assert sec["makespan_s"] == pytest.approx(max(r["response_ns"] for r in recs) / 1e9)
    assert sec["decisions_per_s"] == pytest.approx(10 / sec["makespan_s"])
    assert sec["completed_req_s"] == pytest.approx(11 / sec["makespan_s"])


def test_miss_rate_thresholds_are_strict():
    assert metrics.miss_rates([25, 25.001, 50, 100, 500, 501]) == {
        "25": 5 / 6,
        "50": 3 / 6,
        "100": 2 / 6,
        "250": 2 / 6,
        "500": 1 / 6,
    }
    assert metrics.miss_rates([]) == dict.fromkeys(["25", "50", "100", "250", "500"], 0.0)


def test_pooling_concatenates_samples_never_averages_percentiles():
    a = [rec(i, x, start_ns=0) for i, x in enumerate([10, 11, 12, 13])]
    b = [rec(i, x, start_ns=10**12) for i, x in enumerate([200, 300, 400, 500])]
    pooled = metrics.summarize([(a, 0), (b, 10**12)])
    assert pooled["systems"]["decision_latency"]["p99_ms"] == pytest.approx(
        stats([10, 11, 12, 13, 200, 300, 400, 500])["p99_ms"]
    )
    assert pooled["game"]["trains"] == 8 and pooled["game"]["late"] == 4
    ms_a = metrics.summarize([(a, 0)])["systems"]["secondary"]["makespan_s"]
    ms_b = metrics.summarize([(b, 10**12)])["systems"]["secondary"]["makespan_s"]
    assert pooled["systems"]["secondary"]["makespan_s"] == pytest.approx(ms_a + ms_b)


def test_decision_disagreements_join_on_schedule_index():
    gpu = [rec(0, 10, answer="A"), rec(1, 10, answer="B"), rec(2, 10, cls="medium_1q", target="gpu")]
    same = [rec(0, 90, answer="A"), rec(1, 150, answer="B")]
    other = [rec(0, 10, answer="C"), rec(1, 10, answer="B")]
    assert metrics.decision_disagreements([gpu, same]) == 0  # latency never enters
    assert metrics.decision_disagreements([gpu, other]) == 1
    assert metrics.decision_disagreements([gpu]) == 0


def test_timetable_hash_is_relative_to_the_round_start():
    a = [rec(i, 10, start_ns=0) for i in range(3)]
    b = [rec(i, 50, start_ns=5 * 10**12) for i in range(3)]
    assert metrics.timetable_sha256(a, 0) == metrics.timetable_sha256(b, 5 * 10**12)
    c = [rec(i, 10, start_ns=0, offset_ms=i * 11) for i in range(3)]
    assert metrics.timetable_sha256(c, 0) != metrics.timetable_sha256(a, 0)
