"""The GPU-ANE interference research harness (research/gpu-ane-interference/): its tracing only
observes, and its interval and statistics helpers are right. No model loading."""

from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "research" / "gpu-ane-interference" / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"interference_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def jobtrace():
    return _load("jobtrace")


@pytest.fixture(scope="module")
def analyze():
    return _load("analyze")


class FakeWorker:
    """The DeviceWorker surface jobtrace.tag_dispatcher wraps: _run(job_id, rows)."""

    kind = "ane"

    def __init__(self, jobtrace, fail=False):
        self.jobtrace, self.fail = jobtrace, fail
        self.seen = []

    def _run(self, job_id, rows):
        self.seen.append((job_id, rows, self.jobtrace.current_request_id()))
        if self.fail:
            raise RuntimeError("device error")
        return ("logits", "act", 1, 2)


def test_dispatcher_tag_exposes_the_job_id_and_changes_nothing(jobtrace):
    w = FakeWorker(jobtrace)
    jobtrace.tag_dispatcher(w)
    rows = [{"ids": [1, 2]}]
    assert w._run(41, rows) == ("logits", "act", 1, 2)  # the result is returned unchanged
    assert w.seen == [(41, rows, 41)]  # same arguments; the forward saw its request id
    assert jobtrace.current_request_id() is None  # cleared after the job


def test_dispatcher_tag_propagates_errors_and_clears(jobtrace):
    w = FakeWorker(jobtrace, fail=True)
    jobtrace.tag_dispatcher(w)
    with pytest.raises(RuntimeError, match="device error"):
        w._run(7, [{}])
    assert w.seen[0][2] == 7 and jobtrace.current_request_id() is None


def test_worker_connection_tags_received_jobs_only(jobtrace):
    class Conn:
        def __init__(self, msgs):
            self.msgs, self.sent = list(msgs), []

        def recv(self):
            return self.msgs.pop(0)

        def send(self, m):
            self.sent.append(m)

    conn = jobtrace._TaggingConnection(Conn([(12, ["row"]), None]))
    assert conn.recv() == (12, ["row"]) and jobtrace.current_request_id() == 12
    conn.send((12, "ok", ()))  # other calls pass through
    assert conn._conn.sent == [(12, "ok", ())]
    assert conn.recv() is None and jobtrace.current_request_id() == 12  # the close message is not a job


def test_request_id_is_per_thread(jobtrace):
    w = FakeWorker(jobtrace)
    jobtrace.tag_dispatcher(w)
    seen = {}

    def other():
        seen["other"] = jobtrace.current_request_id()

    w._run(5, [])
    t = threading.Thread(target=other)
    t.start()
    t.join()
    assert seen["other"] is None


def test_clock_is_system_wide_monotonic(jobtrace):
    info = jobtrace.clock_info()
    assert info["monotonic"]
    if sys.platform == "darwin":  # parent and worker timestamps are only comparable on this clock
        assert info["implementation"] == "mach_absolute_time()"


def test_overlap_fraction_and_busy_union(analyze):
    busy = analyze.busy_union([(5, 7), (0, 2), (1, 3)])
    assert busy == [[0, 3], [5, 7]]
    starts = [b[0] for b in busy]
    assert analyze.overlap_fraction(0, 10, busy, starts) == pytest.approx(0.5)
    assert analyze.overlap_fraction(3, 5, busy, starts) == 0.0
    assert analyze.overlap_fraction(6, 6.5, busy, starts) == 1.0
    assert analyze.overlap_fraction(2, 6, busy, starts) == pytest.approx(0.5)
    assert analyze.overlap_fraction(4, 4, busy, starts) == 0.0


def test_describe_flags_an_unreliable_p99(analyze):
    small = analyze.describe(range(100))
    assert small["n"] == 100 and small["p99_tail_n"] < 10 and small["p99_reliable"] is False
    big = analyze.describe(range(2000))
    assert big["p99_reliable"] is True
    assert big["p99_ci95"][0] <= big["p99"] <= big["p99_ci95"][1]


def test_ratio_ci_brackets_the_ratio(analyze):
    import numpy as np

    rng = np.random.default_rng(0)
    solo, conc = rng.normal(10, 0.5, 500), rng.normal(12, 0.5, 500)
    r = analyze.ratio_ci(conc, solo)
    assert r["ci95"][0] < r["ratio"] < r["ci95"][1]
    assert r["ratio"] == pytest.approx(1.2, abs=0.02)


def test_columnar_round_trip(analyze):
    cols = {"a": [1, 2], "b": [None, "x"]}
    assert analyze.rows(cols) == [{"a": 1, "b": None}, {"a": 2, "b": "x"}]
    assert analyze.rows({}) == []


def test_compact_encoding_round_trips_every_request(analyze):
    compact = _load("compact")
    recs = [
        {
            "arrival": 10.0,
            "queue_enter": 10.5,
            "device_start": 11.0,
            "device_end": 21.03,
            "response": 21.2,
            "forward_ms": 9.98,
            "device": "ane",
            "in_window": True,
            "match": True,
            "seed": 3,
            "estimate_ms": 9.93,
        },
        {"arrival": 12.5, "response": 13.0, "device": "gpu", "in_window": False, "match": False, "seed": 4},
    ]
    phases = analyze.Phases({"kind": ["ane"], "t0": [11.01], "t1": [21.0], "cpu_ms": [0.5], "dev": [[[11.2, 20.9]]]})
    out = compact.decode(compact.encode(recs, phases))
    assert len(out) == 2
    a, b = out
    for k in ("arrival", "queue_enter", "device_start", "device_end", "response", "forward_ms", "estimate_ms"):
        assert a[k] == pytest.approx(recs[0][k], abs=0.006)
    assert a["device_exec"] == pytest.approx(9.7, abs=0.006) and a["host"] == pytest.approx(0.29, abs=0.006)
    assert a["match"] is True and a["seed"] == 3 and a["device"] == "ane"
    assert b["response"] == pytest.approx(13.0) and "device_start" not in b and b["match"] is False


def test_contention_fit_separates_additive_from_proportional():
    contention = _load("contention")
    a, d = contention._fit([(12.0, 20.0), (71.0, 79.0)])  # the same +8 ms at both lengths
    assert a == pytest.approx(0.0) and d == pytest.approx(8.0)
    a, d = contention._fit([(10.0, 11.0), (70.0, 77.0)])  # +10% at both lengths
    assert a == pytest.approx(0.1) and d == pytest.approx(0.0)
    a, d = contention._fit([(10.0, 9.9), (70.0, 69.0)])  # faster: clamped, never negative
    assert a == 0.0 and d == 0.0
