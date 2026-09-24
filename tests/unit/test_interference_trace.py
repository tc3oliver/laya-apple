"""The GPU-ANE interference research harness (research/gpu-ane-interference/): its tracing only
observes, and its interval and statistics helpers are right. No model loading."""

from __future__ import annotations

import importlib.util
import sys
import threading
from concurrent.futures import Future
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
    """The DeviceWorker surface the tracer wraps: submit, _run, backlog_ms, kind."""

    kind = "gpu"

    def __init__(self, fail=False):
        self.fail = fail
        self.submitted = []

    def backlog_ms(self):
        return 3.5

    def submit(self, rows, estimate_ms):
        self.submitted.append((rows, estimate_ms))
        fut = Future()
        fut.set_result("queued")
        return fut

    def _run(self, rows):
        if self.fail:
            raise RuntimeError("device error")
        return ("logits", "act", 1.25)


def test_tracer_records_times_and_returns_results_unchanged(jobtrace):
    w, jobs = FakeWorker(), jobtrace.JobTable()
    jobtrace.instrument_worker(w, jobs)
    rows = [{"ids": [1, 2]}]
    rec = jobs.open(rows, arrival=1.0)
    fut = w.submit(rows, 12.0)
    assert fut.result() == "queued"
    assert w.submitted == [(rows, 12.0)]  # the original submit ran with the same arguments
    assert w._run(rows) == ("logits", "act", 1.25)
    assert rec["estimate_ms"] == 12.0 and rec["backlog_at_enter_ms"] == 3.5 and rec["device"] == "gpu"
    assert rec["arrival"] <= rec["queue_enter"] <= rec["device_start"] <= rec["device_end"]
    assert jobs.close(rec) is rec and jobs.get(rows) is None


def test_tracer_propagates_device_errors_and_still_stamps(jobtrace):
    w, jobs = FakeWorker(fail=True), jobtrace.JobTable()
    jobtrace.instrument_worker(w, jobs)
    rows = [{}]
    rec = jobs.open(rows)
    w.submit(rows, 1.0)
    with pytest.raises(RuntimeError, match="device error"):
        w._run(rows)
    assert rec["device_end"] >= rec["device_start"]


def test_product_path_opens_a_fresh_record_even_when_an_id_is_reused(jobtrace):
    w, jobs = FakeWorker(), jobtrace.JobTable()
    jobtrace.instrument_worker(w, jobs)
    rows = [{}]
    stale = jobs.open(rows, arrival=0.0)  # a finished request whose record is not closed yet
    jobs.expect_new()
    w.submit(rows, 2.0)  # Laya.submit path: same id, new request
    fresh = jobs.last_opened()
    assert fresh is not stale and fresh["estimate_ms"] == 2.0 and "estimate_ms" not in stale


def test_last_opened_is_per_thread(jobtrace):
    jobs = jobtrace.JobTable()
    mine = jobs.open([1])
    seen = {}
    t = threading.Thread(target=lambda: seen.setdefault("other", jobs.last_opened()))
    t.start()
    t.join()
    assert jobs.last_opened() is mine and seen["other"] is None


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
