"""Request lifecycle tracing on real worker processes (execution="workers")."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from laya_apple import Laya, scheduling
from laya_apple.workload import make_request

pytestmark = [pytest.mark.integration, pytest.mark.ane]
MODEL = "laya-typed-decisions"

STAMPS = (
    "submit_ns",
    "prepared_ns",
    "routed_ns",
    "queue_enter_ns",
    "dispatch_ns",
    "service_start_ns",
    "service_end_ns",
    "received_ns",
    "response_ns",
)


def load(placement, trace):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    laya = Laya.from_pretrained(
        MODEL, device="auto", execution="workers", ane_placement=placement, local_files_only=True, trace=trace
    )
    if not laya.ane_state.buckets:
        laya.close()
        pytest.skip("no validated ANE artifacts for the workers test model")
    return laya


@pytest.fixture(scope="module", params=["thread", "process"])
def pair(request):
    """Two instances with the same ANE placement: trace=None and a recording callback."""
    traces = []
    plain, traced = load(request.param, None), load(request.param, traces.append)
    yield plain, traced, traces
    plain.close()
    traced.close()


@pytest.fixture(scope="module")
def requests(pair):
    tok, cfg = pair[0].tokenizer, pair[0].config
    return {
        "short": make_request(tok, cfg, 128, 1, seed=1),
        "tie": make_request(tok, cfg, 100, 1, seed=4),
        "long": make_request(tok, cfg, 1024, 1, seed=2),
        "multi": make_request(tok, cfg, 96, 3, seed=3),
    }


def test_each_request_traces_once_in_lifecycle_order(pair, requests):
    _, laya, traces = pair
    traces.clear()
    results = [laya.predict(context=s, questions=q) for s, q in requests.values()]
    assert [t.request_id for t in traces] == [r.runtime.request_id for r in results]
    for t, r in zip(traces, results):
        stamps = [getattr(t, s) for s in STAMPS]
        assert stamps == sorted(stamps), dict(zip(STAMPS, stamps))
        assert t.target == r.runtime.device and t.routing_reason == r.runtime.routing_reason
        assert (t.sequence_length, t.question_count) == (r.runtime.sequence_length, r.runtime.question_count)
        # the trace and RuntimeInfo measure the same intervals from the same timestamps
        assert t.queue_ms == pytest.approx(r.runtime.queue_wait_ms, abs=1e-6)
        assert t.service_ms == pytest.approx(r.runtime.device_ms, abs=1e-6)
        assert t.e2e_ms == pytest.approx(r.runtime.latency_ms, abs=1e-6)
        assert t.service_estimate_ms > 0
    gpu = next(t for t in traces if t.target == "gpu")
    # the GPU worker is a separate process: its forward starts after the parent dispatched
    # the job and the request crossed the IPC, and the reply arrives after it ends
    assert gpu.dispatch_ms > 0 and gpu.return_ms > 0


def test_tracing_changes_no_decision_and_no_answer(pair, requests):
    """Sequential (idle-queue) requests: routing is deterministic, answers bit-identical."""
    plain, traced, _ = pair
    for name, (s, q) in requests.items():
        a, b = plain.predict(context=s, questions=q), traced.predict(context=s, questions=q)
        assert (a.runtime.device, a.runtime.routing_reason, a.runtime.buckets) == (
            b.runtime.device,
            b.runtime.routing_reason,
            b.runtime.buckets,
        ), name
        assert a.answers == b.answers, name


def test_trace_records_the_snapshot_the_router_used(pair, requests, monkeypatch):
    """A busy GPU (several long jobs) and an idle ANE when a short request routes."""
    _, laya, traces = pair
    seen = []
    real = scheduling.decide_queued

    def spy(*args, **kwargs):
        seen.append((kwargs["gpu_backlog_ms"], kwargs["ane_backlog_ms"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(scheduling, "decide_queued", spy)
    traces.clear()
    long_s, long_q = requests["long"]
    futs = [laya.submit(context=long_s, questions=long_q) for _ in range(4)]
    s, q = requests["short"]
    short = laya.submit(context=s, questions=q)
    rid = short.result(60).runtime.request_id
    for f in futs:
        f.result(60)
    t = next(t for t in traces if t.request_id == rid)
    assert seen[-1] == (t.gpu_backlog_ms, t.ane_backlog_ms)
    # the dispatcher may not have taken the first long job yet: count it either way
    assert t.gpu.queued_jobs + t.gpu.running >= 2 and t.gpu_backlog_ms > 0
    assert t.ane_queue_depth == 0
    assert len(traces) == 5 and len({x.request_id for x in traces}) == 5


def test_concurrent_load_traces_every_request_once(pair, requests):
    _, laya, traces = pair
    traces.clear()
    names = [n for n in requests for _ in range(10)]
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda n: laya.predict(context=requests[n][0], questions=requests[n][1]), names))
    assert sorted(t.request_id for t in traces) == sorted(r.runtime.request_id for r in results)
    for t in traces:
        stamps = [getattr(t, s) for s in STAMPS]
        assert stamps == sorted(stamps)
