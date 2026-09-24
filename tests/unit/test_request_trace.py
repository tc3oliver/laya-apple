"""Request lifecycle tracing (laya_apple/trace.py) through Laya.submit and DeviceWorker.

No model loading: both devices are thread-placed DeviceWorkers around a fake backend whose
forward can be held open, so the queue state the router sees is set up exactly. The real
worker-process path is covered by tests/integration/test_request_trace_workers.py.
"""

from __future__ import annotations

import threading
import warnings

import pytest

from laya_apple import executor, model, routing, scheduling
from laya_apple.backends.coreml_ane import ANEShapes
from laya_apple.executor import DeviceWorker
from laya_apple.registry import resolve, routing_table
from laya_apple.trace import QueueSnapshot, RequestTrace

MODEL = "laya-typed-decisions"


class FakeBackend:
    """forward returns the request's token count; `gate`, when set, holds it until released."""

    def __init__(self, kind):
        self.name, self.device = ("mlx", "gpu") if kind == "gpu" else ("coreml", "ane")
        self.gate: threading.Event | None = None
        self.entered = threading.Semaphore(0)

    def forward(self, rows):
        self.entered.release()
        if self.gate is not None:
            assert self.gate.wait(10)
        return [len(r["ids"]) for r in rows], None


@pytest.fixture
def fake_backends(monkeypatch):
    made = {}

    def load(kind, args):
        made[kind] = FakeBackend(kind)
        return made[kind]

    monkeypatch.setattr(executor, "load_backend", load)
    monkeypatch.setattr(executor, "warm", lambda kind, backend, pad_id: None)
    monkeypatch.setattr(executor, "backend_info", lambda kind, backend: {"name": backend.name})
    monkeypatch.setattr(model, "format_answers", lambda prep, logits, act, cal: {"n": logits})
    return made


class Prep:
    def __init__(self, length, questions=1):
        self.items = [{"ids": [0] * length, "markers": [1, 2], "qtype": 0} for _ in range(questions)]
        self.sequence_length, self.question_count, self.input_tokens = length, questions, length * questions


def make_laya(trace=None):
    """A workers-mode Laya over two thread-placed fake devices (no checkpoint)."""
    spec = resolve(MODEL)
    laya = model.Laya.__new__(model.Laya)
    laya.spec, laya.device, laya.dtype, laya.execution = spec, "auto", "float16", "workers"
    laya._closed, laya._ane_thread, laya._ane_dead_warned = False, None, False
    laya._trace, laya._trace_warned = trace, False
    laya._service = scheduling.ServiceModel(routing_table()["models"][spec.name]["service_ms"])
    laya._workers = {k: DeviceWorker(k, {"pad_id": 0}, placement="thread") for k in ("gpu", "ane")}
    laya.mlx = model._GPUView(spec, "float16")
    laya.ane = ANEShapes(spec, spec.ane_buckets, {b: "0" * 64 for b in spec.ane_buckets}, {})
    laya.ane_state = routing.AneState(None, spec.auto_ane_buckets)
    laya._tie_buckets = ()
    laya.calibration = None
    laya.prepare = lambda context, questions: questions  # the tests pass a Prep as `questions`
    return laya


# ----------------------------------------------------------------------------- RequestTrace


def test_durations_come_from_the_raw_timestamps():
    ms = 1_000_000
    t = RequestTrace(
        request_id=7,
        sequence_length=128,
        question_count=1,
        target="ane",
        routing_reason=routing.ANE_AUTO,
        service_estimate_ms=4.0,
        gpu=QueueSnapshot(12.5, 2, True),
        ane=None,
        submit_ns=0,
        prepared_ns=1 * ms,
        routed_ns=2 * ms,
        queue_enter_ns=3 * ms,
        dispatch_ns=5 * ms,
        service_start_ns=6 * ms,
        service_end_ns=10 * ms,
        received_ns=11 * ms,
        response_ns=13 * ms,
    )
    assert (t.prepare_ms, t.routing_ms, t.enqueue_ms, t.queue_ms) == (1, 1, 1, 2)
    assert (t.dispatch_ms, t.service_ms, t.return_ms, t.postprocess_ms, t.e2e_ms) == (1, 4, 1, 2, 13)
    assert (t.gpu_backlog_ms, t.gpu_queue_depth, t.gpu_running) == (12.5, 2, True)
    assert (t.ane_backlog_ms, t.ane_queue_depth, t.ane_running) == (0.0, 0, False)
    d = t.to_dict()
    assert d["gpu"] == {"backlog_ms": 12.5, "queued_jobs": 2, "running": True} and d["ane"] is None
    assert "e2e_ms" not in d  # durations are derived, never stored
    with pytest.raises(AttributeError):
        t.target = "gpu"  # immutable


# ----------------------------------------------------------------------------- DeviceWorker


def test_snapshot_reads_backlog_depth_and_running_together(fake_backends):
    w = DeviceWorker("gpu", {"pad_id": 0}, placement="thread")
    try:
        backend = fake_backends["gpu"]
        backend.gate = threading.Event()
        assert w.snapshot() == QueueSnapshot(0.0, 0, False)
        first = w.submit([{"ids": [0]}], 50.0, 1)
        assert backend.entered.acquire(timeout=5)  # job 1 is running and held
        w.submit([{"ids": [0]}], 20.0, 2)
        w.submit([{"ids": [0]}], 30.0, 3)
        snap = w.snapshot()
        assert snap.queued_jobs == 2 and snap.running
        assert 50.0 <= snap.backlog_ms <= 100.0  # 20 + 30 queued, plus what is left of 50
        assert w.backlog_ms() <= snap.backlog_ms  # the running estimate only drains
        backend.gate.set()
        logits, act, enq, dispatch, start, end, received = first.result(5)
        assert logits == [1] and enq <= dispatch <= start <= end <= received
    finally:
        w.close()
    assert w.snapshot().queued_jobs == 0 and not w.snapshot().running


def test_process_protocol_carries_the_request_id(monkeypatch):
    """Process placement: the job id on the wire is the request id, and the reply's
    forward timestamps come back unchanged (no extra message)."""
    sent = []

    class Conn:
        def send(self, msg):
            sent.append(msg)

        def recv(self):
            job_id, rows = sent[-1]
            return job_id, "ok", ("logits", "act", 100, 200)

    w = DeviceWorker.__new__(DeviceWorker)
    w.kind, w._backend, w._conn = "gpu", None, Conn()
    assert w._run(4242, ["row"]) == ("logits", "act", 100, 200)
    assert sent == [(4242, ["row"])]


# ----------------------------------------------------------------------------- Laya.submit


def test_one_trace_per_request_in_lifecycle_order(fake_backends):
    traces = []
    laya = make_laya(traces.append)
    try:
        results = [laya.submit(questions=Prep(n)).result(5) for n in (64, 128, 1024)]
    finally:
        laya.close()
    ids = [t.request_id for t in traces]
    assert ids == [r.runtime.request_id for r in results] and ids == list(range(ids[0], ids[0] + 3))
    assert [t.target for t in traces] == ["ane", "ane", "gpu"]
    for t, r in zip(traces, results):
        stamps = [
            t.submit_ns,
            t.prepared_ns,
            t.routed_ns,
            t.queue_enter_ns,
            t.dispatch_ns,
            t.service_start_ns,
            t.service_end_ns,
            t.received_ns,
            t.response_ns,
        ]
        assert stamps == sorted(stamps) and stamps[0] > 0
        assert t.routing_reason == r.runtime.routing_reason and t.target == r.runtime.device
        assert t.queue_ms == pytest.approx(r.runtime.queue_wait_ms, abs=1e-9)
        assert t.service_ms == pytest.approx(r.runtime.device_ms, abs=1e-9)
        assert t.e2e_ms == pytest.approx(r.runtime.latency_ms, abs=1e-9)
        assert (t.gpu_backlog_ms, t.ane_backlog_ms) == (r.runtime.gpu_backlog_ms, r.runtime.ane_backlog_ms)


def test_router_and_trace_share_one_snapshot(fake_backends, monkeypatch):
    """GPU running with jobs queued, ANE idle: the backlogs the router decided on are the
    ones recorded, from the same snapshot objects the workers returned."""
    seen, snaps, traces = [], [], []
    real_decide, real_snapshot = scheduling.decide_queued, DeviceWorker.snapshot

    def spy_decide(*args, **kwargs):
        seen.append((kwargs["gpu_backlog_ms"], kwargs["ane_backlog_ms"]))
        return real_decide(*args, **kwargs)

    def spy_snapshot(self):
        snap = real_snapshot(self)
        snaps.append((self.kind, snap))
        return snap

    monkeypatch.setattr(scheduling, "decide_queued", spy_decide)
    monkeypatch.setattr(DeviceWorker, "snapshot", spy_snapshot)
    laya = make_laya(traces.append)
    gpu = fake_backends["gpu"]
    gpu.gate = threading.Event()
    try:
        futs = [laya.submit(questions=Prep(1024))]
        assert gpu.entered.acquire(timeout=5)
        futs += [laya.submit(questions=Prep(1024)) for _ in range(2)]
        snaps.clear()
        futs.append(laya.submit(questions=Prep(128)))  # the request under test
        (gk, g), (ak, a) = snaps
        assert (gk, ak) == ("gpu", "ane")
        assert g.running and g.queued_jobs == 2 and g.backlog_ms > 0
        assert a == QueueSnapshot(0.0, 0, False)
        futs[-1].result(5)  # the ANE is idle: it completes while the GPU is still held
        t = traces[-1]
        assert t.gpu is g and t.ane is a  # the very objects the router read
        assert seen[-1] == (t.gpu_backlog_ms, t.ane_backlog_ms) == (g.backlog_ms, 0.0)
        assert t.target == "ane" and t.routing_reason == routing.ANE_AUTO
        gpu.gate.set()
        for f in futs:
            f.result(5)
    finally:
        gpu.gate.set()
        laya.close()
    assert len(traces) == 4 and len({t.request_id for t in traces}) == 4


def test_tracing_changes_no_decision_and_no_answer(fake_backends):
    """The same deterministic sequence with trace=None and with a callback."""
    lengths = [32, 64, 96, 128, 200, 256, 512, 1024, 128, 64]

    def run(trace):
        laya = make_laya(trace)
        try:
            out = []
            for n in lengths:
                r = laya.submit(questions=Prep(n)).result(5)
                out.append((r.runtime.device, r.runtime.routing_reason, r.runtime.buckets, r.answers))
            return out
        finally:
            laya.close()

    assert run(None) == run(lambda t: None)


def test_callback_errors_never_fail_requests(fake_backends):
    calls = []

    def bad(trace):
        calls.append(trace.request_id)
        raise RuntimeError("callback bug")

    laya = make_laya(bad)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = [laya.submit(questions=Prep(64)).result(5) for _ in range(3)]
    finally:
        laya.close()
    assert [r.answers for r in results] == [{"n": [64]}] * 3
    assert calls == [r.runtime.request_id for r in results]
    assert sum("trace callback raised RuntimeError" in str(w.message) for w in caught) == 1


def test_failed_requests_emit_no_trace(fake_backends):
    traces = []
    laya = make_laya(traces.append)
    fake_backends["gpu"].forward = lambda rows: (_ for _ in ()).throw(RuntimeError("device error"))
    try:
        with pytest.raises(RuntimeError, match="device error"):
            laya.submit(questions=Prep(1024)).result(5)
    finally:
        laya.close()
    assert traces == []


def test_request_ids_are_unique_across_instances(fake_backends):
    a, b = make_laya(), make_laya()
    try:
        ids = [laya.submit(questions=Prep(64)).result(5).runtime.request_id for laya in (a, b, a, b)]
    finally:
        a.close()
        b.close()
    assert len(set(ids)) == 4 and ids == sorted(ids)


def test_trace_requires_workers():
    with pytest.raises(ValueError, match="workers"):
        model.Laya(resolve(MODEL), None, "auto", dtype="float16", batch_size=1, trace=lambda t: None)
