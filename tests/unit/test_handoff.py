"""The staged Core ML handoff (laya_apple/handoff.py) and its wiring, without models or the ANE.

- the state machine: transitions, the guard boundary (forward 64 sync, 65 async), episodes,
  re-arm, disable, concurrency, corruption detection, and decision-for-decision equality with
  the research copy (research/coreml-staged-handoff/scripts/handoff.py);
- GpuActivity fed through DeviceWorker.activity: success, error, cancellation, dead worker,
  and jobs failed by close();
- ANEBackend.forward's path selection with fake sync and async models: one decision per
  forward, an async failure raised, the handoff disabled and warned once;
- the async model's load failure, and the Completion's exactly-one-callback rule;
- Laya's eligibility rules and wiring (_start_workers with fake workers), background start-up,
  info(), and close() mid-episode;
- the research harness contract (prod_run.py): the class-level decide() wrapper and
  backend.handoff.snapshot().
"""

from __future__ import annotations

import importlib.util
import json
import random
import threading
import time
import warnings
from concurrent.futures import CancelledError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from laya_apple import executor, handoff, model, routing
from laya_apple.backends import coreml_ane, coreml_async
from laya_apple.backends.coreml_ane import ANEBackend, ANEShapes
from laya_apple.errors import BackendUnavailableError, LayaAppleError
from laya_apple.executor import DeviceWorker
from laya_apple.registry import ANE_MAX_OPTIONS, resolve

S = 1_000_000_000


class Clock:
    def __init__(self):
        self.t = 10 * S

    def __call__(self):
        return self.t

    def step(self, s: float):
        self.t += int(s * S)


def machine(guard=64, gap_s=1.0):
    clock = Clock()
    gpu = handoff.GpuActivity(clock)
    return handoff.StagedHandoff(gpu, guard, gap_s=gap_s, clock=clock), gpu, clock


def paths(m, clock, n, dt=0.01):
    out = []
    for _ in range(n):
        out.append(m.decide()[0])
        clock.step(dt)
    return out


# ----------------------------------------------------------------------------- state machine


def test_defaults_are_the_confirmed_h64_cell():
    m = handoff.StagedHandoff(handoff.GpuActivity())
    assert (m.guard, m.gap_s, m.state) == (64, 1.0, handoff.ARMED) and handoff.GUARD == 64


def test_armed_runs_sync_without_gpu():
    m, _, clock = machine()
    assert paths(m, clock, 100) == ["sync"] * 100
    assert m.state == handoff.ARMED and m.episodes == 0


@pytest.mark.parametrize("guard", [32, 64])
def test_guard_boundary(guard):
    m, gpu, clock = machine(guard)
    gpu.started()
    rows = []
    for _ in range(guard + 3):
        rows.append(m.decide())
        clock.step(0.01)
    assert rows[guard - 2] == ("sync", handoff.SYNC_GUARD, guard - 1)
    assert rows[guard - 1] == ("sync", handoff.ASYNC_STEADY, guard)  # forward N is still sync
    assert rows[guard] == ("async", handoff.ASYNC_STEADY, guard)  # forward N + 1 is the first async
    assert [r[0] for r in rows] == ["sync"] * guard + ["async"] * 3
    assert m.episodes == 1


def test_gpu_active_within_gap_after_last_end():
    m, gpu, clock = machine(2)
    gpu.started()
    gpu.ended()
    clock.step(0.5)
    assert paths(m, clock, 3) == ["sync", "sync", "async"]


def test_repeated_episodes_rearm():
    m, gpu, clock = machine(2)
    for episode in range(1, 4):
        gpu.started()
        assert paths(m, clock, 3) == ["sync", "sync", "async"]
        gpu.ended()
        clock.step(1.5)  # hetero -> solo_short
        assert paths(m, clock, 2) == ["sync"] * 2 and m.state == handoff.ARMED
        assert m.episodes == episode


def test_short_pause_rearms_while_gpu_stays_busy():
    m, gpu, clock = machine(2)
    gpu.started()
    assert paths(m, clock, 3) == ["sync", "sync", "async"]
    clock.step(2.0)  # no ANE forward for longer than the gap, GPU busy throughout
    assert paths(m, clock, 3) == ["sync", "sync", "async"]
    assert m.episodes == 2


def test_hetero_ends_before_guard():
    m, gpu, clock = machine()
    gpu.started()
    assert paths(m, clock, 63) == ["sync"] * 63 and m.count == 63
    gpu.ended()
    clock.step(1.5)
    assert paths(m, clock, 5) == ["sync"] * 5 and m.state == handoff.ARMED and m.count == 0
    gpu.started()
    p = paths(m, clock, 65)
    assert p[:64] == ["sync"] * 64 and p[64] == "async"  # the count restarted
    assert m.episodes == 2 and m.forwards == {"sync": 63 + 5 + 64, "async": 1}


def test_gpu_inflight_never_negative():
    clock = Clock()
    gpu = handoff.GpuActivity(clock)
    gpu.started()
    gpu.started()
    gpu.ended()
    clock.step(5)
    assert gpu.active(clock(), S) and gpu.inflight == 1
    gpu.ended()
    gpu.ended()
    clock.step(5)
    assert not gpu.active(clock(), S) and gpu.inflight == 0


def test_concurrent_decisions_count_exactly():
    guard = 500
    gpu = handoff.GpuActivity()
    gpu.started()
    m = handoff.StagedHandoff(gpu, guard)
    out: list = []
    lock = threading.Lock()

    def worker():
        mine = [m.decide()[0] for _ in range(200)]
        with lock:
            out.extend(mine)

    ts = [threading.Thread(target=worker) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert out.count("sync") == guard and out.count("async") == 1600 - guard
    assert m.episodes == 1 and m.forwards == {"sync": guard, "async": 1600 - guard}
    assert m.snapshot()["consistent"] is True


def test_disable_pins_to_sync_and_reports_once():
    m, gpu, clock = machine(1)
    gpu.started()
    assert paths(m, clock, 2) == ["sync", "async"]
    assert m.disable("boom") is True
    assert m.disable("again") is False
    assert paths(m, clock, 5) == ["sync"] * 5
    snap = m.snapshot()
    assert snap["enabled"] is False and snap["disabled"] is True
    assert snap["state"] == handoff.DISABLED and snap["disabled_reason"] == "boom"
    assert snap["forwards"] == {"sync": 6, "async": 1}  # forwards after disable still count, as sync


def test_snapshot_is_a_json_copy():
    m, gpu, clock = machine(2)
    snap = m.snapshot()
    snap["forwards"]["sync"] = 99
    assert m.snapshot()["forwards"]["sync"] == 0
    assert set(snap) == {
        "enabled",
        "disabled",
        "disabled_reason",
        "state",
        "guard",
        "gap_s",
        "count",
        "episodes",
        "forwards",
        "consistent",
    }
    assert json.loads(json.dumps(snap)) == snap


def test_guard_must_be_positive():
    with pytest.raises(ValueError):
        handoff.StagedHandoff(handoff.GpuActivity(), 0)


# ----------------------------------------------------------------------------- DeviceWorker hooks


class FakeBackend:
    def __init__(self, kind):
        self.name, self.device = ("mlx", "gpu") if kind == "gpu" else ("coreml", "ane")
        self.gate: threading.Event | None = None
        self.entered = threading.Semaphore(0)
        self.error: BaseException | None = None
        self.handoff = None
        self.paths: list = []

    def forward(self, rows):
        self.entered.release()
        if self.handoff is not None:
            self.paths.append(self.handoff.decide()[0])
        if self.gate is not None:
            assert self.gate.wait(10)
        if self.error is not None:
            raise self.error
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
    return made


ROW = [{"ids": [0] * 8, "markers": [1, 2], "qtype": 0}]


def gpu_worker(fake_backends):
    w = DeviceWorker("gpu", {"pad_id": 0}, placement="thread")
    w.activity = handoff.GpuActivity()
    return w, fake_backends["gpu"]


def test_activity_on_success(fake_backends):
    w, b = gpu_worker(fake_backends)
    b.gate = threading.Event()
    fut = w.submit(ROW, 1.0, 1)
    assert b.entered.acquire(timeout=5)
    assert w.activity.inflight == 1
    b.gate.set()
    fut.result(5)
    assert w.activity.inflight == 0 and w.activity._last_end_ns is not None
    w.close()


def test_activity_on_error(fake_backends):
    w, b = gpu_worker(fake_backends)
    b.error = RuntimeError("mlx failed")
    with pytest.raises(RuntimeError):
        w.submit(ROW, 1.0, 1).result(5)
    assert w.activity.inflight == 0
    w.close()


def test_activity_on_cancellation(fake_backends):
    w, b = gpu_worker(fake_backends)
    b.gate = threading.Event()
    first = w.submit(ROW, 1.0, 1)
    assert b.entered.acquire(timeout=5)
    queued = w.submit(ROW, 1.0, 2)
    assert w.activity.inflight == 2
    assert queued.cancel()
    assert w.activity.inflight == 1
    b.gate.set()
    first.result(5)
    with pytest.raises(CancelledError):
        queued.result(5)
    w.close()
    assert w.activity.inflight == 0


def test_activity_on_dead_worker(fake_backends):
    w, _ = gpu_worker(fake_backends)
    w._dead = BackendUnavailableError("gone")
    with pytest.raises(BackendUnavailableError):
        w.submit(ROW, 1.0, 1).result(5)
    assert w.activity.inflight == 0 and w.activity._last_end_ns is None  # never counted as activity
    assert not w.activity.active(time.monotonic_ns(), 10**9)
    w.close()


def test_dead_gpu_worker_cannot_start_a_handoff_episode(fake_backends):
    w, _ = gpu_worker(fake_backends)
    w._dead = BackendUnavailableError("gone")
    h = handoff.StagedHandoff(w.activity, guard=2)
    for i in range(5):
        with pytest.raises(BackendUnavailableError):
            w.submit(ROW, 1.0, i).result(5)
        assert h.decide()[:2] == ("sync", handoff.ARMED)
    assert h.episodes == 0
    w.close()


def test_activity_on_jobs_failed_by_close(fake_backends):
    w, b = gpu_worker(fake_backends)
    b.gate = threading.Event()
    running = w.submit(ROW, 1.0, 1)
    assert b.entered.acquire(timeout=5)
    queued = [w.submit(ROW, 1.0, i) for i in range(2, 5)]
    assert w.activity.inflight == 4
    w.close(timeout=0.2)  # the running job holds the dispatcher: the queued ones are failed
    for f in queued:
        with pytest.raises(BackendUnavailableError):
            f.result(5)
    assert w.activity.inflight == 1
    b.gate.set()
    running.result(5)
    assert w.activity.inflight == 0


def test_no_activity_by_default(fake_backends):
    w = DeviceWorker("gpu", {"pad_id": 0}, placement="thread")
    assert w.activity is None
    w.submit(ROW, 1.0, 1).result(5)
    w.close()


def test_load_backend_passes_async_models(monkeypatch):
    seen = {}

    class Spy:
        def __init__(self, *a, **kw):
            seen.update(kw)

    monkeypatch.setattr(coreml_ane, "ANEBackend", Spy)
    args = {"model": "laya", "checkpoint": "/x", "pad_id": 0, "local_attention": 128, "buckets": [64], "strict": False}
    executor.load_backend("ane", args)
    assert seen["async_models"] is False
    executor.load_backend("ane", dict(args, async_models=True))
    assert seen["async_models"] is True


# ----------------------------------------------------------------------------- ANEBackend.forward


class FakeHost:
    def __init__(self):
        self.embedding = np.zeros((4, 8), np.float16)
        self.type_embedding = np.zeros((2, 8), np.float16)

    def window(self, L):
        return np.ones((L, L), bool)

    def tail(self, outputs, rows):
        lg = np.full((1, ANE_MAX_OPTIONS), outputs["tag"], np.float32)
        return lg, np.zeros((1, 3), np.float32)


class FakeModel:
    def __init__(self, tag, error=None):
        self.tag, self.error, self.calls = tag, error, 0

    def predict(self, feats):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return {"tag": self.tag}


def fake_ane(guard=2, async_error=None):
    b = ANEBackend.__new__(ANEBackend)
    ANEShapes.__init__(b, resolve("laya"), [64, 128], {64: "a" * 64, 128: "b" * 64}, {})
    b.pad_id, b.host = 0, FakeHost()
    b.models = {64: FakeModel(0.0), 128: FakeModel(0.0)}
    b.async_models = {64: FakeModel(1.0, async_error), 128: FakeModel(1.0, async_error)}
    clock = Clock()
    gpu = handoff.GpuActivity(clock)
    b.handoff = handoff.StagedHandoff(gpu, guard, clock=clock)
    return b, gpu, clock


def rows(n, length=40):
    return [{"ids": [0] * length, "markers": [1, 2], "qtype": 0} for _ in range(n)]


def test_forward_without_handoff_is_sync():
    b, gpu, _ = fake_ane()
    b.handoff = None
    gpu.started()
    logits, _ = b.forward(rows(2))
    assert (logits == 0.0).all() and b.async_models[64].calls == 0


def test_forward_takes_one_decision_per_multi_item_forward():
    b, gpu, clock = fake_ane(guard=2)
    gpu.started()
    for expected in (0.0, 0.0, 1.0):
        items = rows(2) + rows(1, 100)  # three predicts, two buckets, one decision
        logits, act = b.forward(items)
        assert (logits == expected).all() and act.shape == (3, 3)
        clock.step(0.01)
    assert b.handoff.forwards == {"sync": 2, "async": 1}
    assert b.models[64].calls == 4 and b.models[128].calls == 2
    assert b.async_models[64].calls == 2 and b.async_models[128].calls == 1


def test_forward_armed_without_gpu_is_sync():
    b, _, clock = fake_ane(guard=1)
    for _ in range(5):
        assert (b.forward(rows(1))[0] == 0.0).all()
        clock.step(0.01)
    assert b.handoff.state == handoff.ARMED and b.async_models[64].calls == 0


def test_async_predict_failure_raises_disables_and_warns_once():
    err = LayaAppleError("Core ML async predict failed: boom")
    b, gpu, clock = fake_ane(guard=1, async_error=err)
    gpu.started()
    b.forward(rows(1))  # the guard forward: sync
    clock.step(0.01)
    with pytest.warns(RuntimeWarning, match="laya-apple: the asynchronous Core ML predict failed") as rec:
        with pytest.raises(LayaAppleError, match="boom"):
            b.forward(rows(1))
    assert len(rec) == 1
    assert b.models[64].calls == 1  # the failed request was not re-run on the sync path
    snap = b.handoff.snapshot()
    assert snap["enabled"] is False and "boom" in snap["disabled_reason"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(3):
            clock.step(0.01)
            assert (b.forward(rows(1))[0] == 0.0).all()  # pinned to sync, no second warning
    assert b.async_models[64].calls == 1
    assert b.handoff.forwards == {"sync": 4, "async": 1} and b.handoff.snapshot()["disabled"] is True


def test_forward_with_no_async_models_never_decides():
    b, gpu, _ = fake_ane()
    b.async_models = {}
    gpu.started()
    b.forward(rows(1))
    assert b.handoff.forwards == {"sync": 0, "async": 0}


def test_async_load_failure_is_recorded(monkeypatch):
    b, _, _ = fake_ane()
    b.async_models = {}
    b.probes = {64: {"cpu_ms": 10.0}, 128: {"cpu_ms": 10.0}}

    def fail(path):
        raise BackendUnavailableError("pyobjc-framework-CoreML is not importable")

    monkeypatch.setattr(coreml_async, "AsyncPrebindModel", fail)
    monkeypatch.setattr(coreml_ane, "artifact_dir", lambda spec, bucket: Path("/nonexistent"))
    b._load_async(b.spec)
    assert b.async_models == {} and isinstance(b.async_error, BackendUnavailableError)


def load_async_with(monkeypatch, sync_out, async_out):
    """_load_async on bucket 64 with two check inputs ("pad", "real"); *_out(feats) -> outputs."""
    b, _, _ = fake_ane()
    b.async_models = {}
    b.probes = {64: {"cpu_ms": 10.0}}
    b.models = {64: SimpleNamespace(predict=sync_out)}
    b._check_features = lambda bucket: [{"which": "pad"}, {"which": "real"}]
    fake = SimpleNamespace(predict=async_out)
    monkeypatch.setattr(coreml_async, "AsyncPrebindModel", lambda path: fake)
    monkeypatch.setattr(coreml_ane, "artifact_dir", lambda spec, bucket: Path("/nonexistent"))
    monkeypatch.setattr(coreml_ane, "_fastest_ms", lambda m, f, runs: 1.0)
    b._load_async(b.spec)
    return b, fake


def zeros(f):
    return {"x": np.zeros(4, np.float32)}


@pytest.mark.parametrize(
    "async_out",
    [
        lambda f: {"x": np.ones(4, np.float32)},  # different values
        lambda f: {"x": np.float32([0, 0, 0, np.float32(1e-7)])},  # a tiny difference is still a difference
        lambda f: {"x": np.zeros(4, np.float64)},  # dtype
        lambda f: {"x": np.zeros((1, 4), np.float32)},  # shape
        lambda f: {"y": np.zeros(4, np.float32)},  # name
        lambda f: {"x": np.zeros(4, np.float32) + (f["which"] == "real")},  # only the non-pad row differs
    ],
)
def test_async_load_requires_identical_outputs(monkeypatch, async_out):
    b, _ = load_async_with(monkeypatch, zeros, async_out)
    assert b.async_models == {} and "not identical" in str(b.async_error)


def test_check_features_include_a_non_pad_row(monkeypatch):
    b, _, _ = fake_ane()
    seen = []
    real = coreml_ane.ane_features

    def spy(rows, *a, **kw):
        seen.append(rows[0]["ids"])
        return real(rows, *a, **kw)

    monkeypatch.setattr(coreml_ane, "ane_features", spy)
    feats = b._check_features(64)
    assert len(feats) == 2 and len(seen) == 2
    ids, pad = seen  # the non-pad row is built first, then the pad probe
    assert pad == [0] * 64 and len(ids) == 64
    assert all(0 <= i < len(b.host.embedding) for i in ids) and any(i != b.pad_id for i in ids)


def test_async_load_accepts_identical_outputs(monkeypatch):
    b, fake = load_async_with(monkeypatch, zeros, zeros)
    assert b.async_models == {64: fake} and b.async_error is None
    assert b.async_probes[64]["ratio"] == 0.1


def test_async_load_rejects_a_slow_model(monkeypatch):
    b, _ = load_async_with(monkeypatch, zeros, zeros)
    monkeypatch.setattr(coreml_ane, "_fastest_ms", lambda m, f, runs: 9.0)  # ratio 0.9 > 0.8
    b.async_models = {}
    b._load_async(b.spec)
    assert b.async_models == {} and "not running on the Neural Engine" in str(b.async_error)


# ----------------------------------------------------------------------------- Completion


def test_completion_one_callback_per_submit():
    c = coreml_async.Completion()
    c.submit(lambda handler: handler("out", None), timeout=1)
    assert (c.submits, c.callbacks, c.anomalies, c.state) == (1, 1, 0, "idle")
    c.handler("dup", None)  # a duplicate: counted, changes nothing
    assert (c.callbacks, c.anomalies, c.state) == (2, 1, "idle")


def test_completion_error_raises():
    c = coreml_async.Completion()
    with pytest.raises(LayaAppleError, match="async predict failed"):
        c.submit(lambda handler: handler(None, "bad input"), timeout=1)
    c.submit(lambda handler: handler("out", None), timeout=1)  # a Core ML error is not final


def test_completion_timeout_refuses_later_submits():
    c = coreml_async.Completion()
    held = []
    with pytest.raises(LayaAppleError, match="did not complete"):
        c.submit(held.append, timeout=0.05)
    held[0]("late", None)  # the late callback is an anomaly
    assert c.anomalies == 1 and c.state == "timed_out"
    with pytest.raises(LayaAppleError, match="timed out earlier"):
        c.submit(lambda handler: handler("out", None), timeout=1)


def test_callback_racing_a_timeout_cannot_wake_the_next_submit():
    c = coreml_async.Completion()
    c.arm()
    c.handler("out", None)  # completes just as the waiter's timeout expires
    c.event.wait = lambda timeout: False  # the waiter saw the timeout first
    c.wait(0.01)  # the state is idle: the predict completed, no timeout is raised
    del c.event.wait
    c.arm()
    assert not c.event.is_set()  # nothing left over wakes the next submit
    with pytest.raises(LayaAppleError, match="did not complete"):
        c.wait(0.05)


def test_completion_send_failure_resets():
    c = coreml_async.Completion()

    def send(handler):
        raise RuntimeError("send failed")

    with pytest.raises(RuntimeError):
        c.submit(send, timeout=1)
    c.submit(lambda handler: handler("out", None), timeout=1)


def test_completion_refuses_overlapping_submit():
    c = coreml_async.Completion()
    c.arm()
    with pytest.raises(LayaAppleError, match="before the previous one completed"):
        c.arm()


def test_unavailable_without_pyobjc(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "CoreML", None)
    assert "pyobjc-framework-CoreML" in coreml_async.unavailable_reason()
    with pytest.raises(BackendUnavailableError):
        coreml_async.AsyncPrebindModel("/nonexistent")


# ----------------------------------------------------------------------------- Laya wiring


class FakeWorker:
    """Stands in for DeviceWorker in Laya._start_workers."""

    made: list = []
    async_error: BaseException | None = None

    def __init__(self, kind, args, *, placement="process", wait=True):
        self.kind, self.args, self.placement = kind, args, placement
        self.activity = None
        self.closed = False
        self.info = None
        self._backend = None
        if kind == "ane":
            self._backend = SimpleNamespace(
                async_models={64: object()} if args.get("async_models") and not self.async_error else {},
                async_error=self.async_error if args.get("async_models") else None,
                handoff=None,
            )
        FakeWorker.made.append(self)

    def wait_ready(self):
        if self.kind == "ane":
            self.info = {
                "offered": [64, 96, 128],
                "artifact_sha256": {64: "a" * 64, 96: "b" * 64, 128: "c" * 64},
                "load_errors": {},
                "probes": {},
            }
        else:
            self.info = {"name": "mlx"}

    def close(self, timeout=10.0):
        self.closed = True


@pytest.fixture
def fake_workers(monkeypatch):
    FakeWorker.made = []
    FakeWorker.async_error = None
    monkeypatch.setattr(executor, "DeviceWorker", FakeWorker)
    monkeypatch.setattr(model, "_coremltools_available", lambda: True)
    monkeypatch.setattr(coreml_async, "unavailable_reason", lambda: None)
    return FakeWorker


def make_laya(name="laya", *, device="auto", execution="workers", placement=None, ane_handoff=False, startup="wait"):
    laya = model.Laya.__new__(model.Laya)
    laya.spec = resolve(name)
    laya.checkpoint = Path("/nonexistent")
    laya.device, laya.execution, laya.dtype = device, execution, "float16"
    laya.ane_placement = placement or model.ane_placement_for(name)
    laya.ane_handoff, laya._handoff = ane_handoff, None
    laya.ane_startup, laya._ane_thread = startup, None
    laya.tokenizer = SimpleNamespace(pad_token_id=0)
    laya._encoder_cfg = {"local_attention": 128}
    laya.routing_profile = "shipped"
    laya.ane_state = routing.AneState(routing.RUNTIME_UNAVAILABLE)
    laya.mlx = laya.ane = None
    laya._workers, laya._tie_buckets = {}, ()
    laya._closed = False
    return laya


def workers_of(fake):
    return {w.kind: w for w in fake.made}


def test_opted_in_laya_wires_one_gpu_activity(fake_workers):
    laya = make_laya("laya", ane_handoff=True)
    laya._start_workers(16)
    w = workers_of(fake_workers)
    assert w["ane"].args["async_models"] is True
    assert isinstance(w["gpu"].activity, handoff.GpuActivity)
    assert w["ane"]._backend.handoff is laya._handoff and laya._handoff._gpu is w["gpu"].activity
    assert laya._handoff.enabled and laya._handoff.guard == 64
    info = laya.info()
    assert info["ane_handoff"]["enabled"] is True and info["ane_handoff"]["state"] == handoff.ARMED
    json.dumps(info)  # serve and the research harness serialise info()
    json_ok = json.dumps(w["ane"].args)  # process workers need JSON-serialisable args
    assert "async_models" in json_ok


def test_multilingual_process_placement_never_uses_handoff(fake_workers):
    laya = make_laya("laya-multilingual")
    assert laya.ane_placement == "process"
    laya._start_workers(16)
    w = workers_of(fake_workers)
    assert "async_models" not in w["ane"].args and w["gpu"].activity is None
    assert laya._handoff is None and "ane_handoff" not in laya.info()


@pytest.mark.parametrize("kw", [dict(), dict(device="ane"), dict(device="gpu"), dict(placement="process")])
def test_ineligible_or_off_loads_no_async_models(fake_workers, kw):
    laya = make_laya("laya", **kw)
    laya._start_workers(16)
    for w in fake_workers.made:
        assert "async_models" not in w.args and w.activity is None
    assert laya._handoff is None


def test_default_is_off_and_touches_nothing(fake_workers, monkeypatch):
    def never():
        raise AssertionError("the default must not import pyobjc")

    monkeypatch.setattr(coreml_async, "unavailable_reason", never)
    laya = make_laya("laya")
    laya._start_workers(16)
    w = workers_of(fake_workers)
    assert laya._handoff is None and "async_models" not in w["ane"].args and w["gpu"].activity is None
    assert w["ane"]._backend.handoff is None
    assert "ane_handoff" not in laya.info()


def test_default_signature_is_off():
    import inspect

    for fn in (model.Laya.__init__, model.Laya.from_pretrained):
        assert inspect.signature(fn).parameters["ane_handoff"].default is False


def test_forced_handoff_without_pyobjc_raises(fake_workers, monkeypatch):
    monkeypatch.setattr(coreml_async, "unavailable_reason", lambda: "pyobjc-framework-CoreML is not importable")
    with pytest.raises(BackendUnavailableError, match="ane_handoff=True"):
        make_laya("laya", ane_handoff=True)._start_workers(16)


def test_forced_handoff_without_ane_raises(fake_workers, monkeypatch):
    monkeypatch.setattr(model, "_coremltools_available", lambda: False)
    with pytest.raises(BackendUnavailableError, match="ANE is not started"):
        make_laya("laya", ane_handoff=True)._start_workers(16)


@pytest.mark.parametrize(
    "kw",
    [
        dict(execution="inline"),
        dict(device="ane"),
        dict(device="gpu"),
        dict(model_id="laya-multilingual"),
        dict(model_id="laya-multilingual", ane_placement="thread"),  # measured placement, not an override
        dict(ane_placement="process"),
    ],
)
def test_forced_handoff_ineligible_arguments_raise(kw):
    kw = dict(kw)
    name, device = kw.pop("model_id", "laya"), kw.pop("device", "auto")
    kw.setdefault("execution", "workers")
    with pytest.raises(ValueError, match="ane_handoff=True"):
        model.Laya(resolve(name), Path("/nonexistent"), device, dtype="float16", batch_size=1, ane_handoff=True, **kw)


@pytest.mark.parametrize("value", [None, "yes", 1])
def test_invalid_handoff_value(value):
    with pytest.raises(ValueError, match="ane_handoff must be"):
        model.Laya(resolve("laya"), Path("/nonexistent"), "auto", dtype="float16", batch_size=1, ane_handoff=value)


def test_inline_default_has_no_handoff():
    laya = make_laya("laya", execution="inline")
    assert laya._handoff_activity(True) is None and laya._handoff is None


def test_forced_handoff_async_load_failure_raises(fake_workers):
    fake_workers.async_error = BackendUnavailableError("Core ML could not load")
    laya = make_laya("laya", ane_handoff=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # raised, not warned
        with pytest.raises(BackendUnavailableError, match="Core ML could not load"):
            laya._start_workers(16)
    assert workers_of(fake_workers)["ane"]._backend.handoff is None
    assert laya.ane is None


@pytest.mark.parametrize("entry", ["init", "from_pretrained"])
def test_handoff_with_background_startup_is_rejected(entry, monkeypatch):
    """A background start-up cannot raise to the caller, so ane_handoff=True refuses it outright."""
    if entry == "init":
        with pytest.raises(ValueError, match='ane_startup="wait"'):
            model._check_ane_handoff(True, "laya", "workers", "auto", "auto", "background")
        return
    monkeypatch.setattr(model, "checkpoint_path", lambda *a, **k: pytest.fail("download attempted"))
    with pytest.raises(ValueError, match='ane_startup="wait"'):
        model.Laya.from_pretrained(
            "laya", device="auto", execution="workers", ane_handoff=True, ane_startup="background"
        )


def test_background_startup_without_handoff_is_unchanged():
    model._check_ane_handoff(False, "laya", "workers", "auto", "auto", "background")  # no error


def test_close_mid_episode(fake_backends):
    laya = make_laya("laya", ane_handoff=True)
    laya._workers = {k: DeviceWorker(k, {"pad_id": 0}, placement="thread") for k in ("gpu", "ane")}
    activity = handoff.GpuActivity()
    laya._workers["gpu"].activity = activity
    laya._handoff = handoff.StagedHandoff(activity, guard=2)
    fake_backends["ane"].handoff = laya._handoff
    gpu = fake_backends["gpu"]
    gpu.gate = threading.Event()
    gpu_futs = [laya._workers["gpu"].submit(ROW, 1.0, i) for i in range(3)]
    assert gpu.entered.acquire(timeout=5)
    for i in range(4):
        laya._workers["ane"].submit(ROW, 1.0, 10 + i).result(5)
    assert fake_backends["ane"].paths == ["sync", "sync", "async", "async"]
    assert laya._handoff.state == handoff.ASYNC_STEADY
    threading.Timer(0.2, gpu.gate.set).start()
    laya.close()
    for f in gpu_futs:
        assert f.done()
    assert activity.inflight == 0
    assert laya._workers == {} and laya.info()["ane_handoff"]["state"] == handoff.ASYNC_STEADY


# ----------------------------------------------------------------------------- production = research


def _research_handoff():
    path = Path(__file__).resolve().parents[2] / "research/coreml-staged-handoff/scripts/handoff.py"
    spec = importlib.util.spec_from_file_location("research_staged_handoff_for_prod_tests", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("seed", range(5))
def test_same_decisions_as_the_research_state_machine(seed):
    """Random GPU and forward timelines: every (path, state, count) of the production machine
    equals the research copy's (research/coreml-staged-handoff/scripts/handoff.py, guard 64)."""
    research = _research_handoff()
    rng = random.Random(seed)
    clock = Clock()
    prod_gpu, res_gpu = handoff.GpuActivity(clock), research.GpuActivity(clock)
    prod = handoff.StagedHandoff(prod_gpu, clock=clock)
    res = research.StagedHandoff(64, res_gpu, clock=clock)
    inflight = 0
    for _ in range(5000):
        r = rng.random()
        if r < 0.05:
            prod_gpu.started(), res_gpu.started()
            inflight += 1
        elif r < 0.10 and inflight:
            prod_gpu.ended(), res_gpu.ended()
            inflight -= 1
        else:
            assert prod.decide() == res.decide()
        clock.step(rng.choices([0.001, 0.01, 0.2, 0.9, 1.0, 1.0000001, 1.5, 3.0], [40, 40, 5, 2, 2, 2, 1, 1])[0])
    assert prod.episodes == res.episodes and prod.episodes > 1
    assert prod.forwards["async"] > 0


def test_gap_is_inclusive_at_one_second():
    m, gpu, clock = machine(2)
    gpu.started()
    assert m.decide() == ("sync", handoff.SYNC_GUARD, 1)
    clock.step(1.0)  # exactly the gap since the last forward: the same episode
    assert m.decide() == ("sync", handoff.ASYNC_STEADY, 2)
    gpu.ended()
    clock.step(1.0)  # GPU idle for exactly the gap: still active
    assert m.decide() == ("async", handoff.ASYNC_STEADY, 2)
    clock.step(1.001)  # GPU idle for more than the gap
    assert m.decide() == ("sync", handoff.ARMED, 0)


def test_forward_63_64_65_at_the_default_guard():
    m, gpu, clock = machine()
    gpu.started()
    rows_ = [m.decide() for _ in range(66)]
    assert rows_[62] == ("sync", handoff.SYNC_GUARD, 63)
    assert rows_[63] == ("sync", handoff.ASYNC_STEADY, 64)
    assert rows_[64] == ("async", handoff.ASYNC_STEADY, 64)
    assert rows_[65] == ("async", handoff.ASYNC_STEADY, 64)


def test_concurrent_episodes_with_gpu_hooks_stay_consistent():
    """Decisions on 4 threads while 4 others start and end GPU jobs: counters add up, no state is
    ever inconsistent, and nothing is disabled."""
    gpu = handoff.GpuActivity()
    m = handoff.StagedHandoff(gpu)
    stop = threading.Event()
    seen: list = []
    lock = threading.Lock()

    def feeder():
        while not stop.is_set():
            gpu.started()
            gpu.ended()

    def decider():
        mine = []
        for _ in range(2000):
            path, state, count = m.decide()
            assert state in handoff.STATES and 0 <= count <= m.guard
            mine.append(path)
        with lock:
            seen.extend(mine)

    feeders = [threading.Thread(target=feeder) for _ in range(4)]
    deciders = [threading.Thread(target=decider) for _ in range(4)]
    for t in feeders + deciders:
        t.start()
    for t in deciders:
        t.join()
    stop.set()
    for t in feeders:
        t.join()
    snap = m.snapshot()
    assert snap["consistent"] and not snap["disabled"] and gpu.inflight == 0
    assert snap["forwards"] == {"sync": seen.count("sync"), "async": seen.count("async")}
    assert sum(snap["forwards"].values()) == 8000


def test_corrupted_state_is_detected_disabled_and_warned_once():
    m, gpu, clock = machine(4)
    gpu.started()
    assert paths(m, clock, 2) == ["sync", "sync"]
    m.count = 99  # impossible in SYNC_GUARD
    with pytest.warns(RuntimeWarning, match="handoff state corrupted") as rec:
        assert m.decide() == ("sync", handoff.DISABLED, 0)
    assert len(rec) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert paths(m, clock, 3) == ["sync"] * 3
    snap = m.snapshot()
    assert snap["disabled"] and "count 99" in snap["disabled_reason"] and snap["consistent"]


def test_unknown_state_is_detected():
    m, gpu, clock = machine(4)
    m.state = "bogus"
    with pytest.warns(RuntimeWarning, match="unknown state"):
        assert m.decide()[0] == "sync"
    assert m.snapshot()["disabled"]


# ----------------------------------------------------------------------------- worker edge cases


def test_cancelled_ane_job_takes_no_decision(fake_backends):
    ane = DeviceWorker("ane", {"pad_id": 0}, placement="thread")
    b = fake_backends["ane"]
    gpu = handoff.GpuActivity()
    gpu.started()
    b.handoff = handoff.StagedHandoff(gpu, guard=2)
    b.gate = threading.Event()
    first = ane.submit(ROW, 1.0, 1)
    assert b.entered.acquire(timeout=5)
    queued = ane.submit(ROW, 1.0, 2)
    assert queued.cancel()
    b.gate.set()
    first.result(5)
    ane.submit(ROW, 1.0, 3).result(5)
    assert b.paths == ["sync", "sync"] and b.handoff.forwards == {"sync": 2, "async": 0}
    ane.close()


def test_activity_when_the_worker_dies_with_jobs_queued(fake_backends):
    w, b = gpu_worker(fake_backends)
    b.gate = threading.Event()
    running = w.submit(ROW, 1.0, 1)
    assert b.entered.acquire(timeout=5)
    queued = [w.submit(ROW, 1.0, i) for i in range(2, 4)]
    w._dead = BackendUnavailableError("worker exited")  # as the dispatcher records an EOF
    b.gate.set()
    running.result(5)
    for f in queued:
        with pytest.raises(BackendUnavailableError, match="not running"):
            f.result(5)
    late = w.submit(ROW, 1.0, 9)
    with pytest.raises(BackendUnavailableError):
        late.result(5)
    assert w.activity.inflight == 0 and w.activity._last_end_ns is not None
    w.close()


# ----------------------------------------------------------------------------- research harness contract


def test_class_level_decide_wrapper_sees_every_forward(monkeypatch):
    """prod_run.py wraps StagedHandoff.decide on the class and reads backend.handoff.snapshot()."""
    logged = []
    base = handoff.StagedHandoff.decide

    def decide(self):
        out = base(self)
        logged.append(out)
        return out

    monkeypatch.setattr(handoff.StagedHandoff, "decide", decide)
    b, gpu, clock = fake_ane(guard=2)
    gpu.started()
    for _ in range(3):
        b.forward(rows(1))
        clock.step(0.01)
    assert [p for p, _, _ in logged] == ["sync", "sync", "async"]
    assert {s for _, s, _ in logged} <= {"armed", "sync_guard", "async_steady"}
    snap = b.handoff.snapshot()
    assert snap["guard"] == 2 and not snap["disabled"] and snap["disabled_reason"] is None


def test_prod_run_reads_the_attached_handoff(fake_workers):
    laya = make_laya("laya", ane_handoff=True)
    laya._start_workers(16)
    snap = laya._workers["ane"]._backend.handoff.snapshot()
    assert snap["guard"] == handoff.GUARD and snap["state"] == handoff.ARMED
    assert snap == laya.info()["ane_handoff"]


def test_from_pretrained_rejects_ineligible_handoff_before_any_download(monkeypatch):
    def never(*a, **kw):
        raise AssertionError("must not resolve or download the checkpoint")

    monkeypatch.setattr(model, "checkpoint_path", never)
    for kw in (
        dict(execution="inline"),
        dict(execution="workers", device="gpu"),
        dict(execution="workers", device="ane"),
        dict(execution="workers", ane_placement="process"),
    ):
        with pytest.raises(ValueError, match="ane_handoff=True"):
            model.Laya.from_pretrained("laya", ane_handoff=True, **kw)
    with pytest.raises(ValueError, match="ane_handoff=True"):
        model.Laya.from_pretrained("laya-multilingual", execution="workers", ane_handoff=True)
    with pytest.raises(ValueError, match="ane_handoff must be"):
        model.Laya.from_pretrained("laya", execution="workers", ane_handoff="true")
    with pytest.raises(ValueError, match="ane_placement must be"):
        model.Laya.from_pretrained("laya", execution="workers", ane_placement="gpu", ane_handoff=True)


def test_repeated_episodes_through_the_backend():
    b, gpu, clock = fake_ane(guard=3)
    for episode in range(1, 6):
        gpu.started()
        got = []
        for _ in range(5):
            got.append(float(b.forward(rows(1))[0][0, 0]))
            clock.step(0.01)
        assert got == [0.0, 0.0, 0.0, 1.0, 1.0]  # 3 sync, then async, in every episode
        gpu.ended()
        clock.step(1.5)
        assert float(b.forward(rows(1))[0][0, 0]) == 0.0 and b.handoff.state == handoff.ARMED
        clock.step(1.5)
        assert b.handoff.episodes == episode
    assert b.handoff.forwards == {"sync": 5 * 4, "async": 5 * 2}
