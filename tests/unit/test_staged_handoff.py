"""Unit tests for research/coreml-staged-handoff/ (research only): the handoff state machine
(boundaries, episodes, re-arm, GPU activity, concurrency), the fixed policies, the run order, and
run_config.py's import without PyObjC and its per-bucket model switch."""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "research" / "coreml-staged-handoff" / "scripts"
S = 1_000_000_000


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


handoff = _load("staged_handoff_for_tests", SCRIPTS / "handoff.py")
design = _load("staged_handoff_design_for_tests", SCRIPTS / "design.py")


class Clock:
    def __init__(self):
        self.t = 10 * S

    def __call__(self):
        return self.t

    def step(self, s: float):
        self.t += int(s * S)


def machine(guard=32, gap_s=1.0):
    clock = Clock()
    gpu = handoff.GpuActivity(clock)
    return handoff.StagedHandoff(guard, gpu, gap_s=gap_s, clock=clock), gpu, clock


def paths(m, clock, n, dt=0.01):
    out = []
    for _ in range(n):
        out.append(m.decide()[0])
        clock.step(dt)
    return out


def test_armed_runs_sync_without_gpu():
    m, _, clock = machine()
    assert paths(m, clock, 100) == ["sync"] * 100
    assert m.state == handoff.ARMED and m.episodes == 0


@pytest.mark.parametrize("guard", [32, 64])
def test_guard_boundary(guard):
    m, gpu, clock = machine(guard)
    gpu.started()
    p = paths(m, clock, guard + 3)
    assert p[:guard] == ["sync"] * guard  # forward N is still sync
    assert p[guard:] == ["async"] * 3  # forward N + 1 is the first async
    assert m.episodes == 1


def test_count_and_state_at_boundary():
    m, gpu, clock = machine(32)
    gpu.started()
    rows = []
    for _ in range(33):
        rows.append(m.decide())
        clock.step(0.01)
    assert rows[30] == ("sync", handoff.SYNC_GUARD, 31)
    assert rows[31] == ("sync", handoff.ASYNC_STEADY, 32)
    assert rows[32] == ("async", handoff.ASYNC_STEADY, 32)


def test_gpu_active_within_gap_after_last_end():
    m, gpu, clock = machine(2)
    gpu.started()
    gpu.ended()  # GPU idle, but its last job ended just now
    clock.step(0.5)
    assert paths(m, clock, 3) == ["sync", "sync", "async"]


def test_gpu_idle_longer_than_gap_rearms():
    m, gpu, clock = machine(2)
    gpu.started()
    assert paths(m, clock, 3) == ["sync", "sync", "async"]
    gpu.ended()
    clock.step(1.5)  # hetero -> solo_short
    assert paths(m, clock, 3) == ["sync"] * 3
    assert m.state == handoff.ARMED
    gpu.started()  # hetero again: a new guard
    assert paths(m, clock, 3) == ["sync", "sync", "async"]
    assert m.episodes == 2


def test_short_pause_rearms_while_gpu_stays_busy():
    m, gpu, clock = machine(2)
    gpu.started()
    assert paths(m, clock, 3) == ["sync", "sync", "async"]
    clock.step(2.0)  # hetero -> solo_long -> hetero: shorts pause, GPU busy throughout
    assert paths(m, clock, 3) == ["sync", "sync", "async"]
    assert m.episodes == 2


def test_hetero_ends_before_guard():
    m, gpu, clock = machine(32)
    gpu.started()
    assert paths(m, clock, 10) == ["sync"] * 10
    gpu.ended()
    clock.step(1.5)
    assert paths(m, clock, 5) == ["sync"] * 5 and m.state == handoff.ARMED
    gpu.started()
    p = paths(m, clock, 33)
    assert p[:32] == ["sync"] * 32 and p[32] == "async"  # the count restarted


def test_gap_is_inclusive_at_one_second():
    m, gpu, clock = machine(1)
    gpu.started()
    gpu.ended()
    clock.step(1.0)
    assert m.decide()[1] == handoff.ASYNC_STEADY  # guard 1: the episode started, then ASYNC_STEADY


def test_gpu_inflight_counts_concurrent_jobs():
    clock = Clock()
    gpu = handoff.GpuActivity(clock)
    gpu.started()
    gpu.started()
    gpu.ended()
    clock.step(5)
    assert gpu.active(clock(), S)  # one still in flight
    gpu.ended()
    gpu.ended()  # an extra end never goes negative
    clock.step(5)
    assert not gpu.active(clock(), S)


def test_concurrent_decisions_count_exactly():
    guard = 500
    gpu = handoff.GpuActivity()
    gpu.started()
    m = handoff.StagedHandoff(guard, gpu)
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
    assert m.episodes == 1


def test_guard_must_be_positive():
    with pytest.raises(ValueError):
        handoff.StagedHandoff(0, handoff.GpuActivity())


def test_fixed_policies():
    assert handoff.Fixed("sync").decide()[0] == "sync"
    assert handoff.Fixed("async").decide()[0] == "async"
    with pytest.raises(ValueError):
        handoff.Fixed("other")


# ------------------------------------------------------------------ design


def test_round1_is_mirrored_and_balanced():
    r = design.ROUND1
    assert [c for c, _ in r] == ["H32", "A", "H64", "B", "B", "H64", "A", "H32"]
    for c in design.CELLS:
        assert sum(1 for x, _ in r if x == c) == 2
    assert len(set(r)) == len(r)


def test_round2():
    assert design.round2("H64") == (("H64", 3), ("A", 3), ("H64", 4), ("A", 4), ("H64", 5))
    with pytest.raises(ValueError):
        design.round2("B")


# ------------------------------------------------------------------ run_config


def test_run_config_imports_without_pyobjc_and_switches_models():
    rc = _load("staged_handoff_rc_for_tests", SCRIPTS / "run_config.py")
    assert rc.CELLS == ("A", "B", "H32", "H64")

    class M:
        def __init__(self, name):
            self.name = name

        def predict(self, feats):
            return self.name

    route = rc.Route()
    hm = rc.HandoffModel(M("sync"), M("async"), route)
    assert hm.predict({}) == "sync"
    route.path = "async"
    assert hm.predict({}) == "async"
    assert hm.name == "sync"  # other attributes come from the sync model
    gpu = rc.handoff.GpuActivity()
    assert isinstance(rc.make_policy("H64", gpu), rc.handoff.StagedHandoff)
    assert rc.make_policy("H64", gpu).guard == 64
    assert rc.make_policy("A", gpu).decide()[0] == "sync"
    assert rc.make_policy("B", gpu).decide()[0] == "async"
