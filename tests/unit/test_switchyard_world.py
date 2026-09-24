"""The switchyard-v1 workload (laya_apple/demos/switchyard/world.py): pure and seeded."""

from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

import pytest

from laya_apple.demos.switchyard import driver, world
from laya_apple.workload import arrivals

ROOT = Path(__file__).resolve().parents[2]


def _bench_concurrency():
    spec = importlib.util.spec_from_file_location("bench_concurrency", ROOT / "scripts" / "bench_concurrency.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("seed", [0, 7, 11, 12, 111])
@pytest.mark.parametrize("rate", [20.0, 40.0, 80.0])
@pytest.mark.parametrize("bursty", [True, False])
def test_arrivals_are_bit_identical_to_the_v1_benchmark_script(seed, rate, bursty):
    script = _bench_concurrency().arrivals(rate, 60, seed=seed, bursty=bursty)
    ours = arrivals(rate, 60, seed=seed, bursty=bursty)
    assert ours == script  # exact float equality, element by element
    assert all(0 < t < 60 for t in ours) and ours == sorted(ours)


def test_twelve_core_patterns_with_exactly_one_clear_platform():
    assert len(world.PATTERNS) == 12 == len(set(world.PATTERNS))
    for states in world.PATTERNS:
        assert Counter(states)[world.CLEAR] == 1
        assert set(states) <= {world.CLEAR, world.OCCUPIED, world.CLOSED}
    assert Counter(p.index(world.CLEAR) for p in world.PATTERNS) == {0: 4, 1: 4, 2: 4}


def test_schedule_is_deterministic_and_seeded():
    a, b = world.schedule(11, 20), world.schedule(11, 20)
    assert a == b and world.schedule_sha256(a) == world.schedule_sha256(b)
    assert world.schedule_sha256(world.schedule(12, 20)) != world.schedule_sha256(a)
    assert [x.offset_s for x in a] == arrivals(world.RATE_REQ_S, 20, seed=11, bursty=True)
    assert [x.index for x in a] == list(range(len(a)))


def test_streams_are_independent():
    """Class picks and train contents come from their own Random streams."""
    base = world.schedule(11, 30)
    assert [x.cls for x in world.schedule(11, 60)[: len(base)]] == [x.cls for x in base]
    trains = [x.train for x in base if x.train]
    assert [x.train for x in world.schedule(11, 60) if x.train][: len(trains)] == trains


def test_mix_and_trains():
    items = world.schedule(11, 60)
    share = Counter(x.cls for x in items)
    assert set(share) == {c for c, _ in world.MIX}
    assert 0.5 < share[world.TRAIN] / len(items) < 0.7
    for x in items:
        assert (x.train is not None) == (x.cls == world.TRAIN)
        if x.train:
            t = x.train
            assert 1 <= t.line <= world.LINES and 1000 <= t.train_id <= 9999
            assert t.platforms[t.oracle] == world.CLEAR
            for p, other in zip(world.PLATFORMS, t.occupants):
                assert (other is not None) == (t.platforms[p] == world.OCCUPIED)
                assert other != t.train_id


def test_warmup_uses_its_own_schedule():
    warm = world.warmup_schedule(11)
    assert warm == world.schedule(11 + world.WARMUP_SEED_OFFSET, world.WARMUP_S)
    assert max(x.offset_s for x in warm) < world.WARMUP_S


def test_train_prompt_wording():
    t = world.Train(
        line=3,
        train_id=4127,
        pattern=world.PATTERNS.index(("occupied", "closed", "clear")),
        occupants=(2290, None, None),
    )
    context, questions = world.train_prompt(t)
    assert context == (
        "Line 3 inbound. Train 4127 is approaching the junction. Platform A is occupied by train 2290. "
        "Platform B is closed for maintenance. Platform C is clear and open."
    )
    assert questions == {
        "platform": {
            "type": "choice",
            "instructions": "Train 4127 must be routed to the clear platform. Which platform is clear?",
            "criteria": ["A", "B", "C"],
        }
    }
    assert t.oracle == "C"


def test_prompt_template_hash_is_stable():
    h = world.prompt_template_sha256()
    assert len(h) == 64 and h == world.prompt_template_sha256()


def test_oracle_cases_cover_every_pattern():
    cases = world.oracle_cases()
    assert len(cases) == 240
    assert Counter(c.pattern for c in cases) == dict.fromkeys(range(12), 20)
    assert len({(c.line, c.train_id, c.occupants) for c in cases}) > 200  # surface variants differ


def test_offered_load():
    items = world.schedule(11, 60)
    o = world.offered(items, 60)
    assert o["requests"] == len(items) and o["trains"] == sum(x.cls == "train" for x in items)
    assert o["req_s"] == pytest.approx(len(items) / 60)


@pytest.mark.parametrize(
    ("seed", "ready", "order"),
    [
        (10, True, ["gpu_only", "hybrid"]),
        (11, True, ["hybrid", "gpu_only"]),
        (11, False, ["gpu_only"]),
        (12, False, ["gpu_only"]),
    ],
)
def test_round_order_follows_seed_parity(seed, ready, order):
    assert driver.round_order(seed, ready) == order


def test_frozen_switchyard_v1_fingerprints():
    """switchyard-v1 is frozen: changing any of these means a new workload version (and new
    results), never an edit to these values."""
    items = world.schedule()
    assert world.offered(items, world.DURATION_S)["requests"] == 2410
    assert world.offered(items, world.DURATION_S)["trains"] == 1422
    assert world.schedule_sha256(items) == "bf84ed5699072c8d561ae8c65b61bd6fd71b89c2d467f2b2c5686caad0ccf99d"
    assert world.prompt_template_sha256() == "a7569c1f9b8e5592b7fc1e7a14f78d099ba26cee64037fa596998db45084e7fe"
    assert (world.WORKLOAD_ID, world.WORKLOAD_VERSION, world.DEFAULT_SEED, world.DURATION_S) == (
        "switchyard-v1",
        1,
        11,
        60.0,
    )
