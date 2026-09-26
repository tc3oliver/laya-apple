"""Unit tests for research/coreml-adaptive-breaker/scripts/replay.py (research only): the candidate
detector rules and the latency ground truth, on synthetic prepare / e2e series."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "research" / "coreml-adaptive-breaker" / "scripts"
S = 1_000_000_000
MS = 1_000_000


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rp = _load("breaker_replay_for_tests", SCRIPTS / "replay.py")


def _trip(name, prep):
    prep = np.asarray(prep, dtype=float)
    return rp.DETECTORS[name](prep > rp.HOST_SLOW_MS, prep)


def test_consecutive_rules_need_an_unbroken_run():
    assert _trip("C2", [0.1, 0.5, 0.1, 0.5, 0.5]) == 4
    assert _trip("C3", [0.5, 0.5, 0.1, 0.5, 0.5, 0.1]) is None
    assert _trip("C3", [0.1, 0.5, 0.5, 0.5]) == 3
    assert _trip("C2", [0.3, 0.3, 0.3]) is None  # the threshold is strict


def test_rolling_rules_count_inside_the_window():
    assert _trip("R3of5", [0.5, 0.1, 0.5, 0.1, 0.5]) == 4
    assert _trip("R3of5", [0.5, 0.1, 0.1, 0.1, 0.1, 0.5, 0.1, 0.5]) is None
    assert _trip("R4of8", [0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5]) == 6


def test_ewma_and_median_rules():
    assert _trip("EWMA", [0.1, 0.1, 2.0]) == 2
    assert _trip("EWMA", [0.1] * 5) is None
    assert _trip("MED5", [0.5, 0.1, 0.1, 0.1, 0.5]) is None
    assert _trip("MED5", [0.5, 0.5, 0.1, 0.5, 0.1]) == 4


def _rq(e2e_by_second, rate=100, t_a=10 * S):
    sub, e2e = [], []
    for k, v in enumerate(e2e_by_second):
        for j in range(rate):
            sub.append(t_a + k * S + j * (S // rate))
            e2e.append(v)
    return {"submit": np.array(sub), "e2e": np.array(e2e, dtype=float)}, t_a, t_a + len(e2e_by_second) * S


def test_ground_truth_needs_two_consecutive_slow_spans():
    rq, t_a, end = _rq([10, 13, 10, 13, 10])
    gt = rp.ground_truth(rq, t_a, end, 10.0)
    assert gt["slow_spans"] == 2 and not gt["sustained"]
    rq, t_a, end = _rq([10, 10, 13, 13, 13])
    gt = rp.ground_truth(rq, t_a, end, 10.0)
    assert gt["sustained"] and gt["longest_run"] == 3
    # 1 s windows starting up to 0.4 s earlier still hold more than half slow requests
    assert gt["onset_ns"] == t_a + 2 * S - 4 * S // 10


def test_onset_never_precedes_the_async_start():
    rq, t_a, end = _rq([13, 13, 10])
    gt = rp.ground_truth(rq, t_a, end, 10.0)
    assert gt["onset_ns"] == t_a
