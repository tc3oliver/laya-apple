"""Unit tests for benchmarks/ane-equal-load-cpu/analyze.py (benchmark analysis only)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "ane-equal-load-cpu" / "analyze.py"
spec = importlib.util.spec_from_file_location("equal_load_cpu_analyze", PATH)
analyze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)


def window(samples, roles, measure=(1.0, 3.0)):
    return {"cell": "x", "cycle": 0, "measure": list(measure), "tree_cpu": {"roles": roles, "samples": samples}}


def test_cpu_is_interpolated_at_the_measurement_bounds():
    # parent burns 1 CPU-s per s, the worker 0.5; samples straddle the bounds
    samples = [(t, {"10": 1.0 * t, "11": 0.5 * t}) for t in (0.5, 1.5, 2.5, 3.5)]
    cpu = analyze.window_cpu(window(samples, {"parent": 10, "gpu:gpu": 11}))
    assert cpu["parent"] == pytest.approx(2.0)
    assert cpu["gpu:gpu"] == pytest.approx(1.0)
    assert cpu["other"] == 0.0


def test_unknown_children_count_as_other_and_unreadable_samples_are_skipped():
    samples = [(t, {"10": t, "12": 0.25 * t, "13": None}) for t in (0.0, 2.0, 4.0)]
    cpu = analyze.window_cpu(window(samples, {"parent": 10}))
    assert cpu["other"] == pytest.approx(0.5)
    assert set(cpu) == {"parent", "other"}


def test_a_process_not_sampled_across_the_whole_span_is_left_out():
    # pid 14 only appears after the window started: its CPU cannot be bounded, so it is not guessed
    samples = [(0.0, {"10": 0.0}), (2.0, {"10": 2.0, "14": 5.0}), (4.0, {"10": 4.0, "14": 6.0})]
    cpu = analyze.window_cpu(window(samples, {"parent": 10}))
    assert cpu == {"other": 0.0, "parent": pytest.approx(2.0)}


def test_a_role_not_sampled_across_the_span_is_an_error():
    samples = [(0.0, {"10": 0.0}), (2.0, {"10": 2.0, "11": 5.0}), (4.0, {"10": 4.0, "11": 6.0})]
    with pytest.raises(ValueError, match="gpu:gpu"):
        analyze.window_cpu(window(samples, {"parent": 10, "gpu:gpu": 11}))
