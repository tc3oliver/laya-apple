"""The answer-level parity gate and helpers of benchmarks/compare-v1.4 (no models needed)."""

import importlib.util
import sys
from pathlib import Path

import pytest

from laya_apple.parity import TOLERANCE

HERE = Path(__file__).resolve().parents[1] / "benchmarks" / "compare-v1.4"


def _load(name):
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(f"compare_v14_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


analyze = _load("analyze")
adapter = _load("adapter")


def _choice(probs, act=1.0):
    return adapter.normalize_answer(
        {
            "type": "choice",
            "choice": max(probs, key=probs.get),
            "probabilities": probs,
            "action": {"act_probability": act},
        }
    )


def test_gate_tolerance_is_the_fp16_parity_tolerance():
    assert analyze.TOL == TOLERANCE["float16"]


def test_normalize_noul_and_laya_fast_extension():
    a = adapter.normalize_answer({"type": "noul", "noul": 0.25, "rl_agent": {"act_probability": 0.9}})
    assert a["labels"] == ["false", "true"]
    assert a["probs"] == pytest.approx([0.75, 0.25])
    assert a["act_probability"] == 0.9


def test_identical_answers_pass():
    ref = {"q": _choice({"a": 0.7, "b": 0.3})}
    res = analyze.gate(analyze.compare_answers(ref, ref))
    assert res["passed"] and res["prob_max_abs"] == 0 and res["hard_mismatches"] == 0


def test_hard_mismatch_vs_near_tie_flip():
    ref_far = {"q": _choice({"a": 0.7, "b": 0.3})}
    ref_tie = {"q": _choice({"a": 0.51, "b": 0.49})}
    got_far = {"q": _choice({"a": 0.3, "b": 0.7})}
    got_tie = {"q": _choice({"a": 0.495, "b": 0.505})}
    hard = analyze.gate(analyze.compare_answers(ref_far, got_far))
    assert hard["hard_mismatches"] == 1 and not hard["passed"]
    flip = analyze.gate([dict(r, case="c") for r in analyze.compare_answers(ref_tie, got_tie)])
    assert flip["hard_mismatches"] == 0 and len(flip["near_tie_flips"]) == 1
    assert flip["passed"]  # a near-tie flip inside tolerance is listed, not failed


def test_probability_and_action_errors_fail_the_gate():
    ref = {"q": _choice({"a": 0.7, "b": 0.3})}
    assert not analyze.gate(analyze.compare_answers(ref, {"q": _choice({"a": 0.67, "b": 0.33})}))["passed"]
    assert not analyze.gate(analyze.compare_answers(ref, {"q": _choice({"a": 0.7, "b": 0.3}, act=0.9)}))["passed"]
    assert not analyze.gate(analyze.compare_answers(ref, ref), repeat_identical=False)["passed"]


def test_missing_question_and_label_mismatch_fail():
    ref = {"q": _choice({"a": 0.7, "b": 0.3})}
    assert not analyze.gate(analyze.compare_answers(ref, {}))["passed"]
    res = analyze.gate(analyze.compare_answers(ref, {"q": _choice({"b": 0.3, "a": 0.7})}))
    assert not res["passed"] and len(res["label_mismatches"]) == 1


def test_window_power_uses_samples_inside_the_window():
    samples = [[t * 10**9, 0, {"cpu": 5.0 * t, "gpu": 5.0 * t}] for t in range(10)]
    e = {"mean_power_w": 99.0, "samples": samples}
    w = {"t_start_uptime_ns": 2 * 10**9, "t_end_uptime_ns": 7 * 10**9}
    assert analyze.window_power(e, w) == (pytest.approx(10.0), "window (interpolated)")
    # edges between samples are interpolated, not dropped
    w2 = {"t_start_uptime_ns": 2_300_000_000, "t_end_uptime_ns": 6_700_000_000}
    assert analyze.window_power(e, w2)[0] == pytest.approx(10.0)
    # no sample after the window end: fall back, labelled as the sampler's whole-life mean
    w3 = {"t_start_uptime_ns": 2 * 10**9, "t_end_uptime_ns": 12 * 10**9}
    assert analyze.window_power(e, w3) == (99.0, "sampler whole-life mean")


def test_percentile_interpolates():
    assert analyze.pct([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == 3.0
    assert analyze.pct(list(map(float, range(101))), 0.99) == pytest.approx(99.0)


def test_golden_sets_are_split_by_upstream_version():
    assert analyze.case_group("goldens", "drift-noul-labels").endswith("after 0.3.5")
    assert analyze.case_group("goldens", "lang-en").endswith("0.3.5-identical")
    assert analyze.case_group("shapes", "mixed3") == "shapes"
