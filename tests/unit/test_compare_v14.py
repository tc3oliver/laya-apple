"""The answer-level parity gate, helpers and task scheduling of benchmarks/compare-v1.4 (no models needed)."""

import argparse
import importlib.util
import sys
import threading
import time
from pathlib import Path

import pytest

from laya_apple.parity import TOLERANCE

HERE = Path(__file__).resolve().parents[2] / "benchmarks" / "compare-v1.4"


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


# ------------------------------------------------------------------------------ drive.py scheduling
drive = _load("drive")
ANE_CONFIGS = {("laya-apple", "auto"), ("laya-coreml", "ane"), ("laya-fast", "fast")}


def _args(**kw):
    d = dict(runtimes=None, models=None, rounds=3, warmup=20, window_s=20.0, cooldown_s=15.0, energy_cmd=None)
    d.update(kw)
    return argparse.Namespace(**d)


def test_parity_schedule_runs_ane_configurations_alone():
    cfgs = drive.configs()
    batch, serial = drive.parity_schedule(cfgs)
    assert drive.ANE_PARITY == ANE_CONFIGS
    assert {c[:2] for c in serial} == ANE_CONFIGS
    assert serial == [c for c in cfgs if c[:2] in ANE_CONFIGS]  # matrix order
    assert not {c[:2] for c in batch} & ANE_CONFIGS
    # every configuration plus one upstream reference per model, each exactly once
    refs = {("upstream", "cpu-fp32", m) for _, _, m in cfgs}
    assert sorted(batch + serial) == sorted(set(cfgs) | refs)
    est = [drive.parity_task_s(*c) for c in batch]
    assert est == sorted(est, reverse=True)  # longest first


def test_parity_schedule_follows_the_restricted_matrix():
    batch, serial = drive.parity_schedule(drive.configs(["laya-apple"], ["laya"]))
    assert batch == [("upstream", "cpu-fp32", "laya"), ("laya-apple", "gpu", "laya")]
    assert serial == [("laya-apple", "auto", "laya")]


def test_makespan():
    assert drive.makespan([4, 3, 2, 1], 1) == 10
    assert drive.makespan([4, 3, 2, 1], 2) == 5
    assert drive.makespan([4, 3, 2, 1], 8) == 4
    assert drive.makespan([], 4) == 0


def test_estimate_parity_is_fully_serial_with_one_job():
    one = drive.estimate(_args(parity_jobs=1))
    four = drive.estimate(_args(parity_jobs=4))
    assert one["parity_s"] == one["parity_all_serial_s"] == four["parity_all_serial_s"]
    assert four["parity_s"] < one["parity_s"]
    assert four["latency_s"] == one["latency_s"]  # latency is never run concurrently
    assert four["parity_serial_ane_tasks"] == 5


def test_parity_phase_never_overlaps_ane_tasks(monkeypatch, tmp_path):
    lock, running, log = threading.Lock(), [], []

    def fake_run_parity(a, rt, var, m, sets, out_dir):
        with lock:
            running.append((rt, var, m))
            log.append(((rt, var, m), set(running)))
        time.sleep(0.02)
        with lock:
            running.remove((rt, var, m))

    monkeypatch.setattr(drive, "fixture_sets", lambda a, run_dir: {})
    monkeypatch.setattr(drive, "run_parity", fake_run_parity)
    cfgs = drive.configs()
    drive.run_parity_phase(_args(parity_jobs=3), cfgs, tmp_path)
    batch, serial = drive.parity_schedule(cfgs)
    assert [c for c, _ in log[len(batch) :]] == serial  # ANE tasks last, in matrix order
    assert max(len(r) for _, r in log) <= 3
    assert max(len(r) for _, r in log[: len(batch)]) > 1  # the batch did run concurrently
    for c, r in log:
        if c[:2] in ANE_CONFIGS:
            assert r == {c}


def test_completed_skips_valid_outputs_and_refuses_partial_ones(tmp_path):
    out = tmp_path / "x.json"
    assert not drive.completed(out)
    out.write_text("{}")
    assert drive.completed(out)
    out.write_text('{"cases": [')
    with pytest.raises(SystemExit):
        drive.completed(out)


def test_phase_all_refuses_an_existing_run(tmp_path):
    (tmp_path / "r1").mkdir()
    with pytest.raises(SystemExit, match="never overwritten"):
        drive.open_run(_args(raw=str(tmp_path), run_id="r1", phase="all", base=str(tmp_path)))
