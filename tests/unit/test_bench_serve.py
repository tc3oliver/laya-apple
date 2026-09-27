"""Unit tests for the pure logic of scripts/bench_serve.py (serve decisions beside a local LLM)."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("bench_serve", ROOT / "scripts" / "bench_serve.py")
bs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bs)

MIX = {"short_1q": 0.8, "mixed_3q": 0.2}
VARIANTS = {"short_1q": 7, "mixed_3q": 4}


def choice(probs, act=0.5, label=None):
    label = label or max(probs, key=probs.get)
    return {"type": "choice", "choice": label, "probabilities": probs, "action": {"act_probability": act}}


# --- percentiles ---------------------------------------------------------------------------


def test_percentile_matches_numpy_linear_and_handles_empty():
    xs = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert bs.percentile(xs, 50) == 3.0
    assert bs.percentile(xs, 99) == pytest.approx(float(np.percentile(xs, 99)))
    assert bs.percentile([], 99) is None


def test_latency_stats_empty_and_filled():
    assert bs.latency_stats([]) == {"n": 0, "p50_ms": None, "p99_ms": None, "max_ms": None}
    s = bs.latency_stats(list(range(1, 101)))
    assert s["n"] == 100 and s["p50_ms"] == pytest.approx(50.5) and s["max_ms"] == 100.0
    assert s["p99_ms"] == pytest.approx(99.01)


# --- arrival schedule ----------------------------------------------------------------------


def test_schedule_is_deterministic_sorted_and_within_the_window():
    a = bs.client_schedule(8, 1.0, 60, 11, MIX, VARIANTS)
    b = bs.client_schedule(8, 1.0, 60, 11, MIX, VARIANTS)
    assert a == b
    ts = [x["t"] for x in a]
    assert ts == sorted(ts) and all(0 <= t < 60 for t in ts)
    assert {x["client"] for x in a} == set(range(8))
    assert all(0 <= x["variant"] < VARIANTS[x["cls"]] for x in a)


def test_schedule_rate_and_mix_are_near_the_offered_load():
    s = bs.client_schedule(8, 1.0, 600, 3, MIX, VARIANTS)
    assert len(s) / 600 == pytest.approx(8.0, rel=0.08)
    short = sum(x["cls"] == "short_1q" for x in s) / len(s)
    assert short == pytest.approx(0.8, abs=0.03)


def test_schedule_differs_by_seed_and_each_client_is_its_own_stream():
    a = bs.client_schedule(2, 1.0, 60, 11, MIX, VARIANTS)
    b = bs.client_schedule(2, 1.0, 60, 12, MIX, VARIANTS)
    assert a != b
    # adding a client never changes the existing clients' timetables
    c = bs.client_schedule(3, 1.0, 60, 11, MIX, VARIANTS)
    assert [x for x in c if x["client"] < 2] == a


# --- reference comparison ------------------------------------------------------------------


def test_identical_answers_pass_with_zero_error():
    ref = {"q0": choice({"a": 0.7, "b": 0.3}), "q1": {"type": "noul", "noul": 0.9, "action": {"act_probability": 1}}}
    r = bs.compare_answers(ref, json.loads(json.dumps(ref)))
    assert r == {"prob_err": 0.0, "act_err": 0.0, "hard": [], "flips": []}


def test_probability_and_action_errors_are_max_abs():
    ref = {"q0": choice({"a": 0.70, "b": 0.30}, act=0.50)}
    got = {"q0": choice({"a": 0.69, "b": 0.31}, act=0.53)}
    r = bs.compare_answers(ref, got)
    assert r["prob_err"] == pytest.approx(0.01) and r["act_err"] == pytest.approx(0.03)
    assert r["hard"] == [] and r["flips"] == []


def test_decision_flip_is_hard_outside_the_near_tie_band_and_listed_inside():
    wide = bs.compare_answers({"q": choice({"a": 0.6, "b": 0.4})}, {"q": choice({"a": 0.4, "b": 0.6})})
    assert wide["hard"] == ["q"] and wide["flips"] == []
    # reference margin 0.02 < 2 * 0.02: a near-tie flip, listed not failed
    near = bs.compare_answers({"q": choice({"a": 0.51, "b": 0.49})}, {"q": choice({"a": 0.49, "b": 0.51})})
    assert near["hard"] == [] and near["flips"] == ["q"]


def test_score_and_noul_decisions():
    score = {"type": "score", "score": 1.2, "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}}
    score2 = dict(score, probabilities={"0": 0.1, "1": 0.3, "2": 0.6})
    assert bs.compare_answers({"s": score}, {"s": score2})["hard"] == ["s"]
    noul = {"type": "noul", "noul": 0.8}
    assert bs.compare_answers({"n": noul}, {"n": {"type": "noul", "noul": 0.2}})["hard"] == ["n"]
    assert bs.compare_answers({"n": noul}, {"n": {"type": "noul", "noul": 0.79}})["prob_err"] == pytest.approx(0.01)


def test_missing_extra_or_retyped_questions_are_hard_mismatches():
    ref = {"q0": choice({"a": 0.7, "b": 0.3})}
    assert bs.compare_answers(ref, {})["hard"] == ["q0"]
    assert bs.compare_answers(ref, {"q0": {"type": "noul", "noul": 0.7}})["hard"] == ["q0"]
    assert bs.compare_answers(ref, {**ref, "zz": ref["q0"]})["hard"] == ["zz"]


# --- LLM token attribution and saturation --------------------------------------------------


def test_tokens_are_attributed_by_streamed_text_share():
    chunks = [(10, 10), (20, 30), (30, 60)]  # 100 chars in total
    assert bs.tokens_in_window(chunks, 50, 0, 100) == pytest.approx(50)
    assert bs.tokens_in_window(chunks, 50, 15, 30) == pytest.approx(15)  # the 30-char chunk only
    assert bs.tokens_in_window(chunks, None, 0, 100) == 0.0
    assert bs.tokens_in_window([], 50, 0, 100) == 0.0


def test_coverage_merges_overlaps_and_clips_to_the_window():
    assert bs.coverage([(0, 50), (40, 80)], 0, 100) == pytest.approx(0.8)
    assert bs.coverage([(-10, 20), (90, 200)], 0, 100) == pytest.approx(0.3)
    assert bs.coverage([], 0, 100) == 0.0
    assert bs.coverage([(0, 10)], 5, 5) == 0.0


# --- plan ----------------------------------------------------------------------------------


def test_full_plan_is_abba_with_llm_alone_between_blocks_and_alternating_windows():
    steps = bs.plan(2)
    blocks = [s for s in steps if s["kind"] == "block"]
    assert [b["config"] for b in blocks] == ["gpu", "auto", "auto", "gpu", "auto", "gpu", "gpu", "auto"]
    assert all(steps[i]["kind"] == "llm_alone" for i in range(0, len(steps), 2))
    assert len(steps) == 2 * len(blocks) + 1
    for cfg in ("gpu", "auto"):
        firsts = [b["windows"][0] for b in blocks if b["config"] == cfg]
        assert firsts.count("decisions") == firsts.count("decisions_llm")


def test_smoke_plan_has_one_window_per_cell():
    cells = [("llm_alone", None)] + [(w, s["config"]) for s in bs.plan(1, smoke=True)[1:] for w in s["windows"]]
    assert sorted(cells, key=str) == sorted(
        [
            ("llm_alone", None),
            ("decisions", "gpu"),
            ("decisions_llm", "gpu"),
            ("decisions", "auto"),
            ("decisions_llm", "auto"),
        ],
        key=str,
    )


def test_estimate_grows_with_the_window():
    a = argparse.Namespace(max_tokens=256, llm_warmup=15, window=60, llm_streams=2, cooldown=5, decision_warmup=3)
    b = argparse.Namespace(**{**vars(a), "window": 120})
    assert bs.estimate_seconds(bs.plan(2), b) > bs.estimate_seconds(bs.plan(2), a) > 0


# --- API key -------------------------------------------------------------------------------


def test_api_key_from_env_then_settings_file(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"auth": {"api_key": "from-file"}}))
    assert bs.read_api_key(str(settings), env={"LLM_API_KEY": "from-env"}) == "from-env"
    assert bs.read_api_key(str(settings), env={}) == "from-file"
    assert bs.read_api_key(None, env={}) is None
    settings.write_text(json.dumps({"auth": {}}))
    assert bs.read_api_key(str(settings), env={}) is None


def test_serve_log_has_no_machine_paths():
    text = f"{ROOT}/laya_apple/model.py:191: RuntimeWarning: x\n{Path.home()}/.cache/y\n"
    out = bs.scrub_paths(text)
    assert out == "<repo>/laya_apple/model.py:191: RuntimeWarning: x\n~/.cache/y\n"


def snap(state="async_healthy", episodes=3, trips=0, sync=192, asyn=500, enabled=True):
    return {
        "enabled": enabled,
        "consistent": True,
        "state": state,
        "episodes": episodes,
        "breaker": {"trips": trips},
        "forwards": {"sync": sync, "async": asyn},
    }


def test_handoff_snapshot_reads_the_models_health_entry():
    health = {"ane": {"laya": {"status": "ready", "handoff": snap()}}}
    assert bs.handoff_snapshot(health, "laya") == snap()
    assert bs.handoff_snapshot(health, "laya-typed-decisions") is None
    assert bs.handoff_snapshot({"ane": {"laya": {"status": "ready"}}}, "laya") is None  # gpu / older serve
    assert bs.handoff_snapshot({"error": "ConnectError"}, "laya") is None
    assert bs.handoff_snapshot(None, "laya") is None


def test_handoff_delta_is_the_windows_share_of_cumulative_counters():
    d = bs.handoff_delta(snap("armed", 3, 0, 192, 500), snap("breaker_open", 5, 1, 400, 900))
    assert d["present"] and d["enabled_before"] and d["enabled_after"]
    assert (d["state_before"], d["state_after"]) == ("armed", "breaker_open")
    assert (d["episodes"], d["trips"], d["forwards_sync"], d["forwards_async"]) == (2, 1, 208, 400)
    assert d["async_share"] == pytest.approx(400 / 608)


def test_handoff_delta_missing_or_disabled_snapshots():
    assert bs.handoff_delta(None, snap()) == {"present": False}
    disabled = {"enabled": False, "disabled": True, "disabled_reason": "no pyobjc"}
    d = bs.handoff_delta(disabled, disabled)
    assert d["present"] and d["enabled_after"] is False and d["disabled_reason"] == "no pyobjc"
    assert d["episodes"] is None and d["forwards_async"] is None and d["async_share"] is None
    idle = bs.handoff_delta(snap(), snap())  # no ANE forward in the window
    assert idle["forwards_sync"] == 0 and idle["async_share"] is None
