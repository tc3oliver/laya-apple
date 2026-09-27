"""Unit tests for research/intra-request-split/ (research only): the split policy, output merging,
the ABBA pairing, the gates' three-way rules and the analysis on synthetic runs. No hardware."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "research" / "intra-request-split" / "scripts"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sys.path.insert(0, str(SCRIPTS))
try:
    split = _load("irs_split_for_tests", SCRIPTS / "split.py")
    design = _load("irs_design_for_tests", SCRIPTS / "design.py")
    analyze = _load("irs_analyze_for_tests", SCRIPTS / "analyze.py")
finally:
    # The scripts import each other by plain name (common, design, split). Other research tracks'
    # tests use the same names for their own scripts: leave nothing of ours behind for them.
    sys.path.remove(str(SCRIPTS))
    for _name in ("common", "design", "split"):
        _mod = sys.modules.get(_name)
        if _mod is not None and Path(getattr(_mod, "__file__", "") or "").parent == SCRIPTS:
            del sys.modules[_name]
CRITERIA = json.loads((SCRIPTS.parent / "criteria.json").read_text())


# laya's shipped forward P50s (routing.json service_ms), as the product's ServiceModel reads them
def _service():
    from laya_apple.registry import routing_table
    from laya_apple.scheduling import ServiceModel

    return ServiceModel(routing_table()["models"]["laya"]["service_ms"])


# ------------------------------------------------------------------ policy


def test_single_question_is_never_split():
    s = _service()
    p = split.plan_split([512], [2], (64, 128, 512), s.gpu_ms, s.ane_ms)
    assert p.k == 0 and p.gpu_rows == (0,)
    p = split.plan_split([64], [2], (64, 128), s.gpu_ms, s.ane_ms)
    assert p.k == 0


def test_8x512_takes_some_rows_to_the_ane_and_predicts_a_shorter_makespan():
    s = _service()
    lengths = [512, 509, 510, 508, 511, 507, 512, 506]
    p = split.plan_split(lengths, [3] * 8, (64, 96, 128, 512), s.gpu_ms, s.ane_ms)
    assert 0 < p.k < 8
    assert p.makespan_ms < p.gpu_only_ms
    assert set(p.ane_rows) | set(p.gpu_rows) == set(range(8)) and not set(p.ane_rows) & set(p.gpu_rows)
    assert all(b == 512 for b in p.buckets)
    # the ANE takes the longest rows first
    ane_len = sorted((lengths[r] for r in p.ane_rows), reverse=True)
    assert min(ane_len) >= max(lengths[r] for r in p.gpu_rows) or len(set(lengths)) < 8


def test_makespan_is_the_minimum_over_k_and_ties_keep_the_gpu():
    gpu = lambda length, n: 10.0 * n  # noqa: E731
    ane = lambda b: 10.0  # noqa: E731
    p = split.plan_split([64, 64], [2, 2], (64,), gpu, ane)
    assert p.k == 1 and p.makespan_ms == 10.0
    # equal costs for k=0 and k=1 -> k=0 (today's path)
    p = split.plan_split([64, 64], [2, 2], (64,), lambda length, n: 10.0 if n == 2 else 10.0, ane)
    assert p.k == 0


def test_backlogs_move_rows_away_from_a_busy_device():
    s = _service()
    lengths = [128] * 8
    idle = split.plan_split(lengths, [3] * 8, (64, 96, 128), s.gpu_ms, s.ane_ms)
    busy_ane = split.plan_split(lengths, [3] * 8, (64, 96, 128), s.gpu_ms, s.ane_ms, ane_backlog_ms=500)
    busy_gpu = split.plan_split(lengths, [3] * 8, (64, 96, 128), s.gpu_ms, s.ane_ms, gpu_backlog_ms=500)
    assert busy_ane.k == 0
    assert busy_gpu.k > idle.k


def test_rows_that_fit_no_bucket_or_have_too_many_options_stay_on_the_gpu():
    s = _service()
    p = split.plan_split([300, 300], [2, 2], (64, 128), s.gpu_ms, s.ane_ms)
    assert p.k == 0
    p = split.plan_split([64, 64], [40, 40], (64,), s.gpu_ms, s.ane_ms)
    assert p.k == 0


def test_merge_outputs_places_rows_and_pads():
    a = (np.array([[1.0, 2.0]], np.float32), np.array([[0.1, 0.2]], np.float32))
    b = (np.full((2, 32), 5.0, np.float32), np.ones((2, 2), np.float32))
    logits, act = split.merge_outputs(3, [((1,), *a), ((0, 2), *b)])
    assert logits.shape == (3, 32)
    assert logits[1, 0] == 1.0 and logits[1, 2] == split.NEG
    assert (logits[0] == 5.0).all() and (act[2] == 1.0).all()
    with pytest.raises(ValueError):
        split.merge_outputs(3, [((0,), *a)])


# ------------------------------------------------------------------ design and gates


def test_adjacent_pairs_follow_the_abba_blocks():
    assert design.adjacent_pairs(["ours", "lf", "lf", "ours"], "ours", "lf") == [(1, 2), (4, 3)]
    assert design.adjacent_pairs(["lf", "ours", "ours", "lf"], "ours", "lf") == [(2, 1), (3, 4)]
    for block in CRITERIA["protocol"]["mix"]["blocks"]:
        pairs = design.adjacent_pairs(block, "split", "base")
        assert len(pairs) == 2 and all(abs(a - b) == 1 for a, b in pairs)
    with pytest.raises(ValueError):
        design.adjacent_pairs(["ours", "ours", "lf", "lf", "lf"], "ours", "lf")


def test_run_ids_follow_the_block_order():
    ids = design.run_ids({"ours": "laya", "lf": "laya-fast"}, CRITERIA["protocol"]["blocks"]["order"])
    assert ids[:4] == ["laya-b1-p1", "laya-fast-b1-p2", "laya-fast-b1-p3", "laya-b1-p4"]
    assert ids[4:] == ["laya-fast-b2-p1", "laya-b2-p2", "laya-b2-p3", "laya-fast-b2-p4"]


def test_three_way_rules():
    ci = lambda lo, hi: {"lo": lo, "hi": hi}  # noqa: E731
    assert analyze.below(ci(0.8, 0.99), 1.0) == "PASS"
    assert analyze.below(ci(1.0, 1.2), 1.0) == "FAIL"
    assert analyze.below(ci(0.9, 1.0), 1.0) == "INCONCLUSIVE"
    assert analyze.not_above(ci(0.9, 1.05), 1.05) == "PASS"
    assert analyze.not_above(ci(1.06, 1.2), 1.05) == "FAIL"
    assert analyze.not_above(ci(1.0, 1.1), 1.05) == "INCONCLUSIVE"
    assert design.mix_futility([1.3, 1.25], 1.2) and not design.mix_futility([1.3, 1.1], 1.2)


# ------------------------------------------------------------------ analysis on synthetic runs


def _window(cycle, wl, arm, p50):
    return {"cycle": cycle, "workload": wl, "arm": arm, "summary": {"p50_ms": p50}, "requests": []}


def _ours(run_id, split_p50, gpu_p50=240.0, cycles=5):
    w = []
    for c in range(1, cycles + 1):
        for wl in ("8x128", "8x512", "1x512", "32x64"):
            j = 1 + 0.001 * c
            w += [
                _window(c, wl, "split", split_p50[wl] * j),
                _window(c, wl, "gpu", gpu_p50 * j),
                _window(c, wl, "ane", 2 * gpu_p50),
            ]
    return {"run_id": run_id, "model": "laya", "windows": w, "correctness": []}


def _lf(run_id, p50, cycles=5):
    return {
        "run_id": run_id,
        "windows": [_window(c, "8x512", "laya_fast", p50 * (1 + 0.002 * c)) for c in range(1, cycles + 1)],
    }


def test_g1_and_g2_on_synthetic_runs():
    fast = {"8x128": 200.0, "8x512": 170.0, "1x512": 240.0, "32x64": 200.0}
    ours = {k: _ours(k, fast) for k in ("laya-b1-p1", "laya-b1-p4", "laya-b2-p2", "laya-b2-p3")}
    lf = {k: _lf(k, 220.0) for k in ("laya-fast-b1-p2", "laya-fast-b1-p3", "laya-fast-b2-p1", "laya-fast-b2-p4")}
    r1 = analyze.g1_ratios(ours, lf, CRITERIA, 2, "laya")
    assert len(r1) == 20
    st = analyze.stats([x["ratio"] for x in r1], 0.95)
    assert analyze.below(st["t"], 1.0) == "PASS"
    r2 = analyze.g2_ratios(list(ours.values()), "8x512")
    assert len(r2) == 20 and all(x["ratio"] < 1 for x in r2)
    slow = _lf("x", 150.0)
    lf_slow = {k: {**slow, "run_id": k} for k in lf}
    st = analyze.stats([x["ratio"] for x in analyze.g1_ratios(ours, lf_slow, CRITERIA, 2, "laya")], 0.95)
    assert analyze.below(st["t"], 1.0) == "FAIL"


def test_g3_fails_on_a_split_hard_mismatch_and_is_inconclusive_on_a_shared_exceedance():
    ok = {"prob_err": 0.01, "act_err": 0.01, "hard": [], "flips": []}
    run = {
        "run_id": "laya-b1-p1",
        "model": "laya",
        "windows": [],
        "correctness": [
            {"workload": "8x512", "seed": 0, "arm": "split", **ok},
            {"workload": "8x512", "seed": 0, "arm": "gpu", **ok},
        ],
    }
    tol = CRITERIA["gates"]["G3_correctness"]
    assert analyze.g3([run], {}, tol)["verdict"] == "PASS"
    bad = json.loads(json.dumps(run))
    bad["correctness"][0]["hard"] = ["q3"]
    assert analyze.g3([bad], {}, tol)["verdict"] == "FAIL"
    shared = json.loads(json.dumps(run))
    shared["correctness"][0]["prob_err"] = 0.03
    shared["correctness"][1]["prob_err"] = 0.025
    assert analyze.g3([shared], {}, tol)["verdict"] == "INCONCLUSIVE"
    alone = json.loads(json.dumps(run))
    alone["correctness"][0]["prob_err"] = 0.03
    assert analyze.g3([alone], {}, tol)["verdict"] == "FAIL"


def test_g4_pairs_rounds_and_counts_late_trains():
    blocks = CRITERIA["protocol"]["mix"]["blocks"]
    rounds, i = [], 0
    for b, block in enumerate(blocks, 1):
        for p, cfg in enumerate(block, 1):
            p99 = 50.0 if cfg == "base" else 50.5
            rounds.append(
                {
                    "index": i,
                    "block": b,
                    "position": p,
                    "config": cfg,
                    "oracle_wrong": 0,
                    "summary": {"train": {"p99_ms": p99 * (1 + 0.001 * i)}, "late": 0, "lag_p99_ms": 1.0},
                    "multi_question_checks": [],
                }
            )
            i += 1
    mix = {
        "mix": "switchyard",
        "model": "laya-typed-decisions",
        "protocol": CRITERIA["protocol"]["mix"],
        "rounds": rounds,
    }
    gate = CRITERIA["gates"]["G4a_switchyard_mix"]
    res = analyze.g4(mix, gate, 0.95)
    assert res["stats"]["n"] == 6 and res["verdict"] == "PASS"
    rounds[1]["summary"]["late"] = 3
    assert analyze.g4(mix, gate, 0.95)["verdict"] == "FAIL"


def _screen_runs(split_p50s, gpu_p50=240.0, lf_p50=220.0, hard=()):
    windows = []
    for c, s in enumerate(split_p50s, 1):
        windows += [
            {**_window(c, "8x512", "split", s), "requests": [{"seed": 0, "k": 3, "hard": list(hard)}]},
            _window(c, "8x512", "gpu", gpu_p50),
        ]
    ours = {"run_id": "screen-laya", "model": "laya", "windows": windows, "correctness": []}
    lf = {"run_id": "screen-laya-fast", "windows": [_window(c, "8x512", "laya_fast", lf_p50) for c in (1, 2, 3)]}
    return {"screen-laya": ours, "screen-laya-fast": lf}


def test_screen_is_stop_only():
    add = json.loads((SCRIPTS.parent / "criteria-addendum-1.json").read_text())
    assert add["screen"]["cycles"] == 6 and add["screen"]["arms"] == ["split", "gpu"]
    ok = analyze.screen(_screen_runs([170.0] * 6), add)
    assert ok["decision"] == "SCREEN CONTINUE"
    # one pair not faster, or a geometric mean not below 0.95: S1
    assert "S1" in analyze.screen(_screen_runs([170.0] * 5 + [241.0]), add)["stopped_by"]
    assert "S1" in analyze.screen(_screen_runs([230.0] * 6), add)["stopped_by"]
    # a split hard mismatch: S2
    assert "S2" in analyze.screen(_screen_runs([170.0] * 6, hard=["q1"]), add)["stopped_by"]
    # split median >= 1.10 x laya-fast median: S3
    assert analyze.screen(_screen_runs([170.0] * 6, lf_p50=150.0), add)["stopped_by"] == ["S3"]
