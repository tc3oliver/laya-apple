"""Unit tests for research/coreml-prebind-full-protocol/ (R1, research only): the run order, the
sequential-look rule, analyze.py end to end on synthetic full-protocol runs, and run_config.py's
import without PyObjC or hardware."""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "research" / "coreml-prebind-full-protocol" / "scripts"
MS = 1_000_000
CONDS = ("solo_short", "solo_long", "hetero", "gpu_only")
MODELS = ("laya", "laya-typed-decisions")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sys.path.insert(0, str(SCRIPTS))
design = _load("r1_design_for_tests", SCRIPTS / "design.py")
analyze = _load("r1_analyze_for_tests", SCRIPTS / "analyze.py")


# ------------------------------------------------------------------ design


def test_every_configuration_has_the_same_mean_position_per_block_and_stage():
    for stage in (1, 2, 3):
        firsts, lasts = Counter(), Counter()
        for b in design.stage_blocks(stage):
            runs = design.block_runs("laya", b)
            pos: dict[str, list[int]] = {}
            for i, (_, c, _) in enumerate(runs, start=1):
                pos.setdefault(c, []).append(i)
            assert {c: np.mean(p) for c, p in pos.items()} == {"A": 3.5, "C": 3.5, "PB": 3.5}
            firsts[runs[0][1]] += 1
            lasts[runs[-1][1]] += 1
        assert firsts == lasts == Counter({"A": 1, "C": 1, "PB": 1})


def test_rounds_pair_within_a_block():
    for b in range(1, 10):
        rounds: dict[str, set] = {}
        for _, c, r in design.block_runs("laya", b):
            rounds.setdefault(c, set()).add(r)
        assert all(v == {2 * b - 1, 2 * b} for v in rounds.values())


def test_stage_runs_interleave_models_block_by_block():
    runs = design.stage_runs(2, list(MODELS))
    assert len(runs) == 36
    assert [runs[6 * i][0] for i in range(6)] == ["laya", "laya-typed-decisions"] * 3
    assert all(r[0] == runs[6 * (i // 6)][0] for i, r in enumerate(runs))
    assert {r[2] for r in runs} == set(range(7, 13))
    assert design.config_name("laya", "P") == "A" and design.config_name("laya", "PB") == "PB"


def test_rounds_through_and_pairs_at():
    assert design.rounds_through(1) == tuple(range(1, 7))
    assert design.rounds_through(3) == tuple(range(1, 19))
    assert [design.pairs_at(s) for s in (1, 2, 3)] == [18, 36, 54]
    with pytest.raises(ValueError):
        design.rounds_through(4)
    assert design.run_file("laya", "A", 7) == "laya-A-r7.json.gz"


# ------------------------------------------------------------------ synthetic full-protocol runs


def _fake_run(config, short, long_, short_p99s, *, gpu_return_us=50, native=True, client_ms=50):
    """#77's layout: all four conditions in bench_concurrency's alternating order, 3 cycles."""
    windows_t, part_windows, res_windows = [], [], []
    trace = {c: [] for c in _run_config().TRACE_COLUMNS}
    forwards_ane, predicts, gpu_recv = [], [], []
    t, rid = 1_000 * MS, 0
    for cycle in range(3):
        for cond in CONDS if cycle % 2 == 0 else tuple(reversed(CONDS)):
            lo, hi = t, t + 200 * MS
            t = hi + 300 * MS
            streams = {"solo_short": ["short"], "solo_long": ["long"]}.get(cond, ["short", "long"])
            inst = "gpu" if cond == "gpu_only" else "auto"
            for s in streams:
                windows_t.append({"stream": s, "instance": inst, "start_ns": lo, "end_ns": hi})
            pw = {"cycle": cycle, "condition": cond, "streams": {}}
            for s in streams:
                dev = "gpu" if inst == "gpu" or s == "long" else "ane"
                p99 = short_p99s[cycle] if (cond == "hetero" and s == "short") else (10.0 if s == "short" else 40.0)
                pw["streams"][s] = {"req_s": 100.0 if s == "short" else 25.0, "p99_ms": p99, "p50_ms": 9.0}
                pw["streams"][s].update(devices={dev: 20}, mismatches=0, latency_ms=[1.0, 2.0])
            part_windows.append(pw)
            if inst == "gpu":
                continue
            if "short" in streams:
                for k in range(8):  # ANE forwards, 20 ms apart
                    s0 = lo + k * 20 * MS
                    forwards_ane.append((s0, s0 + 9 * MS, 600_000))
                    p = (s0 + MS, s0 + 2 * MS, s0 + 2 * MS + 1, s0 + 8 * MS, s0 + 8 * MS + 10, s0 + 8 * MS + 50)
                    predicts.append(p if native else (p[0], 0, 0, 0, 0, p[5]))
                    if cond == "hetero":
                        rid += 1
                        row = (rid, "ane", short, s0 - MS, s0 - MS, s0 - MS, s0 - MS, s0, s0, s0 + 9 * MS)
                        for c, v in zip(trace, (*row, s0 + 9 * MS, s0 - MS + (9 + k % 5) * MS)):
                            trace[c].append(v)
            if "long" in streams:
                for k in range(3):  # GPU completions
                    g0 = lo + k * 50 * MS
                    recv = g0 + 40 * MS
                    gpu_recv.append(recv)
                    if cond == "hetero":
                        rid += 1
                        row = (rid, "gpu", long_, g0, g0, g0, g0, g0, g0, recv - 50_000)
                        for c, v in zip(trace, (*row, recv, recv + 200_000)):
                            trace[c].append(v)
            if cond == "hetero":
                res_windows.append(
                    {
                        "start_ns": lo,
                        "end_ns": hi,
                        "instance": "auto",
                        "streams": {"short": 20, "long": 5},
                        "before": {"t_ns": lo - 100 * MS, "threads": {"client-short": 0}},
                        "after": {"t_ns": hi + 100 * MS, "threads": {"client-short": client_ms * MS}},
                    }
                )
    return {
        "args": {"short": short, "long": long_, "seconds": 20.0, "cycles": 3},  # campaign values; windows are short
        "part_a": {"windows": part_windows},
        "windows_t": windows_t,
        "gpu_return": {"received_ns": gpu_recv, "return_us": [gpu_return_us] * len(gpu_recv)},
        "research": {
            "config": config,
            "laya_apple": "x",
            "pyobjc": "x",
            "coremltools": "x",
            "mlx": "x",
            "gil_probe": None,
            "workers": {"ane": {"placement": "thread"}, "gpu": {"placement": "process"}},
            "forwards": {"ane": forwards_ane, "gpu": [(g - 40 * MS, g - 50_000, 2 * MS) for g in gpu_recv]},
            "predicts": predicts,
            "crossings": {"128": {"objc_send": 1, "ctypes": 1, "pool_push": 2, "pool_pop": 1, "total": 5}},
            "backings": None,
            "trace": trace,
            "windows": res_windows,
        },
    }


def _run_config():
    return _load("r1_run_config_for_tests", SCRIPTS / "run_config.py")


def _write(raw: Path, model: str, stages, ratio):
    """Every run of the given stages; ratio(config, round, cycle) scales the hetero short P99
    (P's is 10 ms)."""
    short, long_, _ = design.MODELS[model]
    for s in stages:
        for rnd in range(max(design.rounds_through(s)) - 5, max(design.rounds_through(s)) + 1):
            for c in design.CONFIGS:
                name = design.config_name(model, c)
                p99s = [10.0 * (1.0 if c == "P" else ratio(c, rnd, k)) for k in range(3)]
                run = _fake_run(name, short, long_, p99s, gpu_return_us=5000 if c == "P" else 50, native=c != "P")
                with gzip.open(raw / design.run_file(model, name, rnd), "wt") as fh:
                    json.dump(run, fh)


def flat(v):
    return lambda c, rnd, k: v


def noisy(lo, hi, then=None, after_round=None):
    """Alternating ratios per pair (wide CI); `then` from round after_round + 1 on."""

    def f(c, rnd, k):
        if then is not None and rnd > after_round:
            return then
        return hi if ((rnd - 1) * 3 + k) % 2 == 0 else lo

    return f


@pytest.fixture
def raw(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    return d


def test_empty_raw_requires_stage_1_and_renders(raw, tmp_path):
    res = analyze.summarise(raw)
    assert res["next"] == {"stage": 1, "models": list(MODELS)}
    assert res["check"] is None
    assert "stage 1 incomplete" in res["outcome"]
    md = analyze.tables(res)
    assert "Outcome:" in md
    json.dumps(res)


def test_pass_at_stage_1_is_final_and_later_runs_are_not_used(raw):
    _write(raw, "laya", (1, 2), flat(1.0))
    _write(raw, "laya-typed-decisions", (1,), flat(1.0))
    res = analyze.summarise(raw)
    m = res["models"]["laya"]
    assert m["final"]["stage"] == 1 and m["final"]["verdict"] == "PASS"
    assert list(m["looks"]) == ["1"]  # stage 2 exists on disk but is never looked at
    assert len(m["unused_runs"]) == 18 and "laya-PB-r7.json.gz" in m["unused_runs"]
    assert res["next"] is None
    assert res["outcome"].startswith("PB PASS on laya and laya-typed-decisions: proceed to R2")
    assert "C regression not reproduced" in m["final"]["interpretation"]
    v = m["looks"]["1"]["verdicts"]["PB"]
    assert v["pairs"] == 18 and v["checks"]["gpu_completion_isolation"] == "PASS"
    b = v["bonferroni_sensitivity_not_the_verdict"]["short_p99"]
    assert b["confidence"] == pytest.approx(1 - 0.05 / 3) and b["would_be"] == "PASS"


def test_inconclusive_at_stage_1_requires_stage_2_for_that_model_only(raw):
    _write(raw, "laya", (1,), noisy(0.85, 1.25))
    _write(raw, "laya-typed-decisions", (1,), flat(1.0))
    res = analyze.summarise(raw)
    assert res["models"]["laya"]["looks"]["1"]["verdicts"]["PB"]["verdict"] == "INCONCLUSIVE"
    assert res["models"]["laya"]["status"] == "extension required (stage 2)"
    assert res["models"]["laya-typed-decisions"]["final"]["verdict"] == "PASS"
    assert res["next"] == {"stage": 2, "models": ["laya"]}
    assert res["outcome"].startswith("extension required: stage 2 (36 pairs) for laya")


def test_inconclusive_then_pass_at_stage_2(raw):
    _write(raw, "laya", (1, 2), noisy(0.85, 1.25, then=0.9, after_round=6))
    _write(raw, "laya-typed-decisions", (1,), flat(1.0))
    res = analyze.summarise(raw)
    m = res["models"]["laya"]
    assert m["looks"]["1"]["verdicts"]["PB"]["verdict"] == "INCONCLUSIVE"
    assert m["final"]["stage"] == 2 and m["final"]["verdict"] == "PASS" and m["final"]["pairs"] == 36
    assert res["next"] is None and "proceed to R2" in res["outcome"]


def test_fail_on_one_model_stops_extension_for_the_other(raw):
    _write(raw, "laya", (1,), flat(1.2))
    _write(raw, "laya-typed-decisions", (1,), noisy(0.85, 1.25))
    res = analyze.summarise(raw)
    assert res["models"]["laya"]["final"]["verdict"] == "FAIL"
    assert res["models"]["laya-typed-decisions"]["needs_stage"] == 2
    assert res["next"] is None
    assert res["outcome"].startswith("PB FAIL on laya: R1 FAIL; a protocol-history causal experiment")


def test_fail_still_completes_stage_1_for_the_other_model(raw):
    _write(raw, "laya", (1,), flat(1.2))
    _write(raw, "laya-typed-decisions", (1,), flat(1.0))
    for c in ("A", "C", "PB"):
        (raw / design.run_file("laya-typed-decisions", c, 6)).unlink()
    res = analyze.summarise(raw)
    assert res["models"]["laya"]["final"]["verdict"] == "FAIL"
    assert res["next"] == {"stage": 1, "models": ["laya-typed-decisions"]}
    assert "R1 FAIL" in res["outcome"] and "completing started stage 1 for laya-typed-decisions" in res["outcome"]


def test_fail_completes_a_started_stage_but_starts_no_new_one(raw):
    _write(raw, "laya", (1,), flat(1.2))
    _write(raw, "laya-typed-decisions", (1, 2), noisy(0.85, 1.25))
    for rnd in range(8, 13):  # stage 2 started: only round 7 exists
        for c in ("A", "C", "PB"):
            (raw / design.run_file("laya-typed-decisions", c, rnd)).unlink()
    res = analyze.summarise(raw)
    assert res["models"]["laya-typed-decisions"]["needs_stage_started"] is True
    assert res["next"] == {"stage": 2, "models": ["laya-typed-decisions"]}
    assert "completing started stage 2" in res["outcome"]
    for c in ("A", "C", "PB"):  # not started: nothing more runs
        (raw / design.run_file("laya-typed-decisions", c, 7)).unlink()
    res = analyze.summarise(raw)
    assert res["next"] is None and res["outcome"].startswith("PB FAIL on laya: R1 FAIL")


def test_routing_failures_and_crashed_runs_are_listed(raw):
    _write(raw, "laya", (1,), flat(1.0))
    _write(raw, "laya-typed-decisions", (1,), flat(1.0))
    path = raw / design.run_file("laya", "C", 2)
    with gzip.open(path, "rt") as fh:
        run = json.load(fh)
    w = [w for w in run["part_a"]["windows"] if w["condition"] == "hetero" and w["cycle"] == 1][0]
    w["streams"]["short"]["devices"] = {"ane": 19, "gpu": 1}
    with gzip.open(path, "wt") as fh:
        json.dump(run, fh)
    (raw / "failed").mkdir()
    (raw / "failed" / "laya-PB-r3.1.log").write_text("boom\n")
    res = analyze.summarise(raw)
    assert res["models"]["laya"]["report"]["routing_failures"] == [
        {"config": "C", "round": 2, "cycle": 1, "stream": "short", "devices": {"ane": 19, "gpu": 1}}
    ]
    assert res["failed_runs"] == ["laya-PB-r3.1.log"]
    md = analyze.tables(res)
    assert "C round 2 cycle 1" in md and "laya-PB-r3.1.log" in md


def test_runs_with_other_window_lengths_are_rejected(raw):
    _write(raw, "laya", (1,), flat(1.0))
    path = raw / design.run_file("laya", "A", 1)
    with gzip.open(path, "rt") as fh:
        run = json.load(fh)
    run["args"]["seconds"] = 2.0
    with gzip.open(path, "wt") as fh:
        json.dump(run, fh)
    with pytest.raises(AssertionError):
        analyze.summarise(raw, ["laya"])


def test_inconclusive_at_cap(raw):
    _write(raw, "laya", (1, 2, 3), noisy(0.72, 1.6))
    _write(raw, "laya-typed-decisions", (1,), flat(1.0))
    res = analyze.summarise(raw)
    m = res["models"]["laya"]
    assert list(m["looks"]) == ["1", "2", "3"]
    assert m["final"]["verdict"] == "INCONCLUSIVE" and m["final"]["text"] == "INCONCLUSIVE at cap (54 pairs)"
    assert res["next"] is None
    assert res["outcome"].startswith("PB INCONCLUSIVE at the cap on laya: 1.5.0 blocked")


def test_end_to_end_with_window_history_and_tables(raw, tmp_path):
    # C: +20% short P99 after solo_long (even cycles) only; PB: +2% everywhere
    def ratio(c, rnd, k):
        if c == "C":
            return 1.2 if k % 2 == 0 else 1.0
        return 1.02

    _write(raw, "laya", (1,), ratio)
    _write(raw, "laya-typed-decisions", (1,), flat(1.0))
    (raw / "check.json").write_text(
        json.dumps({"PB_bit_identical_everywhere": True, "C_bit_identical_everywhere": True, "models": {}})
    )
    res = analyze.summarise(raw)
    m = res["models"]["laya"]
    assert m["final"]["verdict"] == "PASS" and m["final"]["C_reference"] == "FAIL"
    assert "reproduced under the full protocol and PB avoids it" in m["final"]["interpretation"]
    h = m["report"]["window_history"]
    assert h["order_as_expected"] is True
    assert h["vs_P"]["C"]["solo_long"]["short_p99_geomean_ratio"] == pytest.approx(1.2)
    assert h["vs_P"]["C"]["gpu_only"]["short_p99_geomean_ratio"] == pytest.approx(1.0)
    assert h["vs_P"]["C"]["solo_long"]["pairs"] == 12 and h["vs_P"]["C"]["gpu_only"]["pairs"] == 6
    assert h["vs_P"]["PB"]["gpu_only"]["short_p99_geomean_ratio"] == pytest.approx(1.02)
    e = m["report"]["extras"]
    assert e["PB"]["gil_wait"]["n"] == 6 * 3 * 8  # hetero windows only: 6 rounds x 3 cycles x 8 forwards
    assert e["P"]["windows"]["slow_cpu"] is None and e["PB"]["windows"]["slow_cpu"]["flagged_windows"] == 0
    assert e["PB"]["forwards"]["ane"]["n"] == 6 * 3 * 8
    assert m["report"]["routing_failures"] == []
    assert res["failed_runs"] == []
    md = analyze.tables(res)
    assert "PB vs A" in md and "solo_long" in md and "Bonferroni" in md and "bit-identical everywhere True" in md
    out = tmp_path / "out"
    out.mkdir()
    sys.argv = ["analyze.py", "--raw", str(raw), "--out", str(out)]
    analyze.main()
    sys.argv = ["analyze.py", "--raw", str(raw), "--out", str(out), "--check"]
    analyze.main()
    assert (out / "results.json").exists() and (out / "tables.md").read_text() == md


def test_preceding_condition_follows_bench_concurrency_order():
    run = _fake_run("A", 128, 512, [10.0, 10.0, 10.0])
    assert analyze.preceding(run) == {0: "solo_long", 1: "gpu_only", 2: "solo_long"}
    assert [analyze.expected_before(k) for k in range(3)] == ["solo_long", "gpu_only", "solo_long"]


# ------------------------------------------------------------------ run_config.py


def test_run_config_imports_without_pyobjc_and_finds_hetero_windows():
    rc = _run_config()
    assert set(rc.CONFIGS) == {"A", "C", "PB"}
    run = _fake_run("A", 128, 512, [10.0, 10.0, 10.0])
    bounds = rc.hetero_bounds(run)
    assert len(bounds) == 3
    assert bounds == analyze.gate57.hetero_bounds(run)
    assert all(
        w["stream"] in ("short", "long") for w in run["windows_t"] if (w["start_ns"], w["end_ns"]) in set(bounds)
    )
