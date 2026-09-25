"""Unit tests for the pure logic of research/coreml-prebind-predict/ (research only): collision
and overlap computation, stage splitting, the crossing counter, the probe grid, and analyze.py
end to end on synthetic runs."""

from __future__ import annotations

import ctypes
import gzip
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "research" / "coreml-prebind-predict" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import crossings  # noqa: E402
import derive  # noqa: E402
import gate  # noqa: E402
import probe  # noqa: E402

MS = 1_000_000


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ collisions


def test_collide_fixed_window_edges_inclusive():
    start = 100 * MS
    completions = [start - derive.BEFORE_NS, start + derive.AFTER_NS]
    assert derive.collide_fixed([start], completions[:1]).tolist() == [True]
    assert derive.collide_fixed([start], completions[1:]).tolist() == [True]
    assert derive.collide_fixed([start], [start - derive.BEFORE_NS - 1, start + derive.AFTER_NS + 1]).tolist() == [
        False
    ]


def test_collide_fixed_unsorted_completions_and_many_starts():
    starts = [10 * MS, 20 * MS, 30 * MS]
    completions = [30 * MS + 100_000, 9 * MS + 500_000]  # unsorted on purpose
    assert derive.collide_fixed(starts, completions).tolist() == [True, False, True]
    assert derive.collide_fixed(starts, []).tolist() == [False, False, False]


def test_overlap_counts_against_brute_force():
    rng = np.random.default_rng(1)
    spans = np.sort(rng.integers(0, 1000, (200, 2)), axis=1)
    spans[:, 1] += 1  # non-empty spans
    iv = np.sort(rng.integers(0, 1000, (300, 2)), axis=1)
    brute = [int(((iv[:, 0] < b) & (iv[:, 1] > a)).sum()) for a, b in spans]
    assert derive.overlap_counts(spans, iv).tolist() == brute


def test_overlap_counts_touching_is_not_overlap_and_rejects_inverted():
    assert derive.overlap_counts([[10, 20]], [[0, 10], [20, 30], [5, 11]]).tolist() == [1]
    with pytest.raises(ValueError):
        derive.overlap_counts([[0, 1]], [[5, 4]])


def test_odds_ratio_and_tail_association():
    exposed = [True] * 10 + [False] * 90
    outcome = [True] * 8 + [False] * 2 + [True] * 2 + [False] * 88
    assert derive.odds_ratio(exposed, outcome) == pytest.approx((8.5 * 88.5) / (2.5 * 2.5))
    a = derive.tail_association(np.array([1, 2, 3, 10, 11]) * MS, [False, False, True, True, False], 5 * MS)
    assert a["n"] == 5 and a["tail_n"] == 2 and a["tail_fraction"] == pytest.approx(0.4)
    assert a["tail_colliding_fraction"] == pytest.approx(0.5)
    assert a["collide_fraction"] == pytest.approx(0.4)


# ------------------------------------------------------------------ stages and joins


def test_stages_with_native_stamps():
    s = derive.stages((0, 100), (10, 18, 20, 70, 75, 80))
    assert s == {"features": 10, "tail": 20, "predict": 70, "pre": 10, "native": 50, "reacquire": 5, "post": 5}
    assert sum(s[k] for k in derive.STAGES) == 100


def test_stages_without_native_stamps_coremltools():
    s = derive.stages((0, 100), (10, 0, 0, 0, 0, 80))
    assert s["predict"] == 70 and s["features"] == 10 and s["tail"] == 20
    assert s["pre"] is s["native"] is s["reacquire"] is s["post"] is None


def test_join_predicts_matches_containing_forward_only():
    forwards = [(0, 100), (200, 300), (400, 500)]
    predicts = [(10, 0, 0, 0, 0, 90), (310, 0, 0, 0, 0, 320), (410, 0, 0, 0, 0, 490)]
    assert derive.join_predicts(forwards, predicts).tolist() == [0, -1, 2]
    assert derive.join_predicts(forwards, np.zeros((0, 6))).tolist() == [-1, -1, -1]


def test_windows_and_thread_delta_and_slow_cpu():
    bounds = [(0, 10), (20, 30)]
    assert derive.in_windows([0, 9, 10, 25, 30], bounds).tolist() == [True, True, False, True, False]
    assert derive.window_index([5, 15, 29], bounds).tolist() == [0, -1, 1]
    d = derive.thread_cpu_delta({"threads": {"a": 5, "gone": 1}}, {"threads": {"a": 8, "new": 4}})
    assert d == {"a": 3, "new": 4}
    assert derive.slow_cpu([130, 130], [100, 100]) == {"ratio": pytest.approx(1.3), "flag": True}
    assert derive.slow_cpu([120], [100])["flag"] is False
    assert derive.slow_cpu([], [100]) == {"ratio": None, "flag": None}


def test_lateness_summary():
    s = derive.lateness_summary([0.5 * MS, 2 * MS, 0.1 * MS, 0.2 * MS])
    assert s["n"] == 4 and s["over_1ms_fraction"] == pytest.approx(0.25)
    assert derive.lateness_summary([])["p50_ms"] is None


# ------------------------------------------------------------------ paired gate


@pytest.mark.parametrize(
    "df, expected",
    [(1, 12.7062), (2, 4.3027), (3, 3.1824), (5, 2.5706), (11, 2.2010), (17, 2.1098), (30, 2.0423)],
)
def test_t_critical_matches_tables(df, expected):
    assert gate.t_critical(df) == pytest.approx(expected, abs=1e-4)


def test_t_interval_on_log_ratios():
    ratios = [1.0, 1.1, 0.95, 1.05, 1.02, 0.98]
    x = np.log(ratios)
    hw = 2.5706 * x.std(ddof=1) / np.sqrt(6)
    ci = gate.t_interval(ratios)
    assert ci["n"] == 6
    assert ci["geomean"] == pytest.approx(np.exp(x.mean()))
    assert ci["lo"] == pytest.approx(np.exp(x.mean() - hw), rel=1e-5)
    assert ci["hi"] == pytest.approx(np.exp(x.mean() + hw), rel=1e-5)
    with pytest.raises(ValueError):
        gate.t_interval([1.0])


def test_bootstrap_is_seeded_and_brackets_the_geomean():
    ratios = [1.0, 1.1, 0.95, 1.05, 1.02, 0.98]
    a, b = gate.bootstrap_interval(ratios), gate.bootstrap_interval(ratios)
    assert a == b and a["resamples"] == 10_000
    g = gate.t_interval(ratios)["geomean"]
    assert a["lo"] <= g <= a["hi"]
    assert gate.bootstrap_interval(ratios, seed=1) != a


@pytest.mark.parametrize(
    "lo, hi, upper, lower",
    [
        (1.00, 1.05, "PASS", "PASS"),  # upper bound exactly at the budget passes
        (1.051, 1.2, "FAIL", "PASS"),
        (1.00, 1.06, "INCONCLUSIVE", "PASS"),
        (0.90, 0.949, "PASS", "FAIL"),
        (0.94, 0.96, "PASS", "INCONCLUSIVE"),
        (0.95, 0.99, "PASS", "PASS"),  # lower bound exactly at the budget passes
    ],
)
def test_three_way_verdicts(lo, hi, upper, lower):
    ci = {"lo": lo, "hi": hi}
    assert gate.upper_verdict(ci, 1.05) == upper
    assert gate.lower_verdict(ci, 0.95) == lower


def test_combine():
    assert gate.combine(["PASS", "PASS"]) == "PASS"
    assert gate.combine(["PASS", "INCONCLUSIVE"]) == "INCONCLUSIVE"
    assert gate.combine(["INCONCLUSIVE", "FAIL", "PASS"]) == "FAIL"


def _hetero_run(p99s, req=(100.0, 25.0), long_p99=40.0):
    return {
        "part_a": {
            "windows": [{"cycle": k, "condition": "solo_short", "streams": {}} for k in range(len(p99s))]
            + [
                {
                    "cycle": k,
                    "condition": "hetero",
                    "streams": {
                        "short": {"p99_ms": p, "req_s": req[0]},
                        "long": {"p99_ms": long_p99, "req_s": req[1]},
                    },
                }
                for k, p in enumerate(p99s)
            ]
        }
    }


def test_pair_windows_matches_round_and_cycle():
    prod = [_hetero_run([10, 11, 12]), _hetero_run([13, 14, 15])]
    cand = [_hetero_run([20, 22, 24]), _hetero_run([26, 28, 30])]
    pairs = gate.pair_windows(prod, cand)
    assert [(r, k) for r, k, _, _ in pairs] == [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    assert gate.pair_ratios(pairs)["short_p99"] == pytest.approx([2.0] * 6)
    assert gate.pair_ratios(pairs)["aggregate"] == pytest.approx([1.0] * 6)


def test_pair_windows_rejects_mismatched_cycles_and_rounds():
    with pytest.raises(ValueError):
        gate.pair_windows([_hetero_run([10, 11])], [_hetero_run([10, 11, 12])])
    with pytest.raises(ValueError):
        gate.pair_windows([_hetero_run([10])], [])


def test_paired_verdict_inconclusive_when_noisy():
    rng = np.random.default_rng(3)
    prod = [_hetero_run([10.0] * 3), _hetero_run([10.0] * 3)]
    cand = [_hetero_run(list(10.0 * np.exp(rng.normal(0, 0.15, 3)))) for _ in range(2)]
    v = gate.paired_verdict(prod, cand, gate_limits(), correctness=True, isolation=True)
    assert v["checks"]["short_p99"] == "INCONCLUSIVE" and v["verdict"] == "INCONCLUSIVE"
    v = gate.paired_verdict(prod, cand, gate_limits(), correctness=False, isolation=True)
    assert v["verdict"] == "FAIL"


def gate_limits():
    return {"aggregate_min_ratio": 0.95, "p99_max_ratio": 1.05, "return_p50_max_ms": 1.0, "return_min_gain": 5.0}


# ------------------------------------------------------------------ crossing counter and probe


def test_crossings_counts_ctypes_calls_on_this_thread_only():
    libc = ctypes.CDLL(None)
    libc.abs.argtypes = [ctypes.c_int]

    def work():
        return libc.abs(-3) + libc.abs(-4) + abs(-5)  # the builtin abs is not a crossing

    result, counts = crossings.count(work)
    assert result == 12
    sites = counts.pop("sites")
    assert counts == {
        "objc_send": 0,
        "ctypes": 2,
        "pool_push": 0,
        "pool_pop": 0,
        "total": 2,
        "reentries": 0,
        "nested_sends": 0,
    }
    assert list(sites.values()) == [2] and next(iter(sites)).startswith("ctypes abs <- ")


def test_crossings_counts_python_callbacks_inside_a_native_call():
    """qsort with a Python comparator: every comparison re-enters Python (and takes the GIL)
    from inside the one native call, which a plain call count would not show."""
    libc = ctypes.CDLL(None)
    cmp_t = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
    calls = []

    def compare(a, b):
        calls.append(1)
        return a[0] - b[0]

    comparator = cmp_t(compare)
    values = (ctypes.c_int * 5)(5, 1, 4, 2, 3)
    libc.qsort.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, cmp_t]

    def sort():
        libc.qsort(values, 5, ctypes.sizeof(ctypes.c_int), comparator)

    _, counts = crossings.count(sort)
    assert list(values) == [1, 2, 3, 4, 5]
    assert counts["ctypes"] == 1 and counts["total"] == 1
    assert counts["reentries"] == len(calls) > 0


def test_crossings_classify_by_type_name_and_caller():
    class native_selector:  # noqa: N801  the PyObjC type's name is what classify reads
        def __call__(self):
            return None

    sel = native_selector()
    assert crossings.classify(sel) == "objc_send"
    assert crossings.classify(sel, "autorelease_pool.__enter__") == "pool_push"
    assert crossings.classify(len) is None

    assert crossings.classify(autorelease_pool().__exit__) == "pool_pop"


class autorelease_pool:  # noqa: N801  same qualified name as objc.autorelease_pool
    def __exit__(self, *a):
        return None


def test_probe_next_tick_skips_missed_ticks():
    assert probe.next_tick(1000, 1200, 1000) == 2000
    assert probe.next_tick(1000, 3500, 1000) == 4000
    assert probe.next_tick(1000, 2000, 1000) == 3000


def test_probe_thread_cpu_sees_this_thread():
    import threading

    snap = probe.snapshot()
    assert threading.current_thread().name in snap["threads"]
    assert all(v >= 0 for v in snap["threads"].values())


# ------------------------------------------------------------------ analyze.py on synthetic runs


def _fake_run(config, short, long_, *, ane_p99=10.0, native=True, in_process=True, client_ms=50):
    base = 1_000 * MS
    windows_t, part_windows, res_windows = [], [], []
    gpu_recv, gpu_ret = [], []
    trace = {c: [] for c in _run_config().TRACE_COLUMNS}
    forwards_ane, predicts = [], []
    t = base
    rid = 0
    for i in range(3):  # hetero-only: one hetero window per cycle
        cond = "hetero"
        lo, hi = t, t + 1000 * MS
        streams = ["short", "long"]
        for s in streams:
            windows_t.append({"stream": s, "instance": "auto", "start_ns": lo, "end_ns": hi})
        pw = {"cycle": i, "condition": cond, "streams": {}}
        for s in streams:
            dev = "ane" if s == "short" else "gpu"
            pw["streams"][s] = {"req_s": 100.0, "p99_ms": ane_p99 if s == "short" else 40.0, "p50_ms": 9.0}
            pw["streams"][s].update(devices={dev: 100}, mismatches=0)
        part_windows.append(pw)
        for k in range(50):  # ANE forwards, 20 ms apart
            s0 = lo + k * 20 * MS
            forwards_ane.append((s0, s0 + 9 * MS, 600_000))
            p = (s0 + MS, s0 + 2 * MS, s0 + 2 * MS + 1, s0 + 8 * MS, s0 + 8 * MS + 10, s0 + 8 * MS + 50)
            predicts.append(p if native else (p[0], 0, 0, 0, 0, p[5]))
            if cond == "hetero":
                rid += 1
                row = (rid, "ane", short, s0 - MS, s0 - MS, s0 - MS, s0 - MS, s0, s0, s0 + 9 * MS)
                lat = (9 + k % 5) * MS
                for c, v in zip(trace, (*row, s0 + 9 * MS, s0 - MS + lat)):
                    trace[c].append(v)
        if cond == "hetero":
            for k in range(20):  # GPU completions, 50 ms apart, some near ANE starts
                g0 = lo + k * 50 * MS
                recv = g0 + 40 * MS
                gpu_recv.append(recv)
                gpu_ret.append(50)
                rid += 1
                row = (rid, "gpu", long_, g0, g0, g0, g0, g0, g0, recv - 50_000)
                for c, v in zip(trace, (*row, recv, recv + 200_000)):
                    trace[c].append(v)
        snap_b = {"t_ns": lo - 100 * MS, "threads": {"client-short": 0, "gil-probe": 0}}
        snap_a = {"t_ns": hi + 100 * MS, "threads": {"client-short": client_ms * MS, "gil-probe": 2 * MS}}
        res_windows.append(
            {
                "start_ns": lo,
                "end_ns": hi,
                "instance": "auto",
                "streams": {s: 100 for s in streams},
                "before": snap_b,
                "after": snap_a,
                "gil_probe": derive.lateness_summary([300_000, 400_000, 2 * MS]),
            }
        )
        t = hi + 3000 * MS
    return {
        "args": {"short": short, "long": long_, "seconds": 1.0, "cycles": 3},
        "part_a": {"windows": part_windows},
        "windows_t": windows_t,
        "gpu_return": {"received_ns": gpu_recv, "return_us": gpu_ret},
        "research": {
            "config": config,
            "laya_apple": "x",
            "pyobjc": "x",
            "coremltools": "x",
            "mlx": "x",
            "workers": {
                "ane": {"placement": "thread" if in_process else "process"},
                "gpu": {"placement": "process"},
            },
            "forwards": {"ane": forwards_ane, "gpu": [(g - 40 * MS, g - 50_000, 2 * MS) for g in gpu_recv]},
            "predicts": predicts if in_process else None,
            "crossings": {"128": {"objc_send": 1, "ctypes": 1, "pool_push": 2, "pool_pop": 1, "total": 5}}
            if in_process
            else None,
            "backings": None,
            "trace": trace,
            "windows": res_windows,
        },
    }


def _run_config():
    return _load("prebind_run_config_for_tests", SCRIPTS / "run_config.py")


def test_analyze_end_to_end_on_synthetic_runs(tmp_path):
    analyze = _load("prebind_analyze_for_tests", SCRIPTS / "analyze.py")
    raw = tmp_path / "raw"
    raw.mkdir()
    runs = {
        "A": dict(native=False),
        "C": dict(ane_p99=12.0, client_ms=70),  # +20%: fails short P99; client CPU 1.4x P: flagged
        "PB": dict(ane_p99=10.2),  # +2%: passes
    }
    for c, kw in runs.items():
        for i in (1, 2):
            with gzip.open(raw / f"laya-{c}-r{i}.json.gz", "wt") as fh:
                json.dump(_fake_run(c, 128, 512, **kw), fh)
    res = analyze.summarise(raw, (1, 2), ["laya"])
    v = res["models"]["laya"]["verdicts"]
    assert v["C"]["checks"]["short_p99"] == "FAIL" and v["C"]["verdict"] == "FAIL"
    assert v["PB"]["checks"]["short_p99"] == "PASS"
    assert v["PB"]["pairs"] == 6
    assert res["per_model"]["PB"]["laya"] == v["PB"]["verdict"]
    assert v["PB"]["stats"]["short_p99"]["t"]["geomean"] == pytest.approx(1.02)
    e = res["models"]["laya"]["extras"]
    assert e["A"]["stages"]["predict"]["n"] > 0 and "native" not in e["A"]["stages"]
    assert e["PB"]["stages"]["reacquire"]["p50_ms"] == pytest.approx(10 / 1e6)
    assert e["PB"]["gil_wait"]["n"] == 300  # 2 runs x 3 hetero windows x 50 forwards
    assert 0 <= e["PB"]["collisions"]["collide_fraction"] <= 1
    assert e["PB"]["windows"]["slow_cpu"]["flagged_windows"] == 0  # same client CPU as P
    assert e["C"]["windows"]["slow_cpu"]["flagged_windows"] == 6
    assert e["A"]["windows"]["slow_cpu"] is None  # P is the reference
    md = analyze.tables(res)
    assert "PB vs A" in md and "Outcome:" in md
