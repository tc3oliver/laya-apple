"""Unit tests for research/coreml-slow-state-trigger/ (research only): the slow-window classifier,
every branch of the decision rule, PB-W's window-order counter, PB-H's GIL-holding wrapper,
run_config.py's import without PyObjC or hardware, and analyze.py end to end on synthetic runs."""

from __future__ import annotations

import ctypes
import gzip
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "research" / "coreml-slow-state-trigger" / "scripts"
MS = 1_000_000
CONDS = ("solo_short", "solo_long", "hetero", "gpu_only")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sys.path.insert(0, str(SCRIPTS))
design = _load("sst_design_for_tests", SCRIPTS / "design.py")
analyze = _load("sst_analyze_for_tests", SCRIPTS / "analyze.py")
run_config = _load("sst_run_config_for_tests", SCRIPTS / "run_config.py")

NORMAL, SLOW = [False] * 3, [True] * 3


# ------------------------------------------------------------------ classifier


def test_classifier_edges():
    assert design.SLOW_P99_MS == 13.0
    assert not design.is_slow(12.99)
    assert design.is_slow(13.0)
    assert design.is_slow(21.2) and not design.is_slow(10.4)


def test_round_one_order_and_files():
    assert design.ROUND1 == ("A", "PB-R", "PB-H", "PB-W", "PB-P")
    assert design.run_file("PB-H", 2) == "laya-PB-H-r2.json.gz"
    assert (design.SHORT, design.LONG, design.CYCLES, design.SECONDS) == (128, 512, 3, 20.0)


# ------------------------------------------------------------------ decide


def r1(**cells):
    """Round-1 slow flags: every cell normal unless given."""
    return {(c, 1): cells.get(c.replace("-", "_"), NORMAL) for c in design.ROUND1}


def test_running_round_1_asks_for_every_cell_in_order():
    d = analyze.decide({})
    assert d["outcome"] is None and d["next"] == [(c, 1) for c in design.ROUND1]
    partial = {("A", 1): NORMAL, ("PB-R", 1): SLOW}
    assert analyze.decide(partial)["next"] == [(c, 1) for c in design.ROUND1]


def test_negative_control_slow_stops():
    d = analyze.decide(r1(A=[False, False, True], PB_R=SLOW))
    assert d["outcome"] == "negative_control_slow" and d["next"] is None
    assert d["text"] == "negative control slow: no causal conclusion"


def test_positive_control_rerun_then_outcome_5():
    s = r1(PB_R=[True, False, False])  # 1 of 3: not slow
    d = analyze.decide(s)
    assert d["outcome"] is None and d["next"] == [("PB-R", 2)]
    s[("PB-R", 2)] = [False, True, False]
    d = analyze.decide(s)
    assert d["outcome"] == 5 and d["next"] is None
    assert d["text"] == "positive control not reproduced: no causal conclusion"


def test_positive_control_slow_only_in_rerun_stops_for_a_human():
    s = r1(PB_R=NORMAL)
    s[("PB-R", 2)] = [True, True, False]
    d = analyze.decide(s)
    assert d["outcome"] == "positive_control_rerun_slow" and d["next"] is None
    assert "human decision" in d["text"]


def test_candidates_go_to_round_2_in_fixed_order():
    s = r1(PB_R=[True, True, False], PB_H=NORMAL, PB_W=SLOW, PB_P=NORMAL)
    d = analyze.decide(s)
    assert d["candidates"] == ["PB-H", "PB-P"]
    assert d["next"] == [("A", 2), ("PB-R", 2), ("PB-H", 2), ("PB-P", 2)]
    assert d["outcome"] is None


@pytest.mark.parametrize("cand, oid", [("PB-H", 1), ("PB-W", 2), ("PB-P", 3)])
def test_round_2_confirms_outcomes_1_to_3(cand, oid):
    others = {c: SLOW for c in design.CANDIDATES if c != cand}
    s = {**r1(PB_R=SLOW), **{(c, 1): v for c, v in others.items()}}
    s[(cand, 1)] = NORMAL
    s.update({("A", 2): NORMAL, ("PB-R", 2): [True, False, True], (cand, 2): NORMAL})
    d = analyze.decide(s)
    assert d["outcome"] == [oid] and d["confirmed"] == [cand] and d["next"] is None
    assert d["interpretations"][cand] == analyze.OUTCOMES[oid]
    assert "PB-R 5 of 6 slow" in d["text"]


def test_interpretation_texts():
    assert analyze.OUTCOMES[1].startswith("GIL release is one necessary condition")
    assert analyze.OUTCOMES[2].startswith("hetero transition / scheduler residency state is key")
    assert analyze.OUTCOMES[3].endswith("the probe is not a production fix")
    assert analyze.OUTCOMES[4] == "stop tuning the Python thread path; next: native/Swift worker architecture research"


def test_round_2_not_confirmed_branches():
    base = r1(PB_R=[True, True, False], PB_H=NORMAL, PB_W=SLOW, PB_P=SLOW)
    ok = {("A", 2): NORMAL, ("PB-R", 2): [True, True, False], ("PB-H", 2): NORMAL}  # PB-R 4 of 6
    assert analyze.decide({**base, **ok})["outcome"] == [1]
    cases = [
        {("PB-H", 2): [False, False, True]},  # candidate slow once in round 2
        {("PB-R", 2): [True, False, False]},  # PB-R 3 of 6
        {("A", 2): [False, True, False]},  # A slow in round 2
    ]
    for c in cases:
        d = analyze.decide({**base, **ok, **c})
        assert d["outcome"] == "not_confirmed" and d["confirmed"] == [] and d["next"] is None
        assert "PB-H not confirmed" in d["text"] and not d["interpretations"]
        assert d["text"].endswith(analyze.PARTIAL)


def test_round_2_mixed_confirmation():
    s = r1(PB_R=SLOW, PB_H=NORMAL, PB_W=NORMAL, PB_P=SLOW)
    s.update({("A", 2): NORMAL, ("PB-R", 2): SLOW, ("PB-H", 2): NORMAL, ("PB-W", 2): [True, False, False]})
    d = analyze.decide(s)
    assert d["outcome"] == [1] and d["confirmed"] == ["PB-H"]
    assert "PB-H confirmed; PB-W not confirmed" in d["text"]


def test_outcome_4_and_partial():
    d = analyze.decide(r1(PB_R=SLOW, PB_H=[True, True, False], PB_W=SLOW, PB_P=[False, True, True]))
    assert d["outcome"] == 4 and d["next"] is None
    d = analyze.decide(r1(PB_R=SLOW, PB_H=[True, False, False], PB_W=SLOW, PB_P=SLOW))  # PB-H 1 of 3
    assert d["outcome"] == "partial" and d["next"] is None
    assert d["text"] == "partial: reported, no interpretation; next step by human decision"


def test_never_a_third_round():
    s = r1(PB_R=SLOW, PB_H=NORMAL, PB_W=SLOW, PB_P=SLOW)
    s.update({(c, 2): NORMAL for c in design.CELLS})
    s.update({(c, 3): NORMAL for c in design.CELLS})
    assert analyze.decide(s)["next"] is None


# ------------------------------------------------------------------ run_config.py


def test_run_config_imports_without_pyobjc():
    assert list(run_config.CELLS) == list(design.CELLS)
    assert run_config.CELLS["A"]["config"] == "A"
    assert {run_config.CELLS[c]["config"] for c in design.CELLS[1:]} == {"PB"}
    assert [c for c, v in run_config.CELLS.items() if v["warmup"]] == ["PB-W"]
    assert [c for c, v in run_config.CELLS.items() if v["gil_probe"]] == ["PB-P"]
    assert [c for c, v in run_config.CELLS.items() if v["shim_gil"] == "held"] == ["PB-H"]
    assert run_config.WARMUP_S == 2.0
    code = (
        "import sys, importlib.util as u; sys.modules['objc'] = None; sys.modules['CoreML'] = None; "
        f"s = u.spec_from_file_location('rc', {str(SCRIPTS / 'run_config.py')!r}); m = u.module_from_spec(s); "
        "s.loader.exec_module(m); print(sorted(m.CELLS))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert "PB-H" in out


def test_window_order_matches_bench_concurrency_part_a():
    order = run_config.window_order(3)
    assert len(order) == 12
    assert [c for k, c in order if k == 0] == list(CONDS)
    assert [c for k, c in order if k == 1] == list(reversed(CONDS))
    hetero = [i for i, (_, c) in enumerate(order) if c == "hetero"]
    assert hetero == [2, 5, 10]  # one per cycle: 3rd of cycle 0, 2nd of cycle 1, 3rd of cycle 2
    assert [order[i][0] for i in hetero] == [0, 1, 2]


def test_window_counter_names_the_upcoming_window():
    c = run_config.WindowCounter(2)
    seen = [c.next() for _ in range(8)]
    assert [x for x in seen if x[1] == "hetero"] == [(0, "hetero"), (1, "hetero")]
    assert seen[:3] == [(0, "solo_short"), (0, "solo_long"), (0, "hetero")]
    assert seen[4:6] == [(1, "gpu_only"), (1, "hetero")]
    with pytest.raises(RuntimeError):
        c.next()


def test_pbh_wrapper_holds_the_gil_at_the_same_address():
    lib = ctypes.CDLL(None)
    fn = lib.labs
    fn.restype, fn.argtypes = ctypes.c_long, [ctypes.c_long]
    assert not fn._flags_ & ctypes._FUNCFLAG_PYTHONAPI  # CDLL: the GIL is released
    held = run_config.gil_holding(fn)
    assert held._flags_ & ctypes._FUNCFLAG_PYTHONAPI  # PYFUNCTYPE: the GIL is kept
    assert ctypes.cast(held, ctypes.c_void_p).value == ctypes.cast(fn, ctypes.c_void_p).value
    assert held.restype is fn.restype and list(held.argtypes) == list(fn.argtypes)
    assert held(-7) == 7


def test_hold_gil_replaces_every_bucket_call():
    lib = ctypes.CDLL(None)
    fn = lib.labs
    fn.restype, fn.argtypes = ctypes.c_long, [ctypes.c_long]

    class M:
        _call = fn

    class Backend:
        models = {64: M(), 128: M()}

    be = Backend()
    run_config.hold_gil(be, fn)
    assert all(m._call._flags_ & ctypes._FUNCFLAG_PYTHONAPI for m in be.models.values())
    assert all(m._call(-3) == 3 for m in be.models.values())
    other = type("Other", (), {"_call": lambda *a: None})()
    with pytest.raises(AssertionError):
        run_config.hold_gil(type("B", (), {"models": {1: other}})(), fn)


# ------------------------------------------------------------------ synthetic runs


def _fake_run(cell, short_p99s, *, mismatches=0):
    """R1's layout: all four conditions in bench_concurrency's alternating order, 3 cycles."""
    windows_t, part_windows, res_windows = [], [], []
    trace = {c: [] for c in run_config.TRACE_COLUMNS}
    forwards_ane, predicts, gpu_recv, warmups = [], [], [], []
    native = cell != "A"
    t, rid = 1_000 * MS, 0
    for cycle, cond in run_config.window_order(3):
        lo, hi = t, t + 200 * MS
        t = hi + 300 * MS
        streams = {"solo_short": ["short"], "solo_long": ["long"]}.get(cond, ["short", "long"])
        inst = "gpu" if cond == "gpu_only" else "auto"
        het = cond == "hetero"
        if het and cell == "PB-W":
            warmups.append(
                {
                    "cycle": cycle,
                    "before": "hetero",
                    "start_ns": lo - 250 * MS,
                    "end_ns": lo - 50 * MS,
                    "streams": {s: {"n": 10, "devices": {}, "mismatches": 0} for s in ("short", "long")},
                }
            )
        for s in streams:
            windows_t.append({"stream": s, "instance": inst, "start_ns": lo, "end_ns": hi})
        pw = {"cycle": cycle, "condition": cond, "streams": {}}
        for s in streams:
            dev = "gpu" if inst == "gpu" or s == "long" else "ane"
            p99 = (short_p99s[cycle] if het else 10.0) if s == "short" else 40.0
            pw["streams"][s] = {"req_s": 100.0 if s == "short" else 25.0, "p99_ms": p99, "p50_ms": 9.0}
            bad = mismatches if het and s == "short" and cycle == 0 else 0
            pw["streams"][s].update(devices={dev: 20}, mismatches=bad, latency_ms=[1.0, 2.0])
        part_windows.append(pw)
        if inst == "gpu":
            continue
        if "short" in streams:
            for k in range(8):
                s0 = lo + k * 20 * MS
                forwards_ane.append((s0, s0 + 9 * MS, 600_000))
                p = (s0 + MS, s0 + 2 * MS, s0 + 2 * MS + 1, s0 + 8 * MS, s0 + 8 * MS + 10, s0 + 8 * MS + 50)
                predicts.append(p if native else (p[0], 0, 0, 0, 0, p[5]))
                if het:
                    rid += 1
                    row = (rid, "ane", 128, s0 - MS, s0 - MS, s0 - MS, s0 - MS, s0, s0, s0 + 9 * MS)
                    for c, v in zip(trace, (*row, s0 + 9 * MS, s0 - MS + (9 + k % 5) * MS)):
                        trace[c].append(v)
        if "long" in streams:
            for k in range(3):
                g0 = lo + k * 50 * MS
                recv = g0 + 40 * MS
                gpu_recv.append(recv)
                if het:
                    rid += 1
                    row = (rid, "gpu", 512, g0, g0, g0, g0, g0, g0, recv - 50_000)
                    for c, v in zip(trace, (*row, recv, recv + 200_000)):
                        trace[c].append(v)
        if het:
            res_windows.append(
                {
                    "start_ns": lo,
                    "end_ns": hi,
                    "instance": "auto",
                    "streams": {"short": 20, "long": 5},
                    "before": {"t_ns": lo - 100 * MS, "threads": {"client-short": 0, "client-long": 0}},
                    # R1's closing snapshot: the short client has already exited
                    "after": {"t_ns": hi + 100 * MS, "threads": {"client-long": 90 * MS}},
                    "after_by_stream": {
                        "short": {"t_ns": hi + 50 * MS, "threads": {"client-short": 300 * MS, "client-long": 80 * MS}},
                        "long": {"t_ns": hi + 100 * MS, "threads": {"client-long": 90 * MS}},
                    },
                    **(
                        {"gil_probe": {"n": 200, "p50_ms": 0.1, "p99_ms": 0.9, "mean_ms": 0.2, "over_1ms_fraction": 0}}
                        if cell == "PB-P"
                        else {}
                    ),
                }
            )
    return {
        "args": {"short": 128, "long": 512, "seconds": 20.0, "cycles": 3},
        "part_a": {"windows": part_windows},
        "windows_t": windows_t,
        "gpu_return": {"received_ns": gpu_recv, "return_us": [5000 if cell == "A" else 50] * len(gpu_recv)},
        "research": {
            "experiment": "coreml-slow-state-trigger",
            "cell": cell,
            **run_config.CELLS[cell],
            "run_wall_s": 280.0,
            "laya_apple": "x",
            "pyobjc": "x",
            "coremltools": "x",
            "mlx": "x",
            "gil_probe": {"ticks": 1000} if cell == "PB-P" else None,
            "warmups": warmups if cell == "PB-W" else None,
            "shim_flags": None if cell == "A" else {"128": 5 if cell == "PB-H" else 1},
            "workers": {"ane": {"placement": "thread"}, "gpu": {"placement": "process"}},
            "forwards": {"ane": forwards_ane, "gpu": [(g - 40 * MS, g - 50_000, 2 * MS) for g in gpu_recv]},
            "predicts": predicts,
            "crossings": {"128": {"objc_send": 0, "ctypes": 1, "pool_push": 2, "pool_pop": 1, "total": 4}},
            "backings": None,
            "trace": trace,
            "windows": res_windows,
        },
    }


def _write(raw: Path, cell: str, rnd: int, p99s, **kw):
    with gzip.open(raw / design.run_file(cell, rnd), "wt") as fh:
        json.dump(_fake_run(cell, p99s, **kw), fh)


N3, S3 = [11.0, 12.99, 10.5], [14.0, 13.0, 20.0]


@pytest.fixture
def raw(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    return d


def test_empty_raw_asks_for_round_1_and_renders(raw, tmp_path):
    res = analyze.summarise(raw)
    assert res["next"] == [[c, 1] for c in design.ROUND1] and res["outcome"] == "running: round 1"
    assert res["runs"] == {} and res["cells"] == {}
    assert "not run yet" in analyze.tables(res)
    out = tmp_path / "out"
    out.mkdir()
    script = SCRIPTS / "analyze.py"
    subprocess.run([sys.executable, script, "--raw", raw, "--out", out], check=True, capture_output=True)
    ok = subprocess.run([sys.executable, script, "--raw", raw, "--out", out, "--check"], capture_output=True)
    assert ok.returncode == 0
    (out / "tables.md").write_text("edited\n")
    stale = subprocess.run([sys.executable, script, "--raw", raw, "--out", out, "--check"], capture_output=True)
    assert stale.returncode != 0 and b"stale" in stale.stderr


def test_design_next_prints_cell_round_lines(raw, monkeypatch, capsys):
    monkeypatch.setattr(analyze, "RAW", raw)
    monkeypatch.setitem(sys.modules, "analyze", analyze)
    monkeypatch.setattr(sys, "argv", ["design.py", "next"])
    design.main()
    assert capsys.readouterr().out.splitlines() == [f"{c} 1" for c in design.ROUND1]
    _write(raw, "A", 1, N3)
    _write(raw, "PB-R", 1, S3)
    for c in design.CANDIDATES:
        _write(raw, c, 1, S3)
    design.main()
    assert capsys.readouterr().out == ""  # outcome 4: nothing to run


def test_end_to_end_round_2_confirmation(raw):
    _write(raw, "A", 1, N3)
    _write(raw, "PB-R", 1, S3)
    _write(raw, "PB-H", 1, N3)
    _write(raw, "PB-W", 1, [14.0, 11.0, 15.0])
    _write(raw, "PB-P", 1, N3, mismatches=2)
    res = analyze.summarise(raw)
    assert res["next"] == [["A", 2], ["PB-R", 2], ["PB-H", 2], ["PB-P", 2]]
    assert res["decision"]["round1_slow"] == {"A": 0, "PB-R": 3, "PB-H": 0, "PB-W": 2, "PB-P": 0}
    for c, p in (("A", N3), ("PB-R", [14.0, 11.0, 13.5]), ("PB-H", N3), ("PB-P", [13.2, 11.0, 11.0])):
        _write(raw, c, 2, p)
    _write(raw, "PB-W", 2, N3)  # not in the round-2 plan
    res = analyze.summarise(raw)
    assert res["next"] is None and res["decision"]["confirmed"] == ["PB-H"]
    assert res["decision"]["outcome"] == [1]
    assert res["outcome"] == "round 2: PB-H confirmed; PB-P not confirmed (PB-R 5 of 6 slow, A 0 of 6 slow)"
    assert res["unused_runs"] == ["laya-PB-W-r2.json.gz"]
    w = res["runs"]["laya-PB-R-r1"]["windows"]
    assert [x["slow"] for x in w] == [True, True, True] and [x["cycle"] for x in w] == [0, 1, 2]
    assert w[0]["aggregate_req_s"] == 125.0 and w[0]["client_cpu_ms"] == {"short": 300.0, "long": 90.0}
    assert w[0]["thread_cpu_per_forward_ms"]["ane"] == pytest.approx(0.6)
    assert w[0]["ane_stages_p50_ms"]["native"] == pytest.approx(6.0, abs=1e-5)
    assert res["runs"]["laya-A-r1"]["windows"][0]["ane_stages_p50_ms"]["native"] is None
    p = res["runs"]["laya-PB-P-r1"]
    assert p["mismatches_all_windows"] == 2 and p["gil_probe_ticks"] == 1000
    assert p["windows"][0]["gil_probe"]["p50_ms"] == 0.1
    wu = res["runs"]["laya-PB-W-r1"]["warmups"]
    assert [x["cycle"] for x in wu] == [0, 1, 2]
    assert res["runs"]["laya-PB-R-r1"]["gpu_return_p50_ms"] == pytest.approx(0.05)
    assert res["runs"]["laya-PB-H-r1"]["shim_flags"] == {"128": 5}
    assert res["runs"]["laya-PB-R-r1"]["hetero_predicts_with_native_stamps"] == 24
    assert res["cells"]["PB-R"]["slow_windows"] == 5 and res["cells"]["PB-R"]["windows"] == 6
    md = analyze.tables(res)
    assert "PB-H: GIL release is one necessary condition" in md and "laya-PB-W-r2.json.gz" in md
    json.dumps(res)


def test_client_cpu_falls_back_to_r1_closing_snapshot():
    run = _fake_run("PB-R", N3)
    for w in run["research"]["windows"]:
        del w["after_by_stream"]
    w = analyze.window_records(run)[0]
    assert w["client_cpu_ms"] == {"short": None, "long": 90.0}


def test_second_crash_invalidates(raw):
    (raw / "failed").mkdir()
    for k in (1, 2):
        (raw / "failed" / f"laya-PB-H-r1.{k}.log").write_text("x")
    res = analyze.summarise(raw)
    assert res["next"] is None and res["outcome"].startswith("invalid: second crash of laya-PB-H-r1")
    assert res["failed_runs"] == ["laya-PB-H-r1.1.log", "laya-PB-H-r1.2.log"]


def test_runs_with_other_window_lengths_are_rejected(raw):
    run = _fake_run("A", N3)
    run["args"]["seconds"] = 2.0
    with gzip.open(raw / design.run_file("A", 1), "wt") as fh:
        json.dump(run, fh)
    with pytest.raises(AssertionError):
        analyze.summarise(raw)


def test_run_config_hetero_bounds_match_the_analysis():
    run = _fake_run("PB-R", N3)
    assert run_config.hetero_bounds(run) == analyze.gate57.hetero_bounds(run)
    assert len(run_config.hetero_bounds(run)) == 3
