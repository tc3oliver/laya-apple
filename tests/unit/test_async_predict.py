"""Unit tests for research/coreml-async-predict/ (research only): the slow-window classifier,
every branch of the decision rule, the completion handler logic of prebind_async.py with a fake
model, run_config.py's import without PyObjC or hardware, and analyze.py end to end on synthetic
runs."""

from __future__ import annotations

import gzip
import importlib.util
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "research" / "coreml-async-predict" / "scripts"
MS = 1_000_000
CONDS = ("solo_short", "solo_long", "hetero", "gpu_only")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


design = _load("async_design_for_tests", SCRIPTS / "design.py")
analyze = _load("async_analyze_for_tests", SCRIPTS / "analyze.py")
run_config = _load("async_run_config_for_tests", SCRIPTS / "run_config.py")
prebind_async = _load("async_prebind_async_for_tests", SCRIPTS / "prebind_async.py")

NORMAL, SLOW = [False] * 3, [True] * 3


# ------------------------------------------------------------------ classifier and design


def test_classifier_edges():
    assert design.SLOW_P99_MS == 13.0
    assert not design.is_slow(12.99)
    assert design.is_slow(13.0)


def test_round_orders_and_files():
    assert design.ROUND1 == ("A", "PB-SYNC", "PB-ASYNC")
    assert design.ROUND2 == ("PB-ASYNC", "PB-SYNC", "A")
    assert design.run_file("PB-ASYNC", 2) == "laya-PB-ASYNC-r2.json.gz"
    assert analyze.RUN_NAME.match("laya-PB-ASYNC-r1.json.gz")["cell"] == "PB-ASYNC"
    assert analyze.RUN_NAME.match("laya-PB-SYNC-r2.json.gz")["cell"] == "PB-SYNC"


# ------------------------------------------------------------------ decide


def rnd(r=1, a=NORMAL, sync=SLOW, asy=NORMAL):
    return {("A", r): a, ("PB-SYNC", r): sync, ("PB-ASYNC", r): asy}


GOOD = {("A", 1): 4.0, ("PB-ASYNC", 1): 0.2, ("A", 2): 4.0, ("PB-ASYNC", 2): 0.2}


def test_isolation_edges():
    assert analyze.isolated(5.0, 1.0)  # 1 ms and exactly 5x
    assert not analyze.isolated(5.0, 1.01)
    assert not analyze.isolated(4.99, 1.0)
    assert not analyze.isolated(None, 0.1) and not analyze.isolated(4.0, None)


def test_running_round_1():
    d = analyze.decide({}, {}, {})
    assert d["outcome"] is None and d["next"] == [(c, 1) for c in design.ROUND1]
    assert analyze.decide({("A", 1): NORMAL}, {}, {})["next"] == [(c, 1) for c in design.ROUND1]


def test_round_1_pattern_is_a_screen_only_and_asks_for_the_reversed_replication():
    d = analyze.decide(rnd(sync=[True, True, False]), {}, GOOD)
    assert d["outcome"] == "pattern_round1" and d["text"] == analyze.PATTERN_ROUND1
    assert "strong" not in d["text"] and "solve" not in d["text"]
    assert d["next"] == [("PB-ASYNC", 2), ("PB-SYNC", 2), ("A", 2)]
    assert d["round1"]["holds"] and all(d["round1"]["conditions"].values())


def test_round_2_reproduces_the_pattern_strong():
    s = rnd() | rnd(2, sync=[True, False, True])
    d = analyze.decide(s, {}, GOOD)
    assert d["outcome"] == "strong" and d["next"] is None
    assert d["text"] == "strong causal evidence: PB-ASYNC is the 1.5 leading candidate"


@pytest.mark.parametrize(
    "r2, mism, gpu, failed",
    [
        (rnd(2, asy=[False, True, False]), {}, GOOD, "PB_ASYNC_normal"),
        (rnd(2, sync=[True, False, False]), {}, GOOD, "PB_SYNC_slow"),
        (rnd(2, a=[True, False, False]), {}, GOOD, "A_normal"),
        (rnd(2), {("A", 2): 1}, GOOD, "no_mismatches"),
        (rnd(2), {}, {**GOOD, ("PB-ASYNC", 2): 1.2}, "PB_ASYNC_isolation"),
        (rnd(2), {}, {**GOOD, ("A", 2): 0.9}, "PB_ASYNC_isolation"),  # A/PB-ASYNC 4.5x in round 2
    ],
)
def test_round_2_alone_must_reproduce_every_condition(r2, mism, gpu, failed):
    d = analyze.decide(rnd() | r2, mism, gpu)
    assert d["outcome"] == "round2_not_reproduced" and d["next"] is None
    assert d["text"].startswith("INCONCLUSIVE: round 2 did not reproduce the pattern")
    assert d["round2_pattern"]["conditions"][failed] is False


def test_round_1_results_do_not_rescue_round_2():
    # round 2 is judged alone: a slow PB-SYNC round 1 does not make up for a normal round 2
    d = analyze.decide(rnd(sync=SLOW) | rnd(2, sync=NORMAL), {}, GOOD)
    assert d["outcome"] == "round2_not_reproduced"


def test_never_a_third_round():
    s = rnd() | rnd(2) | {(c, 3): NORMAL for c in design.CELLS}
    assert analyze.decide(s, {}, GOOD)["next"] is None


def test_negative_control_slow():
    d = analyze.decide(rnd(a=[False, False, True]), {}, GOOD)
    assert d["outcome"] == "negative_control_slow" and d["next"] is None
    assert d["text"] == "INCONCLUSIVE: negative control slow, invalid for causal interpretation"


@pytest.mark.parametrize("sync", [NORMAL, [True, False, False]])
def test_positive_control_not_reproduced(sync):
    d = analyze.decide(rnd(sync=sync), {}, GOOD)
    assert d["outcome"] == "positive_control_not_reproduced" and d["next"] is None
    assert d["text"] == "INCONCLUSIVE: positive control not reproduced"


def test_async_slow():
    d = analyze.decide(rnd(asy=[True, True, False]), {}, GOOD)
    assert d["outcome"] == "async_slow" and d["next"] is None
    assert d["text"] == "async does not solve the slow state: resume #89"


@pytest.mark.parametrize(
    "slow, mism, gpu",
    [
        (rnd(asy=[False, True, False]), {}, GOOD),  # PB-ASYNC 1 of 3
        (rnd(), {}, {**GOOD, ("PB-ASYNC", 1): 0.9}),  # normal but isolation lost (4.0 / 0.9 < 5)
        (rnd(), {("PB-SYNC", 1): 2}, GOOD),  # a mismatch
        (rnd(), {("A", 1): 1}, GOOD),  # a mismatch in the negative control's run
    ],
)
def test_mixed(slow, mism, gpu):
    d = analyze.decide(slow, mism, gpu)
    assert d["outcome"] == "mixed" and d["next"] is None and d["text"] == "mixed: not interpreted; human decision"


# ------------------------------------------------------------------ prebind_async.Completion


class _Labels(dict):
    def __missing__(self, k):
        return 0


def _fresh():
    return prebind_async.Completion(stamps=[], callbacks=[], labels=_Labels(), anomalies=[])


class FakeModel:
    """Calls the handler on another thread after `delay`, `times` times (0 = never)."""

    def __init__(self, delay=0.001, times=1, error=None):
        self.delay, self.times, self.error, self.threads = delay, times, error, []

    def send(self, handler):
        def fire():
            for _ in range(self.times):
                time.sleep(self.delay)
                handler(object(), self.error)

        t = threading.Thread(target=fire)
        self.threads.append(t)
        t.start()


def test_one_callback_per_submit_and_event_reuse():
    c, m = _fresh(), FakeModel()
    for _ in range(3):
        sb, sa, cb, wake = c.submit(m.send, timeout=2)
        assert sb <= sa and sb <= cb <= wake
    for t in m.threads:
        t.join()
    assert c.submits == c.callbacks == 3 and len(c._stamps) == 3 and c.state == "idle"
    assert not c._anomalies and all(e == 0 for _, _, e in c._cb)
    assert {tid for _, tid, _ in c._cb} and threading.get_native_id() not in {tid for _, tid, _ in c._cb}
    assert c.output is None  # the handler does not keep the provider


def test_handler_names_its_thread_and_records_the_label():
    c, m = _fresh(), FakeModel()
    names = []
    orig = c._handler

    def spy(o, e):
        orig(o, e)
        names.append(threading.current_thread().name)

    c.handler = spy
    c.submit(m.send, timeout=2)
    m.threads[0].join()
    assert names == [prebind_async.CALLBACK_THREAD_NAME]
    assert sum(c._labels.values()) == 1


def test_duplicate_callback_is_recorded():
    c, m = _fresh(), FakeModel(times=2)
    c.submit(m.send, timeout=2)
    m.threads[0].join()
    assert c.callbacks == 2 and c.submits == 1
    assert [a["kind"] for a in c._anomalies] == ["duplicate"]


def test_timeout_raises_then_late_callback_is_recorded_and_bucket_refuses():
    c, m = _fresh(), FakeModel(delay=0.2)
    with pytest.raises(TimeoutError):
        c.submit(m.send, timeout=0.02)
    m.threads[0].join()
    assert [a["kind"] for a in c._anomalies] == ["late"]
    with pytest.raises(RuntimeError, match="timed out"):
        c.submit(m.send, timeout=1)


def test_submit_before_completion_is_refused():
    c = _fresh()
    c.arm()
    with pytest.raises(RuntimeError, match="before the previous one completed"):
        c.arm()


def test_error_from_core_ml_raises():
    c, m = _fresh(), FakeModel(error="boom")
    with pytest.raises(RuntimeError, match="boom"):
        c.submit(m.send, timeout=2)
    m.threads[0].join()
    assert [e for _, _, e in c._cb] == [1] and c.state == "idle"


def test_keep_output_only_when_asked():
    c, m = _fresh(), FakeModel()
    c.keep_output = True
    c.submit(m.send, timeout=2)
    m.threads[0].join()
    assert c.output is not None


# ------------------------------------------------------------------ run_config.py


def test_run_config_and_prebind_async_import_without_pyobjc():
    assert list(run_config.CELLS) == list(design.CELLS)
    assert run_config.CELLS["PB-ASYNC"]["ane_predict"] == "prebind_async"
    assert run_config.CELLS["PB-SYNC"]["ane_predict"] == "prebind"
    for script in ("run_config.py", "prebind_async.py"):
        code = (
            "import sys, importlib.util as u; sys.modules['objc'] = None; sys.modules['CoreML'] = None; "
            "sys.modules['Foundation'] = None; "
            f"s = u.spec_from_file_location('m', {str(SCRIPTS / script)!r}); m = u.module_from_spec(s); "
            "s.loader.exec_module(m); print('ok')"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
        assert "ok" in out


def test_stage_definitions():
    f = (0, 20 * MS)
    sync = (1 * MS, 2 * MS, 3 * MS, 12 * MS, 13 * MS, 14 * MS)
    s = analyze.stages("prebind", f, sync)
    assert s == {
        "features": MS,
        "pre": MS,
        "submit": None,
        "completion": 9 * MS,
        "handoff": MS,
        "post": MS,
        "tail": 6 * MS,
        "predict": 13 * MS,
    }
    asy = (1 * MS, 2 * MS, 2 * MS + 10_000, 11 * MS, 11 * MS + 20_000, 12 * MS)
    s = analyze.stages("prebind_async", f, asy)
    assert s["pre"] == MS and s["submit"] == 10_000 and s["completion"] == 9 * MS
    assert s["handoff"] == 20_000 and s["post"] == MS - 20_000
    s = analyze.stages("coremltools", f, (MS, 0, 0, 0, 0, 12 * MS))
    assert s["completion"] is None and s["predict"] == 11 * MS


# ------------------------------------------------------------------ synthetic runs


def _fake_run(cell, short_p99s, *, mismatches=0, gpu_us=None):
    """R1's layout: all four conditions in bench_concurrency's alternating order, 3 cycles."""
    windows_t, part_windows, res_windows = [], [], []
    trace = {c: [] for c in run_config.TRACE_COLUMNS}
    forwards_ane, predicts, gpu_recv, cbs = [], [], [], []
    t, rid = 1_000 * MS, 0
    order = [(k, c) for k in range(3) for c in (CONDS if k % 2 == 0 else tuple(reversed(CONDS)))]
    for cycle, cond in order:
        lo, hi = t, t + 200 * MS
        t = hi + 300 * MS
        streams = {"solo_short": ["short"], "solo_long": ["long"]}.get(cond, ["short", "long"])
        inst = "gpu" if cond == "gpu_only" else "auto"
        het = cond == "hetero"
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
                if cell == "A":
                    p = (s0 + MS, 0, 0, 0, 0, s0 + 8 * MS + 50)
                elif cell == "PB-SYNC":
                    p = (s0 + MS, s0 + 2 * MS, s0 + 2 * MS + 1, s0 + 8 * MS, s0 + 8 * MS + 10, s0 + 8 * MS + 50)
                else:
                    p = (
                        s0 + MS,
                        s0 + 2 * MS,
                        s0 + 2 * MS + 5_000,
                        s0 + 8 * MS,
                        s0 + 8 * MS + 10_000,
                        s0 + 8 * MS + 50_000,
                    )
                    if het:
                        cbs.append([s0 + 8 * MS, 777, 0])
                predicts.append(p)
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
            after = {"client-long": 90 * MS, "coreml-callback": 40 * MS, "other": 10 * MS}
            res_windows.append(
                {
                    "start_ns": lo,
                    "end_ns": hi,
                    "instance": "auto",
                    "streams": {"short": 20, "long": 5},
                    "before": {"t_ns": lo - 100 * MS, "threads": {"client-short": 0, "client-long": 0}},
                    "after": {"t_ns": hi + 100 * MS, "threads": after},
                    "after_by_stream": {
                        "short": {"t_ns": hi + 50 * MS, "threads": {"client-short": 300 * MS}},
                        "long": {"t_ns": hi + 100 * MS, "threads": after},
                    },
                }
            )
    cfg = run_config.CELLS[cell]
    if gpu_us is None:
        gpu_us = 5000 if cell == "A" else 50
    return {
        "args": {"short": 128, "long": 512, "seconds": 20.0, "cycles": 3},
        "part_a": {"windows": part_windows},
        "windows_t": windows_t,
        "gpu_return": {"received_ns": gpu_recv, "return_us": [gpu_us] * len(gpu_recv)},
        "research": {
            "experiment": "coreml-async-predict",
            "cell": cell,
            **cfg,
            "run_wall_s": 280.0,
            "laya_apple": "x",
            "pyobjc": "x",
            "coremltools": "x",
            "mlx": "x",
            "workers": {"ane": {"placement": "thread"}, "gpu": {"placement": "process"}},
            "forwards": {"ane": forwards_ane, "gpu": [(g - 40 * MS, g - 50_000, 2 * MS) for g in gpu_recv]},
            "predicts_columns": run_config.PREDICT_COLUMNS[cfg["ane_predict"]],
            "predicts": predicts,
            "crossings": {"128": {"objc_send": 1, "ctypes": 0, "pool_push": 2, "pool_pop": 1, "total": 4}},
            "callbacks_per_forward_at_load": 1.0 if cell == "PB-ASYNC" else None,
            "backings": None if cell == "A" else {"128": {"modes": {"cls": "backed", "logits": "backed"}}},
            "callbacks": None
            if cell != "PB-ASYNC"
            else {
                "columns": run_config.CALLBACK_COLUMNS,
                "hetero": cbs,
                "total": 100,
                "submits": 100,
                "errors": 0,
                "thread_ids": [777],
                "queue_labels": {"com.apple.root.default-qos.overcommit": 100},
                "anomalies": [],
            },
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
    (out / "results.json").write_text("{}\n")
    stale = subprocess.run([sys.executable, script, "--raw", raw, "--out", out, "--check"], capture_output=True)
    assert stale.returncode != 0 and b"stale" in stale.stderr


def test_design_next_prints_cell_round_lines(raw, monkeypatch, capsys):
    monkeypatch.setattr(analyze, "RAW", raw)
    monkeypatch.setitem(sys.modules, "analyze", analyze)
    monkeypatch.setattr(sys, "argv", ["design.py", "next"])
    design.main()
    assert capsys.readouterr().out.splitlines() == [f"{c} 1" for c in design.ROUND1]
    _write(raw, "A", 1, N3)
    _write(raw, "PB-SYNC", 1, S3)
    _write(raw, "PB-ASYNC", 1, N3)
    design.main()
    assert capsys.readouterr().out.splitlines() == ["PB-ASYNC 2", "PB-SYNC 2", "A 2"]


def test_end_to_end_strong(raw):
    _write(raw, "A", 1, N3)
    _write(raw, "PB-SYNC", 1, S3)
    _write(raw, "PB-ASYNC", 1, N3)
    res = analyze.summarise(raw)
    assert res["decision"]["outcome"] == "pattern_round1" and res["decision"]["round1"]["holds"]
    assert res["runs"]["laya-A-r1"]["gpu_return_p50_ms"] == 5.0
    for c, p in (("PB-ASYNC", N3), ("PB-SYNC", [14.0, 11.0, 13.0]), ("A", N3)):
        _write(raw, c, 2, p)
    res = analyze.summarise(raw)
    assert res["next"] is None and res["decision"]["outcome"] == "strong", res["outcome"]
    assert res["unused_runs"] == []
    w = res["runs"]["laya-PB-ASYNC-r1"]["windows"][0]
    assert w["slow"] is False and w["aggregate_req_s"] == 125.0 and w["gpu_return_p50_ms"] == 0.05
    assert w["thread_cpu_ms"] == {"client-short": 300.0, "client-long": 90.0, "coreml-callback": 40.0, "other": 10.0}
    assert w["callbacks"]["n"] == 8 and w["callbacks"]["threads"] == 1
    st = w["ane_stages_p50_ms"]
    assert st["submit"] == pytest.approx(0.005) and st["completion"] == pytest.approx(6.0)
    assert st["handoff"] == pytest.approx(0.01) and st["pre"] == pytest.approx(1.0)
    ws = res["runs"]["laya-PB-SYNC-r1"]["windows"][0]
    assert ws["callbacks"] is None and ws["thread_cpu_ms"]["coreml-callback"] is None
    assert ws["ane_stages_p50_ms"]["submit"] is None and ws["ane_stages_p50_ms"]["completion"] == pytest.approx(6.0)
    assert res["runs"]["laya-A-r1"]["windows"][0]["ane_stages_p50_ms"]["completion"] is None
    r = res["runs"]["laya-PB-ASYNC-r1"]
    assert r["callbacks"]["submits"] == r["callbacks"]["callbacks"] == 100 and r["backing_modes"] == ["backed"]
    assert r["hetero_predicts_with_binding_stamps"] == 24
    assert res["cells"]["PB-SYNC"]["slow_windows"] == 5 and res["cells"]["PB-SYNC"]["windows"] == 6
    md = analyze.tables(res)
    assert analyze.STRONG in md and "laya-PB-ASYNC-r2" in md
    json.dumps(res)


def test_end_to_end_isolation_lost_is_mixed(raw):
    _write(raw, "A", 1, N3)
    _write(raw, "PB-SYNC", 1, S3)
    _write(raw, "PB-ASYNC", 1, N3, gpu_us=2000)
    res = analyze.summarise(raw)
    assert res["decision"]["outcome"] == "mixed" and res["next"] is None


def test_second_crash_invalidates(raw):
    (raw / "failed").mkdir()
    for k in (1, 2):
        (raw / "failed" / f"laya-PB-ASYNC-r1.{k}.log").write_text("x")
    res = analyze.summarise(raw)
    assert res["next"] is None and res["outcome"].startswith("invalid: second crash of laya-PB-ASYNC-r1")


def test_runs_with_other_window_lengths_are_rejected(raw):
    run = _fake_run("A", N3)
    run["args"]["seconds"] = 2.0
    with gzip.open(raw / design.run_file("A", 1), "wt") as fh:
        json.dump(run, fh)
    with pytest.raises(AssertionError):
        analyze.summarise(raw)


def test_run_config_hetero_bounds_match_the_analysis():
    run = _fake_run("PB-ASYNC", N3)
    assert run_config.hetero_bounds(run) == analyze.gate57.hetero_bounds(run)
    assert len(run_config.hetero_bounds(run)) == 3


def test_phase0_record_is_rendered_with_the_stamped_label(raw):
    p0 = {
        "gates": {"1_callable": True, "8_gil": False},
        "stamped": {
            "label": "semantics check, not performance evidence",
            "built": True,
            "native_completion_to_callback_entry": {"n": 200, "p50_ms": 0.003, "p99_ms": 0.01},
        },
    }
    (raw / "phase0.json").write_text(json.dumps(p0))
    md = analyze.tables(analyze.summarise(raw))
    assert "1_callable PASS" in md and "8_gil **FAIL**" in md
    assert "(semantics check, not performance evidence)" in md
