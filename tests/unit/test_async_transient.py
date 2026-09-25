"""Unit tests for research/coreml-async-transient/ (research only): the run order, the window-order
hook, the per-thread counter arithmetic, the transient metrics, the A validity guard, every
evaluator and reading, run_config.py's import without PyObjC, and analyze.py end to end on
synthetic runs."""

from __future__ import annotations

import gzip
import importlib.util
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "research" / "coreml-async-transient" / "scripts"
S = 1_000_000_000
MS = 1_000_000


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


design = _load("transient_design_for_tests", SCRIPTS / "design.py")
analyze = _load("transient_analyze_for_tests", SCRIPTS / "analyze.py")
run_config = _load("transient_run_config_for_tests", SCRIPTS / "run_config.py")
recount = _load("transient_recount_for_tests", SCRIPTS / "recount.py")


# ------------------------------------------------------------------ design and hook


def test_run_order():
    assert design.RUNS == (("A", 1), ("PB-ASYNC", 1), ("PB-ASYNC", 2), ("A", 2))
    assert design.CYCLES == 2 and design.SECONDS == 20.0
    assert design.run_file("PB-ASYNC", 1) == "laya-PB-ASYNC-r1.json.gz"


def test_design_runs_prints_every_run_in_order(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["design.py", "runs"])
    design.main()
    assert capsys.readouterr().out.splitlines() == ["A 1", "PB-ASYNC 1", "PB-ASYNC 2", "A 2"]


def test_window_order_and_counter():
    order = run_config.window_order(2)
    assert order == [
        (0, "solo_short"),
        (0, "solo_long"),
        (0, "hetero"),
        (0, "gpu_only"),
        (1, "gpu_only"),
        (1, "hetero"),
        (1, "solo_long"),
        (1, "solo_short"),
    ]
    assert order[1][1] == "solo_long" and order[4][1] == "gpu_only"  # what each hetero window follows
    c = run_config.WindowCounter(2)
    seen = [c.next() for _ in range(8)]
    assert seen[5] == (5, 1, "hetero")
    with pytest.raises(RuntimeError):
        c.next()


def test_run_config_imports_without_pyobjc():
    assert list(run_config.CELLS) == ["A", "PB-ASYNC"]
    assert run_config.PREDICT_COLUMNS[3] == "native_completion_ns"
    assert run_config.IOREPORT.startswith("IOReport skipped:")
    assert not hasattr(run_config, "traced_index") and not hasattr(run_config, "TRACE_AFTER_S")
    assert callable(recount.set_os_thread_name)
    code = (
        "import sys, importlib.util as u; sys.modules['objc'] = None; sys.modules['CoreML'] = None; "
        f"s = u.spec_from_file_location('m', {str(SCRIPTS / 'run_config.py')!r}); m = u.module_from_spec(s); "
        "s.loader.exec_module(m); print(sorted(m.CELLS))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert "PB-ASYNC" in out


def test_stamped_wrapper_inserts_native_completion():
    stamps, native, out = [], [], []

    class M:
        def predict(self, feats):
            stamps.append((10, 11, 30, 31))
            native.append((29, 30))
            return {"x": 1}

    w = run_config.Stamped(M(), stamps, native, out)
    assert w.predict({}) == {"x": 1}
    entry, sb, sa, nat, cb, wake, exit_ = out[0]
    assert (sb, sa, nat, cb, wake) == (10, 11, 29, 30, 31) and entry <= exit_
    a = run_config.Stamped(type("C", (), {"predict": lambda self, f: {}})(), None, None, out)
    a.predict({})
    assert out[1][1:6] == (0, 0, 0, 0, 0)


# ------------------------------------------------------------------ counters


def _series(name, tid, rows, pid=1):
    return {"name": name, "pid": pid, "tid": tid, "rows": rows}


def test_counter_delta_interpolates_and_metrics():
    t = [0, 100 * MS, 200 * MS]
    # cumulative [ins, cyc, cpu_ns, nj] for P then E
    s = _series(
        "laya-ane-dispatch",
        5,
        [[0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 400, 200, 50, 0, 100, 100, 50, 0], [2, 800, 400, 100, 0, 200, 200, 100, 0]],
    )
    d = analyze.counter_delta(t, [s], 2, 50 * MS, 150 * MS)
    assert list(d) == [400, 200, 50, 0, 100, 100, 50, 0]
    m = analyze.counter_metrics(d, 2, per=4)
    assert m["e_share"] == 0.5 and m["cycle_rate_rel_P"] == 4.0 and m["cycle_rate_rel_E"] == 2.0
    assert m["ipc_P"] == 2.0 and m["ipc"] == pytest.approx(500 / 300)
    assert m["cpu_ms_per_unit"] == pytest.approx(100 / 1e6 / 4)
    assert analyze.counter_delta(t, [], 2, 0, 1) is None


def test_counter_groups_pick_the_windows_own_clients():
    run = {
        "research": {
            "recount": {
                "series": {
                    "1:1": _series("client-short", 1, []),
                    "1:2": _series("client-short", 2, []),
                    "1:3": _series("coreml-callback", 3, []),
                    "9:4": _series("gpu-worker", 4, [], pid=9),
                }
            }
        }
    }
    g = analyze.counter_groups(run, {"tids": {"short": 2, "long": 7}})
    assert [s["tid"] for s in g["client-short"]] == [2] and g["client-long"] == []
    assert len(g["callback"]) == 1 and len(g["gpu-worker"]) == 1


@pytest.mark.skipif(sys.platform != "darwin", reason="proc_pidinfo is macOS only")
def test_sampler_reads_this_process_and_a_child():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1.0)"])
    stop = threading.Event()
    th = threading.Thread(target=lambda: stop.wait(2), name="client-short")
    th.start()
    s = recount.Sampler(extra_pids=lambda: [child.pid], period_ns=50 * MS)
    s.start()
    time.sleep(0.4)
    s.stop()
    stop.set()
    th.join()
    child.wait()
    r = s.record()
    if not r["series"]:
        pytest.skip("PROC_PIDTHREADCOUNTS unavailable on this macOS")
    names = {v["name"] for v in r["series"].values()}
    assert {"MainThread", "client-short"} <= names
    assert r["levels"] >= 1 and len(r["fields_per_level"]) == 4 and r["sampler_cpu_ns"] > 0


# ------------------------------------------------------------------ transient


def _req(shares, per_bin=10, t0=0):
    """Requests per 0.5 s bin with the given host-slow shares."""
    sub, prep = [], []
    for b, sh in enumerate(shares):
        for i in range(per_bin):
            sub.append(t0 + int((b * 0.5 + i * 0.05) * S))
            prep.append(0.5 if i < round(sh * per_bin) else 0.1)
    return np.asarray(sub, np.int64), np.asarray(prep)


def test_transient_recovery_duration_and_presence():
    sub, prep = _req([0.5, 0.3, 0.1, 0.0] + [0.0] * 36)
    client = np.full(len(sub), 10.0)
    client[:10] = 30.0
    tr = analyze.transient(sub, prep, client, 0, 20 * S)
    assert tr["recovery_s"] == 1.5 and tr["duration_s"] == 1.5 and tr["present"]  # 0.1 is not < 10%
    assert tr["peak_short_p99_ms"] == pytest.approx(30.0)
    sub, prep = _req([0.0] * 40)
    tr = analyze.transient(sub, prep, np.full(len(sub), 10.0), 0, 20 * S)
    assert tr["duration_s"] == 0 and not tr["present"]
    sub, prep = _req([0.2] + [0.0] * 39)
    assert analyze.transient(sub, prep, np.ones(len(sub)), 0, 20 * S)["duration_s"] == 0.5  # present at 0.5
    sub, prep = _req([0.0] * 39 + [0.5])  # a late relapse: never recovered
    tr = analyze.transient(sub, prep, np.ones(len(sub)), 0, 20 * S)
    assert not tr["recovered"] and tr["duration_s"] == 20.0
    assert analyze.reference_interval({"recovery_s": 4.0}) == (4.0, 5.0)
    assert analyze.reference_interval({"recovery_s": 4.5}) == (-2.0, 0.0)
    assert analyze.reference_interval({"recovery_s": None}) == (-2.0, 0.0)


# ------------------------------------------------------------------ validity guard and evaluators

CLEAN_WINDOW = {
    "short_p99_ms": 12.0,
    "aggregate_req_s": 122.4,
    "mismatches": 0,
    "routing_failures": [],
    "steady_host_slow_share": 0.01,
}


def _tr(run, cycle, duration, peak, *, native=(9.0, 9.0), cb=(0.05, 0.05), cpu=(0.6, 0.6), cnt=None, steady=None):
    present = duration >= 0.5
    ref_c, tr_c = cnt or (
        {"e_share": 0.1, "cycle_rate_rel_P": 4.0, "ipc": 2.0},
        {"e_share": 0.1, "cycle_rate_rel_P": 4.0, "ipc": 2.0},
    )
    st = steady or {"seconds": 10.0, "requests": 1250, "short_client": [10.0] * 100, "gpu_returns": [0.2] * 50}
    return {
        "run": run,
        "cycle": cycle,
        "transient": {
            "duration_s": duration,
            "present": present,
            "recovered": duration < 20,
            "recovery_s": duration if duration < 20 else None,
            "peak_short_p99_ms": peak,
        },
        "window": dict(CLEAN_WINDOW),
        "periods": {
            "transient": {
                "native": [native[0]] * 10,
                "callback_delay": [cb[0]] * 10,
                "ane_cpu_per_forward": [cpu[0]] * 10,
                "counters": {g: tr_c for g in analyze.SCHED_THREADS},
            },
            "steady": {
                "native": [native[1]] * 10,
                "callback_delay": [cb[1]] * 10,
                "ane_cpu_per_forward": [cpu[1]] * 10,
                "counters": None,
            },
            "reference": {"counters": {g: ref_c for g in analyze.SCHED_THREADS}},
        },
        "_steady": st,
    }


def _a(**window):
    trs = [_tr("A1", k, 0.3, 12.0) for k in (0, 1)] + [_tr("A2", k, 0.0, 12.0) for k in (0, 1)]
    trs[3]["window"].update(window)
    return trs


A_TRS = _a()


def _pb(duration=1.0, peak=20.0, **kw):
    return [_tr(r, k, duration, peak, **kw) for r in ("P1", "P2") for k in (0, 1)]


def test_validity_guard_clean():
    v = analyze.validity(A_TRS)
    assert not v["concern"] and all(x["flags"] == [] for x in v["per_transition"])
    assert len(v["per_transition"]) == 4
    lo, hi = analyze.A_VALIDITY_AGGREGATE_REQ_S
    assert lo == pytest.approx(109.89) and hi == pytest.approx(134.97)
    # the edges that stay clean: P99 just under 13.0, transient exactly 1.0 s, share just under 10%
    edge = _a(short_p99_ms=12.99, steady_host_slow_share=0.099, aggregate_req_s=lo)
    edge[0]["transient"]["duration_s"] = 1.0
    assert not analyze.validity(edge)["concern"]


@pytest.mark.parametrize(
    ("window", "flag"),
    [
        ({"short_p99_ms": 13.0}, "short_p99"),
        ({"steady_host_slow_share": 0.10}, "steady_host_slow"),
        ({"aggregate_req_s": 109.8}, "aggregate"),
        ({"aggregate_req_s": 135.1}, "aggregate"),
        ({"aggregate_req_s": None}, "aggregate"),
        ({"mismatches": 1}, "mismatch_or_routing"),
        ({"routing_failures": ["short"]}, "mismatch_or_routing"),
    ],
)
def test_validity_guard_triggers(window, flag):
    v = analyze.validity(_a(**window))
    assert v["concern"] and v["per_transition"][3]["flags"] == [flag]


def test_validity_guard_transient_trigger_and_it_replaces_the_readings():
    a = _a()
    a[1]["transient"]["duration_s"] = 1.5
    v = analyze.validity(a)
    assert v["concern"] and v["per_transition"][1]["flags"] == ["transient"]
    ev = analyze.evaluate(a, _pb(native=(9.91, 9.0)))
    assert ev["readings"] == [analyze.VALIDITY_CONCERN] and ev["validity"]["concern"]
    assert ev["native"]["rises"] and ev["native"]["transient_mean_ms"] == pytest.approx(9.91)  # still reported
    assert "no causal interpretation" in analyze.VALIDITY_CONCERN
    ev = analyze.evaluate(a, _pb(duration=0.0))  # the guard also overrides "not reproduced"
    assert ev["readings"] == [analyze.VALIDITY_CONCERN] and not ev["phenotype"]


def test_phenotype_not_reproduced():
    ev = analyze.evaluate(A_TRS, _pb(duration=0.0))
    assert ev["readings"] == [analyze.PHENOTYPE_NOT_REPRODUCED] and not ev["phenotype"]
    assert not ev["validity"]["concern"] and len(ev["transitions_used"]) == 4


def test_reading_texts():
    r = analyze.READINGS
    assert r["A"] == "A: Core ML / ANE runtime transition becomes the leading hypothesis"
    assert r["B"] == "B: callback / GIL handoff becomes the leading hypothesis"
    assert r["C"] == "C: supports a perf-level placement / performance-state hypothesis"
    assert r["D"] == "D: the slowdown is not explained by the measured placement / cycle-rate factors"
    assert r["E"] == (
        "E: a production-safe priming / residency strategy is studied next; "
        "the existing 2 s benchmark warm-up is not a fix"
    )
    assert analyze.IPC_TEXT == "; supports further investigation of shared-resource / memory-hierarchy effects"
    texts = " ".join([*r.values(), analyze.IPC_TEXT, analyze.VALIDITY_CONCERN, analyze.PHENOTYPE_NOT_REPRODUCED])
    assert "wakeup" not in texts and "runnable" not in texts and "scheduler" not in texts


def test_reading_a_native_rises_and_b_is_suppressed():
    ev = analyze.evaluate(A_TRS, _pb(native=(9.91, 9.0), cb=(1.0, 0.05)))
    assert ev["native"]["rises"] and analyze.READINGS["A"] in ev["readings"]
    assert analyze.READINGS["B"] not in ev["readings"]
    assert not analyze.evaluate(A_TRS, _pb(native=(9.9, 9.0)))["native"]["rises"]  # exactly 1.10x: not above
    ev = analyze.evaluate(A_TRS, _pb(native=(2.4, 2.1)))  # 1.14x but only + 0.3 ms: not above the floor
    assert not ev["native"]["rises"]


def test_reading_b_callback_rises():
    ev = analyze.evaluate(A_TRS, _pb(cb=(0.26, 0.05)))
    assert ev["callback"]["rises"] and analyze.READINGS["B"] in ev["readings"]
    ev = analyze.evaluate(A_TRS, _pb(cb=(0.2, 0.05)))  # 4x but only + 0.15 ms
    assert not ev["callback"]["rises"]


def test_readings_c_and_d_split_on_placement_or_cycle_rate():
    base = {"e_share": 0.1, "cycle_rate_rel_P": 4.0, "ipc": 2.0}
    ev = analyze.evaluate(
        A_TRS, _pb(cpu=(1.2, 0.6), cnt=(base, {"e_share": 0.35, "cycle_rate_rel_P": 4.0, "ipc": 2.0}))
    )
    assert ev["ane_cpu"]["rises"] and analyze.READINGS["C"] in ev["readings"]
    assert not any(r.startswith(analyze.READINGS["D"]) for r in ev["readings"])
    ev = analyze.evaluate(A_TRS, _pb(cpu=(1.2, 0.6), cnt=(base, {"e_share": 0.1, "cycle_rate_rel_P": 3.2, "ipc": 2.0})))
    assert analyze.READINGS["C"] in ev["readings"]
    ev = analyze.evaluate(A_TRS, _pb(cpu=(1.19, 0.6)))
    assert not ev["ane_cpu"]["rises"]


def test_reading_d_with_and_without_ipc():
    base = {"e_share": 0.1, "cycle_rate_rel_P": 4.0, "ipc": 2.0}
    ev = analyze.evaluate(
        A_TRS, _pb(cpu=(1.2, 0.6), cnt=(base, {"e_share": 0.34, "cycle_rate_rel_P": 3.3, "ipc": 1.6}))
    )
    assert analyze.READINGS["D"] + analyze.IPC_TEXT in ev["readings"]
    assert analyze.READINGS["C"] not in ev["readings"]
    ev = analyze.evaluate(A_TRS, _pb(cpu=(1.2, 0.6)))  # counters unchanged, IPC unchanged
    assert analyze.READINGS["D"] in ev["readings"]
    assert analyze.READINGS["D"] + analyze.IPC_TEXT not in ev["readings"]


def test_reading_e_transient_resolves():
    ev = analyze.evaluate(A_TRS, _pb(duration=1.0))
    assert ev["transient_resolves"] and analyze.READINGS["E"] in ev["readings"]
    slow = {"seconds": 10.0, "requests": 1250, "short_client": [11.0] * 100, "gpu_returns": [0.2] * 50}
    assert not analyze.evaluate(A_TRS, _pb(steady=slow))["transient_resolves"]  # P99 11 > A's 10 x 1.05
    worse = {"seconds": 10.0, "requests": 1250, "short_client": [12.7] * 100, "gpu_returns": [0.2] * 50}
    assert not analyze.evaluate(A_TRS, _pb(steady=worse))["transient_resolves"]
    iso = {"seconds": 10.0, "requests": 1250, "short_client": [10.0] * 100, "gpu_returns": [1.5] * 50}
    assert not analyze.evaluate(A_TRS, _pb(steady=iso))["transient_resolves"]
    thin = {"seconds": 10.0, "requests": 1180, "short_client": [10.0] * 100, "gpu_returns": [0.2] * 50}
    assert not analyze.evaluate(A_TRS, _pb(steady=thin))["transient_resolves"]
    assert not analyze.evaluate(A_TRS, _pb(duration=5.0))["transient_resolves"]


# ------------------------------------------------------------------ end to end


def _fake_run(cell, seconds=20.0):
    order = run_config.window_order(2)
    part, windows_t, rwin, trace = [], [], [], {c: [] for c in run_config.TRACE_COLUMNS}
    fw_ane, fw_gpu, predicts, recv = [], [], [], []
    t, rid = 1000 * S, 0
    series_t = []
    for cycle, cond in order:
        lo, hi = t, t + int(seconds * S)
        t = hi + 3 * S
        streams = {"solo_short": ["short"], "solo_long": ["long"]}.get(cond, ["short", "long"])
        inst = "gpu" if cond == "gpu_only" else "auto"
        het = cond == "hetero"
        for s in streams:
            windows_t.append({"stream": s, "instance": inst, "start_ns": lo, "end_ns": hi})
        pw = {"cycle": cycle, "condition": cond, "streams": {}}
        lat = []
        if het:
            for k in range(int(seconds * 20)):  # a short request every 50 ms
                s0 = lo + k * 50 * MS
                slow = k < 20  # host-slow for the first second
                prep = 600_000 if slow else 100_000
                rid += 1
                svc = s0 + prep + 200_000
                row = (rid, "ane", 128, s0, s0 + prep, s0 + prep, s0 + prep, svc, svc, svc + 9 * MS)
                for c, v in zip(trace, (*row, svc + 9 * MS, svc + 9 * MS + 100_000)):
                    trace[c].append(v)
                lat.append(11.0 if not slow else 15.0)
                fw_ane.append((svc, svc + 9 * MS, 600_000))
                e = svc + 100_000
                predicts.append(
                    (
                        e,
                        e + 50_000,
                        e + 60_000,
                        e + 8 * MS,
                        e + 8 * MS + 50_000,
                        e + 8 * MS + 60_000,
                        e + 8 * MS + 90_000,
                    )
                    if cell == "PB-ASYNC"
                    else (e, 0, 0, 0, 0, 0, e + 8 * MS)
                )
            for k in range(int(seconds * 5)):
                g0 = lo + k * 200 * MS
                rid += 1
                row = (rid, "gpu", 512, g0, g0, g0, g0, g0, g0, g0 + 30 * MS)
                for c, v in zip(trace, (*row, g0 + 30 * MS + 50_000, g0 + 31 * MS)):
                    trace[c].append(v)
                fw_gpu.append((g0, g0 + 30 * MS, 2 * MS))
                recv.append(g0 + 30 * MS + 50_000)
            rwin.append(
                {
                    "start_ns": lo,
                    "end_ns": hi,
                    "instance": "auto",
                    "streams": {"short": len(lat), "long": int(seconds * 5)},
                    "tids": {"short": 100 + cycle, "long": 200 + cycle},
                    "before": {"t_ns": lo - S, "threads": {}},
                    "after": {"t_ns": hi + S, "threads": {"client-long": 50 * MS}},
                    "after_by_stream": {"short": {"t_ns": hi + S, "threads": {"client-short": 400 * MS}}},
                }
            )
            series_t += [lo - 3 * S + i * 100 * MS for i in range(int((seconds + 6) * 10))]
        for s in streams:
            dev = "gpu" if inst == "gpu" or s == "long" else "ane"
            pw["streams"][s] = {
                "req_s": 61.2,  # 122.4 req/s aggregate in a hetero window: #92's A
                "p99_ms": 12.0,
                "p50_ms": 10.0,
                "devices": {dev: 1},
                "mismatches": 0,
                "latency_ms": lat if (het and s == "short") else [30.0],
            }
        part.append(pw)
    tn = np.asarray(series_t, np.int64)
    cum = np.arange(len(tn))

    def rows(scale):
        return [
            [i, int(4000 * c * scale), int(2000 * c * scale), int(500 * c * scale), 0, 0, 0, 0, 0]
            for i, c in enumerate(cum)
        ]

    series = {
        "1:10": _series("laya-ane-dispatch", 10, rows(1.0)),
        "1:100": _series("client-short", 100, rows(0.5)),
        "1:101": _series("client-short", 101, rows(0.5)),
        "1:30": _series("coreml-callback", 30, rows(0.1)),
        "9:40": _series("gpu-worker", 40, rows(2.0), pid=9),
    }
    return {
        "args": {"short": 128, "long": 512, "seconds": seconds, "cycles": 2},
        "part_a": {"windows": part},
        "windows_t": windows_t,
        "gpu_return": {"received_ns": recv, "return_us": [300] * len(recv)},
        "research": {
            "experiment": "coreml-async-transient",
            "cell": cell,
            **run_config.CELLS[cell],
            "run_wall_s": 300.0,
            "window_order": [list(x) for x in order],
            "ioreport": run_config.IOREPORT,
            "workers": {"ane": {"placement": "thread", "pid": None}, "gpu": {"placement": "process", "pid": 9}},
            "forwards": {"ane": fw_ane, "gpu": fw_gpu},
            "predicts_columns": run_config.PREDICT_COLUMNS,
            "predicts": predicts,
            "callbacks": {"submits": 10, "total": 10, "errors": 0} if cell == "PB-ASYNC" else None,
            "recount": {
                "levels": 2,
                "t_ns": tn.tolist(),
                "series": series,
                "errors": {},
                "sampler_cpu_fraction_of_core": 0.001,
                "interval_ns": {"n": 1},
            },
            "trace": trace,
            "windows": rwin,
        },
    }


@pytest.fixture
def raw(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    return d


def _write_all(raw, seconds=20.0):
    for cell, rnd in design.RUNS:
        with gzip.open(raw / design.run_file(cell, rnd), "wt") as fh:
            json.dump(_fake_run(cell, seconds), fh)


def test_empty_raw_and_check(raw, tmp_path):
    res = analyze.summarise(raw)
    assert res["evaluation"] is None and res["outcome"].startswith("running: A r1, PB-ASYNC r1")
    out = tmp_path / "out"
    out.mkdir()
    script = SCRIPTS / "analyze.py"
    subprocess.run([sys.executable, script, "--raw", raw, "--out", out], check=True, capture_output=True)
    ok = subprocess.run([sys.executable, script, "--raw", raw, "--out", out, "--check"], capture_output=True)
    assert ok.returncode == 0
    (out / "tables.md").write_text("x\n")
    stale = subprocess.run([sys.executable, script, "--raw", raw, "--out", out, "--check"], capture_output=True)
    assert stale.returncode != 0 and b"stale" in stale.stderr


def test_end_to_end(raw):
    _write_all(raw)
    res = analyze.summarise(raw)
    pb = res["transitions"]["PB-ASYNC"]
    assert len(pb) == 4 and len(res["transitions"]["A"]) == 4
    assert [t["preceded_by"] for t in pb[:2]] == ["solo_long", "gpu_only"]
    assert all("traced" not in t and "trace" not in t for t in pb)
    t = pb[0]
    assert t["transient"]["duration_s"] == 1.0 and t["transient"]["present"]
    b = t["buckets"]["0-0.5"]
    assert b["short_n"] == 10 and b["host_slow_share"] == 1.0 and b["short_p99_ms"] == pytest.approx(15.0)
    assert b["native_mean_ms"] == pytest.approx(8.0 - 0.06) and b["callback_delay_p50_ms"] == pytest.approx(0.05)
    assert b["handoff_mean_ms"] == pytest.approx(0.01) and b["action_head_mean_ms"] == pytest.approx(0.81)
    assert b["ane_cpu_per_forward_ms"] == pytest.approx(0.6) and b["gpu_return_p50_ms"] == pytest.approx(0.3)
    c = b["counters"]["ane-dispatch"]
    assert c["cycle_rate_rel_P"] == pytest.approx(4.0) and c["ipc"] == pytest.approx(2.0) and c["e_share"] == 0.0
    assert b["counters"]["client-short"] is not None and t["buckets"]["-2-0"]["short_n"] == 0
    assert res["transitions"]["A"][0]["buckets"]["0-0.5"]["native_mean_ms"] is None
    w = res["transitions"]["A"][0]["window"]
    assert w["short_p99_ms"] == 12.0 and w["aggregate_req_s"] == pytest.approx(122.4) and w["mismatches"] == 0
    assert w["routing_failures"] == [] and w["steady_host_slow_share"] == 0.0
    ev = res["evaluation"]
    assert ev["phenotype"] and not ev["validity"]["concern"] and len(ev["transitions_used"]) == 4
    assert analyze.READINGS["E"] in ev["readings"]
    assert all("trace" not in r for r in res["runs"].values())
    md = analyze.tables(res)
    assert "## Validity guard" in md and "ane-dispatch" in md and "Perturbation" not in md
    json.dumps(res, default=analyze._json)


def test_end_to_end_validity_concern(raw):
    _write_all(raw)
    p = raw / design.run_file("A", 2)
    with gzip.open(p, "rt") as fh:
        run = json.load(fh)
    het = [w for w in run["part_a"]["windows"] if w["condition"] == "hetero"]
    het[1]["streams"]["short"]["devices"] = {"gpu": 1}  # a failed routing in A r2, cycle 1
    with gzip.open(p, "wt") as fh:
        json.dump(run, fh)
    res = analyze.summarise(raw)
    assert res["evaluation"]["validity"]["concern"] and res["outcome"] == analyze.VALIDITY_CONCERN
    assert res["transitions"]["A"][3]["window"]["routing_failures"] == ["short"]
    assert res["evaluation"]["transient_resolves"]  # the numbers are still computed and reported


def test_runs_with_other_window_lengths_are_rejected(raw):
    _write_all(raw, seconds=5.0)
    with pytest.raises(AssertionError):
        analyze.summarise(raw)
