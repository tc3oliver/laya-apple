"""Unit tests for research/coreml-staged-handoff/scripts/eval_analyze.py (research only): the H64
production evaluation of evaluation.md on synthetic prod_run.py records (every run passes by
default): the run order and pairing, the runs / reruns / ready subcommands, a PASS phase, each hard
gate H1-H7 failing, tail labels, the latency non-inferiority and its bootstrap, the soak clusters,
validity (runtime tree, dirty, A side, machine), the multilingual smoke and the phase order."""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "research" / "coreml-staged-handoff" / "scripts"
S = 1_000_000_000
MS = 1_000_000
STEP = 50 * MS  # one short request every 50 ms, one long every 100 ms


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("eval_analyze_for_tests", SCRIPTS / "eval_analyze.py")
GUARD = ev.GUARD


def _record(phase, cell):
    """A synthetic prod_run.py record that passes every gate: short latency ~5 ms, prepare 0.1 ms,
    GPU return 0.5 ms (P) / 4.3 ms (A), 20 + 10 req/s in hetero; P logs 64 sync decisions then async
    in each hetero window and "armed" ones in solo_short."""
    cfg = ev.PHASES[phase]
    p = cell == "P"
    t, windows, trace, dec = 100 * S, [], [], []
    sched = ev.prod_run.schedule(cfg["schedule"], cfg["seconds"])
    for i, w in enumerate(sched):
        t += int((w["gap_s"] + 0.5) * S)
        t0, end = t, t + int(w["seconds"] * S)
        t = end
        inst, streams = ev.prod_run.STREAMS[w["condition"]]
        rec = {"index": i, "cycle": w.get("cycle"), "condition": w["condition"], "instance": inst}
        rec.update(start_ns=t0, end_ns=end, streams={})
        for s in streams:
            dev = "gpu" if inst == "gpu" or s == "long" else "ane"
            step = STEP if s == "short" else 2 * STEP
            subs = list(range(t0 + MS, end - 10 * MS, step))
            lat = [5.0 + 0.01 * (k % 7) for k in range(len(subs))]
            rec["streams"][s] = {
                "latency_ms": lat,
                "req_s": 20.0 if s == "short" else 10.0,
                "devices": {dev: len(subs)},
                "mismatches": 0,
            }
            if inst != "auto":
                continue
            for k, sub in enumerate(subs):
                if dev == "ane":
                    resp = sub + int(lat[k] * MS)
                    trace.append(["ane", sub, sub + MS // 10, sub + MS // 5, resp - MS, resp - MS // 2, resp])
                    if not p:
                        continue
                    if w["condition"] != "hetero":
                        dec.append([sub + MS // 10, 0, "armed", 0])
                    elif k < GUARD:
                        dec.append([sub + MS // 10, 0, "async_steady" if k == GUARD - 1 else "sync_guard", k + 1])
                    else:
                        dec.append([sub + MS // 10, 1, "async_steady", GUARD])
                else:
                    ret = MS // 2 if p else 4300_000
                    trace.append(
                        ["gpu", sub, sub + MS // 10, sub + MS, sub + 40 * MS, sub + 40 * MS + ret, sub + 41 * MS]
                    )
        windows.append(rec)
    n_het = sum(w["condition"] == "hetero" for w in sched)
    snap = {
        "enabled": True,
        "disabled": False,
        "disabled_reason": None,
        "state": "async_steady",
        "guard": GUARD,
        "gap_s": 1.0,
        "count": GUARD,
        "episodes": n_het,
        "forwards": {"sync": 1, "async": 1},
        "consistent": True,
    }
    info = {"ane_placement": "process" if phase == "3" else "thread"}
    if p:
        info["ane_handoff"] = snap
    return {
        "experiment": "coreml-staged-handoff production",
        "runtime": {
            "head": "8d9e798e51abaf59a28c3ce6d2d9b27ddccfcbba",
            "laya_apple_tree": ev.TREE,
            "pyproject_blob": ev.PYPROJECT_BLOB,
            "uv_lock_blob": ev.UV_LOCK_BLOB,
            "dirty": False,
        },
        "expect_rejected": (
            {"raised": "ValueError", "message": "ane_handoff=True applies only to ..."}
            if cfg["expect_rejected"]
            else None
        ),
        "args": {
            "cell": cell,
            "model": cfg["model"],
            "short": cfg["short"],
            "long": cfg["long"],
            "schedule": cfg["schedule"],
            "seconds": cfg["seconds"],
            "expect_rejected": cfg["expect_rejected"],
        },
        "info_start": info,
        "info_end": info,
        "workers_alive_at_end": {"ane": True, "gpu": True},
        "handoff_snapshot": snap if p else None,
        "windows": windows,
        "trace_columns": ev.prod_run.TRACE_COLUMNS,
        "trace": sorted(trace, key=lambda r: r[1]),
        "decisions_columns": ["decision_ns", "path", "state", "count"],
        "decisions": dec,
        "run_wall_s": 1.0,
    }


_CACHE: dict = {}


def _run(phase, cell):
    if (phase, cell) not in _CACHE:
        _CACHE[(phase, cell)] = json.dumps(_record(phase, cell))
    return json.loads(_CACHE[(phase, cell)])


SNAP = {
    "uptime": "10:00 up 1 day, load averages: 1.0 1.0 1.0",
    "cpu_user_sys_idle_pct": [5.0, 5.0, 90.0],
    "memory_free_pct": 80,
    "swapusage": "total = 0M",
    "vm_counters": {"Swapins": 0, "Swapouts": 0, "Pageouts": 0},
    "thermal_warning": False,
    "performance_warning": False,
    "power": "Now drawing from 'AC Power'",
    "time_machine_running": False,
    "top_cpu": [[20.0, "python3.12"], [3.0, "WindowServer"]],
}


def _write(raw, phase, cell, rep, mod=None, snap=None):
    run = _run(phase, cell)
    if mod:
        mod(run)
    name = ev.run_name(phase, cell, rep)
    with gzip.open(raw / f"{name}.json.gz", "wt") as fh:
        json.dump(run, fh)
    for tag in ("before", "after"):
        (raw / f"{name}.{tag}.json").write_text(json.dumps(snap or SNAP))


def _write_phases(raw, phases, mods=None):
    mods = mods or {}
    for ph in phases:
        for cell, rep in ev.RUNS[ph]:
            _write(raw, ph, cell, rep, mods.get((ph, cell, rep)))


@pytest.fixture
def raw(tmp_path):
    d = tmp_path / "raw-eval"
    (d / "failed").mkdir(parents=True)
    return d


def _hetero(run):
    return [w for w in run["windows"] if w["condition"] == "hetero"]


def _t_h(run, w):
    ane = [r for r in run["trace"] if r[0] == "ane" and w["start_ns"] <= r[1] < w["end_ns"]]
    return ane[GUARD][1]


def _phase(raw, ph="1"):
    return ev.summarise(raw)["phases"][ph]


# ------------------------------------------------------------------ design and subcommands


def test_run_order_blocks_and_adjacent_pairs():
    for ph in ("1", "2", "4", "5"):
        runs = ev.RUNS[ph]
        assert len(runs) % 4 == 0
        for b in range(0, len(runs), 4):
            block = runs[b : b + 4]
            assert "".join(c for c, _ in block) in ("PAAP", "APPA")
            # each P rN is adjacent to A rN inside its block
            for i in (0, 2):
                assert block[i][1] == block[i + 1][1] and {block[i][0], block[i + 1][0]} == {"P", "A"}
    assert [len(ev.RUNS[p]) for p in ev.ORDER] == [12, 8, 1, 8, 4]


def test_runs_subcommand(capsys):
    assert ev.main(["runs", "1"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[:4] == [
        "mix laya P 1 128 512 20",
        "mix laya A 1 128 512 20",
        "mix laya A 2 128 512 20",
        "mix laya P 2 128 512 20",
    ]
    assert len(lines) == 12 and lines[4] == "mix laya A 3 128 512 20"
    ev.main(["runs", "3"])
    assert capsys.readouterr().out.splitlines() == ["mix laya-multilingual A 1 128 512 5 --expect-rejected"]
    ev.main(["runs", "2"])
    assert capsys.readouterr().out.splitlines()[0] == "mix laya-typed-decisions P 1 128 1024 20"
    ev.main(["runs", "5"])
    assert capsys.readouterr().out.splitlines() == [
        "soak55 laya P 1 128 512 20",
        "soak55 laya A 1 128 512 20",
        "soak55 laya A 2 128 512 20",
        "soak55 laya P 2 128 512 20",
    ]


def test_expected_episode_counts():
    het = {ph: sum(w["condition"] == "hetero" for w in ev.prod_run.schedule(c["schedule"], c["seconds"]))
           for ph, c in ev.PHASES.items()}  # fmt: skip
    assert het == {"1": 2, "2": 2, "3": 2, "4": 6, "5": 66}


# ------------------------------------------------------------------ PASS path


def test_phase1_passes_and_later_phases_wait(raw):
    _write_phases(raw, ["1"])
    res = ev.summarise(raw)
    j = res["phases"]["1"]
    assert j["status"] == "PASS", j["reasons"]
    assert j["latency"]["clusters"] == 6 and j["latency"]["cluster_size"] == 2
    assert j["latency"]["median_ms"] == 0.0 and j["latency"]["pass"]
    assert j["P_episodes"] == 12 and j["tails"]["events"] == []
    assert j["gpu_return_ms"]["P"]["p50"] == pytest.approx(0.5)
    assert j["gpu_return_ms"]["A"]["p50"] == pytest.approx(4.3)
    assert res["phases"]["2"]["status"] == "pending"
    assert all(res["phases"][p]["status"] == "not reached" for p in ("3", "4", "5"))
    assert res["outcome"].startswith("phase 2 pending: mix-laya-typed-decisions-P-r1")
    ep = j["runs"]["mix-laya-P-r1"]["episodes"][0]
    assert ep["structure"]["sync_before_t_h"] == GUARD and ep["t_h_s"] == pytest.approx(3.2011)
    assert not any(k.startswith("_") for k in ep)


def test_full_evaluation_is_product_candidate(raw, tmp_path):
    _write_phases(raw, ev.ORDER)
    out = tmp_path / "out"
    out.mkdir()
    assert ev.main(["--raw", str(raw), "--out", str(out)]) == 0
    md = (out / "eval_tables.md").read_text().splitlines()
    assert md[2] == "Outcome: PRODUCT-CANDIDATE: H64 is the leading 1.5 production candidate."
    res = json.loads((out / "eval_results.json").read_text())
    assert all(res["phases"][p]["status"] == "PASS" for p in ev.ORDER)
    soak = res["phases"]["5"]
    assert soak["latency"]["clusters"] == 22 and soak["latency"]["cluster_size"] == 6
    assert soak["P_episodes"] == 132
    assert res["phases"]["3"]["smoke"]["pass"] is True
    assert ev.main(["--check", "--raw", str(raw), "--out", str(out)]) == 0
    (out / "eval_tables.md").write_text("x")
    assert ev.main(["--check", "--raw", str(raw), "--out", str(out)]) == 1


# ------------------------------------------------------------------ hard gates


def _mismatch(run):
    run["windows"][1]["streams"]["long"]["mismatches"] = 1


def _routing(run):
    run["windows"][0]["streams"]["short"]["devices"] = {"gpu": 3, "ane": 100}


def _dead_worker(run):
    run["workers_alive_at_end"] = {"ane": False, "gpu": True}


def _inconsistent(run):
    run["handoff_snapshot"]["consistent"] = False


def _disabled(run):
    run["handoff_snapshot"].update(enabled=False, disabled=True, disabled_reason="async predict failed")


def _drop_guard_decision(run):
    w = _hetero(run)[0]
    ins = [d for d in run["decisions"] if w["start_ns"] <= d[0] < w["end_ns"]]
    run["decisions"].remove(ins[10])


def _sync_after(run):
    w = _hetero(run)[0]
    th = _t_h(run, w)
    d = next(d for d in run["decisions"] if d[0] > th + S)
    d[1], d[2] = 0, "sync_guard"


def _guard_decisions(run, k=0):
    w = _hetero(run)[k]
    return [d for d in run["decisions"] if w["start_ns"] <= d[0] < w["end_ns"]]


def _leading_armed(run):
    first = _guard_decisions(run)[0]
    i = run["decisions"].index(first)
    run["decisions"][i:i] = [[first[0] - 3000, 0, "armed", 0], [first[0] - 2000, 0, "armed", 0]]


def _armed_inside_guard(run):
    g = _guard_decisions(run)
    run["decisions"].insert(run["decisions"].index(g[10]) + 1, [g[10][0] + 1, 0, "armed", 0])


def _guard_63(run):
    run["decisions"].remove(_guard_decisions(run)[GUARD - 1])


def _guard_65(run):
    g = _guard_decisions(run)
    run["decisions"].insert(run["decisions"].index(g[GUARD - 1]) + 1, [g[GUARD - 1][0] + 1, 0, "async_steady", GUARD])


def _episodes_off(run):
    run["handoff_snapshot"]["episodes"] += 1


def _solo_short_not_armed(run):
    w = next(w for w in run["windows"] if w["condition"] == "solo_short")
    d = next(d for d in run["decisions"] if d[0] >= w["start_ns"])
    d[2] = "sync_guard"


def _gpu_return(run):
    w = _hetero(run)[1]
    for r in run["trace"]:
        if r[0] == "gpu" and w["start_ns"] <= r[5] < w["end_ns"] + S:
            r[5] = r[4] + 1500_000


def _throughput(run):
    for w in _hetero(run):
        for s in w["streams"].values():
            s["req_s"] *= 0.9


def _host_slow(run):
    w = _hetero(run)[0]
    th = _t_h(run, w)
    for r in run["trace"]:
        if r[0] == "ane" and th + 2 * S <= r[1] < th + 3 * S:
            r[2] = r[1] + MS // 2


def _set_latency(run, w, lo, hi, ms):
    """Short requests submitted in [lo, hi) of window w get latency `ms` (client and trace)."""
    ane = [r for r in run["trace"] if r[0] == "ane" and w["start_ns"] <= r[1] < w["end_ns"]]
    lat = w["streams"]["short"]["latency_ms"]
    for k, r in enumerate(ane):
        if lo <= r[1] < hi:
            lat[k] = ms
            r[6] = r[1] + int(ms * MS)


def _slow_state(run):
    w = _hetero(run)[0]
    th = _t_h(run, w)
    _set_latency(run, w, th + S, th + 3 * S + S // 2, 7.0)


def _tail(w):
    """3% of the window's short requests at 10 ms, spread out: P99 up, median and P95 unchanged."""
    lat = w["streams"]["short"]["latency_ms"]
    for k in range(0, len(lat), 33):
        lat[k] = 10.0


def _tails3(run):
    for w in _hetero(run):
        _tail(w)


H_CASES = [
    (("P", 1), _mismatch, "H1"),
    (("P", 2), _routing, "H1"),
    (("P", 3), _dead_worker, "H1"),
    (("P", 4), _inconsistent, "H1"),
    (("P", 5), _disabled, "H1"),
    (("P", 1), _drop_guard_decision, "H2"),
    (("P", 6), _sync_after, "H2"),
    (("P", 2), _episodes_off, "H2"),
    (("P", 2), _armed_inside_guard, "H2"),
    (("P", 4), _guard_63, "H2"),
    (("P", 5), _guard_65, "H2"),
    (("P", 3), _solo_short_not_armed, "H2"),
    (("P", 1), _gpu_return, "H3"),
    (("P", 2), _throughput, "H4"),
    (("P", 3), _host_slow, "H5"),
    (("P", 4), _slow_state, "H6"),
]


@pytest.mark.parametrize("key,mod,gate", H_CASES, ids=[f"{g}-{m.__name__}" for _, m, g in H_CASES])
def test_each_hard_gate_fails(raw, key, mod, gate):
    _write_phases(raw, ["1"], {("1", *key): mod})
    res = ev.summarise(raw)
    j = res["phases"]["1"]
    assert j["status"] == "FAIL"
    assert any(x.startswith(gate) for x in j["reasons"]), j["reasons"]
    assert res["outcome"].startswith("CLOSE: phase 1 failed: ")
    assert res["phases"]["2"]["status"] == "not reached"


def test_leading_armed_decisions_are_allowed(raw):
    _write_phases(raw, ["1"], {("1", "P", r): _leading_armed for r in (1, 4)})
    j = _phase(raw)
    assert j["status"] == "PASS", j["reasons"]
    ep = j["runs"]["mix-laya-P-r1"]["episodes"][0]
    assert ep["structure"]["leading_armed"] == 2 and ep["structure"]["sync_before_t_h"] == GUARD


def test_h7_repeated_tails_fail(raw):
    _write_phases(raw, ["1"], {("1", "P", 1): _tails3, ("1", "P", 3): lambda r: _tail(_hetero(r)[0])})
    j = _phase(raw)
    assert j["tails"]["P"] == 3 and j["tails"]["A"] == 0 and j["tails"]["limit"] == 2
    assert j["status"] == "FAIL" and any(x.startswith("H7") for x in j["reasons"])
    assert not any(x.startswith(("H1", "H2", "H3", "H4", "H5", "H6")) for x in j["reasons"])


def test_h7_limit_scales_with_a_tails(raw):
    mods = {("1", "P", 1): _tails3, ("1", "P", 3): lambda r: _tail(_hetero(r)[0])}
    mods[("1", "A", 2)] = lambda r: _tail(_hetero(r)[0])
    mods[("1", "A", 4)] = lambda r: _tail(_hetero(r)[1])
    _write_phases(raw, ["1"], mods)
    j = _phase(raw)
    assert (j["tails"]["P"], j["tails"]["A"], j["tails"]["limit"]) == (3, 2, 4)
    assert not any(x.startswith("H7") for x in j["reasons"])


def test_a_crash_once_is_rerun_a_p_crash_fails(raw):
    _write_phases(raw, ["1"])
    (raw / "failed" / "mix-laya-A-r1.1.log").write_text("boom")
    assert _phase(raw)["status"] == "PASS"
    (raw / "failed" / "mix-laya-P-r2.1.log").write_text("boom")
    j = _phase(raw)
    assert j["status"] == "FAIL" and any("H1 mix-laya-P-r2: crash" in x for x in j["reasons"])


def test_p_crash_with_missing_file_fails_even_with_runs_pending(raw):
    _write(raw, "1", "P", 1)
    (raw / "failed" / "mix-laya-A-r1.1.log").write_text("boom")
    (raw / "failed" / "mix-laya-P-r2.1.log").write_text("boom")
    j = _phase(raw)
    assert j["status"] == "FAIL" and j["pending"]


# ------------------------------------------------------------------ tail labels and latency


def test_isolated_tail(raw):
    _write_phases(raw, ["1"], {("1", "P", 1): lambda r: _tail(_hetero(r)[0])})
    j = _phase(raw)
    assert j["status"] == "PASS", j["reasons"]
    (t,) = j["tails"]["events"]
    assert (t["run"], t["cell"], t["label"]) == ("mix-laya-P-r1", "P", "isolated tail")


def test_tail_with_other_symptoms(raw):
    def mod(run):
        _tail(_hetero(run)[0])
        w = _hetero(run)[0]
        th = _t_h(run, w)
        for r in run["trace"]:  # one host-slow bin (not two: H5 holds)
            if r[0] == "ane" and th + 11 * S // 10 <= r[1] < th + 14 * S // 10:
                r[2] = r[1] + MS // 2

    _write_phases(raw, ["1"], {("1", "P", 1): mod})
    j = _phase(raw)
    assert j["status"] == "PASS", j["reasons"]
    (t,) = j["tails"]["events"]
    assert t["label"] == "tail with other symptoms" and "1 host-slow bins" in t["why"]


def test_latency_non_inferiority_fails(raw):
    def slower(run):
        for w in _hetero(run):
            w["streams"]["short"]["latency_ms"] = [x + 0.8 for x in w["streams"]["short"]["latency_ms"]]

    _write_phases(raw, ["1"], {("1", "P", r): slower for r in range(1, 7)})
    j = _phase(raw)
    assert j["status"] == "FAIL"
    assert [x.split(":")[0] for x in j["reasons"]] == ["latency non-inferiority"]
    assert j["latency"]["median_ms"] == pytest.approx(0.8)


def test_bootstrap_is_deterministic():
    d = np.random.default_rng(1).normal(0.1, 0.5, size=(22, 6))
    a, b = ev.bootstrap(d), ev.bootstrap(d.copy())
    assert a == b and a["n_clusters"] == 22 and a["cluster_size"] == 6
    assert a["median_ms"] == pytest.approx(float(np.median(d)))
    assert a["cluster_upper_ms"] >= a["median_ms"]
    c = ev.bootstrap(d + 1.0)
    assert c["cluster_upper_ms"] == pytest.approx(a["cluster_upper_ms"] + 1.0)


def test_soak_clusters_are_22_groups_of_6():
    pairs = [{"deltas": [{"p99": 0.01 * k, "median": 0, "p95": 0, "p999": 0, "req_s": 0} for k in range(66)]}] * 2
    lat = ev.latency("5", pairs)
    assert lat["clusters"] == 22 and lat["cluster_size"] == 6 and lat["n"] == 132 and lat["pass"]
    short = [{"deltas": pairs[0]["deltas"][:65]}] * 2
    assert ev.latency("5", short)["pass"] is False


# ------------------------------------------------------------------ validity


def _tree(run):
    run["runtime"]["laya_apple_tree"] = "0" * 40


def _dirty(run):
    run["runtime"]["dirty"] = True


def _a_handoff(run):
    run["info_end"]["ane_handoff"] = None


@pytest.mark.parametrize(
    "key,mod,text",
    [
        (("P", 1), _tree, "runtime"),
        (("A", 3), _dirty, "runtime"),
        (("P", 4), lambda r: r["runtime"].update(uv_lock_blob="0" * 40), "runtime"),
        (("A", 4), lambda r: r["runtime"].update(pyproject_blob=None), "runtime"),
        (("A", 2), _mismatch, "A has 1 mismatches"),
        (("A", 5), _routing, "A has 0 mismatches"),
        (("A", 1), _a_handoff, "a_default_path"),
        (("P", 2), lambda r: r["args"].update(long=1024), "lengths"),
    ],
)
def test_invalid(raw, key, mod, text):
    _write_phases(raw, ["1"], {("1", *key): mod})
    res = ev.summarise(raw)
    j = res["phases"]["1"]
    assert j["status"] == "INVALID" and any(text in x for x in j["reasons"]), j["reasons"]
    assert res["outcome"].startswith("INVALID: phase 1: ")


def test_a_crashed_twice_is_invalid(raw):
    _write_phases(raw, ["1"])
    (raw / "mix-laya-A-r4.json.gz").unlink()
    for k in (1, 2):
        (raw / "failed" / f"mix-laya-A-r4.{k}.log").write_text("boom")
    assert _phase(raw)["status"] == "INVALID"


def test_a_signatures_over_ten_percent_are_invalid(raw):
    def slow_a(run):
        w = _hetero(run)[0]
        _set_latency(run, w, w["start_ns"] + 2 * S, w["start_ns"] + 5 * S, 9.0)

    _write_phases(raw, ["1"], {("1", "A", 1): slow_a, ("1", "A", 2): slow_a})
    j = _phase(raw)
    assert j["A_signatures"]["with_H5_or_H6"] == 2 and j["status"] == "INVALID"


def test_machine_rerun_pending_then_replaced_or_invalid(raw, capsys):
    bad = {**SNAP, "power": "Now drawing from 'Battery Power'"}
    _write_phases(raw, ["1"])
    assert ev.reruns(raw, "1") == []
    _write(raw, "1", "A", 2, snap=bad)
    j = _phase(raw)
    assert j["status"] == "pending" and "mix-laya-A-r2-b" in j["reasons"][0]
    ev.main(["reruns", "1", "--raw", str(raw)])
    assert capsys.readouterr().out.splitlines() == ["mix laya A 2-b 128 512 20"]
    # the -b re-run with a clean machine replaces it
    _write(raw, "1", "A", "2-b")
    assert ev.reruns(raw, "1") == []
    j = _phase(raw)
    assert j["status"] == "PASS" and j["pairs"][1]["A"] == "mix-laya-A-r2-b"
    # a second machine failure is INVALID
    _write(raw, "1", "A", "2-b", snap=bad)
    j = _phase(raw)
    assert j["status"] == "INVALID" and "machine snapshot failed twice" in j["reasons"][0]


def test_reruns_wait_for_the_main_runs(raw):
    bad = {**SNAP, "time_machine_running": True}
    _write(raw, "1", "P", 1, snap=bad)
    assert ev.reruns(raw, "1") == []


def test_superseded_p_run_still_counts_for_hard_gates(raw):
    bad = {**SNAP, "memory_free_pct": 10}
    _write_phases(raw, ["1"])
    _write(raw, "1", "P", 1, mod=_drop_guard_decision, snap=bad)
    _write(raw, "1", "P", "1-b")
    j = _phase(raw)
    assert j["status"] == "FAIL" and any(x.startswith("H2 mix-laya-P-r1:") for x in j["reasons"])


# ------------------------------------------------------------------ multilingual smoke and order


def _through_phase2(raw):
    _write_phases(raw, ["1", "2"])


def test_multilingual_smoke_passes(raw):
    _through_phase2(raw)
    _write_phases(raw, ["3"])
    res = ev.summarise(raw)
    assert res["phases"]["3"]["status"] == "PASS"
    assert res["outcome"].startswith("phase 4 pending")


@pytest.mark.parametrize(
    "mod,check",
    [
        (lambda r: r.update(expect_rejected={"raised": None}), "rejected_with_value_error"),
        (
            lambda r: r.update(expect_rejected={"raised": "BackendUnavailableError", "message": "x"}),
            "rejected_with_value_error",
        ),
        (lambda r: r["info_end"].update(ane_placement="thread"), "process_placement"),
        (lambda r: r["info_end"].update(ane_handoff=None), "no_ane_handoff_key"),
        (_mismatch, "no_mismatch"),
    ],
)
def test_multilingual_smoke_fails(raw, mod, check):
    _through_phase2(raw)
    _write(raw, "3", "A", 1, mod)
    res = ev.summarise(raw)
    j = res["phases"]["3"]
    assert j["status"] == "FAIL" and check in j["smoke"]["failed"]
    assert res["outcome"].startswith("CLOSE: phase 3 failed: multilingual smoke")


def test_multilingual_smoke_crash_fails(raw):
    _through_phase2(raw)
    _write_phases(raw, ["3"])
    (raw / "failed" / "mix-laya-multilingual-A-r1.1.log").write_text("boom")
    assert _phase(raw, "3")["status"] == "FAIL"


def test_phase_order_not_reached_and_ready(raw, capsys):
    _write_phases(raw, ["1", "2"], {("1", "P", 1): _mismatch})
    res = ev.summarise(raw)
    assert res["phases"]["1"]["status"] == "FAIL"
    assert res["phases"]["2"]["status"] == "not reached" and res["phases"]["2"]["judged_status"] == "PASS"
    assert ev.main(["ready", "2", "--raw", str(raw)]) == 1
    assert "phase 1 FAIL" in capsys.readouterr().out
    assert ev.main(["ready", "1", "--raw", str(raw)]) == 1  # a failed phase is not run again


def test_unexpected_files_are_not_judged(raw):
    _write_phases(raw, ["1"])
    _write(raw, "1", "P", 9)
    res = ev.summarise(raw)
    assert res["unexpected_files"] == ["mix-laya-P-r9.json.gz"] and res["phases"]["1"]["status"] == "PASS"


def test_ready_refuses_a_phase_already_failed_invalid_or_with_a_p_crash(raw, capsys):
    _write_phases(raw, ["1"])
    assert ev.main(["ready", "1", "--raw", str(raw)]) == 0
    (raw / "failed" / "mix-laya-P-r3.1.log").write_text("boom")
    assert ev.main(["ready", "1", "--raw", str(raw)]) == 1
    assert "P crash logs: mix-laya-P-r3.1.log" in capsys.readouterr().out
    (raw / "failed" / "mix-laya-P-r3.1.log").unlink()
    _write(raw, "1", "A", 1, _dirty)
    assert ev.main(["ready", "1", "--raw", str(raw)]) == 1
    assert "already INVALID" in capsys.readouterr().out
    _write(raw, "1", "A", 1)
    _write(raw, "1", "P", 1, _mismatch)
    assert ev.main(["ready", "1", "--raw", str(raw)]) == 1
    assert "already FAIL" in capsys.readouterr().out
