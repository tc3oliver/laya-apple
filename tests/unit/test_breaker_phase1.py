"""Unit tests for research/coreml-adaptive-breaker (research only): the frozen breaker state machine
(breaker.py) and Phase 1's analysis (phase1_analyze.py) on synthetic run_config.py records: a
PASS, a fallback that does not recover, a missed slow state, a retry after the trip, request loss,
a crash, and the INCONCLUSIVE rules."""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "research" / "coreml-adaptive-breaker" / "scripts"
S = 1_000_000_000
MS = 1_000_000
TREE = "f" * 40


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bk = _load("adaptive_breaker_for_tests", SCRIPTS / "breaker.py")
pa = _load("adaptive_phase1_for_tests", SCRIPTS / "phase1_analyze.py")


# ------------------------------------------------------------------ the breaker


class Clock:
    def __init__(self):
        self.t = 10 * S

    def __call__(self):
        return self.t


class Gpu:
    def __init__(self):
        self.on = True

    def active(self, now, gap):
        return self.on


def _trace(prep, target="ane", rid=0):
    return SimpleNamespace(target=target, prepare_ms=prep, request_id=rid)


def test_breaker_trips_on_three_consecutive_and_stays_open_for_the_episode():
    clock, gpu = Clock(), Gpu()
    b = bk.Breaker(gpu, clock=clock)
    b.decide()  # the episode starts; C3 is armed 1 s later
    clock.t += S - 10 * MS
    paths = []
    for i, prep in enumerate([0.1, 0.5, 0.5, 0.1, 0.5, 0.5, 0.5, 0.1, 0.1]):
        clock.t += 10 * MS
        paths.append(b.decide()[0])
        b.observe(_trace(prep, rid=i))
    assert paths[:7] == ["async"] * 7  # the trip comes from request 6; its forward already ran async
    assert paths[7:] == ["sync", "sync"]  # no retry after healthy requests
    assert b.trips == [(clock.t - 20 * MS, 1, 6, 0.5)]
    assert b.state == bk.OPEN and b.episodes == 1


def test_breaker_ignores_gpu_requests_and_idle_observations():
    clock, gpu = Clock(), Gpu()
    b = bk.Breaker(gpu, clock=clock)
    for i in range(3):
        b.observe(_trace(0.9, rid=i))  # before any episode: IDLE, not counted
    assert b.state == bk.IDLE and b.count == 0
    b.decide()
    for i in range(5):
        b.observe(_trace(0.9, target="gpu", rid=i))
    assert b.state == bk.HEALTHY and b.count == 0


def test_breaker_rearms_after_a_gap_or_gpu_idle():
    clock, gpu = Clock(), Gpu()
    b = bk.Breaker(gpu, clock=clock)
    b.decide()
    clock.t += S
    for i in range(3):
        b.observe(_trace(0.9, rid=i))
    assert b.decide()[0] == "sync"
    clock.t += 2 * S  # no ANE forward for more than 1 s: a new episode
    assert b.decide() == ("async", bk.HEALTHY, 0) and b.episodes == 2
    for i in range(3):  # not armed yet in the new episode
        b.observe(_trace(0.9, rid=10 + i))
    assert b.state == bk.HEALTHY
    clock.t += S
    for i in range(3):
        b.observe(_trace(0.9, rid=10 + i))
    assert b.decide()[0] == "sync"
    gpu.on = False  # GPU idle: outside an episode R runs async, like B
    clock.t += 10 * MS
    assert b.decide() == ("async", bk.IDLE, 0)
    assert len(b.trips) == 2


def test_breaker_count_resets_on_a_good_request_and_the_threshold_is_strict():
    clock, gpu = Clock(), Gpu()
    b = bk.Breaker(gpu, clock=clock)
    b.decide()
    clock.t += S
    for prep in [0.5, 0.5, 0.3, 0.5, 0.5]:
        b.observe(_trace(prep))
    assert b.state == bk.HEALTHY and b.count == 2


# ------------------------------------------------------------------ Phase 1 records

SPACING = 15 * MS  # one short request every 15 ms, closed-loop-like (response before the next submit)
LONG_SPACING = 45 * MS
TRANSIENT = 4  # every hetero start: the first requests are slow in every cell, A included


def _hetero(t0, cell, rid, mode, n_short=None):
    """One 20 s hetero window with a start transient, simulating the breaker for R. mode:
    healthy; slow (from t0, whole window); recover (slow from t0 until the trip, A after); mid
    (healthy, slow from 8 s, A after the trip); stuck (slow throughout, trips); missed (latency
    slow, prepare normal: C3 never trips); retry (as recover, one async forward after the trip)."""
    end = t0 + 20 * S
    tr, dec, obs, trips = [], [], [], []
    k, t, count = 0, t0, 0
    tripped_at = ep_start = None
    after_trip = 0
    while t + 14 * MS < end and (n_short is None or k < n_short):
        slow_state = {
            "healthy": False,
            "slow": True,
            "recover": tripped_at is None,
            "retry": tripped_at is None,
            "mid": tripped_at is None and t >= t0 + 8 * S,
            "stuck": True,
            "missed": True,
        }[mode]
        transient = k < TRANSIENT
        prep = 0.6 if (transient or (slow_state and mode != "missed")) else 0.12
        e2e = 14.0 if transient else (12.5 if slow_state else 10.0)
        start = t + int(0.3 * MS)
        dnow = start + 1000
        if ep_start is None:
            ep_start = dnow
        if cell == "R":
            if tripped_at is None:
                dec.append([dnow, 1, "healthy", count])
            elif mode == "retry" and after_trip == 5:
                dec.append([dnow, 1, "healthy", 0])
            else:
                dec.append([dnow, 0, "open", count])
            if tripped_at is not None:
                after_trip += 1
        else:
            dec.append([dnow, 0 if cell == "A" else 1, "fixed", 0])
        resp = t + int(e2e * MS)
        tr.append(("ane", t, t + int(prep * MS), start, t + 9 * MS, resp, rid))
        if cell == "R":
            if tripped_at is not None:
                before = "open"
            elif resp - ep_start < S:
                before = "arming"
            else:
                before = "healthy"
                count = count + 1 if prep > 0.3 else 0
            obs.append([resp, rid, prep, before, count])
            if before == "healthy" and count >= 3:
                tripped_at = resp
                trips.append([resp, 0, rid, prep])
        rid += 1
        k += 1
        t += SPACING
    for t in range(t0, end - 40 * MS, LONG_SPACING):
        tr.append(("gpu", t, t + int(0.5 * MS), t + MS, t + 30 * MS, t + 35 * MS, rid))
        rid += 1
    return tr, dec, obs, trips, rid


def _record(cell, modes=("healthy", "healthy"), n_short=None):
    t, rid = 100 * S, 1
    trace, dec, obs, trips, wins, pa_windows = [], [], [], [], [], []
    order = [(0, "solo_short"), (0, "solo_long"), (0, "hetero"), (0, "gpu_only")]
    order += [(1, "gpu_only"), (1, "hetero"), (1, "solo_long"), (1, "solo_short")]
    h = 0
    for cyc, cond in order:
        t += 25 * S
        streams = {s: {"n": 100, "devices": {d: 100}, "mismatches": 0} for s, d in pa.ROUTES[cond].items()}
        if cond == "hetero":
            tr, d, o, tp, rid = _hetero(t, cell, rid, modes[h], n_short)
            for x in tp:
                x[1] = h + 1
            h += 1
            trace += tr
            dec += d
            obs += o
            trips += tp
            wins.append({"start_ns": t, "end_ns": t + 20 * S, "instance": "auto"})
            streams["short"]["n"] = sum(1 for r in tr if r[0] == "ane")
            streams["short"]["devices"] = {"ane": streams["short"]["n"]}
            streams["long"]["n"] = sum(1 for r in tr if r[0] == "gpu")
            streams["long"]["devices"] = {"gpu": streams["long"]["n"]}
        pa_windows.append({"cycle": cyc, "condition": cond, "streams": streams})
    cols = ["target", "submit_ns", "prepared_ns", "service_start_ns", "service_end_ns", "response_ns", "request_id"]
    gpu_rows = [r for r in trace if r[0] == "gpu"]
    return {
        "args": {"cycles": 2, "seconds": 20.0},
        "runtime": {"laya_apple_tree": TREE, "dirty": False},
        "part_a": {"windows": pa_windows},
        "gpu_return": {
            "received_ns": [r[5] for r in gpu_rows],
            "return_us": [50.0 if cell != "A" else 4300.0 for _ in gpu_rows],
        },
        "research": {
            "experiment": "coreml-adaptive-breaker",
            "cell": cell,
            "windows": wins,
            "trace": {c: [r[i] for r in trace] for i, c in enumerate(cols)},
            "handoff": {"episodes": 2, "errors": [], "decisions": dec},
            "breaker": {
                "installed": cell == "R",
                "episodes": sum(
                    1 for k, x in enumerate(dec) if x[2] == "healthy" and (k == 0 or dec[k - 1][2] != "healthy")
                )
                if cell == "R"
                else None,
                "trips": trips,
                "observations": obs,
            },
        },
    }


SNAP = {
    "thermal_warning": False,
    "performance_warning": False,
    "power": "Now drawing from 'AC Power'",
    "memory_free_pct": 80,
    "time_machine_running": False,
    "top_cpu": [],
    "cpu_user_sys_idle_pct": [2.0, 2.0, 96.0],
    "vm_counters": {"Swapins": 0, "Swapouts": 0},
}


def _write(raw, name, rec, snap=SNAP):
    with gzip.open(raw / f"{name}.json.gz", "wt") as fh:
        json.dump(rec, fh)
    for tag in ("before", "after"):
        (raw / f"{name}.{tag}.json").write_text(json.dumps(snap))


def _campaign(tmp_path, r_modes=None, b_modes=("slow", "slow"), skip=()):
    raw = tmp_path / "raw"
    raw.mkdir()
    for cell, rep in pa.RUNS:
        if (cell, rep) in skip:
            continue
        modes = {"A": ("healthy", "healthy"), "B": b_modes, "R": (r_modes or {}).get(rep, ("recover", "recover"))}[cell]
        _write(raw, pa.run_name(cell, rep), _record(cell, modes))
    return raw


@pytest.fixture(autouse=True)
def _tree(monkeypatch):
    monkeypatch.setattr(pa, "pinned_tree", lambda: TREE)


def test_pass_when_every_trip_recovers_within_a_second(tmp_path):
    res = pa.summarise(_campaign(tmp_path))
    assert res["outcome"] == "PASS", (res["hard_failures"], res["failures"], res["invalid"], res["inconclusive"])
    r = res["stats"]["R"]
    assert r["tripped"] == 12 and r["recovered_within_1s"] == 12 and r["confirmed_slow_at_trip"] == 12
    assert r["trip_to_first_a_ms"]["worst"] < 5
    # armed at the episode start + 1 s: the trip comes 3 requests later, onset at the arm time
    assert r["onset_to_trip_ms"]["n"] == 12 and r["onset_to_trip_ms"]["worst"] < 60
    assert all(1.0 < e["trip_s"] < 1.1 for e in res["episodes"] if e["cell"] == "R")
    assert r["trip_to_recovery_ms"]["worst"] < 20
    assert res["stats"]["B"]["sustained"] == 6 and res["stats"]["A"]["sustained"] == 0
    e = next(e for e in res["episodes"] if e["cell"] == "R")
    assert e["first_a_request_id"] > e["trip_request_id"] == e["last_async_request_id"]
    assert res["stats"]["post_fallback"]["throughput_ratio"] == pytest.approx(1.0, abs=0.02)


def test_the_start_transient_never_trips_and_a_healthy_r_is_inconclusive(tmp_path):
    healthy = {rep: ("healthy", "healthy") for rep in range(1, 7)}
    res = pa.summarise(_campaign(tmp_path, r_modes=healthy))
    assert res["stats"]["R"]["tripped"] == 0
    assert res["outcome"] == "INCONCLUSIVE" and any("natural slow state" in x for x in res["inconclusive"])


def test_a_slow_state_that_starts_mid_episode(tmp_path):
    res = pa.summarise(_campaign(tmp_path, r_modes={2: ("mid", "recover")}))
    assert res["outcome"] == "PASS", (res["hard_failures"], res["failures"])
    e = next(e for e in res["episodes"] if e["run"] == "laya-R-r2" and e["window"] == 0)
    assert 8.0 <= e["onset_s"] < 8.05 and e["onset_to_trip_ms"] < 60 and e["confirmed_slow_at_trip"] is False
    assert e["trip_to_recovery_ms"] < 20


def test_fail_when_the_slow_state_outlives_the_fallback(tmp_path):
    res = pa.summarise(_campaign(tmp_path, r_modes={3: ("recover", "stuck")}))
    assert res["outcome"] == "FAIL"
    assert any("laya-R-r3 w1: no sustained A-like recovery" in x for x in res["failures"])
    assert any("persists after the fallback" in x for x in res["failures"])


def test_fail_when_a_sustained_slow_state_never_trips(tmp_path):
    res = pa.summarise(_campaign(tmp_path, r_modes={2: ("missed", "recover")}))
    assert res["outcome"] == "FAIL"
    assert any("missed" in x for x in res["failures"])


def test_async_retry_after_the_trip_is_a_hard_failure(tmp_path):
    res = pa.summarise(_campaign(tmp_path, r_modes={1: ("retry", "recover")}))
    assert res["outcome"] == "FAIL"
    assert any("async decision after the trip" in x for x in res["hard_failures"])


def test_request_loss_and_missing_observations_are_hard_failures(tmp_path):
    raw = _campaign(tmp_path)
    rec = _record("R", ("recover", "recover"))
    rec["part_a"]["windows"][2]["streams"]["short"]["n"] += 1
    _write(raw, "laya-R-r4", rec)
    res = pa.summarise(raw)
    assert res["outcome"] == "FAIL" and any("request_loss" in x for x in res["hard_failures"])
    rec = _record("R", ("recover", "recover"))
    del rec["research"]["breaker"]["observations"][500]
    _write(raw, "laya-R-r4", rec)
    res = pa.summarise(raw)
    assert any("observations cover" in x for x in res["hard_failures"])


def test_a_trip_on_the_last_request_of_a_window_is_not_a_failure(tmp_path):
    raw = _campaign(tmp_path)
    n = next(i for i in range(200) if _hetero(0, "R", 0, "recover", i)[3])  # requests up to the trip
    _write(raw, "laya-R-r6", _record("R", ("recover", "recover"), n_short=n))
    res = pa.summarise(raw)
    e = next(e for e in res["episodes"] if e["run"] == "laya-R-r6")
    assert e["trips"] == 1 and e.get("no_request_after_trip")
    assert not res["hard_failures"], res["hard_failures"]


def test_crashes_stop_the_campaign_with_an_outcome(tmp_path):
    raw = _campaign(tmp_path, skip={("A", 2), ("R", 6), ("B", 3)})
    (raw / "failed").mkdir()
    (raw / "failed" / "laya-A-r2.1.log").write_text("boom")
    assert pa.summarise(raw)["outcome"] == "PENDING"  # an A crash is re-run in place
    (raw / "failed" / "laya-A-r2.2.log").write_text("boom")
    res = pa.summarise(raw)
    assert res["outcome"] == "INCONCLUSIVE" and not res["pending"]
    (raw / "failed" / "laya-R-r5.1.log").write_text("boom")
    res = pa.summarise(raw)
    assert res["outcome"] == "FAIL" and "laya-R-r5 crashed" in res["hard_failures"]


def test_a_crash_of_a_machine_rerun_counts(tmp_path):
    raw = _campaign(tmp_path)
    bad = dict(SNAP, time_machine_running=True)
    _write(raw, "laya-R-r1", _record("R", ("recover", "recover")), snap=bad)
    (raw / "failed").mkdir()
    (raw / "failed" / "laya-R-r1-b.1.log").write_text("boom")
    res = pa.summarise(raw)
    assert res["outcome"] == "FAIL" and "laya-R-r1-b crashed" in res["hard_failures"]


def test_inconclusive_without_the_b_phenotype(tmp_path):
    res = pa.summarise(_campaign(tmp_path, b_modes=("healthy", "healthy")))
    assert res["outcome"] == "INCONCLUSIVE" and "B phenotype" in res["inconclusive"][0]


def test_machine_failure_asks_for_a_rerun_and_twice_is_inconclusive(tmp_path):
    raw = _campaign(tmp_path)
    bad = dict(SNAP, time_machine_running=True)
    _write(raw, "laya-B-r2", _record("B", ("slow", "slow")), snap=bad)
    assert pa.summarise(raw)["outcome"] == "PENDING"
    _write(raw, "laya-B-r2-b", _record("B", ("slow", "slow")), snap=bad)
    res = pa.summarise(raw)
    assert res["outcome"] == "INCONCLUSIVE" and any("machine failed twice" in x for x in res["invalid"])


def test_runtime_tree_mismatch_is_invalid(tmp_path):
    raw = _campaign(tmp_path)
    rec = _record("A")
    rec["runtime"]["dirty"] = True
    _write(raw, "laya-A-r3", rec)
    assert pa.summarise(raw)["outcome"] == "INCONCLUSIVE"


def test_recovery_point_needs_a_full_second_of_good_windows():
    env = {"med": 10.0, "p95": 10.0, "span": 400.0}
    n = 200
    sub = [k * SPACING for k in range(n)]
    e2e = [12.5 if 40 <= k < 45 else 10.0 for k in range(n)]  # a brief relapse after the start
    rq = {"submit": np.array(sub), "e2e": np.array(e2e), "prep": np.full(n, 0.1)}
    # a window with 2 or more of the 5 slow requests fails on its P95 (windows 17-43); the first start
    # whose next second of windows all pass is 44
    assert pa.recovery_point(pa.rolling(rq), env) == sub[44]


def test_breaker_arms_one_second_after_the_episode_start():
    clock, gpu = Clock(), Gpu()
    b = bk.Breaker(gpu, clock=clock)
    b.decide()
    for i in range(5):  # the start transient: not counted
        clock.t += 10 * MS
        b.observe(_trace(0.9, rid=i))
    assert b.state == bk.HEALTHY and b.count == 0 and b.observations[-1][3] == bk.ARMING
    clock.t += S
    for i in range(3):
        b.observe(_trace(0.9, rid=10 + i))
    assert b.state == bk.OPEN and b.trips[0][2] == 12
