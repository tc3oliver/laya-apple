"""Unit tests for research/coreml-adaptive-breaker/scripts/val_analyze.py (research only): the 1.5
release validation of validation.md on synthetic val_run.py records: run order and pairing, a PASS
phase, a tripped episode that recovers, an exposed slow state, a slow state that outlives the
fallback, false trips, product value, a retry after the trip, correctness, crashes, machine
re-runs, INVALID pins, the multilingual smoke and the phase order."""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "research" / "coreml-adaptive-breaker" / "scripts"
S, MS = 1_000_000_000, 1_000_000
PINS = {"tree": "a" * 40, "pyproject": "b" * 40, "uv_lock": "c" * 40}
SPACING, LONG_SPACING = 15 * MS, 45 * MS


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


va = _load("adaptive_val_analyze_for_tests", SCRIPTS / "val_analyze.py")
sys.path.insert(0, str(ROOT / "scripts"))
vr = _load("adaptive_val_run_for_tests", SCRIPTS / "val_run.py")


def _window(t0, seconds, cond, inst, cell, mode, rid, trace, decisions, trips):
    """One window's requests. mode (P hetero only): healthy; trip (slow 3 requests after the handoff,
    A-like after the fallback); stuck (slow after the handoff, trips, stays slow); exposed (slow
    latency after the handoff, prepare fast: no trip); false (3 slow prepares, latency normal);
    retry (as trip, one async decision after it)."""
    end = t0 + int(seconds * S)
    streams = {}
    if inst is None:
        return streams, rid
    for s in va.ROUTES[cond]:
        lat, devices = [], {}
        target = va.ROUTES[cond][s]
        step = SPACING if s == "short" else LONG_SPACING
        t, k = t0, 0
        tripped_at = None
        after = 0
        while t + 14 * MS < end:
            prep, e2e = 0.12, 10.0 if s == "short" else 30.0
            if inst == "auto" and target == "ane" and cond in va.HETERO and cell == "P":
                async_phase = k >= va.GUARD
                if (
                    async_phase
                    and tripped_at is None
                    and mode in ("trip", "stuck", "retry", "false")
                    and k < va.GUARD + 3
                ):
                    prep = 0.6
                    e2e = 10.0 if mode == "false" else 12.5
                if mode == "stuck" and async_phase:
                    prep, e2e = 0.6, 12.5
                if mode == "exposed" and async_phase:
                    e2e = 12.5
                start = t + int(0.3 * MS)
                if k < va.GUARD:
                    decisions.append([start + 1000, 0, "safe_sync", k + 1])
                elif tripped_at is None:
                    decisions.append([start + 1000, 1, "async_healthy", va.GUARD])
                elif mode == "retry" and after == 3:
                    decisions.append([start + 1000, 1, "async_healthy", va.GUARD])
                else:
                    decisions.append([start + 1000, 0, "breaker_open", va.GUARD])
                if tripped_at is not None:
                    after += 1
                resp = t + int(e2e * MS)
                trace.append((target, t, t + int(prep * MS), start, t + 9 * MS, t + 9 * MS + 1000, resp, rid))
                if tripped_at is None and k == va.GUARD + 2 and mode in ("trip", "stuck", "retry", "false"):
                    tripped_at = resp - 500
                    trips.append(tripped_at)
            elif inst == "auto":
                if cell == "P" and target == "ane":
                    decisions.append([t + int(0.3 * MS) + 1000, 0, "armed", 0])
                gr = 4.3 if cell == "A" else 0.05
                trace.append(
                    (
                        target,
                        t,
                        t + int(prep * MS),
                        t + int(0.3 * MS),
                        t + 9 * MS,
                        t + 9 * MS + int(gr * MS),
                        t + int(e2e * MS),
                        rid,
                    )
                )
            lat.append(e2e)
            devices[target] = devices.get(target, 0) + 1
            rid += 1
            k += 1
            t += step
        streams[s] = {"latency_ms": lat, "devices": devices, "mismatches": 0, "req_s": len(lat) / seconds}
    # GPU requests of a P hetero window: GPU return isolated
    if cell == "P" and cond in va.HETERO:
        trace[:] = [
            r if r[0] != "gpu" or not (t0 <= r[1] < end) else (r[0], r[1], r[2], r[3], r[4], r[4] + 50_000, r[6], r[7])
            for r in trace
        ]
    return streams, rid


def _record(sched, model, cell, modes=None, snapshot=None):
    modes = list(modes or [])
    t, rid = 100 * S, 1
    trace, decisions, trips, windows = [], [], [], []
    h = 0
    for i, w in enumerate(vr.schedule(sched, 20.0 if sched != "mix" or model != "laya-multilingual" else 5.0)):
        t += int((w["gap_s"] + 0.5) * S)
        inst, _ = vr.STREAMS[w["condition"]]
        mode = "healthy"
        if w["condition"] in va.HETERO:
            mode = modes[h] if h < len(modes) else "healthy"
            h += 1
        streams, rid = _window(t, w["seconds"], w["condition"], inst, cell, mode, rid, trace, decisions, trips)
        windows.append(
            {
                "index": i,
                "condition": w["condition"],
                "instance": inst,
                "start_ns": t,
                "end_ns": t + int(w["seconds"] * S),
                "streams": streams,
            }
        )
        t += int(w["seconds"] * S)
    p = cell == "P" and model != "laya-multilingual"
    info = {"ane_placement": "process" if model == "laya-multilingual" else "thread"}
    if p:
        info["ane_handoff"] = {"enabled": True}
    snap = snapshot or {
        "enabled": True,
        "consistent": True,
        "state": "armed",
        "episodes": h,
        "breaker": {"trips": len(trips)},
    }
    return {
        "runtime": {
            "laya_apple_tree": PINS["tree"],
            "pyproject_blob": PINS["pyproject"],
            "uv_lock_blob": PINS["uv_lock"],
            "dirty": False,
        },
        "args": {"cell": cell, "model": model, "schedule": sched},
        "expect_rejected": {"raised": "ValueError"} if model == "laya-multilingual" else None,
        "info_start": info,
        "info_end": info,
        "workers_alive_at_end": {"ane": True, "gpu": True},
        "handoff_snapshot": snap if p else None,
        "windows": windows,
        "trace_columns": list(va.TRACE),
        "trace": [list(r) for r in sorted(trace, key=lambda r: r[1])],
        "decisions": decisions if p else [],
        "trips": trips if p else [],
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


def _phase(raw, phase, p_modes=None, skip=()):
    for sched, model, *_rest, runs in va.PHASES[phase]:
        for cell, rep in runs:
            if (sched, cell, rep) in skip:
                continue
            modes = (p_modes or {}).get((sched, rep)) if cell == "P" else None
            _write(raw, va.run_name(sched, model, cell, rep), _record(sched, model, cell, modes))


@pytest.fixture
def raw(tmp_path, monkeypatch):
    monkeypatch.setattr(va, "pinned", lambda: PINS)
    d = tmp_path / "raw-val"
    d.mkdir()
    return d


def test_run_lines():
    assert va.run_lines("2") == [
        "product laya P 1 128 512 20.0",
        "product laya A 1 128 512 20.0",
        "product laya P 2 128 512 20.0",
    ]
    assert va.run_lines("3")[-1] == "soak55 laya-typed-decisions P 3 128 1024 20.0"
    assert va.run_lines("4") == ["mix laya-multilingual P 1 128 512 5.0 --expect-rejected"]
    assert va.run_lines("6") == ["soak55 laya P 1 128 512 20.0"]


def test_phase_passes_when_healthy(raw):
    _phase(raw, "2")
    j = va.judge(raw, "2")
    assert j["status"] == "PASS", (j["fail"], j["invalid"])
    st = j["stats"]
    assert st["p_episodes"] == 12 and st["a_episodes"] == 6 and st["untripped"] == 12 and st["trips"] == 0
    assert st["async_residency"] > 0.9
    assert st["gpu_return_p50_ms"]["P_healthy_median"] < 0.1 and st["gpu_return_p50_ms"]["A_median"] > 4


def test_a_trip_that_recovers_passes(raw):
    _phase(raw, "2", {("product", 1): ["trip"]})
    j = va.judge(raw, "2")
    assert j["status"] == "PASS", (j["fail"], j["invalid"])
    st = j["stats"]
    assert st["trips"] == 1 and st["confirmed_slow_trips"] == 1 and st["suspected_false_trips"] == 0
    assert st["trip_to_recovery_ms"]["worst"] < 100


def test_slow_state_after_the_fallback_fails(raw):
    _phase(raw, "2", {("product", 2): ["healthy", "stuck"]})
    j = va.judge(raw, "2")
    assert j["status"] == "FAIL"
    assert any("no sustained A-like recovery" in x for x in j["fail"])
    assert any("persists after the fallback" in x for x in j["fail"])


def test_exposed_slow_state_fails(raw):
    _phase(raw, "2", {("product", 2): ["exposed"]})
    j = va.judge(raw, "2")
    assert j["status"] == "FAIL" and any("exposed" in x for x in j["fail"])


def test_false_trips_are_counted_and_many_fail(raw):
    _phase(raw, "2", {("product", 1): ["false"]})
    j = va.judge(raw, "2")
    assert j["status"] == "PASS" and j["stats"]["suspected_false_trips"] == 1
    _phase(raw, "2", {("product", 1): ["false", "false", "healthy", "false"]})
    j = va.judge(raw, "2")
    assert j["status"] == "FAIL" and any("false trips" in x for x in j["fail"])
    assert any("stayed async" in x for x in j["fail"])  # 9 of 12 untripped < 80%


def test_retry_after_the_trip_fails(raw):
    _phase(raw, "2", {("product", 2): ["retry"]})
    j = va.judge(raw, "2")
    assert j["status"] == "FAIL" and any("retry" in x for x in j["fail"])


def test_p_correctness_and_default_use(raw):
    _phase(raw, "2")
    rec = _record("product", "laya", "P")
    rec["windows"][1]["streams"]["short"]["mismatches"] = 1
    del rec["info_start"]["ane_handoff"]
    _write(raw, "product-laya-P-r2", rec)
    j = va.judge(raw, "2")
    assert j["status"] == "FAIL"
    assert any("correctness" in x for x in j["fail"])
    assert any("did not use adaptive execution" in x for x in j["fail"])


def test_crashes_and_machine_reruns(raw):
    _phase(raw, "2", skip={("product", "A", 1)})
    (raw / "failed").mkdir()
    (raw / "failed" / "product-laya-A-r1.1.log").write_text("x")
    assert va.judge(raw, "2")["status"] == "PENDING"
    (raw / "failed" / "product-laya-A-r1.2.log").write_text("x")
    assert va.judge(raw, "2")["status"] == "INVALID"
    (raw / "failed" / "product-laya-P-r2.1.log").write_text("x")
    j = va.judge(raw, "2")
    assert "product-laya-P-r2 crashed" in j["fail"]


def test_machine_failure_reruns_then_invalid(raw):
    _phase(raw, "2")
    bad = dict(SNAP, time_machine_running=True)
    _write(raw, "product-laya-A-r1", _record("product", "laya", "A"), snap=bad)
    assert va.reruns(raw, "2") == ["product laya A 1-b 128 512 20.0"]
    assert va.judge(raw, "2")["status"] == "PENDING"
    _write(raw, "product-laya-A-r1-b", _record("product", "laya", "A"), snap=bad)
    assert va.judge(raw, "2")["status"] == "INVALID"


def test_pins_must_match(raw):
    _phase(raw, "2")
    rec = _record("product", "laya", "A")
    rec["runtime"]["laya_apple_tree"] = "0" * 40
    _write(raw, "product-laya-A-r1", rec)
    assert va.judge(raw, "2")["status"] == "INVALID"


def test_multilingual_smoke(raw):
    _phase(raw, "4")
    assert va.judge(raw, "4")["status"] == "PASS"
    rec = _record("mix", "laya-multilingual", "P")
    rec["info_end"] = dict(rec["info_end"], ane_handoff={"enabled": True})
    rec["expect_rejected"] = {"raised": None}
    _write(raw, "mix-laya-multilingual-P-r1", rec)
    j = va.judge(raw, "4")
    assert j["status"] == "FAIL" and len(j["fail"]) == 2


def test_soak_uses_earlier_a_references_and_extends_on_a_trigger(raw, monkeypatch):
    # synthetic P and A have equal throughput, so move only the borderline throughput trigger
    monkeypatch.setitem(va.BORDERLINE, "throughput_ratio", 0.99)
    _phase(raw, "2")
    _phase(raw, "5")
    _phase(raw, "6")
    j = va.judge(raw, "6")
    assert j["status"] == "PASS", (j["fail"], j["invalid"])
    assert j["stats"]["p_episodes"] == 66 and j["stats"]["a_episodes"] == 6 + 8 and not j["triggers"]
    assert va.reruns(raw, "6") == []
    _write(raw, "soak55-laya-P-r1", _record("soak55", "laya", "P", ["false"]))
    j = va.judge(raw, "6")
    assert j["status"] == "PENDING" and j["triggers"]  # r1 passes but triggers r2
    assert va.reruns(raw, "6") == ["soak55 laya P 2 128 512 20.0"]
    _write(raw, "soak55-laya-P-r2", _record("soak55", "laya", "P"))
    j = va.judge(raw, "6")
    assert j["status"] == "PASS" and j["stats"]["p_episodes"] == 132


def test_phase_order_and_soak_minimum(raw):
    _phase(raw, "3")
    ok, why = va.ready(raw, "3")
    assert not ok and "phase 2" in why
    _phase(raw, "2")
    assert va.ready(raw, "3")[0]
    res = va.summarise(raw)
    assert res["phases"]["2"]["status"] == "PASS" and res["phases"]["3"]["status"] == "PASS"
    assert res["phases"]["4"]["status"] == "PENDING" and res["phases"]["5"]["status"] == "NOT RUN"
