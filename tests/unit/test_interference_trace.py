"""The GPU-ANE interference research harness (research/gpu-ane-interference/): its tracing only
observes, and its interval and statistics helpers are right. No model loading."""

from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "research" / "gpu-ane-interference" / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"interference_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def jobtrace():
    return _load("jobtrace")


@pytest.fixture(scope="module")
def analyze():
    return _load("analyze")


class FakeWorker:
    """The DeviceWorker surface jobtrace.tag_dispatcher wraps: _run(job_id, rows)."""

    kind = "ane"

    def __init__(self, jobtrace, fail=False):
        self.jobtrace, self.fail = jobtrace, fail
        self.seen = []

    def _run(self, job_id, rows):
        self.seen.append((job_id, rows, self.jobtrace.current_request_id()))
        if self.fail:
            raise RuntimeError("device error")
        return ("logits", "act", 1, 2)


def test_dispatcher_tag_exposes_the_job_id_and_changes_nothing(jobtrace):
    w = FakeWorker(jobtrace)
    jobtrace.tag_dispatcher(w)
    rows = [{"ids": [1, 2]}]
    assert w._run(41, rows) == ("logits", "act", 1, 2)  # the result is returned unchanged
    assert w.seen == [(41, rows, 41)]  # same arguments; the forward saw its request id
    assert jobtrace.current_request_id() is None  # cleared after the job


def test_dispatcher_tag_propagates_errors_and_clears(jobtrace):
    w = FakeWorker(jobtrace, fail=True)
    jobtrace.tag_dispatcher(w)
    with pytest.raises(RuntimeError, match="device error"):
        w._run(7, [{}])
    assert w.seen[0][2] == 7 and jobtrace.current_request_id() is None


def test_worker_connection_tags_received_jobs_only(jobtrace):
    class Conn:
        def __init__(self, msgs):
            self.msgs, self.sent = list(msgs), []

        def recv(self):
            return self.msgs.pop(0)

        def send(self, m):
            self.sent.append(m)

    conn = jobtrace._TaggingConnection(Conn([(12, ["row"]), None]))
    assert conn.recv() == (12, ["row"]) and jobtrace.current_request_id() == 12
    conn.send((12, "ok", ()))  # other calls pass through
    assert conn._conn.sent == [(12, "ok", ())]
    assert conn.recv() is None and jobtrace.current_request_id() == 12  # the close message is not a job


def test_request_id_is_per_thread(jobtrace):
    w = FakeWorker(jobtrace)
    jobtrace.tag_dispatcher(w)
    seen = {}

    def other():
        seen["other"] = jobtrace.current_request_id()

    w._run(5, [])
    t = threading.Thread(target=other)
    t.start()
    t.join()
    assert seen["other"] is None


def test_clock_is_system_wide_monotonic(jobtrace):
    info = jobtrace.clock_info()
    assert info["monotonic"]
    if sys.platform == "darwin":  # parent and worker timestamps are only comparable on this clock
        assert info["implementation"] == "mach_absolute_time()"


def test_overlap_fraction_and_busy_union(analyze):
    busy = analyze.busy_union([(5, 7), (0, 2), (1, 3)])
    assert busy == [[0, 3], [5, 7]]
    starts = [b[0] for b in busy]
    assert analyze.overlap_fraction(0, 10, busy, starts) == pytest.approx(0.5)
    assert analyze.overlap_fraction(3, 5, busy, starts) == 0.0
    assert analyze.overlap_fraction(6, 6.5, busy, starts) == 1.0
    assert analyze.overlap_fraction(2, 6, busy, starts) == pytest.approx(0.5)
    assert analyze.overlap_fraction(4, 4, busy, starts) == 0.0


def test_describe_flags_an_unreliable_p99(analyze):
    small = analyze.describe(range(100))
    assert small["n"] == 100 and small["p99_tail_n"] < 10 and small["p99_reliable"] is False
    big = analyze.describe(range(2000))
    assert big["p99_reliable"] is True
    assert big["p99_ci95"][0] <= big["p99"] <= big["p99_ci95"][1]


def test_ratio_ci_brackets_the_ratio(analyze):
    import numpy as np

    rng = np.random.default_rng(0)
    solo, conc = rng.normal(10, 0.5, 500), rng.normal(12, 0.5, 500)
    r = analyze.ratio_ci(conc, solo)
    assert r["ci95"][0] < r["ratio"] < r["ci95"][1]
    assert r["ratio"] == pytest.approx(1.2, abs=0.02)


def test_columnar_round_trip(analyze):
    cols = {"a": [1, 2], "b": [None, "x"]}
    assert analyze.rows(cols) == [{"a": 1, "b": None}, {"a": 2, "b": "x"}]
    assert analyze.rows({}) == []


def test_compact_encoding_round_trips_every_request(analyze):
    compact = _load("compact")
    recs = [
        {
            "arrival": 10.0,
            "queue_enter": 10.5,
            "device_start": 11.0,
            "device_end": 21.03,
            "response": 21.2,
            "forward_ms": 9.98,
            "device": "ane",
            "in_window": True,
            "match": True,
            "seed": 3,
            "estimate_ms": 9.93,
        },
        {"arrival": 12.5, "response": 13.0, "device": "gpu", "in_window": False, "match": False, "seed": 4},
    ]
    phases = analyze.Phases({"kind": ["ane"], "t0": [11.01], "t1": [21.0], "cpu_ms": [0.5], "dev": [[[11.2, 20.9]]]})
    out = compact.decode(compact.encode(recs, phases))
    assert len(out) == 2
    a, b = out
    for k in ("arrival", "queue_enter", "device_start", "device_end", "response", "forward_ms", "estimate_ms"):
        assert a[k] == pytest.approx(recs[0][k], abs=0.006)
    assert a["device_exec"] == pytest.approx(9.7, abs=0.006) and a["host"] == pytest.approx(0.29, abs=0.006)
    assert a["match"] is True and a["seed"] == 3 and a["device"] == "ane"
    assert b["response"] == pytest.approx(13.0) and "device_start" not in b and b["match"] is False


def test_contention_fit_separates_additive_from_proportional():
    contention = _load("contention")
    a, d = contention._fit([(12.0, 20.0), (71.0, 79.0)])  # the same +8 ms at both lengths
    assert a == pytest.approx(0.0) and d == pytest.approx(8.0)
    a, d = contention._fit([(10.0, 11.0), (70.0, 77.0)])  # +10% at both lengths
    assert a == pytest.approx(0.1) and d == pytest.approx(0.0)
    a, d = contention._fit([(10.0, 9.9), (70.0, 69.0)])  # faster: clamped, never negative
    assert a == 0.0 and d == 0.0


# ----------------------------------------------------------------------------- request ledger analysis


@pytest.fixture(scope="module")
def ledger():
    return _load("ledger")


@pytest.fixture(scope="module")
def analyze_ledger():
    return _load("analyze_ledger")


# phase durations (us) per stream: dispatch, service, return; every request also has prepare 500,
# route 50, enqueue 20, queue 0, postprocess 300
_PHASES = {
    "solo_gpu": (100, 12000, 100),
    "solo_ane": (50, 10000, 50),
    "conc_gpu": (3000, 12500, 2000),
    "conc_ane": (50, 10000, 50),
}


def _trace(rid, target, submit, phases, other_busy):
    d, s, r = phases
    t = {"submit_us": submit}
    t["prepared_us"] = submit + 500
    t["routed_us"] = t["prepared_us"] + 50
    t["queue_enter_us"] = t["routed_us"] + 20
    t["dispatch_us"] = t["queue_enter_us"]
    t["service_start_us"] = t["dispatch_us"] + d
    t["service_end_us"] = t["service_start_us"] + s
    t["received_us"] = t["service_end_us"] + r
    t["response_us"] = t["received_us"] + 300
    other = "ane" if target == "gpu" else "gpu"
    snap = {f"{target}_backlog_ms": 1.0, f"{target}_queued_jobs": 0, f"{target}_running": False}
    snap.update(
        {
            f"{other}_backlog_ms": 9.0 if other_busy else None,
            f"{other}_queued_jobs": 1 if other_busy else None,
            f"{other}_running": True if other_busy else None,
        }
    )
    return {
        "request_id": rid,
        "sequence_length": 128,
        "question_count": 1,
        "target": target,
        "routing_reason": f"{target}_requested",
        "service_estimate_ms": 12.0 if target == "gpu" else 10.0,
        **snap,
        **t,
    }


def _columns(rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    return {k: [r.get(k) for r in rows] for k in keys}


def _synthetic_run():
    """solo:gpu_M (ids 1,2), solo:ane_B (3,4; 4 has no backend event), matrix:gpu_M+ane_B (5-8),
    and one backend forward without a request_id before the first submit."""
    plan = [  # id, device, submit us, phases, window index, stream
        (1, "gpu", 100_000, "solo_gpu", 0, "gpu_M"),
        (2, "gpu", 200_000, "solo_gpu", 0, "gpu_M"),
        (3, "ane", 1_100_000, "solo_ane", 1, "ane_B"),
        (4, "ane", 1_200_000, "solo_ane", 1, "ane_B"),
        (5, "gpu", 2_400_000, "conc_gpu", 2, "gpu_M"),
        (6, "gpu", 2_450_000, "conc_gpu", 2, "gpu_M"),
        (7, "ane", 2_405_000, "conc_ane", 2, "ane_B"),
        (8, "ane", 2_430_000, "conc_ane", 2, "ane_B"),
    ]
    windows = [
        {"cell": "solo:gpu_M", "kind": "solo", "measure": [0.0, 1.0]},
        {"cell": "solo:ane_B", "kind": "solo", "measure": [1.0, 2.0]},
        {"cell": "matrix:gpu_M+ane_B", "kind": "matrix", "measure": [2.0, 3.0]},
    ]
    for w in windows:
        w.update(
            cycle=0,
            t_start=w["measure"][0],
            stop=w["measure"][1],
            spec={},
            conditions_before={},
            aggressors=[],
            streams={},
        )
    traces, backend = (
        [],
        [
            {
                "request_id": None,
                "device": "gpu",
                "pid": 1,
                "tid": 1,
                "rows": 1,
                "max_len": 128,
                "t0_us": 50_000,
                "t1_us": 60_000,
                "cpu_us": 900,
                "spans": [[51_000, 59_000]],
            }
        ],
    )
    for rid, dev, submit, ph, wi, label in plan:
        t = _trace(rid, dev, submit, _PHASES[ph], other_busy=ph.startswith("conc"))
        traces.append(t)
        if rid != 4:
            t0, t1 = t["service_start_us"] + 10, t["service_end_us"] - 10
            backend.append(
                {
                    "request_id": rid,
                    "device": dev,
                    "pid": 2,
                    "tid": 3,
                    "rows": 1,
                    "max_len": 128,
                    "t0_us": t0,
                    "t1_us": t1,
                    "cpu_us": 1500,
                    "spans": [[t0 + 100, t1 - 400]],
                }
            )
        st = windows[wi]["streams"].setdefault(
            label, {"device": dev, "shape": label[-1], "mode": "closed", "rate": 0.0, "errors": [], "records": []}
        )
        st["records"].append(
            {
                "request_id": rid,
                "arrival_us": submit,
                "seed": rid,
                "shape": label[-1],
                "match": True,
                "in_window": True,
                "done_us": t["response_us"],
            }
        )
    for w in windows:
        for st in w["streams"].values():
            st["records"] = _columns(st["records"])
    return {
        "experiment": "gpu-ane-interference",
        "pipeline": "runtime-trace",
        "plan": "device",
        "args": {"model": "laya-typed-decisions", "ane_placement": "thread"},
        "environment": {},
        "t_ref_ns": 0,
        "windows": windows,
        "traces": _columns(traces),
        "backend": _columns(backend),
    }


def test_ledger_join_report_counts(ledger, analyze_ledger):
    records, rep = ledger.build(_synthetic_run())
    assert len(records) == rep["traces"] == 8
    assert rep["backend_events"] == 8 and rep["traces_joined_1to1"] == 7
    assert rep["traces_without_backend_event"] == 1 and rep["traces_with_several_backend_events"] == 0
    assert rep["backend_events_without_request_id"] == {"before first request": 1}
    assert rep["backend_events_without_trace"] == 0 and rep["backend_event_outside_service_window"] == 0
    assert rep["backend_device_mismatch"] == 0 and rep["join_rate"] == pytest.approx(7 / 8)
    join = analyze_ledger.join_section(rep)
    assert join["pass"] is False and join["checks"]["join_rate_is_1"] is False


def test_ledger_identities_and_unattributed(ledger):
    records, _ = ledger.build(_synthetic_run())
    for r in records:
        t = r["timing_ms"]
        assert t["occupancy"] == pytest.approx(t["dispatch"] + t["service"] + t["return"])
        parts = t["prepare"] + t["route"] + t["enqueue"] + t["queue"] + t["occupancy"] + t["postprocess"]
        assert t["e2e"] == pytest.approx(parts)
        if r["backend"] is not None:
            assert t["service"] == pytest.approx(t["device_exec"] + t["host"] + t["unattributed"])
            assert t["unattributed"] == pytest.approx(0.02)  # the hook's span is 10 us inside each end
    (no_be,) = [r for r in records if r["request_id"] == 4]
    assert no_be["timing_ms"]["forward"] is None and no_be["timing_ms"]["unattributed"] == no_be["timing_ms"]["service"]


def test_prediction_error_splits_exactly(ledger, analyze_ledger):
    records, _ = ledger.build(_synthetic_run())
    ledger.annotate_overlap(records)
    for r in records:
        e = analyze_ledger.prediction_errors(r)
        assert e["signed"] == pytest.approx(e["queue_error"] + e["service_error"], abs=1e-9)
    (r5,) = [r for r in records if r["request_id"] == 5]
    e = analyze_ledger.prediction_errors(r5)
    assert e["actual"] == pytest.approx(0.02 + 3.0 + 12.5 + 2.0)  # routed -> received
    assert e["predicted"] == pytest.approx(13.0) and e["signed"] == pytest.approx(4.52)
    assert e["queue_error"] == pytest.approx(0.02 - 1.0) and e["service_error"] == pytest.approx(17.5 - 12.0)
    assert analyze_ledger.group_values(r5)["target|other_device"] == "gpu|busy"
    pred = analyze_ledger.prediction_section({"x": {"plan": "device", "raw_files": ["x"], "records": records}})
    assert pred["identity"]["n"] == 8 and pred["identity"]["max_abs_residual_ms"] < 1e-6
    assert pred["scopes"]["x"]["groups"]["target|other_device"]["gpu|busy"]["n"] == 2


def test_decomposition_delta_and_timeline(analyze_ledger):
    hist = {
        "runs": {
            "device:laya-typed-decisions:thread": {
                "inflation": {
                    "matrix:gpu_M+ane_B": {
                        "gpu_M": {"dispatch_delta_ms": 7.4, "host_delta_ms": 0.02, "service": {"mean": {"ratio": 1.64}}}
                    }
                }
            }
        }
    }
    entry, inwin = analyze_ledger.analyse_run(_synthetic_run(), hist)
    assert len(inwin) == 8
    d = entry["decomposition"]["matrix:gpu_M+ane_B"]["gpu_M"]
    c = d["components"]
    assert c["dispatch"]["delta"]["delta_ms"] == pytest.approx(2.9)
    assert c["return"]["delta"]["delta_ms"] == pytest.approx(1.9)
    assert c["dispatch_plus_return"]["delta"]["delta_ms"] == pytest.approx(4.8)
    assert c["occupancy"]["delta"]["delta_ms"] == pytest.approx(5.3)
    assert c["device_exec"]["delta"]["delta_ms"] == pytest.approx(0.5)  # 11.48 -> 11.98 ms of spans
    assert c["occupancy"]["ratio"]["ratio"] == pytest.approx(17.5 / 12.2)
    assert d["share_of_occupancy_inflation"]["dispatch"] == pytest.approx(2.9 / 5.3)
    assert d["share_of_occupancy_inflation"]["sum"] == pytest.approx(1.0)
    assert d["comparison"]["dispatch"] == {
        "historical_dispatch_delta_ms": 7.4,
        "new_dispatch_plus_return_delta_ms": pytest.approx(4.8),
    }
    assert d["comparison"]["service_ratio"]["historical_service_mean_ratio"] == 1.64
    ane = entry["decomposition"]["matrix:gpu_M+ane_B"]["ane_B"]
    assert ane["excluded_without_backend"] == {"solo": 1, "concurrent": 0}  # request 4: no backend record
    assert ane["components"]["unattributed"]["solo_mean"] == pytest.approx(0.02)
    assert ane["share_of_occupancy_inflation"]["dispatch"] is None  # occupancy did not change
    assert entry["validation"]["ane"]["n_with_backend"] == 3 and entry["validation"]["gpu"]["forward_gt_service"] == 0
    tl = entry["timeline"]
    assert tl["cell"] == "matrix:gpu_M+ane_B" and tl["selection"] == "first matrix window"
    assert tl["t0_ms"] == pytest.approx(2350.0)
    assert [q["request_id"] for q in tl["lanes"]["gpu"]] == [5, 6]
    assert [q["request_id"] for q in tl["lanes"]["ane"]] == [7, 8]
    q5 = tl["lanes"]["gpu"][0]
    assert q5["start_ms"] == pytest.approx(50.0) and dict(q5["segments"])["dispatch"] == pytest.approx(3.0)


def test_ledger_outputs_and_figures_render(analyze_ledger, tmp_path):
    import gzip
    import json

    path = tmp_path / "device-laya-typed-decisions-thread.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump(_synthetic_run(), f)
    res = analyze_ledger.analyse_all([path], None)
    res = json.loads(json.dumps(res, sort_keys=True))  # what figures_ledger reads back
    assert set(res["runs"]) == {"device:laya-typed-decisions:thread"}
    assert "join" in res["runs"]["device:laya-typed-decisions:thread"]
    tables, table_csv = analyze_ledger.render_tables(res), analyze_ledger.render_csv(res)
    assert path.name in tables and "gpu|busy" in table_csv
    figures = _load("figures_ledger")
    for fn in figures.FIGURES.values():
        out = fn(res)
        assert out.startswith("<svg") and "<title" in out


def test_return_alignment_measures_the_wait_for_the_ane_predict(analyze_ledger):
    """GPU forward 1 ends 4 ms before an ANE predict ends and gets its result 0.1 ms after it;
    GPU forward 2 ends while no predict runs."""
    ane = {
        "request_id": 9,
        "device": "ane",
        "pid": 1,
        "tid": 1,
        "rows": 1,
        "max_len": 128,
        "t0_us": 0,
        "t1_us": 20_000,
        "cpu_us": 0,
        "spans": [[1_000, 10_000]],
    }
    run = {"backend": _columns([ane])}
    cell = "matrix:gpu_M+ane_B"
    recs = [
        {"device": "gpu", "cell": cell, "t_ms": {"service_end": 6.0, "received": 10.1}},
        {"device": "gpu", "cell": cell, "t_ms": {"service_end": 15.0, "received": 15.2}},
        {"device": "ane", "cell": cell, "t_ms": {"service_end": 10.0, "received": 10.0}},
    ]
    e = analyze_ledger.return_alignment_section(run, recs)[cell]
    assert e["gpu_requests"] == 2 and e["ended_inside_ane_predict"] == 1
    assert e["received_minus_predict_end_ms"]["50"] == pytest.approx(0.1)
    assert e["return_mean_ms"] == pytest.approx(4.1) and e["predict_remaining_mean_ms"] == pytest.approx(4.0)
    assert e["return_mean_ms_when_no_predict_running"] == pytest.approx(0.2)
