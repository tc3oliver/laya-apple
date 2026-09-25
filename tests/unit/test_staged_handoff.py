"""Unit tests for research/coreml-staged-handoff/ (research only): the handoff state machine
(boundaries, episodes, re-arm, GPU activity, concurrency), the fixed policies, the run order,
run_config.py's import without PyObjC and its per-bucket model switch, and analyze.py on
synthetic runs (#94's synthetic run with handoff decisions): every gate, structural validity, A
validity, the B phenotype, leader selection, the stop rule and round 2 with its fallback."""

from __future__ import annotations

import gzip
import importlib.util
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "research" / "coreml-staged-handoff" / "scripts"
S = 1_000_000_000


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


handoff = _load("staged_handoff_for_tests", SCRIPTS / "handoff.py")
design = _load("staged_handoff_design_for_tests", SCRIPTS / "design.py")


class Clock:
    def __init__(self):
        self.t = 10 * S

    def __call__(self):
        return self.t

    def step(self, s: float):
        self.t += int(s * S)


def machine(guard=32, gap_s=1.0):
    clock = Clock()
    gpu = handoff.GpuActivity(clock)
    return handoff.StagedHandoff(guard, gpu, gap_s=gap_s, clock=clock), gpu, clock


def paths(m, clock, n, dt=0.01):
    out = []
    for _ in range(n):
        out.append(m.decide()[0])
        clock.step(dt)
    return out


def test_armed_runs_sync_without_gpu():
    m, _, clock = machine()
    assert paths(m, clock, 100) == ["sync"] * 100
    assert m.state == handoff.ARMED and m.episodes == 0


@pytest.mark.parametrize("guard", [32, 64])
def test_guard_boundary(guard):
    m, gpu, clock = machine(guard)
    gpu.started()
    p = paths(m, clock, guard + 3)
    assert p[:guard] == ["sync"] * guard  # forward N is still sync
    assert p[guard:] == ["async"] * 3  # forward N + 1 is the first async
    assert m.episodes == 1


def test_count_and_state_at_boundary():
    m, gpu, clock = machine(32)
    gpu.started()
    rows = []
    for _ in range(33):
        rows.append(m.decide())
        clock.step(0.01)
    assert rows[30] == ("sync", handoff.SYNC_GUARD, 31)
    assert rows[31] == ("sync", handoff.ASYNC_STEADY, 32)
    assert rows[32] == ("async", handoff.ASYNC_STEADY, 32)


def test_gpu_active_within_gap_after_last_end():
    m, gpu, clock = machine(2)
    gpu.started()
    gpu.ended()  # GPU idle, but its last job ended just now
    clock.step(0.5)
    assert paths(m, clock, 3) == ["sync", "sync", "async"]


def test_gpu_idle_longer_than_gap_rearms():
    m, gpu, clock = machine(2)
    gpu.started()
    assert paths(m, clock, 3) == ["sync", "sync", "async"]
    gpu.ended()
    clock.step(1.5)  # hetero -> solo_short
    assert paths(m, clock, 3) == ["sync"] * 3
    assert m.state == handoff.ARMED
    gpu.started()  # hetero again: a new guard
    assert paths(m, clock, 3) == ["sync", "sync", "async"]
    assert m.episodes == 2


def test_short_pause_rearms_while_gpu_stays_busy():
    m, gpu, clock = machine(2)
    gpu.started()
    assert paths(m, clock, 3) == ["sync", "sync", "async"]
    clock.step(2.0)  # hetero -> solo_long -> hetero: shorts pause, GPU busy throughout
    assert paths(m, clock, 3) == ["sync", "sync", "async"]
    assert m.episodes == 2


def test_hetero_ends_before_guard():
    m, gpu, clock = machine(32)
    gpu.started()
    assert paths(m, clock, 10) == ["sync"] * 10
    gpu.ended()
    clock.step(1.5)
    assert paths(m, clock, 5) == ["sync"] * 5 and m.state == handoff.ARMED
    gpu.started()
    p = paths(m, clock, 33)
    assert p[:32] == ["sync"] * 32 and p[32] == "async"  # the count restarted


def test_gap_is_inclusive_at_one_second():
    m, gpu, clock = machine(1)
    gpu.started()
    gpu.ended()
    clock.step(1.0)
    assert m.decide()[1] == handoff.ASYNC_STEADY  # guard 1: the episode started, then ASYNC_STEADY


def test_gpu_inflight_counts_concurrent_jobs():
    clock = Clock()
    gpu = handoff.GpuActivity(clock)
    gpu.started()
    gpu.started()
    gpu.ended()
    clock.step(5)
    assert gpu.active(clock(), S)  # one still in flight
    gpu.ended()
    gpu.ended()  # an extra end never goes negative
    clock.step(5)
    assert not gpu.active(clock(), S)


def test_concurrent_decisions_count_exactly():
    guard = 500
    gpu = handoff.GpuActivity()
    gpu.started()
    m = handoff.StagedHandoff(guard, gpu)
    out: list = []
    lock = threading.Lock()

    def worker():
        mine = [m.decide()[0] for _ in range(200)]
        with lock:
            out.extend(mine)

    ts = [threading.Thread(target=worker) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert out.count("sync") == guard and out.count("async") == 1600 - guard
    assert m.episodes == 1


def test_guard_must_be_positive():
    with pytest.raises(ValueError):
        handoff.StagedHandoff(0, handoff.GpuActivity())


def test_fixed_policies():
    assert handoff.Fixed("sync").decide()[0] == "sync"
    assert handoff.Fixed("async").decide()[0] == "async"
    with pytest.raises(ValueError):
        handoff.Fixed("other")


# ------------------------------------------------------------------ design


def test_round1_is_mirrored_and_balanced():
    r = design.ROUND1
    assert [c for c, _ in r] == ["H32", "A", "H64", "B", "B", "H64", "A", "H32"]
    for c in design.CELLS:
        assert sum(1 for x, _ in r if x == c) == 2
    assert len(set(r)) == len(r)


def test_round2_fallback_has_its_own_a_runs():
    assert design.round2("H64", fallback=True) == (("H64", 3), ("A", 5), ("H64", 4), ("A", 6), ("H64", 5))
    with pytest.raises(ValueError):
        design.round2("H32", fallback=True)


def test_round2():
    assert design.round2("H64") == (("H64", 3), ("A", 3), ("H64", 4), ("A", 4), ("H64", 5))
    with pytest.raises(ValueError):
        design.round2("B")


# ------------------------------------------------------------------ run_config


def test_run_config_imports_without_pyobjc_and_switches_models():
    rc = _load("staged_handoff_rc_for_tests", SCRIPTS / "run_config.py")
    assert rc.CELLS == ("A", "B", "H32", "H64")

    class M:
        def __init__(self, name):
            self.name = name

        def predict(self, feats):
            return self.name

    route = rc.Route()
    hm = rc.HandoffModel(M("sync"), M("async"), route)
    assert hm.predict({}) == "sync"
    route.path = "async"
    assert hm.predict({}) == "async"
    assert hm.name == "sync"  # other attributes come from the sync model
    gpu = rc.handoff.GpuActivity()
    assert isinstance(rc.make_policy("H64", gpu), rc.handoff.StagedHandoff)
    assert rc.make_policy("H64", gpu).guard == 64
    assert rc.make_policy("A", gpu).decide()[0] == "sync"
    assert rc.make_policy("B", gpu).decide()[0] == "async"


# ------------------------------------------------------------------ analyze.py on synthetic runs

analyze = _load("staged_handoff_analyze_for_tests", SCRIPTS / "analyze.py")
t94 = _load("staged_handoff_t94_for_tests", ROOT / "tests" / "unit" / "test_async_transient.py")
MS = 1_000_000


def _fake_run(cell, slow_s=1.0, guard=None):
    """#94's synthetic run (a short request every 50 ms in each 20 s hetero window, one ANE forward
    per request) as cell A / B / H32 / H64, with one decision per forward: H's first `guard`
    forwards of each hetero window sync (their predicts without a native stamp), async after.
    Host-slow requests for the first slow_s seconds of each hetero window."""
    run = t94._fake_run("A" if cell == "A" else "PB-ASYNC")
    res = run["research"]
    res["base"] = {"experiment": res["experiment"], "cell": "PB-ASYNC"}
    res["experiment"], res["cell"] = "coreml-staged-handoff", cell
    for s in res["recount"]["series"].values():  # 50 ms CPU per 100 ms sample: active threads
        s["rows"] = [[r[0], *(v * 100_000 for v in r[1:])] for r in s["rows"]]
    tr = res["trace"]
    for w in res["windows"]:
        for i, sub in enumerate(tr["submit_ns"]):
            if tr["target"][i] == "ane" and w["start_ns"] <= sub < w["end_ns"]:
                slow = sub < w["start_ns"] + int(slow_s * S)
                tr["prepared_ns"][i] = sub + (600_000 if slow else 100_000)
    n = guard if guard is not None else analyze.GUARDS.get(cell)
    rows = []
    pr = res["predicts"]
    for w in res["windows"]:
        fw = sorted(x[0] for x in res["forwards"]["ane"] if w["start_ns"] <= x[0] < w["end_ns"])
        for k, ns in enumerate(fw):
            if cell == "A":
                rows.append([ns, 0, "fixed", 0])
            elif cell == "B":
                rows.append([ns, 1, "fixed", 0])
            elif k < n:
                rows.append([ns, 0, "async_steady" if k == n - 1 else "sync_guard", k + 1])
                j = next(i for i, p in enumerate(pr) if p[0] == ns + 100_000)
                pr[j] = (pr[j][0], 0, 0, 0, 0, 0, pr[j][6])
            else:
                rows.append([ns, 1, "async_steady", n])
    res["handoff"] = {
        "cell": cell,
        "guard": analyze.GUARDS.get(cell),
        "installed": True,
        "errors": [],
        "episodes": 2 if cell in analyze.GUARDS else 0,
        "forwards": len(rows),
        "forwards_sync": sum(1 for r in rows if r[1] == 0),
        "forwards_async": sum(1 for r in rows if r[1] == 1),
        "decisions": rows,
    }
    return run


@pytest.fixture(scope="module")
def a_ref():
    a = [analyze.run_stats(_fake_run("A"), f"laya-A-r{i}", "A") for i in (1, 2)]
    return analyze.references(a)


def _gates(run, a_ref, cell="H32", crashed=False):
    st = analyze.run_stats(run, f"laya-{cell}-r1", cell, crashed)
    return st, [analyze.gates(t, st, a_ref) for t in st["transitions"]]


def test_references_and_allowed(a_ref):
    assert a_ref["A_onset_p99_ms"] == pytest.approx(15.0) and a_ref["A_window_p99_ms"] == 12.0
    assert a_ref["A_predict_mean_ms"] == pytest.approx(8.0) and a_ref["A_agg_req_s"] == pytest.approx(122.4)
    assert analyze.allowed(12.0) == pytest.approx(13.0)  # the absolute rule is more permissive here
    assert analyze.allowed(30.0) == pytest.approx(31.5)  # the ratio is more permissive here


def test_gates_pass_and_measures(a_ref):
    st, gs = _gates(_fake_run("H32"), a_ref)
    t = st["transitions"][0]
    s = t["structure"]
    assert s["valid"] and s["episodes"] == 1 and s["sync_before_t_h"] == 32
    assert t["t_h_s"] == pytest.approx(1.6003)  # forward 33's decision (submit 1.6 s + 0.3 ms)
    assert t["native_mean_ms"] == pytest.approx(7.94) and t["native_n"] == 400 - 32
    assert t["transient"]["duration_s"] == 1.0 and t["transient_from_th"]["duration_s"] == 0.0
    assert t["gpu_return_ms"]["p50"] == pytest.approx(0.3) and t["gpu_return_ms"]["from_s"] == pytest.approx(2.6003)
    assert [g["pass"] for g in gs] == [True, True] and all(not g["failed"] for g in gs)
    assert set(gs[0]["gates"]) == set(analyze.GATES)
    assert t["mechanism"]["p_dominant_stays"]


def test_gates_pending_without_references():
    _, gs = _gates(_fake_run("H32"), None)
    assert gs[0]["pass"] is None
    assert set(gs[0]["pending"]) == {"onset_p99", "window_p99", "native_ratio", "native_plus", "throughput"}


def _het(run):
    return [w for w in run["part_a"]["windows"] if w["condition"] == "hetero"]


def _onset(run):
    lat = _het(run)[0]["streams"]["short"]["latency_ms"]
    lat[:80] = [30.0] * 80  # the first 4 s of cycle 0


def _window(run):
    _het(run)[1]["streams"]["short"]["p99_ms"] = 13.5


def _native(delta_ms):
    def mod(run):
        pr = run["research"]["predicts"]
        for i, p in enumerate(pr):
            if p[3] > 0:
                pr[i] = (p[0], p[1], p[2], p[3] + int(delta_ms * MS), *p[4:])

    return mod


def _gpu(run):
    run["gpu_return"]["return_us"] = [2000] * len(run["gpu_return"]["return_us"])


def _throughput(run):
    for s in _het(run)[0]["streams"].values():
        s["req_s"] = 55.0


def _mismatch(run):
    run["part_a"]["windows"][0]["streams"]["short"]["mismatches"] = 1  # solo_short: every window counts


def _routing(run):
    run["part_a"]["windows"][0]["streams"]["short"]["devices"] = {"gpu": 1}


def _steady(run):
    tr = run["research"]["trace"]
    t0 = run["research"]["windows"][1]["start_ns"]
    for i, sub in enumerate(tr["submit_ns"]):
        if tr["target"][i] == "ane" and t0 + 12 * S <= sub < t0 + 14 * S:
            tr["prepared_ns"][i] = sub + 600_000


@pytest.mark.parametrize(
    "mod, kw, failed",
    [
        (_onset, {}, {0: ["onset_p99"]}),
        (_window, {}, {1: ["window_p99"]}),
        (_native(1.0), {}, {0: ["native_ratio", "native_plus"], 1: ["native_ratio", "native_plus"]}),
        (_native(0.41), {}, {0: ["native_plus"], 1: ["native_plus"]}),  # both must hold
        (None, {"slow_s": 6.0}, {0: ["transient_from_th"], 1: ["transient_from_th"]}),
        (_steady, {}, {1: ["transient_from_th", "steady_host_slow"]}),
        (_gpu, {}, {0: ["gpu_return_p50"], 1: ["gpu_return_p50"]}),
        (_throughput, {}, {0: ["throughput"]}),
        (_mismatch, {}, {0: ["mismatches"], 1: ["mismatches"]}),
        (_routing, {}, {0: ["routing"], 1: ["routing"]}),
    ],
)
def test_each_gate_fails(a_ref, mod, kw, failed):
    run = _fake_run("H32", **kw)
    if mod:
        mod(run)
    _, gs = _gates(run, a_ref)
    assert {i: g["failed"] for i, g in enumerate(gs) if g["failed"]} == failed
    assert all(g["pass"] is (i not in failed) for i, g in enumerate(gs))


def test_crash_fails_correctness(a_ref):
    _, gs = _gates(_fake_run("H32"), a_ref, crashed=True)
    assert [g["failed"] for g in gs] == [["crash"], ["crash"]]


def _drop_guard_decision(run):
    rows = run["research"]["handoff"]["decisions"]
    del rows[5]  # cycle 0: 31 sync forwards before t_h


def _second_episode(run):
    rows = run["research"]["handoff"]["decisions"]
    rows[450][1:] = [0, "sync_guard", 1]  # cycle 1: a second episode starts after the first


def _no_handoff(run):
    rows = run["research"]["handoff"]["decisions"]
    for r in rows[400:]:
        r[1:] = [0, "sync_guard", 1 if r is rows[400] else 2]


@pytest.mark.parametrize(
    "mod, cycle, check",
    [
        (_drop_guard_decision, 0, lambda s: s["sync_before_t_h"] == 31),
        (_second_episode, 1, lambda s: s["episodes"] == 2),
        (_no_handoff, 1, lambda s: s["t_h_ns"] is None),
    ],
)
def test_structural_invalidity(a_ref, mod, cycle, check):
    run = _fake_run("H32")
    mod(run)
    st, gs = _gates(run, a_ref)
    s = st["transitions"][cycle]["structure"]
    assert not s["valid"] and check(s)
    assert "structure" in gs[cycle]["failed"] and gs[cycle]["pass"] is False
    assert gs[1 - cycle]["pass"]


def test_armed_decision_inside_the_guard_is_invalid(a_ref):
    run = _fake_run("H32")
    run["research"]["handoff"]["decisions"][10][2] = "armed"
    st, _ = _gates(run, a_ref)
    assert not st["transitions"][0]["structure"]["valid"]


def test_mechanism_reading_on_e_cores(a_ref):
    run = _fake_run("H32")
    for s in run["research"]["recount"]["series"].values():
        s["rows"] = [[r[0], 0, 0, 0, 0, *r[1:5]] for r in s["rows"]]  # all CPU on E
    st, gs = _gates(run, a_ref)
    m = st["transitions"][0]["mechanism"]
    assert not m["p_dominant_stays"] and m["spans"]["parent"]["0-0.5"]["e_share"] == pytest.approx(1.0)
    assert m["switch_s"]["parent"] == "never"
    assert all(g["pass"] for g in gs)  # report only: never a gate


def test_a_validity_and_b_phenotype():
    a = analyze.run_stats(_fake_run("A"), "laya-A-r1", "A")
    assert analyze.a_validity([a])["valid"]
    run = _fake_run("A", slow_s=1.5)
    _het(run)[0]["streams"]["short"]["p99_ms"] = 13.0
    run["part_a"]["windows"][0]["streams"]["short"]["mismatches"] = 1
    v = analyze.a_validity([analyze.run_stats(run, "laya-A-r2", "A")])
    assert not v["valid"]
    assert v["per_transition"][0]["failed"] == ["short_p99", "transient", "no_mismatch"]
    b = [analyze.run_stats(_fake_run("B", slow_s=s), f"laya-B-r{i}", "B") for i, s in ((1, 2.0), (2, 1.5))]
    bp = analyze.b_phenotype(b)
    assert bp["n_ge_2s"] == 2 and bp["reproduced"]
    assert b[0]["transitions"][0]["t_h_s"] == 0.0 and b[0]["transitions"][0]["structure"]["valid"] is None
    assert not analyze.b_phenotype([b[1], b[1]])["reproduced"]


# end to end: files in a raw directory


def _write(raw, cell, rep, mod=None, **kw):
    run = _fake_run(cell, **kw)
    if mod:
        mod(run)
    with gzip.open(raw / design.run_file(cell, rep), "wt") as fh:
        json.dump(run, fh)


@pytest.fixture
def raw(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    return d


def _round1(raw, h32=None, h64=None, a2=None, b_slow=3.0):
    for cell, rep in design.ROUND1:
        mod = {"H32": h32, "H64": h64, "A": a2 if rep == 2 else None}.get(cell)
        _write(raw, cell, rep, mod, **({"slow_s": b_slow} if cell == "B" else {}))


def test_partial_raw_is_pending(raw):
    res = analyze.summarise(raw)
    assert res["outcome"].startswith("round 1 pending: H32 r1, A r1") and res["round2"] == []
    _write(raw, "H32", 1)
    res = analyze.summarise(raw)
    assert "H32 r1" not in res["outcome"] and res["round1"]["candidates"]["H32"]["transitions"][0]["pending"]
    assert "pending" in analyze.tables(res)
    json.dumps(res, default=analyze.a94._json)


def test_round1_leader_h32_before_h64(raw):
    _round1(raw)
    res = analyze.summarise(raw)
    r1 = res["round1"]
    assert r1["A_validity"]["valid"] and r1["B_phenotype"]["reproduced"]
    assert r1["candidates"]["H32"]["pass"] and r1["candidates"]["H64"]["pass"]
    assert r1["leader"] == "H32" and r1["decision"].startswith("leader H32; round 2 runs: H32 r3, A r3")
    assert res["outcome"].startswith("round 2 (H32) pending: H32 r3")
    md = analyze.tables(res)
    assert "### H32" in md and "A validity: **True**" in md and "B phenotype: **True**" in md


def test_round1_leader_h64_when_h32_fails(raw):
    _round1(raw, h32=_window)
    r1 = analyze.summarise(raw)["round1"]
    assert r1["leader"] == "H64"
    assert r1["candidates"]["H32"]["failing"] == ["laya-H32-r1 c1: window_p99", "laya-H32-r2 c1: window_p99"]


def test_round1_stop(raw):
    _round1(raw, h32=_gpu, h64=_throughput)
    res = analyze.summarise(raw)
    assert res["round1"]["leader"] is None and res["outcome"] == analyze.STOP


def test_round1_a_invalid_is_inconclusive(raw):
    _round1(raw, a2=_window)  # A r2's cycle-1 window P99 13.5 ms >= 13 ms
    res = analyze.summarise(raw)
    assert res["outcome"] == analyze.INCONCLUSIVE_A and res["round1"]["leader"] is None
    assert res["round1"]["candidates"]["H32"]["pass"]  # still reported, not judged


def test_round1_b_phenotype_missing_is_inconclusive(raw):
    _round1(raw, b_slow=1.0)
    res = analyze.summarise(raw)
    assert res["outcome"] == analyze.INCONCLUSIVE_B and res["round1"]["B_phenotype"]["n_ge_2s"] == 0


def test_round2_fallback_and_replication(raw):
    _round1(raw)
    for cell, rep in design.round2("H32"):
        _write(raw, cell, rep, _gpu if (cell, rep) == ("H32", 4) else None)
    res = analyze.summarise(raw)
    assert [r["candidate"] for r in res["round2"]] == ["H32", "H64"]
    assert res["round2"][0]["passed"] is False and res["outcome"].startswith("round 2 (H64) pending: H64 r3")
    assert len(res["round2"][0]["references"]["transitions"]) == 8
    for cell, rep in design.round2("H64", fallback=True):
        _write(raw, cell, rep)
    res = analyze.summarise(raw)
    assert res["round2"][1]["fallback"] and len(res["round2"][1]["references"]["transitions"]) == 8
    assert res["outcome"] == analyze.REPLICATES.format(c="H64") and res["round2"][1]["passed"]
    assert res["unexpected_files"] == [] and "## Round 2: H64" in analyze.tables(res)


def test_round2_h32_a_invalid_with_a_independent_failure_runs_fallback(raw):
    _round1(raw)
    for cell, rep in design.round2("H32"):
        mod = _window if (cell, rep) == ("A", 3) else _gpu if (cell, rep) == ("H32", 4) else None
        _write(raw, cell, rep, mod)
    res = analyze.summarise(raw)
    assert res["round2"][0]["valid"] is False
    assert "gpu_return_p50" in " ".join(res["round2"][0]["a_independent_failures"])
    assert "H64 fallback runs (addendum 4)" in res["outcome"] and "round 2 (H64) pending" in res["outcome"]


def test_round2_failure_closes(raw):
    _round1(raw, h32=_gpu)
    for cell, rep in design.round2("H64"):
        _write(raw, cell, rep, _throughput if cell == "H64" and rep == 5 else None)
    res = analyze.summarise(raw)
    assert res["outcome"] == f"no candidate replicates: {analyze.CLOSED}"


def test_crash_log_and_protocol_deviation(raw):
    _round1(raw)
    (raw / "failed").mkdir()
    (raw / "failed" / "laya-H32-r2.1.log").write_text("x\n")
    (raw / "failed" / "laya-A-r1.1.log").write_text("x\n")
    res = analyze.summarise(raw)
    assert res["runs"]["laya-H32-r2"]["crashed"] and not res["runs"]["laya-A-r1"]["crashed"]
    assert res["round1"]["leader"] == "H64"
    run = _fake_run("H64")
    run["args"]["seconds"] = 5.0
    with gzip.open(raw / design.run_file("H64", 1), "wt") as fh:
        json.dump(run, fh)
    res = analyze.summarise(raw)
    assert "protocol deviation (not judged): H64 r1" in res["outcome"]


def test_check_mode(raw, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    script = SCRIPTS / "analyze.py"
    subprocess.run([sys.executable, script, "--raw", raw, "--out", out], check=True, capture_output=True)
    ok = subprocess.run([sys.executable, script, "--raw", raw, "--out", out, "--check"], capture_output=True)
    assert ok.returncode == 0
    (out / "tables.md").write_text("x\n")
    stale = subprocess.run([sys.executable, script, "--raw", raw, "--out", out, "--check"], capture_output=True)
    assert stale.returncode != 0 and b"stale" in stale.stderr


def test_rejects_other_experiments(raw):
    run = _fake_run("B")
    run["research"]["experiment"] = "coreml-dependency-qos"
    with gzip.open(raw / design.run_file("B", 1), "wt") as fh:
        json.dump(run, fh)
    with pytest.raises(AssertionError):
        analyze.summarise(raw)


# ------------------------------------------------------------------ prod_analyze.py on synthetic records

prod = _load("staged_handoff_prod_analyze_for_tests", SCRIPTS / "prod_analyze.py")
STEP = 50 * MS  # one closed-loop short request every 50 ms, one long every 100 ms


def _prod_record(phase, cell):
    """A synthetic prod_run.py record of `phase` as cell A / P that passes every gate: short
    latency 5 ms, prepare 0.1 ms, ANE forward 2 ms, GPU return 0.5 ms, 100 + 20 req/s in hetero;
    P logs N sync decisions then async in each hetero window, "armed" ones in solo_short."""
    cfg = prod.PHASES[phase]
    enabled = cell == "P" and phase != "5b"
    t, windows, trace, dec = 100 * S, [], [], []
    for i, w in enumerate(prod.prod_run.schedule(cfg["schedule"], cfg["seconds"])):
        t += int((w["gap_s"] + 0.5) * S)
        t0, end = t, t + int(w["seconds"] * S)
        t = end
        inst, streams = prod.prod_run.STREAMS[w["condition"]]
        rec = {"index": i, "cycle": w.get("cycle"), "condition": w["condition"], "instance": inst}
        rec.update(start_ns=t0, end_ns=end, streams={})
        for s in streams:
            dev = "gpu" if inst == "gpu" or s == "long" else "ane"
            step = STEP if s == "short" else 2 * STEP
            subs = list(range(t0 + MS, end - 10 * MS, step))
            rec["streams"][s] = {
                "latency_ms": [5.0] * len(subs),
                "p99_ms": 5.0,
                "req_s": 100.0 if s == "short" else 20.0,
                "devices": {dev: len(subs)},
                "mismatches": 0,
            }
            if inst != "auto":
                continue
            for k, sub in enumerate(subs):
                if dev == "ane":
                    trace.append(
                        [
                            "ane",
                            sub,
                            sub + MS // 10,
                            sub + MS // 5,
                            sub + 2 * MS + MS // 5,
                            sub + 2400_000,
                            sub + 2500_000,
                        ]
                    )
                    if not enabled:
                        continue
                    if w["condition"] != "hetero":
                        dec.append([sub, 0, "armed", 0])
                    elif k < prod.GUARD:
                        dec.append([sub, 0, "async_steady" if k == prod.GUARD - 1 else "sync_guard", k + 1])
                    else:
                        dec.append([sub, 1, "async_steady", prod.GUARD])
                else:
                    trace.append(
                        ["gpu", sub, sub + MS // 10, sub + MS, sub + 40 * MS, sub + 40 * MS + MS // 2, sub + 41 * MS]
                    )
        windows.append(rec)
    return {
        "experiment": "coreml-staged-handoff production",
        "args": {
            "cell": cell,
            "model": cfg["model"],
            "short": cfg["short"],
            "long": cfg["long"],
            "schedule": cfg["schedule"],
            "seconds": cfg["seconds"],
        },
        "info_end": {"ane_placement": "process" if phase == "5b" else "thread"},
        "workers_alive_at_end": {"ane": True, "gpu": True},
        "handoff_snapshot": {"guard": prod.GUARD, "disabled": False, "disabled_reason": None} if enabled else None,
        "windows": windows,
        "trace_columns": prod.prod_run.TRACE_COLUMNS,
        "trace": sorted(trace, key=lambda r: r[1]),
        "decisions_columns": ["decision_ns", "path", "state", "count"],
        "decisions": dec,
        "run_wall_s": 1.0,
    }


_RECORDS: dict = {}


def _record(phase, cell):
    if (phase, cell) not in _RECORDS:
        _RECORDS[(phase, cell)] = json.dumps(_prod_record(phase, cell))
    return json.loads(_RECORDS[(phase, cell)])


def _write_prod(raw, phase, cell, rep, mod=None):
    run = _record(phase, cell)
    if mod:
        mod(run)
    with gzip.open(raw / (prod.run_name(phase, cell, rep) + ".json.gz"), "wt") as fh:
        json.dump(run, fh)


def _write_phases(raw, phases, mods=None):
    mods = mods or {}
    for ph in phases:
        for cell, rep in prod.RUNS[ph]:
            _write_prod(raw, ph, cell, rep, mods.get((ph, cell, rep)))


def _hetero(run):
    return [w for w in run["windows"] if w["condition"] == "hetero"]


def _rows(run, lo, hi, target="ane"):
    return [r for r in run["trace"] if r[0] == target and lo <= r[1] < hi]


def _slow_client(lo_s, hi_s, ms):
    def mod(run):
        for w in _hetero(run):
            subs = [r[1] for r in _rows(run, w["start_ns"], w["end_ns"])]
            lat = w["streams"]["short"]["latency_ms"]
            for k, sub in enumerate(subs):
                if w["start_ns"] + lo_s * S <= sub < w["start_ns"] + hi_s * S:
                    lat[k] = ms

    return mod


def _slow_prepare(lo_s, hi_s, every=1):
    def mod(run):
        for w in _hetero(run):
            for k, r in enumerate(_rows(run, w["start_ns"] + int(lo_s * S), w["start_ns"] + int(hi_s * S))):
                if k % every == 0:
                    r[2] = r[1] + MS

    return mod


def _slow_forward(run):
    for w in _hetero(run):
        for r in _rows(run, w["start_ns"] + 2 * S, w["end_ns"]):
            r[4] += MS


def _slow_gpu_return(run):
    for r in run["trace"]:
        if r[0] == "gpu":
            r[5] += 2 * MS


def _low_throughput(run):
    for w in _hetero(run):
        w["streams"]["long"]["req_s"] = 5.0


def _mismatch(run):
    run["windows"][0]["streams"]["short"]["mismatches"] = 1


def _misroute(run):
    w = next(w for w in run["windows"] if w["condition"] == "solo_long")
    w["streams"]["long"]["devices"] = {"gpu": 10, "ane": 1}


def _dead_worker(run):
    run["workers_alive_at_end"] = {"ane": False, "gpu": True}


def _disabled(run):
    run["handoff_snapshot"] = {"guard": prod.GUARD, "disabled": True, "disabled_reason": "worker error"}


def _drop_guard_decision(run):
    w = _hetero(run)[0]
    k = next(i for i, d in enumerate(run["decisions"]) if d[0] >= w["start_ns"] and d[3] == 5)
    del run["decisions"][k]


PROD_GATE_FAILS = [
    (_slow_client(0.0, 1.0, 12.0), "onset_p99"),
    (_slow_client(15.0, 16.0, 12.0), "window_p99"),
    (_slow_forward, "forward_ratio"),
    (_slow_forward, "forward_plus"),
    (_slow_prepare(1.7, 3.7), "transient_from_th"),
    (_slow_prepare(10.0, 20.0, every=5), "steady_host_slow"),
    (_slow_gpu_return, "gpu_return_p50"),
    (_low_throughput, "throughput"),
    (_mismatch, "mismatches"),
    (_misroute, "routing"),
    (_dead_worker, "workers_alive"),
    (_disabled, "handoff_enabled"),
    (_drop_guard_decision, "structure"),
]


@pytest.fixture
def raw_prod(tmp_path):
    d = tmp_path / "raw-prod"
    d.mkdir()
    return d


def test_prod_phase4_passes_with_measures(raw_prod):
    _write_phases(raw_prod, ["4"])
    res = prod.summarise(raw_prod)
    ph = res["phases"]["4"]
    assert ph["status"] == "PASS" and ph["A_validity"]["valid"]
    assert ph["judgement"]["n"] == ph["judgement"]["n_expected"] == 4
    ref = ph["references"]["all"]
    assert ref["A_window_p99_ms"] == 5.0 and ref["A_forward_mean_ms"] == pytest.approx(2.0)
    assert ref["A_agg_req_s"] == 120.0
    t = res["runs"]["mix-laya-P-r1"]["transitions"][0]
    assert t["latency_source"] == "client" and t["structure"]["valid"] and t["structure"]["sync_before_t_h"] == 32
    assert t["t_h_s"] == pytest.approx(0.001 + 32 * 0.05)
    assert t["gpu_return_ms"]["p50"] == pytest.approx(0.5) and t["transient_from_th"]["duration_s"] == 0.0
    assert res["phases"]["5"]["status"] == "pending" and res["outcome"].startswith("phase 5 pending")
    assert res["phases"]["6"]["status"] == "not reached"
    json.dumps(res, default=prod.a94._json)
    assert "Phase 4" in prod.tables(res)


def test_prod_latency_source_falls_back_to_trace():
    run = _record("4", "P")
    _hetero(run)[0]["streams"]["short"]["latency_ms"].pop()
    st = prod.run_stats(run, "mix-laya-P-r1", "4", "P")
    t = st["transitions"][0]
    assert t["latency_source"] == "trace" and t["onset_p99_ms"] == pytest.approx(2.5)


@pytest.mark.parametrize(("mod", "gate"), PROD_GATE_FAILS)
def test_prod_each_gate_fails(raw_prod, mod, gate):
    _write_phases(raw_prod, ["4"], {("4", "P", 2): mod})
    res = prod.summarise(raw_prod)
    ph = res["phases"]["4"]
    assert ph["status"] == "FAIL" and res["outcome"].startswith("phase 4 FAIL: " + prod.CLOSED)
    failed = {g for x in ph["judgement"]["transitions"] if x["run"] == "mix-laya-P-r2" for g in x["failed"]}
    assert gate in failed
    assert all(x["pass"] for x in ph["judgement"]["transitions"] if x["run"] == "mix-laya-P-r1")
    assert all(res["phases"][p]["status"] == "not reached" for p in ("5", "5b", "6", "7"))


def test_prod_crash_log_fails(raw_prod):
    _write_phases(raw_prod, ["4"])
    (raw_prod / "failed").mkdir()
    (raw_prod / "failed" / "mix-laya-P-r1.1.log").write_text("x\n")
    (raw_prod / "failed" / "mix-laya-A-r1.1.log").write_text("x\n")
    ph = prod.summarise(raw_prod)["phases"]["4"]
    assert ph["status"] == "FAIL" and {"crash"} == {g for x in ph["judgement"]["transitions"] for g in x["failed"]}


def test_prod_structure_second_episode_fails(raw_prod):
    def second(run):
        w = _hetero(run)[1]
        run["decisions"].append([w["end_ns"] - 5 * MS, 0, "sync_guard", 1])
        run["decisions"].sort(key=lambda d: d[0])

    _write_phases(raw_prod, ["4"], {("4", "P", 1): second})
    ph = prod.summarise(raw_prod)["phases"]["4"]
    x = next(x for x in ph["judgement"]["transitions"] if x["run"] == "mix-laya-P-r1" and x["index"] == 5)
    assert ph["status"] == "FAIL" and x["failed"] == ["structure"]


def test_prod_a_invalid_is_inconclusive(raw_prod):
    _write_phases(raw_prod, ["4"], {("4", "A", 2): _slow_prepare(0.0, 3.0)})
    res = prod.summarise(raw_prod)
    ph = res["phases"]["4"]
    assert ph["status"] == "INCONCLUSIVE" and not ph["A_validity"]["valid"]
    assert res["outcome"].startswith("INCONCLUSIVE: phase 4") and prod.CLOSED not in res["outcome"]
    assert res["phases"]["5"]["status"] == "not reached"


def test_prod_protocol_deviation_is_not_judged(raw_prod):
    def other_guard(run):
        run["handoff_snapshot"]["guard"] = 64

    _write_phases(raw_prod, ["4"], {("4", "P", 1): other_guard})
    ph = prod.summarise(raw_prod)["phases"]["4"]
    assert ph["status"] == "pending" and "protocol deviation (not judged): mix-laya-P-r1" in ph["reason"]


def test_prod_phase_gating(raw_prod):
    _write_phases(raw_prod, ["4", "5", "5b", "6", "7"], {("5", "P", 1): _slow_gpu_return})
    res = prod.summarise(raw_prod)
    st = {p: j["status"] for p, j in res["phases"].items()}
    assert st == {"4": "PASS", "5": "FAIL", "5b": "PASS", "6": "not reached", "7": "not reached"}
    assert res["phases"]["6"]["judged_status"] == "PASS" and res["outcome"].startswith("phase 5 FAIL")


def test_prod_all_phases_pass_is_product_candidate(raw_prod):
    _write_phases(raw_prod, ["4", "5", "5b", "6", "7"])
    res = prod.summarise(raw_prod)
    assert res["outcome"] == prod.CANDIDATE, res["outcome"]
    assert res["phases"]["6"]["judgement"]["n"] == 12 and res["phases"]["6"]["non_hetero"]["pass"]
    sk = res["phases"]["7"]["soak"]
    assert sk["pass"] and res["phases"]["7"]["judgement"]["n"] == 60 and sk["first15_p99_ms"] == 5.0
    assert "Soak gates" in prod.tables(res)


def test_prod_phase6_non_hetero_fails(raw_prod):
    def slow_solo(run):
        w = next(w for w in run["windows"] if w["condition"] == "solo_short")
        w["streams"]["short"]["latency_ms"] = [7.0] * len(w["streams"]["short"]["latency_ms"])

    _write_phases(raw_prod, ["4", "5", "5b", "6"], {("6", "P", 2): slow_solo})
    ph = prod.summarise(raw_prod)["phases"]["6"]
    assert ph["status"] == "FAIL" and ph["judgement"]["pass"]
    assert ph["non_hetero"]["failing"] == ["product-laya-P-r2 w2 short: p99"]


def test_prod_soak_rearm_fails(raw_prod):
    def not_rearmed(run):
        w = _hetero(run)[10]
        run["decisions"].append([w["start_ns"], 0, "armed", 0])  # the window starts without an episode
        run["decisions"].sort(key=lambda d: d[0])

    _write_phases(raw_prod, ["4", "5", "5b", "6", "7"], {("7", "P", 1): not_rearmed})
    res = prod.summarise(raw_prod)
    ph = res["phases"]["7"]
    assert ph["status"] == "FAIL" and ph["judgement"]["pass"] and ph["soak"]["failed"] == ["rearm"]
    assert ph["soak"]["rearm_failures"] == [21] and res["outcome"].startswith("phase 7 FAIL")


def test_prod_soak_degradation_fails(raw_prod):
    def degrade(run):
        for w in _hetero(run)[-15:]:
            w["streams"]["short"]["latency_ms"] = [5.5] * len(w["streams"]["short"]["latency_ms"])

    _write_phases(raw_prod, ["4", "5", "5b", "6", "7"], {("7", "P", 1): degrade})
    ph = prod.summarise(raw_prod)["phases"]["7"]
    assert ph["status"] == "FAIL" and ph["judgement"]["pass"] and ph["soak"]["failed"] == ["degradation"]
    assert ph["soak"]["last15_p99_ms"] == 5.5


def test_prod_multilingual_handoff_enabled_fails(raw_prod):
    def enabled(run):
        run["handoff_snapshot"] = {"guard": prod.GUARD, "disabled": False}

    _write_phases(raw_prod, ["4", "5"])
    _write_phases(raw_prod, ["5b"], {("5b", "P", 1): enabled})
    res = prod.summarise(raw_prod)
    assert res["phases"]["5b"]["status"] == "FAIL" and res["phases"]["5b"]["checks"]["failed"] == [
        "handoff_not_enabled"
    ]
    assert res["outcome"].startswith("phase 5b FAIL") and res["phases"]["6"]["status"] == "not reached"


def test_prod_check_mode(raw_prod, tmp_path):
    _write_phases(raw_prod, ["4"])
    out = tmp_path / "out"
    out.mkdir()
    script = SCRIPTS / "prod_analyze.py"
    subprocess.run([sys.executable, script, "--raw", raw_prod, "--out", out], check=True, capture_output=True)
    ok = subprocess.run([sys.executable, script, "--raw", raw_prod, "--out", out, "--check"], capture_output=True)
    assert ok.returncode == 0
    (out / "prod_tables.md").write_text("x\n")
    stale = subprocess.run([sys.executable, script, "--raw", raw_prod, "--out", out, "--check"], capture_output=True)
    assert stale.returncode != 0 and b"stale" in stale.stderr
