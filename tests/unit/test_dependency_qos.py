"""Unit tests for research/coreml-dependency-qos/ (research only): the run order and round-2 order,
the override registry (fake and real pthreads), the W classification, B validity, the guards, O's
structural validity, every outcome row, the stop rule, the leading candidate, run_config.py's
import without PyObjC and its override proxy, and analyze.py end to end on synthetic runs (#94's
synthetic PB-ASYNC run, reused from its tests)."""

from __future__ import annotations

import gzip
import importlib.util
import json
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "research" / "coreml-dependency-qos" / "scripts"
S = 1_000_000_000


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


design = _load("depqos_design_for_tests", SCRIPTS / "design.py")
analyze = _load("depqos_analyze_for_tests", SCRIPTS / "analyze.py")
qos = _load("depqos_qos_for_tests", SCRIPTS / "qos.py")
t94 = _load("depqos_t94_for_tests", ROOT / "tests" / "unit" / "test_async_transient.py")

# ------------------------------------------------------------------ design


def test_round1_order_and_files():
    assert design.ROUND1 == (("B", 1), ("O", 1), ("Q", 1))
    assert design.CYCLES == 2 and design.SECONDS == 20.0
    assert design.run_file("O", 2) == "laya-O-r2.json.gz"


@pytest.mark.parametrize(
    "passing, runs",
    [
        (["O", "Q"], (("Q", 2), ("O", 2), ("B", 2))),
        (["Q", "O"], (("Q", 2), ("O", 2), ("B", 2))),
        (["O"], (("O", 2), ("B", 2))),
        (["Q"], (("Q", 2), ("B", 2))),
        ([], ()),
    ],
)
def test_round2_order(passing, runs):
    assert design.round2(passing) == runs


def test_round2_rejects_non_candidates():
    with pytest.raises(ValueError):
        design.round2(["B"])


def test_design_cli(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sys, "argv", ["design.py", "runs", "1"])
    design.main()
    assert capsys.readouterr().out.splitlines() == ["B 1", "O 1", "Q 1"]
    monkeypatch.setattr(sys, "argv", ["design.py", "runs", "2", "--raw", str(tmp_path)])
    design.main()
    assert capsys.readouterr().out == ""  # no round 1 data: round 2 does not run


# ------------------------------------------------------------------ override registry


class _Fake:
    def __init__(self, null_at=(), rc_at=()):
        self.n, self.null_at, self.rc_at, self.freed = 0, set(null_at), set(rc_at), []

    def start(self, target, q, rel):
        assert q == qos.QOS_CLASS_USER_INITIATED and rel == 0
        self.n += 1
        return None if self.n in self.null_at else 1000 + self.n

    def end(self, h):
        self.freed.append(h)
        return 5 if h - 1000 in self.rc_at else 0


def _reg(fake):
    return qos.OverrideRegistry(qos.QOS_CLASS_USER_INITIATED, 0, start_fn=fake.start, end_fn=fake.end)


def test_registry_bookkeeping():
    fake = _Fake()
    r = _reg(fake)
    toks = [r.start(7) for _ in range(3)]
    assert toks == [0, 1, 2] and r.outstanding == 3
    for t in toks:
        assert r.end(t) == 0
    assert r.end_all() == 0
    s = r.summary()
    assert (s["starts"], s["ends"], s["null_starts"], s["end_errors"], s["outstanding_at_end"], s["outstanding"]) == (
        3,
        3,
        0,
        0,
        0,
        0,
    )
    rec = r.record()["overrides"]
    assert rec["ended_by"] == ["client"] * 3 and rec["end_rc"] == [0, 0, 0]
    assert all(
        a <= b <= c <= d for a, b, c, d in zip(rec["start_call_ns"], rec["start_ns"], rec["end_ns"], rec["end_ret_ns"])
    )


def test_registry_null_start_double_end_rc_and_end_all():
    fake = _Fake(null_at={2}, rc_at={3})
    r = _reg(fake)
    a, b, c, d = (r.start(7) for _ in range(4))
    assert b is None and r.end(b) is None  # the failed start has nothing to end
    assert r.end(a) == 0 and r.end(a) is None  # the second end never reaches libSystem
    assert fake.freed.count(1001) == 1
    assert r.end(c) == 5  # a non-zero end result
    assert r.end_all() == 1 and r.end_all() == 0  # d was left outstanding
    s = r.summary()
    assert s["null_starts"] == 1 and s["double_ends"] == 1 and s["end_errors"] == 2
    assert s["starts"] == 3 and s["ends"] == 2 and s["ended_by_end_all"] == 1
    assert s["outstanding_at_end"] == 1 and s["outstanding"] == 0
    assert r.record()["overrides"]["ended_by"] == ["client", "client", "end_all"]


def test_registry_no_target_is_a_null_start():
    r = _reg(_Fake())
    assert r.start(None) is None and r.summary()["null_starts"] == 1


def test_registry_is_thread_safe():
    r = _reg(_Fake())

    def work():
        for _ in range(200):
            r.end(r.start(7))

    ts = [threading.Thread(target=work) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    s = r.summary()
    assert s["starts"] == s["ends"] == 1600 and s["end_errors"] == 0 and r.outstanding == 0


@pytest.mark.skipif(sys.platform != "darwin", reason="pthread QoS calls are macOS only")
def test_real_override_and_requested_qos():
    here = qos.pthread_self()
    assert qos.thread_id(here) == threading.get_native_id()
    r = qos.OverrideRegistry()
    t = r.start(here)  # the current thread
    assert t is not None and r.end(t) == 0
    box, go, done = {}, threading.Event(), threading.Event()

    def helper():
        box["pt"] = qos.pthread_self()
        box["before"] = qos.get_qos(box["pt"])
        go.set()
        done.wait(10)
        box["rc"] = qos.set_qos_self(qos.QOS_CLASS_USER_INITIATED, 0)
        box["after"] = qos.get_qos(box["pt"])

    th = threading.Thread(target=helper)
    th.start()
    assert go.wait(10)
    tok = r.start(box["pt"])  # another thread, blocked on an Event
    assert tok is not None
    assert qos.get_qos(box["pt"])["qos"] == box["before"]["qos"]  # an override is not a requested QoS
    assert r.end(tok) == 0
    done.set()
    th.join()
    assert box["rc"] == 0 and box["after"]["qos"] == qos.QOS_CLASS_USER_INITIATED
    s = r.summary()
    assert s["starts"] == s["ends"] == 2 and s["null_starts"] == s["end_errors"] == 0


# ------------------------------------------------------------------ classification and outcomes


@pytest.mark.parametrize(
    "e, cls",
    [
        (0.0, "avoided"),
        (0.10, "avoided"),
        (0.1001, "partial"),
        (0.4999, "partial"),
        (0.50, "E-resident"),
        (1.0, "E-resident"),
        (None, "unassessable"),
    ],
)
def test_classify_edges(e, cls):
    assert analyze.classify(e) == cls


@pytest.mark.parametrize(
    "mech, lat, chain, ok, key",
    [
        (True, True, False, True, "PASS"),
        (True, True, True, True, "PASS"),
        (False, True, False, True, "LAT_ONLY"),
        (False, True, True, False, "LAT_ONLY"),
        (True, False, True, True, "MECH_CHAIN"),
        (True, False, True, False, "MECH_CHAIN"),
        (True, False, False, True, "MECH_ONLY"),
        (False, False, False, True, "NEITHER"),
        (False, False, True, False, "NEITHER"),
    ],
)
def test_outcome_rows(mech, lat, chain, ok, key):
    assert analyze.outcome(mech, lat, chain, ok) == analyze.OUTCOMES[key]


def test_outcome_guard_row_names_the_failed_guard():
    out = analyze.outcome(True, True, False, False, ["throughput"])
    assert out.startswith(analyze.OUTCOMES["GUARD"]) and "throughput" in out


def test_outcome_texts():
    assert analyze.OUTCOMES["MECH_CHAIN"].startswith("dispatcher-only QoS changes placement but is insufficient")
    assert analyze.OUTCOMES["MECH_ONLY"].startswith("QoS affects placement but is insufficient; not productionized")
    assert analyze.STOP.startswith("STOP this USER_INITIATED dispatcher-QoS route")
    for text in [*analyze.OUTCOMES.values(), analyze.STOP, analyze.CHAIN_FOLLOW_UP, analyze.HUMAN]:
        assert "heuristic" not in text and "wakeup" not in text and "root cause" not in text


@pytest.mark.parametrize(
    "classes, text",
    [
        (["E-resident", "partial"], "changed nothing"),
        (["unassessable", "E-resident"], "changed nothing"),
        (["avoided", "partial"], "changed placement in some transitions"),
        (["avoided", "avoided"], "changed placement in both"),
    ],
)
def test_placement_change(classes, text):
    assert analyze.placement_change(classes) == text


# ------------------------------------------------------------------ B validity and guards


def _st(cell="B", classes=("E-resident", "E-resident"), durations=(3.0, 3.0), chain=(False, False), **kw):
    st = {
        "cell": cell,
        "transitions": [
            {"class": c, "transient": {"duration_s": d}, "chain_e": ch} for c, d, ch in zip(classes, durations, chain)
        ],
        "mismatches": 0,
        "routing_failures": [],
        "gpu_return_p50_ms": 0.3,
        "aggregate_req_s_per_window": [120.0, 120.0],
        "aggregate_req_s_pooled": 120.0,
        "native_mean_ms": 8.0,
        "override_validity": {"ok": True} if cell == "O" else None,
        "q_readback": {"ok": True} if cell == "Q" else None,
    }
    st.update(kw)
    return st


def test_b_valid():
    v = analyze.b_validity(_st(classes=("partial", "E-resident"), aggregate_req_s_per_window=[96.5, 141.7]))
    assert v["valid"] and v["failed"] == []


@pytest.mark.parametrize(
    "kw, flag",
    [
        ({"classes": ("partial", "avoided")}, "e_resident_transition"),
        ({"classes": ("unassessable", "unassessable")}, "e_resident_transition"),
        ({"mismatches": 1}, "no_mismatch"),
        ({"gpu_return_p50_ms": 1.01}, "gpu_return_p50"),
        ({"gpu_return_p50_ms": None}, "gpu_return_p50"),
        ({"aggregate_req_s_per_window": [96.4, 120.0]}, "aggregate_in_range"),
        ({"aggregate_req_s_per_window": [120.0, 141.8]}, "aggregate_in_range"),
        ({"aggregate_req_s_per_window": [120.0]}, "aggregate_in_range"),
    ],
)
def test_b_validity_triggers(kw, flag):
    v = analyze.b_validity(_st(**kw))
    assert not v["valid"] and v["failed"] == [flag]


@pytest.mark.parametrize(
    "kw, failed",
    [
        ({}, []),
        ({"native_mean_ms": 8.48}, ["native"]),  # 1.06x and +0.48 ms
        ({"native_mean_ms": 8.25}, []),  # +0.25 ms, 1.03x
        ({"gpu_return_p50_ms": 1.0}, []),
        ({"gpu_return_p50_ms": 1.2}, ["gpu_return_p50"]),
        ({"mismatches": 2}, ["correctness"]),
        ({"routing_failures": ["0:hetero:short"]}, ["correctness"]),
        ({"aggregate_req_s_pooled": 114.0}, []),  # exactly 0.95x
        ({"aggregate_req_s_pooled": 113.9}, ["throughput"]),
    ],
)
def test_guards(kw, failed):
    assert analyze.guards(_st("Q", **kw), _st())["failed"] == failed


def test_native_guard_needs_both_ratio_and_plus():
    b = _st(native_mean_ms=2.0)
    assert analyze.guards(_st("Q", native_mean_ms=2.2), b)["pass"]  # 1.10x but only +0.2 ms
    b = _st(native_mean_ms=10.0)
    assert analyze.guards(_st("Q", native_mean_ms=10.4), b)["pass"]  # +0.4 ms but only 1.04x
    assert not analyze.guards(_st("Q", native_mean_ms=10.6), b)["pass"]  # 1.06x and +0.6 ms


def test_candidate_specific_guards():
    b = _st()
    assert analyze.guards(_st("O", override_validity={"ok": False}), b)["failed"] == ["override_validity"]
    assert analyze.guards(_st("O", override_validity=None), b)["failed"] == ["override_validity"]
    assert analyze.guards(_st("Q", q_readback={"ok": False}), b)["failed"] == ["q_readback"]
    assert "q_readback" not in analyze.guards(_st("O"), b)["checks"]


def test_q_readback():
    ok = {"set": {"readback": {"qos": 0x19}}, "at_end": {"qos": 0x19}}
    assert analyze.q_readback(ok)["ok"]
    assert not analyze.q_readback({"set": {"readback": {"qos": 0x15}}, "at_end": {"qos": 0x19}})["ok"]
    assert not analyze.q_readback({"set": {"readback": {"qos": 0x19}}, "at_end": None})["ok"]


# ------------------------------------------------------------------ O structural validity


def _ov(n=2, **kw):
    base = 1000 * S
    o = {
        "start_call_ns": [base + i * S for i in range(n)],
        "start_ns": [base + i * S + 10 for i in range(n)],
        "end_ns": [base + i * S + 1000 for i in range(n)],
        "end_ret_ns": [base + i * S + 1010 for i in range(n)],
        "end_rc": [0] * n,
        "ended_by": ["client"] * n,
    }
    ov = {
        "starts": n,
        "ends": n,
        "null_starts": 0,
        "end_errors": 0,
        "double_ends": 0,
        "ended_by_end_all": 0,
        "outstanding_at_end": 0,
        "outstanding": 0,
        "overrides": o,
    }
    ov.update(kw)
    return ov


BOUNDS = [(1000 * S, 1000 * S + S // 2), (1001 * S, 1001 * S + S // 2)]
DISP = {"captured_ns": 999 * S, "exit_ns": 2000 * S}


def test_override_validity_ok():
    v = analyze.override_validity(_ov(), DISP, BOUNDS)
    assert v["ok"], v
    assert analyze.override_validity(_ov(), {"captured_ns": 999 * S, "exit_ns": None}, BOUNDS)["ok"]


@pytest.mark.parametrize(
    "ov_kw, disp, bounds, flag",
    [
        ({"null_starts": 1}, DISP, BOUNDS, "no_null_start"),
        ({"ends": 1}, DISP, BOUNDS, "one_end_each"),
        ({"double_ends": 1, "end_errors": 1}, DISP, BOUNDS, "one_end_each"),
        ({"end_errors": 1}, DISP, BOUNDS, "end_rc_zero"),
        ({"outstanding_at_end": 1}, DISP, BOUNDS, "none_outstanding"),
        ({}, {"captured_ns": 1000 * S + 5, "exit_ns": 2000 * S}, BOUNDS, "lifetime"),
        ({}, {"captured_ns": 999 * S, "exit_ns": 1000 * S + 500}, BOUNDS, "lifetime"),
        ({}, None, BOUNDS, "lifetime"),
        ({}, DISP, [*BOUNDS, (1500 * S, 1501 * S)], "active_each_hetero_window"),
        ({}, DISP, [], "active_each_hetero_window"),
    ],
)
def test_override_validity_triggers(ov_kw, disp, bounds, flag):
    v = analyze.override_validity(_ov(**ov_kw), disp, bounds)
    assert not v["ok"] and not v["checks"][flag]


def test_override_validity_rc_and_missing_record():
    ov = _ov()
    ov["overrides"]["end_rc"][1] = 3
    assert not analyze.override_validity(ov, DISP, BOUNDS)["checks"]["end_rc_zero"]
    assert not analyze.override_validity(None, DISP, BOUNDS)["ok"]
    assert not analyze.override_validity(_ov(0, starts=0, ends=0), DISP, BOUNDS)["ok"]


def test_override_cost():
    c = analyze.override_cost_us(_ov())
    assert c["start_p50"] == pytest.approx(0.01) and c["end_p99"] == pytest.approx(0.01)
    assert analyze.override_cost_us(None) is None


# ------------------------------------------------------------------ rounds, stop rule, leading


def _round(b=None, o=None, q=None):
    stats = {"B": b or _st()}
    if o is not None:
        stats["O"] = o
    if q is not None:
        stats["Q"] = q
    return analyze.judge_round(stats)


PASSING = {"classes": ("avoided", "avoided"), "durations": (0.5, 1.0)}
STUCK = {"classes": ("E-resident", "partial"), "durations": (3.0, 3.0)}


def test_round_pass_and_round2_trigger():
    r = _round(o=_st("O", **PASSING), q=_st("Q", **STUCK))
    assert r["B"]["valid"] and r["passing"] == ["O"]
    j = r["candidates"]["O"]
    assert j["MECH"] and j["LAT"] and j["outcome"] == analyze.OUTCOMES["PASS"]
    d = analyze.round1_decision(r)
    assert d["round2"] == ["O"] and not d["stop"] and d["next"] == "round 2 runs: O r2, B r2"


def test_round_with_invalid_b_judges_nothing():
    r = _round(b=_st(mismatches=1), o=_st("O", **PASSING), q=_st("Q", **PASSING))
    assert not r["B"]["valid"] and r["passing"] == []
    assert all(x["outcome"] == analyze.B_INVALID for x in r["candidates"].values())
    d = analyze.round1_decision(r)
    assert d["round2"] == [] and not d["stop"] and d["next"] == analyze.B_INVALID


def test_stop_rule_when_both_changed_nothing():
    r = _round(o=_st("O", **STUCK), q=_st("Q", classes=("partial", "unassessable")))
    d = analyze.round1_decision(r)
    assert d["stop"] and d["next"] == analyze.STOP and d["round2"] == []


def test_no_stop_when_one_changed_some_transitions():
    r = _round(o=_st("O", classes=("avoided", "partial")), q=_st("Q", **STUCK))
    assert r["candidates"]["O"]["placement_change"] == "changed placement in some transitions"
    d = analyze.round1_decision(r)
    assert not d["stop"] and d["next"] == analyze.HUMAN


def test_chain_follow_up():
    o = _st("O", classes=("avoided", "avoided"), durations=(3.0, 1.0), chain=(True, False))
    r = _round(o=o, q=_st("Q", **STUCK))
    assert r["candidates"]["O"]["outcome"] == analyze.OUTCOMES["MECH_CHAIN"] and r["candidates"]["O"]["CHAIN_E"]
    assert analyze.round1_decision(r)["next"] == analyze.CHAIN_FOLLOW_UP
    o = _st("O", classes=("avoided", "avoided"), durations=(3.0, 1.0))
    assert _round(o=o, q=_st("Q", **STUCK))["candidates"]["O"]["outcome"] == analyze.OUTCOMES["MECH_ONLY"]


def test_latency_without_mech_and_guard_failure():
    r = _round(o=_st("O", classes=("partial", "avoided"), durations=(1.0, 0.5)), q=_st("Q", **PASSING, mismatches=1))
    assert r["candidates"]["O"]["outcome"] == analyze.OUTCOMES["LAT_ONLY"]
    q = r["candidates"]["Q"]
    assert q["outcome"].startswith(analyze.OUTCOMES["GUARD"]) and "correctness" in q["outcome"] and not q["pass"]
    assert analyze.round1_decision(r)["next"] == analyze.HUMAN


def test_leading_prefers_o():
    r1 = _round(o=_st("O", **PASSING), q=_st("Q", **PASSING))
    r2 = _round(o=_st("O", **PASSING), q=_st("Q", **PASSING))
    lead = analyze.leading(r1, r2)
    assert lead["leading"] == "O" and lead["fallback"] == "Q" and "(Q is the fallback)" in lead["text"]


def test_leading_q_alone_and_inconclusive():
    r1 = _round(o=_st("O", **PASSING), q=_st("Q", **PASSING))
    r2 = _round(o=_st("O", **STUCK), q=_st("Q", **PASSING))
    lead = analyze.leading(r1, r2)
    assert lead["leading"] == "Q" and lead["fallback"] is None
    assert lead["per_candidate"] == {"O": analyze.R2_INCONCLUSIVE, "Q": "leads"}


def test_round2_with_invalid_b_leaves_every_pass_inconclusive():
    r1 = _round(o=_st("O", **PASSING))
    r2 = _round(b=_st(classes=("avoided", "avoided")), o=_st("O", **PASSING))
    lead = analyze.leading(r1, r2)
    assert lead["leading"] is None and lead["per_candidate"] == {"O": analyze.R2_INCONCLUSIVE}


# ------------------------------------------------------------------ run_config


def test_run_config_imports_without_pyobjc():
    code = (
        "import sys, importlib.util as u; sys.modules['objc'] = None; sys.modules['CoreML'] = None; "
        f"s = u.spec_from_file_location('m', {str(SCRIPTS / 'run_config.py')!r}); m = u.module_from_spec(s); "
        "s.loader.exec_module(m); print(m.CELLS, m.BASE_CELL)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert "('B', 'O', 'Q') PB-ASYNC" in out


def test_override_proxy_ends_in_finally():
    rc = _load("depqos_run_config_for_tests", SCRIPTS / "run_config.py")
    fake = _Fake()
    reg = _reg(fake)
    reqs = {"token": [], "device": []}

    class R:
        class runtime:
            device = "ane"

    class Laya:
        device = "auto"
        fail = False

        def predict(self, **kw):
            if self.fail:
                raise RuntimeError("boom")
            return R()

    inner = Laya()
    p = rc.OverrideLaya(inner, reg, [7], reqs)
    assert p.device == "auto" and isinstance(p.predict(context=1, questions=2), R)
    inner.fail = True
    with pytest.raises(RuntimeError):
        p.predict(context=1, questions=2)
    s = reg.summary()
    assert s["starts"] == s["ends"] == 2 and reg.outstanding == 0
    assert reqs == {"token": [0, 1], "device": ["ane", None]}


# ------------------------------------------------------------------ end to end


def _fake_run(cell, e_disp=(0.9, 0.9), e_chain=(0.0, 0.0), slow_s=1.0, mismatches=0):
    """#94's synthetic PB-ASYNC run as cell B / O / Q: per hetero window, the ANE dispatcher's and
    the chain threads' E share (uniform over that window's samples) and host-slow requests for the
    first slow_s seconds."""
    run = t94._fake_run("PB-ASYNC")
    res = run["research"]
    res["base"] = {"experiment": res["experiment"], "cell": "PB-ASYNC"}
    res["experiment"], res["cell"] = "coreml-dependency-qos", cell
    rc = res["recount"]
    n = len(rc["t_ns"]) // 2  # samples per hetero window, in cycle order
    for s in rc["series"].values():
        e = e_disp if s["name"] == "laya-ane-dispatch" else e_chain
        cp = ce = 0
        rows = []
        for i in range(len(rc["t_ns"])):
            share = e[min(i // n, 1)]
            cp += int(500 * (1 - share))
            ce += int(500 * share)
            rows.append([i, 4000 * i, 2000 * i, cp, 0, 0, 0, ce, 0])
        s["rows"] = rows
    tr = res["trace"]
    for w in res["windows"]:
        for i, sub in enumerate(tr["submit_ns"]):
            if tr["target"][i] == "ane" and w["start_ns"] <= sub < w["start_ns"] + int(slow_s * S):
                tr["prepared_ns"][i] = sub + 600_000
    run["part_a"]["windows"][0]["streams"]["short"]["mismatches"] = mismatches
    q = {
        "cell": cell,
        "dispatcher": {
            "thread_name": "laya-ane-dispatch",
            "native_id": 10,
            "pthread_threadid": 10,
            "captured_ns": 0,
            "exit_ns": 10**15,
        },
        "at_load": {"rc": 0, "qos": 0x15, "relpri": 0},
        "set": None,
        "at_end": {"rc": 0, "qos": 0x15, "relpri": 0},
        "errors": [],
        "override": None,
    }
    if cell == "Q":
        q["set"] = {"rc": 0, "readback": {"rc": 0, "qos": 0x19, "relpri": 0}}
        q["at_end"]["qos"] = 0x19
    if cell == "O":
        ov = _ov(0, starts=0, ends=0)
        for w in res["windows"]:
            for k in range(3):
                s0 = w["start_ns"] + k * S
                for c, v in zip(ov["overrides"], (s0, s0 + 1000, s0 + 5 * 10**6, s0 + 5 * 10**6 + 1000, 0, "client")):
                    ov["overrides"][c].append(v)
        ov["starts"] = ov["ends"] = len(ov["overrides"]["start_ns"])
        ov["short_requests"] = {"token": list(range(ov["starts"])), "device": ["ane"] * ov["starts"]}
        q["override"] = ov
    res["qos"] = q
    return run


def _write(raw, cell, rnd, **kw):
    with gzip.open(raw / design.run_file(cell, rnd), "wt") as fh:
        json.dump(_fake_run(cell, **kw), fh)


@pytest.fixture
def raw(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    return d


AVOID = {"e_disp": (0.0, 0.05), "slow_s": 0.5}


def test_end_to_end_measures(raw):
    _write(raw, "B", 1)
    _write(raw, "O", 1, e_disp=(0.0, 0.3), e_chain=(0.8, 0.0), slow_s=3.0)
    _write(raw, "Q", 1, **AVOID)
    res = analyze.summarise(raw)
    runs = res["rounds"]["1"]["runs"]
    b = runs["B"]
    t = b["transitions"][0]
    assert t["class"] == "E-resident" and t["W"]["ane-dispatch_e_share"] == pytest.approx(0.9, abs=0.01)
    assert t["onset"]["ane-dispatch_e_share"] == pytest.approx(0.9, abs=0.01)
    assert t["onset"]["ane_cpu_per_forward_ms"] == pytest.approx(0.6) and t["onset"]["transient_s"] == 1.0
    assert "0.5-1" in t["buckets"] and t["transient"]["peak_short_p99_ms"] is not None
    assert b["dispatcher"]["ok"] and b["native_mean_ms"] == pytest.approx(7.94)
    assert (
        b["gpu_return_p50_ms"] == pytest.approx(0.3) and b["aggregate_req_s_per_window"] == [pytest.approx(122.4)] * 2
    )
    o = runs["O"]
    assert [x["class"] for x in o["transitions"]] == ["avoided", "partial"]
    assert o["transitions"][0]["chain_e"] and not o["transitions"][1]["chain_e"]
    assert o["override_validity"]["ok"] and o["override"]["cost_us"]["start_p50"] == pytest.approx(1.0)
    assert runs["Q"]["q_readback"]["ok"]
    j = res["rounds"]["1"]["judgement"]
    assert j["B"]["valid"] and j["passing"] == ["Q"]
    assert j["candidates"]["O"]["outcome"] == analyze.OUTCOMES["NEITHER"]
    assert j["candidates"]["O"]["placement_change"] == "changed placement in some transitions"
    assert res["outcome"] == "round 1 passed Q; running: Q r2, B r2"
    assert analyze.round2_trigger(raw) == ["Q"]
    md = analyze.tables(res)
    assert "## Round 1: B validity" in md and "W E share" in md
    json.dumps(res, default=analyze.a94._json)


def test_end_to_end_two_rounds_and_design_cli(raw, monkeypatch, capsys):
    _write(raw, "B", 1)
    _write(raw, "O", 1, **AVOID)
    _write(raw, "Q", 1, **AVOID)
    monkeypatch.setattr(sys, "argv", ["design.py", "runs", "2", "--raw", str(raw)])
    design.main()
    assert capsys.readouterr().out.splitlines() == ["Q 2", "O 2", "B 2"]
    for c in ("Q", "O"):
        _write(raw, c, 2, **AVOID)
    _write(raw, "B", 2)
    res = analyze.summarise(raw)
    assert res["leading"]["leading"] == "O" and res["leading"]["fallback"] == "Q"
    assert res["outcome"].startswith("leading candidate: O (Q is the fallback); a leading candidate is not")
    assert "## Leading candidate" in analyze.tables(res)


def test_end_to_end_stop_rule(raw):
    _write(raw, "B", 1)
    _write(raw, "O", 1, e_disp=(0.9, 0.3))
    _write(raw, "Q", 1)
    res = analyze.summarise(raw)
    assert res["outcome"] == analyze.STOP and res["rounds"]["2"] is None


def test_end_to_end_invalid_b(raw):
    _write(raw, "B", 1, e_disp=(0.3, 0.2))
    _write(raw, "O", 1, **AVOID)
    _write(raw, "Q", 1, **AVOID)
    res = analyze.summarise(raw)
    assert res["outcome"] == analyze.B_INVALID
    assert res["rounds"]["1"]["judgement"]["B"]["failed"] == ["e_resident_transition"]


def test_end_to_end_chain(raw):
    _write(raw, "B", 1)
    _write(raw, "O", 1, e_disp=(0.0, 0.0), e_chain=(0.9, 0.9), slow_s=4.0)
    _write(raw, "Q", 1)
    res = analyze.summarise(raw)
    assert res["rounds"]["1"]["judgement"]["candidates"]["O"]["outcome"] == analyze.OUTCOMES["MECH_CHAIN"]
    assert res["outcome"] == analyze.CHAIN_FOLLOW_UP


def test_check_mode(raw, tmp_path):
    _write(raw, "B", 1)
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
    run["research"]["experiment"] = "coreml-async-transient"
    with gzip.open(raw / design.run_file("B", 1), "wt") as fh:
        json.dump(run, fh)
    with pytest.raises(AssertionError):
        analyze.summarise(raw)


def test_fake_run_shares_are_uniform():
    run = _fake_run("B", e_disp=(0.2, 0.7))
    rows = np.asarray(run["research"]["recount"]["series"]["1:10"]["rows"])
    assert rows[:, 7].max() > 0 and rows[:, 3].max() > 0
