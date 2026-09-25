"""Unit tests for research/coreml-prebind-full-protocol/ (R1, research only): the run order, the
sequential-look rule, analyze.py end to end on synthetic full-protocol runs, and run_config.py's
import without PyObjC or hardware."""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "research" / "coreml-prebind-full-protocol" / "scripts"
MS = 1_000_000
CONDS = ("solo_short", "solo_long", "hetero", "gpu_only")
MODELS = ("laya", "laya-typed-decisions")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sys.path.insert(0, str(SCRIPTS))
design = _load("r1_design_for_tests", SCRIPTS / "design.py")
analyze = _load("r1_analyze_for_tests", SCRIPTS / "analyze.py")


# ------------------------------------------------------------------ design


def test_blocks_are_abba_balanced_and_rounds_pair_within_a_block():
    for b, order in design.BLOCK_ORDERS.items():
        pos: dict[str, list[int]] = {}
        for i, c in enumerate(order, start=1):
            pos.setdefault(c, []).append(i)
        mean = (len(order) + 1) / 2
        assert all(np.mean(p) == mean for p in pos.values())
        rounds: dict[str, set] = {}
        for _, c, r in design.block_runs("laya", b):
            rounds.setdefault(c, set()).add(r)
        assert all(v == {2 * b - 1, 2 * b} for v in rounds.values())
    assert set(design.BLOCK_ORDERS[1]) == {"P", "C", "PB"}
    assert set(design.BLOCK_ORDERS[2]) == set(design.BLOCK_ORDERS[3]) == {"P", "PB"}  # no C after block 1
    assert design.BLOCK_ORDERS[2] == ("PB", "P", "P", "PB") and design.BLOCK_ORDERS[3] == ("P", "PB", "PB", "P")


def test_campaign_order_looks_and_files():
    assert design.blocks() == [(m, b) for b in (1, 2, 3) for m in MODELS]
    assert [design.pairs_at(b) for b in (1, 2, 3)] == [6, 12, 18]
    assert design.LOOKS == {1: ("futility", 6), 2: ("futility", 12), 3: ("final", 18)}
    assert design.rounds_through(2) == (1, 2, 3, 4)
    assert design.config_name("laya", "P") == "A" and design.run_file("laya", "A", 5) == "laya-A-r5.json.gz"
    files = design.designed_files()
    assert len(files) == 2 * (6 + 4 + 4) and "laya-C-r3.json.gz" not in files
    with pytest.raises(ValueError):
        design.block_runs("laya", 4)


# ------------------------------------------------------------------ synthetic full-protocol runs


def _fake_run(
    config,
    short,
    long_,
    short_p99s,
    *,
    long_p99s=None,
    req=None,
    mismatches=0,
    gpu_return_us=50,
    native=True,
    client_ms=50,
):
    """#77's layout: all four conditions in bench_concurrency's alternating order, 3 cycles."""
    windows_t, part_windows, res_windows = [], [], []
    trace = {c: [] for c in _run_config().TRACE_COLUMNS}
    forwards_ane, predicts, gpu_recv = [], [], []
    t, rid = 1_000 * MS, 0
    for cycle in range(3):
        for cond in CONDS if cycle % 2 == 0 else tuple(reversed(CONDS)):
            lo, hi = t, t + 200 * MS
            t = hi + 300 * MS
            streams = {"solo_short": ["short"], "solo_long": ["long"]}.get(cond, ["short", "long"])
            inst = "gpu" if cond == "gpu_only" else "auto"
            for s in streams:
                windows_t.append({"stream": s, "instance": inst, "start_ns": lo, "end_ns": hi})
            pw = {"cycle": cycle, "condition": cond, "streams": {}}
            for s in streams:
                dev = "gpu" if inst == "gpu" or s == "long" else "ane"
                het = cond == "hetero"
                p99 = 10.0 if s == "short" else 40.0
                if het:
                    p99 = short_p99s[cycle] if s == "short" else (long_p99s[cycle] if long_p99s else 40.0)
                rate = (100.0 if s == "short" else 25.0) * (req[cycle] if het and req else 1.0)
                pw["streams"][s] = {"req_s": rate, "p99_ms": p99, "p50_ms": 9.0}
                bad = mismatches if cond == "solo_short" and cycle == 0 else 0  # outside hetero: still counts
                pw["streams"][s].update(devices={dev: 20}, mismatches=bad, latency_ms=[1.0, 2.0])
            part_windows.append(pw)
            if inst == "gpu":
                continue
            if "short" in streams:
                for k in range(8):  # ANE forwards, 20 ms apart
                    s0 = lo + k * 20 * MS
                    forwards_ane.append((s0, s0 + 9 * MS, 600_000))
                    p = (s0 + MS, s0 + 2 * MS, s0 + 2 * MS + 1, s0 + 8 * MS, s0 + 8 * MS + 10, s0 + 8 * MS + 50)
                    predicts.append(p if native else (p[0], 0, 0, 0, 0, p[5]))
                    if cond == "hetero":
                        rid += 1
                        row = (rid, "ane", short, s0 - MS, s0 - MS, s0 - MS, s0 - MS, s0, s0, s0 + 9 * MS)
                        for c, v in zip(trace, (*row, s0 + 9 * MS, s0 - MS + (9 + k % 5) * MS)):
                            trace[c].append(v)
            if "long" in streams:
                for k in range(3):  # GPU completions
                    g0 = lo + k * 50 * MS
                    recv = g0 + 40 * MS
                    gpu_recv.append(recv)
                    if cond == "hetero":
                        rid += 1
                        row = (rid, "gpu", long_, g0, g0, g0, g0, g0, g0, recv - 50_000)
                        for c, v in zip(trace, (*row, recv, recv + 200_000)):
                            trace[c].append(v)
            if cond == "hetero":
                res_windows.append(
                    {
                        "start_ns": lo,
                        "end_ns": hi,
                        "instance": "auto",
                        "streams": {"short": 20, "long": 5},
                        "before": {"t_ns": lo - 100 * MS, "threads": {"client-short": 0}},
                        "after": {"t_ns": hi + 100 * MS, "threads": {"client-short": client_ms * MS}},
                    }
                )
    return {
        "args": {"short": short, "long": long_, "seconds": 20.0, "cycles": 3},  # campaign values; windows are short
        "part_a": {"windows": part_windows},
        "windows_t": windows_t,
        "gpu_return": {"received_ns": gpu_recv, "return_us": [gpu_return_us] * len(gpu_recv)},
        "research": {
            "config": config,
            "laya_apple": "x",
            "pyobjc": "x",
            "coremltools": "x",
            "mlx": "x",
            "gil_probe": None,
            "workers": {"ane": {"placement": "thread"}, "gpu": {"placement": "process"}},
            "forwards": {"ane": forwards_ane, "gpu": [(g - 40 * MS, g - 50_000, 2 * MS) for g in gpu_recv]},
            "predicts": predicts,
            "crossings": {"128": {"objc_send": 1, "ctypes": 1, "pool_push": 2, "pool_pop": 1, "total": 5}},
            "backings": None,
            "trace": trace,
            "windows": res_windows,
        },
    }


def _run_config():
    return _load("r1_run_config_for_tests", SCRIPTS / "run_config.py")


def _write(raw: Path, model: str, blocks, ratio=None, **kw):
    """Every run of the given blocks. ratio(config, round, cycle) -> short P99 multiplier, or a dict
    with any of short / long / req multipliers of the hetero window (P's: 10 ms, 40 ms, 1.0)."""
    short, long_, _ = design.MODELS[model]
    ratio = ratio or flat(1.0)
    for b in blocks:
        for _, name, rnd in design.block_runs(model, b):
            c = "P" if name == "A" else name
            ms = [ratio(c, rnd, k) if c != "P" else 1.0 for k in range(3)]
            ms = [m if isinstance(m, dict) else {"short": m} for m in ms]
            run = _fake_run(
                name,
                short,
                long_,
                [10.0 * m.get("short", 1.0) for m in ms],
                long_p99s=[40.0 * m.get("long", 1.0) for m in ms],
                req=[m.get("req", 1.0) for m in ms],
                gpu_return_us=5000 if c == "P" else kw.get("pb_gpu_us", 50),
                mismatches=kw.get("mismatches", {}).get(c, 0),
                native=c != "P",
            )
            with gzip.open(raw / design.run_file(model, name, rnd), "wt") as fh:
                json.dump(run, fh)


def flat(v):
    return lambda c, rnd, k: v


def alt(lo, hi, then=None, from_round=None):
    """Alternating ratios per pair (a wide interval); `then` from round from_round on."""

    def f(c, rnd, k):
        if then is not None and rnd >= from_round:
            return then
        return hi if ((rnd - 1) * 3 + k) % 2 == 0 else lo

    return f


@pytest.fixture
def raw(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    return d


def _both(raw, blocks, laya=None, typed=None, **kw):
    _write(raw, "laya", blocks, laya, **kw)
    _write(raw, "laya-typed-decisions", blocks, typed)


# ------------------------------------------------------------------ sequencing


def test_empty_raw_runs_laya_block_1_first_and_renders(raw):
    res = analyze.summarise(raw)
    assert res["next"] == {"model": "laya", "block": 1} and res["outcome"] == "running: next block 1 of laya"
    assert res["check"] is None and res["failed_runs"] == []
    assert "Outcome:" in analyze.tables(res)
    json.dumps(res)


def test_n6_look_waits_for_both_block_1s_then_blocks_alternate(raw):
    _write(raw, "laya", (1,))
    res = analyze.summarise(raw)
    assert res["next"] == {"model": "laya-typed-decisions", "block": 1}
    assert res["models"]["laya"]["looks"] == {}  # no n=6 look before both block 1s exist
    _write(raw, "laya-typed-decisions", (1,))
    res = analyze.summarise(raw)
    assert all(res["models"][m]["looks"]["6"]["decision"] == "CONTINUE" for m in MODELS)
    assert all(res["models"][m]["C_reference"]["pairs"] == 6 for m in MODELS)
    assert res["next"] == {"model": "laya", "block": 2}
    _write(raw, "laya", (2,))
    res = analyze.summarise(raw)
    assert res["models"]["laya"]["looks"]["12"]["decision"] == "CONTINUE"
    assert "12" not in res["models"]["laya-typed-decisions"]["looks"]
    assert res["next"] == {"model": "laya-typed-decisions", "block": 2}
    _write(raw, "laya-typed-decisions", (2,))
    _write(raw, "laya", (3,))
    res = analyze.summarise(raw)
    assert res["next"] == {"model": "laya-typed-decisions", "block": 3}
    assert "18" not in res["models"]["laya"]["looks"]  # the final look waits for both block 3s


def test_futility_never_passes(raw):
    _both(raw, (1, 2))
    res = analyze.summarise(raw)
    for m in MODELS:
        for lk in res["models"][m]["looks"].values():
            assert lk["decision"] == "CONTINUE" and "verdict" not in lk
            assert {v["verdict_99"] for v in lk["intervals_99"].values()} == {"PASS"}


@pytest.mark.parametrize(
    "laya, kw, triggered",
    [
        (flat(1.3), {}, ["short_p99"]),
        (flat({"long": 1.3}), {}, ["long_p99"]),
        (flat({"req": 0.8}), {}, ["aggregate_throughput"]),
        (flat(1.0), {"mismatches": {"PB": 1}}, ["correctness"]),
        (flat(1.0), {"mismatches": {"P": 1}}, ["correctness"]),
        (flat(1.0), {"pb_gpu_us": 1500}, ["gpu_completion_isolation"]),  # P50 > 1 ms
        (flat(1.0), {"pb_gpu_us": 1000 + 1}, ["gpu_completion_isolation"]),
    ],
)
def test_futility_stop_at_n6_on_each_criterion(raw, laya, kw, triggered):
    _both(raw, (1, 2, 3), laya=laya, **kw)  # later blocks exist but must not be used
    res = analyze.summarise(raw)
    lk = res["models"]["laya"]["looks"]["6"]
    assert lk["decision"] == "FUTILITY STOP" and lk["triggered"] == triggered
    assert res["next"] is None and "12" not in res["models"]["laya"]["looks"]
    assert res["outcome"] == (
        f"FUTILITY STOP at n=6 on laya: {', '.join(triggered)}; R1 FAIL; protocol-history experiment next"
    )
    assert "laya-PB-r3.json.gz" in res["models"]["laya"]["unused_runs"]
    assert "laya-typed-decisions-PB-r3.json.gz" in res["models"]["laya-typed-decisions"]["unused_runs"]
    assert res["models"]["laya"]["interpretation"].startswith("C ")


def test_isolation_gain_below_5x_is_futile(raw):
    _both(raw, (1,), pb_gpu_us=1250)  # PB P50 1.25 ms: over 1 ms; gain 4x
    assert analyze.summarise(raw)["models"]["laya"]["looks"]["6"]["triggered"] == ["gpu_completion_isolation"]


def test_futility_stop_at_n12_not_at_n6(raw):
    _both(raw, (1, 2), laya=alt(1.0, 1.3, then=1.3, from_round=3))
    res = analyze.summarise(raw)
    looks = res["models"]["laya"]["looks"]
    assert looks["6"]["decision"] == "CONTINUE"
    assert looks["12"]["decision"] == "FUTILITY STOP" and looks["12"]["triggered"] == ["short_p99"]
    assert res["next"] is None and res["outcome"].startswith("FUTILITY STOP at n=12 on laya: short_p99; R1 FAIL")
    assert "12" not in res["models"]["laya-typed-decisions"]["looks"]


def test_pb_second_crash_is_futility_and_p_second_crash_is_invalid(raw):
    _both(raw, (1,))
    (raw / "failed").mkdir()
    for a in (1, 2):
        (raw / "failed" / f"laya-typed-decisions-PB-r3.{a}.log").write_text("boom\n")
    res = analyze.summarise(raw)
    assert res["next"] is None
    assert res["outcome"] == (
        "FUTILITY STOP at n=6 on laya-typed-decisions-PB-r3: PB second crash; R1 FAIL; protocol-history experiment next"
    )
    assert res["models"]["laya-typed-decisions"]["crashes"] == {"laya-typed-decisions-PB-r3": 2}
    for p in (raw / "failed").iterdir():
        p.unlink()
    for a in (1, 2):
        (raw / "failed" / f"laya-A-r3.{a}.log").write_text("boom\n")
    res = analyze.summarise(raw)
    assert res["next"] is None and res["outcome"].startswith("invalid: second crash of laya-A-r3")
    (raw / "failed" / "laya-A-r3.2.log").unlink()  # one crash: re-run, the campaign goes on
    res = analyze.summarise(raw)
    assert res["next"] == {"model": "laya", "block": 2} and res["failed_runs"] == ["laya-A-r3.1.log"]


# ------------------------------------------------------------------ the final look


def test_final_pass_on_both(raw):
    _both(raw, (1, 2, 3))
    res = analyze.summarise(raw)
    assert all(res["models"][m]["looks"]["18"]["verdict"] == "PASS" for m in MODELS)
    assert res["models"]["laya"]["looks"]["18"]["pairs"] == 18
    assert res["outcome"] == "PB PASS on laya and laya-typed-decisions at n=18: proceed to R2"
    assert res["next"] is None and "not reproduced" in res["models"]["laya"]["interpretation"]


def test_final_fail_after_two_continues(raw):
    _both(raw, (1, 2, 3), laya=alt(1.0, 1.3, then=1.3, from_round=5))
    res = analyze.summarise(raw)
    looks = res["models"]["laya"]["looks"]
    assert looks["6"]["decision"] == looks["12"]["decision"] == "CONTINUE"
    assert looks["18"]["verdict"] == "FAIL" and looks["18"]["checks"]["short_p99"] == "FAIL"
    assert res["outcome"] == "PB FAIL at n=18 on laya: R1 FAIL; protocol-history experiment next"


def test_final_inconclusive_stops_with_pairs_needed(raw):
    _both(raw, (1, 2, 3), laya=alt(0.85, 1.25))
    res = analyze.summarise(raw)
    lk = res["models"]["laya"]["looks"]["18"]
    assert lk["verdict"] == "INCONCLUSIVE" and list(lk["inconclusive"]) == ["short_p99"]
    d = lk["inconclusive"]["short_p99"]
    sd = float(np.std(np.log([1.25, 0.85] * 9), ddof=1))
    assert d["pair_log_sd"] == pytest.approx(sd)
    n = d["pairs_needed"]
    margin = np.log(1.05) - np.log(d["geomean"])
    assert analyze.gate.t_critical(n - 1) * sd / np.sqrt(n) < margin
    assert analyze.gate.t_critical(n - 2) * sd / np.sqrt(n - 1) >= margin
    assert res["next"] is None
    assert res["outcome"] == (
        "INCONCLUSIVE at n=18 on laya: campaign stopped; a targeted replication needs a new preregistration"
    )
    assert "pairs needed" in analyze.tables(res)


def test_pairs_needed_edges():
    assert analyze.pairs_needed(0.1, 1.05, 1.05) is None  # no margin
    assert analyze.pairs_needed(5.0, 1.049, 1.05) is None  # beyond 10000 pairs
    assert analyze.pairs_needed(0.0, 1.0, 1.05) == 2
    n = analyze.pairs_needed(0.1, 1.0, 1.05)
    brute = next(k for k in range(2, 1000) if analyze.gate.t_critical(k - 1) * 0.1 / np.sqrt(k) < np.log(1.05))
    assert n == brute


# ------------------------------------------------------------------ reporting


def test_end_to_end_with_window_history_and_tables(raw, tmp_path):
    # C: +20% short P99 after solo_long (even cycles), +15% after gpu_only; PB: +2% everywhere
    def ratio(c, rnd, k):
        if c == "C":
            return 1.2 if k % 2 == 0 else 1.15
        return 1.02

    _both(raw, (1, 2, 3), laya=ratio)
    (raw / "check.json").write_text(
        json.dumps({"PB_bit_identical_everywhere": True, "C_bit_identical_everywhere": True, "models": {}})
    )
    res = analyze.summarise(raw)
    m = res["models"]["laya"]
    assert m["looks"]["18"]["verdict"] == "PASS" and m["C_reference"]["verdict"] == "FAIL"
    assert "reproduced under the full protocol and PB avoids it" in m["interpretation"]
    rep = m["report"]
    assert rep["rounds"] == {"P": list(range(1, 7)), "PB": list(range(1, 7)), "C": [1, 2]}
    h = rep["window_history"]
    assert h["order_as_expected"] is True
    assert h["vs_P"]["C"]["solo_long"]["short_p99_geomean_ratio"] == pytest.approx(1.2)
    assert h["vs_P"]["C"]["gpu_only"]["short_p99_geomean_ratio"] == pytest.approx(1.15)
    assert h["vs_P"]["C"]["solo_long"]["pairs"] == 4 and h["vs_P"]["C"]["gpu_only"]["pairs"] == 2
    assert h["vs_P"]["PB"]["gpu_only"]["short_p99_geomean_ratio"] == pytest.approx(1.02)
    e = rep["extras"]
    assert e["PB"]["gil_wait"]["n"] == 6 * 3 * 8  # hetero windows only: 6 rounds x 3 cycles x 8 forwards
    assert e["C"]["gil_wait"]["n"] == 2 * 3 * 8
    assert e["P"]["windows"]["slow_cpu"] is None and e["PB"]["windows"]["slow_cpu"]["flagged_windows"] == 0
    assert rep["routing_failures"] == [] and m["unused_runs"] == []
    md = analyze.tables(res)
    assert "PB (n=18)" in md and "C (n=6)" in md and "solo_long" in md and "bit-identical everywhere True" in md
    assert "Bonferroni" not in md
    out = tmp_path / "out"
    out.mkdir()
    sys.argv = ["analyze.py", "--raw", str(raw), "--out", str(out)]
    analyze.main()
    sys.argv = ["analyze.py", "--raw", str(raw), "--out", str(out), "--check"]
    analyze.main()
    assert (out / "tables.md").read_text() == md


def test_old_design_files_are_listed_unused(raw):
    _both(raw, (1,))
    old = _fake_run("C", 128, 512, [10.0] * 3)
    with gzip.open(raw / "laya-C-r3.json.gz", "wt") as fh:
        json.dump(old, fh)
    res = analyze.summarise(raw)
    assert res["models"]["laya"]["unused_runs"] == ["laya-C-r3.json.gz"]
    assert res["models"]["laya-typed-decisions"]["unused_runs"] == []  # laya-* does not match laya-typed-*
    assert res["next"] == {"model": "laya", "block": 2}


def test_routing_failures_and_crashed_runs_are_listed(raw):
    _both(raw, (1,))
    path = raw / design.run_file("laya", "C", 2)
    with gzip.open(path, "rt") as fh:
        run = json.load(fh)
    w = [w for w in run["part_a"]["windows"] if w["condition"] == "hetero" and w["cycle"] == 1][0]
    w["streams"]["short"]["devices"] = {"ane": 19, "gpu": 1}
    with gzip.open(path, "wt") as fh:
        json.dump(run, fh)
    (raw / "failed").mkdir()
    (raw / "failed" / "laya-PB-r3.1.log").write_text("boom\n")
    res = analyze.summarise(raw)
    assert res["models"]["laya"]["report"]["routing_failures"] == [
        {"config": "C", "round": 2, "cycle": 1, "stream": "short", "devices": {"ane": 19, "gpu": 1}}
    ]
    assert res["failed_runs"] == ["laya-PB-r3.1.log"]
    md = analyze.tables(res)
    assert "C round 2 cycle 1" in md and "laya-PB-r3.1.log" in md


def test_runs_with_other_window_lengths_are_rejected(raw):
    _both(raw, (1,))
    path = raw / design.run_file("laya", "A", 1)
    with gzip.open(path, "rt") as fh:
        run = json.load(fh)
    run["args"]["seconds"] = 2.0
    with gzip.open(path, "wt") as fh:
        json.dump(run, fh)
    with pytest.raises(AssertionError):
        analyze.summarise(raw)


def test_preceding_condition_follows_bench_concurrency_order():
    run = _fake_run("A", 128, 512, [10.0, 10.0, 10.0])
    assert analyze.preceding(run) == {0: "solo_long", 1: "gpu_only", 2: "solo_long"}
    assert [analyze.expected_before(k) for k in range(3)] == ["solo_long", "gpu_only", "solo_long"]


# ------------------------------------------------------------------ run_config.py


def test_run_config_imports_without_pyobjc_and_finds_hetero_windows():
    rc = _run_config()
    assert set(rc.CONFIGS) == {"A", "C", "PB"}
    run = _fake_run("A", 128, 512, [10.0, 10.0, 10.0])
    bounds = rc.hetero_bounds(run)
    assert len(bounds) == 3
    assert bounds == analyze.gate57.hetero_bounds(run)
