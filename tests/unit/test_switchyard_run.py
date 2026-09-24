"""Switchyard end to end without a model: driver, result.json, trace.jsonl, replay and CLI.

A fake workers-mode Laya answers every train with its oracle and emits a RequestTrace whose
timestamps put GPU trains at GPU_LATENCY_MS and ANE trains at ANE_LATENCY_MS after submit.
The warmup and the lead-in are shortened; everything else is the real code path.
"""

from __future__ import annotations

import itertools
import json
import re
import time
from concurrent.futures import Future
from queue import SimpleQueue
from types import SimpleNamespace

import pytest

from laya_apple.benchmark import stats
from laya_apple.demos.switchyard import ane_state, driver, metrics, replay, result, world
from laya_apple.trace import RequestTrace

GPU_LATENCY_MS, ANE_LATENCY_MS, BACKGROUND_MS = 130.0, 12.0, 300.0
DURATION = 0.3
IDS = itertools.count(1)
READY = ane_state.AneStatus("ready", artifacts={"64": "a" * 64})
MACHINE = {
    "platform": {"soc": "Test SoC"},
    "memory_gb": 16,
    "python": "3.12",
    "mlx": "0",
    "routing_profile_validated": True,
    "calibrated_profile": False,
}
MODEL = {"name": world.MODEL, "revision": "f" * 40, "weights_sha256": "0" * 64, "ane_artifacts": {}}


class FakeLaya:
    def __init__(self, config, trace, *, use_ane=True, events=None, fail=None, ane_budget=None):
        self.config_label, self.trace, self.use_ane, self.fail = config, trace, use_ane, fail
        self.ane_budget = ane_budget  # trains the ANE serves before its worker "dies" (None: all)
        self.device = "gpu" if config == "gpu_only" else "auto"
        self.tokenizer = self.config = None
        self.ane = SimpleNamespace(artifact_sha256={64: "b" * 64}) if config == "hybrid" else None
        self.events = events if events is not None else []
        self.events.append(("load", config))
        self.submitted = []

    def info(self):
        return {"device": self.device, "execution": "workers", "ane_placement": "thread"}

    def submit(self, context=None, questions=None):
        now = time.monotonic_ns()
        self.submitted.append(now)
        rid = next(IDS)
        train = world.QUESTION_ID in questions
        target = "ane" if train and self.device == "auto" and self.use_ane else "gpu"
        reason = "validated_short_single_question_path" if target == "ane" else "gpu_requested"
        if target == "ane" and self.ane_budget is not None:
            if self.ane_budget <= 0:
                target, reason = "gpu", "ane_runtime_unavailable"
            self.ane_budget -= 1
        total = int((ANE_LATENCY_MS if target == "ane" else GPU_LATENCY_MS if train else BACKGROUND_MS) * 1e6)
        us = 1000
        t = RequestTrace(
            request_id=rid,
            sequence_length=64,
            question_count=1,
            target=target,
            routing_reason=reason,
            service_estimate_ms=9.0,
            gpu=None,
            ane=None,
            submit_ns=now,
            prepared_ns=now + us,
            routed_ns=now + 2 * us,
            queue_enter_ns=now + 3 * us,
            dispatch_ns=now + total // 2,
            service_start_ns=now + total // 2 + us,
            service_end_ns=now + total - 2 * us,
            received_ns=now + total - us,
            response_ns=now + total,
        )
        fut = Future()
        if self.fail is not None and len(self.submitted) == self.fail:
            fut.set_exception(RuntimeError("device error"))
            return fut
        answers = {}
        if train:
            choice = re.search(r"Platform (\w) is clear", context).group(1)
            answers = {
                world.QUESTION_ID: {"choice": choice, "probabilities": {"A": 0.1, "B": 0.1, "C": 0.1} | {choice: 0.8}}
            }
        self.trace(t)
        fut.set_result(SimpleNamespace(runtime=SimpleNamespace(request_id=rid), answers=answers))
        return fut

    def close(self):
        self.events.append(("close", self.config_label))


@pytest.fixture
def fake(monkeypatch):
    """Patch model loading and payload building; returns the list of load/close events."""
    events, opts = [], {"use_ane": True, "ane_budget": None, "fail_load": None, "fail_at": {}}
    monkeypatch.setattr(world, "WARMUP_S", 0.15)
    monkeypatch.setattr(driver, "LEAD_S", 0.01)
    monkeypatch.setattr(
        driver,
        "payloads",
        lambda items, tok, cfg: [world.train_prompt(a.train) if a.train else (a.cls, {"q0": {}}) for a in items],
    )

    def load(config, traces, local_files_only=False):
        if opts["fail_load"] == config:
            raise RuntimeError(f"cannot load {config}")
        return FakeLaya(
            config,
            traces.put,
            use_ane=opts["use_ane"],
            events=events,
            ane_budget=opts["ane_budget"],
            fail=opts["fail_at"].get(config),
        )

    monkeypatch.setattr(driver, "load", load)
    return SimpleNamespace(events=events, opts=opts)


@pytest.fixture
def static(tmp_path):
    d = tmp_path / "static"
    d.mkdir()
    (d / "index.html").write_text(
        "<html><style>/*__SWITCHYARD_CSS__*/</style>"
        '<script type="application/json" id="switchyard-data">__SWITCHYARD_DATA__</script>'
        "<script>/*__SWITCHYARD_JS__*/</script></html>"
    )
    (d / "style.css").write_text("body{color:red}")
    (d / "app.js").write_text('const raw = "__SWITCHYARD_DATA__"; /*__SWITCHYARD_CSS__*/')
    return d


_RUNS: dict = {}


def cached_run(seed=11):
    """One fake run per seed for the tests that only read it (each run takes ~1 s)."""
    if seed not in _RUNS:
        _RUNS[seed] = driver.run(seed, DURATION, READY)
    return _RUNS[seed]


def build(run):
    return result.build(run, machine_info=MACHINE, model=MODEL, created_utc="2026-01-01T00:00:00+00:00")


# ----------------------------------------------------------------------------- driver


def test_open_loop_submits_on_schedule_never_early(monkeypatch):
    monkeypatch.setattr(driver, "LEAD_S", 0.01)
    laya = FakeLaya("gpu_only", lambda t: None)
    items = world.schedule(11, 0.4)
    reqs = [world.train_prompt(a.train) if a.train else ("x", {"q0": {}}) for a in items]
    start, futures = driver.open_loop(laya, items, reqs)
    assert len(futures) == len(items) == len(laya.submitted)
    lags = [(s - (start + round(a.offset_s * 1e9))) / 1e6 for s, a in zip(laya.submitted, items)]
    assert min(lags) >= 0
    assert max(lags) < 50  # generous: a loaded CI runner may deschedule the submitter


def test_join_builds_trace_records_and_classifies():
    traces = SimpleQueue()
    laya = FakeLaya("gpu_only", traces.put)
    items = world.schedule(11, 0.4)
    reqs = [world.train_prompt(a.train) if a.train else ("x", {"q0": {}}) for a in items]
    futures = [laya.submit(*r) for r in reqs]
    records = driver.join(items, futures, driver.drain(traces), 0, 3, 100.0)
    assert [r["index"] for r in records] == [a.index for a in items]
    for r, a in zip(records, items):
        assert r["round"] == 3 and r["class"] == a.cls and r["arrival_ns"] == round(a.offset_s * 1e9)
        assert set(RequestTrace.__dataclass_fields__) <= set(r)
        if a.train:
            assert (r["line"], r["train_id"], r["pattern"], r["oracle"]) == (
                a.train.line,
                a.train.train_id,
                a.train.pattern,
                a.train.oracle,
            )
            assert r["answer"] == r["oracle"] and set(r["probabilities"]) == {"A", "B", "C"}
            assert r["outcome"] == metrics.outcome(metrics.latency_ms(r), r["answer"], r["oracle"], 100.0)
        else:
            assert "outcome" not in r


def test_join_refuses_gaps():
    traces = SimpleQueue()
    items = world.schedule(11, 0.2)
    reqs = [world.train_prompt(a.train) if a.train else ("x", {"q0": {}}) for a in items]
    futures = [FakeLaya("gpu_only", traces.put).submit(*r) for r in reqs]
    seen = driver.drain(traces)
    pending = list(futures)
    pending[1] = Future()
    with pytest.raises(driver.RoundError, match="did not finish"):
        driver.join(items, pending, seen, 0, 0, 100.0)
    seen.pop(next(iter(seen)))
    with pytest.raises(driver.RoundError, match="has no trace"):
        driver.join(items, futures, seen, 0, 0, 100.0)
    failing = FakeLaya("gpu_only", lambda t: None, fail=1)
    with pytest.raises(driver.RoundError, match="device error"):
        driver.join(items[:1], [failing.submit(*reqs[0])], {}, 0, 0, 100.0)


@pytest.mark.parametrize(("seed", "order"), [(10, ["gpu_only", "hybrid"]), (11, ["hybrid", "gpu_only"])])
def test_run_closes_each_laya_before_loading_the_next(fake, seed, order):
    run = driver.run(seed, DURATION, READY)
    assert [r.config for r in run.rounds] == order and [r.index for r in run.rounds] == [0, 1]
    assert fake.events == [("load", order[0]), ("close", order[0]), ("load", order[1]), ("close", order[1])]
    assert run.ane.state == "ready" and run.ane.warmup_ane_requests > 0
    assert run.configs["hybrid"] == {"device": "auto", "execution": "workers", "ane_placement": "thread"}
    assert run.configs["gpu_only"]["ane_placement"] is None
    assert run.ane_artifacts == {"64": "b" * 64}


def test_run_drops_hybrid_when_the_warmup_never_reaches_the_ane(fake):
    fake.opts["use_ane"] = False
    steps = []
    run = driver.run(11, DURATION, READY, step=lambda *a: steps.append(a[0]))
    assert [r.config for r in run.rounds] == ["gpu_only"]
    assert (run.ane.state, run.ane.reason, run.ane.warmup_ane_requests) == (
        "setup_available",
        ane_state.NO_ANE_WARMUP,
        0,
    )
    assert "ane_skipped" in steps and ("close", "hybrid") in fake.events


def test_run_without_ane_runs_gpu_only(fake):
    run = driver.run(11, DURATION, ane_state.AneStatus("setup_available", ane_state.NO_COREMLTOOLS))
    assert [r.config for r in run.rounds] == ["gpu_only"] and fake.events == [
        ("load", "gpu_only"),
        ("close", "gpu_only"),
    ]


def test_conditions_and_other_laya_processes(monkeypatch):
    import os

    me, parent = os.getpid(), 777001
    ps = (
        f"  {parent} 1 uv run laya-apple switchyard\n"
        f"  {me} {parent} python laya-apple switchyard\n"
        f"  777002 {me} python -c from laya_apple.executor import worker\n"
        "  4242 1 /usr/bin/python3 -m laya_apple.cli benchmark --token secret\n"
        "  4343 1 /System/AmbientDisplayAgent\n"
        "  17 1 zsh\n"
    )
    therm = "Note: No thermal warning level has been recorded\n"
    cpu = " 27.1 /Applications/OrbStack.app/Contents/MacOS/OrbStack Helper\n 0.0 /bin/zsh\n 11.7 claude\n bad\n"
    outputs = {"pid=,ppid=,command=": ps, "%cpu=,comm=": cpu}
    monkeypatch.setattr(driver, "_sh", lambda *cmd: outputs[cmd[2]] if cmd[0] == "ps" else therm)
    assert driver.other_laya_processes() == [{"pid": 4242, "name": "python3"}]
    c = driver.conditions()
    assert len(c["loadavg"]) == 3 and c["pmset_therm"] == ["Note: No thermal warning level has been recorded"]
    assert c["top_cpu"] == [[27.1, "OrbStack Helper"], [11.7, "claude"], [0.0, "zsh"]]


@pytest.mark.parametrize(
    ("command", "match"),
    [
        ("laya-apple switchyard", True),
        ("/x/.venv/bin/python3 /x/.venv/bin/laya-apple benchmark m", True),
        ("python -m laya_apple.cli info", True),
        ("python scripts/bench_concurrency.py --output o", True),
        ("/usr/bin/python3 /repo/scripts/bench_probe.py", True),
        ("/bin/zsh -c cd /Users/me/src/laya-apple && make", False),  # the repo path is not a laya process
        ("uv pip install -p /tmp/laya-apple/bin/python x", False),
        ("python -c from laya_apple.executor import worker", False),
        ("/System/AmbientDisplayAgent", False),
    ],
)
def test_other_laya_processes_match_argv0_argv1_only(command, match):
    assert driver._is_laya(command.split()) is match


# ----------------------------------------------------------------------------- result.json


def test_result_is_schema_valid_and_complete(fake):
    run = cached_run()
    r = build(run)
    assert result.validate(r) == []
    assert r["standard"] is False and r["submission_eligible"] is False  # custom duration
    assert r["design"] == {"ordering": "seed_parity", "counterbalance": "none", "sequence": ["hybrid", "gpu_only"]}
    assert r["workload"]["schedule_sha256"] == world.schedule_sha256(world.schedule(11, DURATION))
    assert r["comparison"] == {"available": True, "same_schedule_sha256": True, "decision_disagreements": 0}
    for rnd in r["rounds"]:
        assert rnd["game"]["trains"] == r["workload"]["offered"]["trains"]
        assert rnd["game"]["misrouted"] == 0 and rnd["game"]["route_accuracy"] == 1.0
    hybrid, gpu = r["configs"]["hybrid"], r["configs"]["gpu_only"]
    assert hybrid["game"]["late"] == 0 and gpu["game"]["late"] == gpu["game"]["trains"]
    assert set(gpu["summary"]) == set(r["rounds"][0]["systems"])  # summary is shaped like a round's systems
    assert r["ane"]["state"] == "ready" and r["ane"]["setup_command"] is None


def test_submission_eligible_is_any_standard_run(fake, monkeypatch):
    """GPU-only Macs can submit: eligibility follows `standard` alone."""
    monkeypatch.setattr(world, "DURATION_S", DURATION)
    both = build(cached_run())
    gpu_only = build(driver.run(11, DURATION, ane_state.AneStatus("setup_available", ane_state.NO_COREMLTOOLS)))
    for r in (both, gpu_only):
        assert r["mode"] == "standard" and r["standard"] is True and r["submission_eligible"] is True
        assert result.validate(r) == []
    assert list(gpu_only["configs"]) == ["gpu_only"]
    custom = build(driver.run(12, DURATION, ane_state.AneStatus("setup_available", ane_state.NO_COREMLTOOLS)))
    assert custom["standard"] is False and custom["submission_eligible"] is False


def test_standard_flag():
    assert result.is_standard(11, 60) and result.is_standard(11, 60.0)
    assert not result.is_standard(12, 60) and not result.is_standard(11, 30)


def test_configs_summary_pools_every_round_of_a_config(fake):
    """ABBA-ready: two rounds of one config pool their samples."""
    run = driver.run(10, DURATION, READY)
    run2 = driver.run(10, DURATION, READY)
    for rnd in reversed(run2.rounds):  # gpu_only, hybrid, hybrid, gpu_only
        rnd.index = len(run.rounds)
        run.rounds.append(rnd)
    r = build(run)
    assert r["configs"]["gpu_only"]["rounds"] == [0, 3] and r["configs"]["hybrid"]["rounds"] == [1, 2]
    mine = [x for x in run.rounds if x.config == "gpu_only"]
    lat = [metrics.latency_ms(rec) for x in mine for rec in x.records if rec["class"] == "train"]
    assert r["configs"]["gpu_only"]["summary"]["decision_latency"]["p99_ms"] == pytest.approx(stats(lat)["p99_ms"])
    assert r["configs"]["gpu_only"]["game"]["trains"] == 2 * r["workload"]["offered"]["trains"]


def test_schema_rejects_a_broken_result(fake):
    r = build(cached_run())
    del r["rounds"][0]["systems"]["miss_rate_at_ms"]["100"]
    r["ane"]["state"] = "maybe"
    problems = result.validate(r)
    assert any("miss_rate_at_ms" in p for p in problems) and any("ane.state" in p for p in problems)
    with pytest.raises(ValueError, match="schema"):
        result.write("unused", r, [])


def test_machine_paths_are_redacted():
    roots = [("/Users/someone/cache/laya", "<LAYA_APPLE_CACHE>"), ("/Volumes/hf", "<HF_HOME>"), ("/Users/someone", "~")]
    obj = {
        "detail": "L64: ArtifactMissingError: no artifact in /Users/someone/cache/laya/artifacts/x",
        "hf": "/Volumes/hf/hub/models--x",
        "/Users/someone/k": [1, "/Users/someone/y"],
    }
    assert result.redact(obj, roots) == {
        "detail": "L64: ArtifactMissingError: no artifact in <LAYA_APPLE_CACHE>/artifacts/x",
        "hf": "<HF_HOME>/hub/models--x",
        "~/k": [1, "~/y"],
    }


def test_result_redacts_ane_detail(fake, monkeypatch):
    from laya_apple import hub

    monkeypatch.setattr(hub, "cache_root", lambda: __import__("pathlib").Path("/somewhere/cache"))
    status = ane_state.AneStatus("setup_available", ane_state.NO_ARTIFACTS, detail="L64: missing in /somewhere/cache/a")
    r = build(driver.run(11, DURATION, status))
    assert r["ane"]["detail"] == "L64: missing in <LAYA_APPLE_CACHE>/a"


def test_write_and_read_round_trip(fake, tmp_path):
    run = cached_run()
    r = build(run)
    result.write(tmp_path, r, run.rounds)
    back, records = result.read(tmp_path)
    assert back == json.loads(json.dumps(r))
    assert len(records) == sum(len(x.records) for x in run.rounds)  # every request traced
    for x in run.rounds:
        s = metrics.summarize([([rec for rec in records if rec["round"] == x.index], x.round_start_ns)])
        assert s["game"] == back["rounds"][x.index]["game"]
        assert s["systems"] == back["rounds"][x.index]["systems"]


# ----------------------------------------------------------------------------- replay


TRAIN_KEYS = [
    "id",
    "line",
    "platforms",
    "oracle",
    "answer",
    "device",
    "outcome",
    "arrival_ms",
    "queue_enter_ms",
    "dispatch_ms",
    "service_start_ms",
    "service_end_ms",
    "response_ms",
    "latency_ms",
    "queue_ms",
    "service_ms",
]
BACKGROUND_KEYS = [
    "class",
    "device",
    "arrival_ms",
    "queue_enter_ms",
    "dispatch_ms",
    "service_start_ms",
    "service_end_ms",
    "response_ms",
    "latency_ms",
]


def test_replay_data_follows_the_contract(fake):
    run = cached_run()
    r = build(run)
    records = [rec for x in run.rounds for rec in x.records]
    data = replay.replay_data(r, records)
    assert (data["format"], data["format_version"]) == ("switchyard-replay", 1) and data["result"] == r
    assert [x["index"] for x in data["rounds"]] == [0, 1]
    for rnd, raw in zip(data["rounds"], run.rounds):
        assert set(rnd) == {"index", "config", "duration_s", "deadline_ms", "trains", "background"}
        assert (rnd["config"], rnd["duration_s"], rnd["deadline_ms"]) == (raw.config, DURATION, 100.0)
        assert rnd["trains"] and rnd["background"]
        for t in rnd["trains"]:
            assert list(t) == TRAIN_KEYS
            assert t["outcome"] in ("delivered", "late", "misrouted") and t["device"] in ("gpu", "ane")
            assert 1 <= t["line"] <= 6 and t["platforms"][t["oracle"]] == "clear"
            for k in TRAIN_KEYS[7:]:
                assert t[k] == round(t[k], 3)
            assert 0 <= t["arrival_ms"] <= t["queue_enter_ms"] <= t["dispatch_ms"] <= t["response_ms"]
        for b in rnd["background"]:
            assert list(b) == BACKGROUND_KEYS and b["class"] in world.BACKGROUND
        arrivals = [t["arrival_ms"] for t in rnd["trains"]]
        assert arrivals == sorted(arrivals)
        # relative to the round start: the scheduled offsets, to the microsecond
        expected = [round(a.offset_s * 1e3, 3) for a in run.schedule if a.train]
        assert arrivals == pytest.approx(expected, abs=1e-3)


def test_replay_round_trip_recomputes_the_result(fake):
    """Every headline number on the card can be recomputed from the replay data alone."""
    run = cached_run()
    r = build(run)
    data = replay.replay_data(r, [rec for x in run.rounds for rec in x.records])
    for rnd, res in zip(data["rounds"], r["rounds"]):
        trains = rnd["trains"]
        for outcome in ("delivered", "late", "misrouted"):
            assert sum(t["outcome"] == outcome for t in trains) == res["game"][outcome]
        assert len(trains) == res["game"]["trains"]
        lat = [t["latency_ms"] for t in trains]
        q = [t["queue_ms"] for t in trains]
        for key in ("p50_ms", "p95_ms", "p99_ms", "max_ms", "mean_ms"):
            assert stats(lat)[key] == pytest.approx(res["systems"]["decision_latency"][key], abs=1e-3)
            assert stats(q)[key] == pytest.approx(res["systems"]["queue_wait"][key], abs=1e-3)
        assert metrics.miss_rates(lat) == res["systems"]["miss_rate_at_ms"]
        assert sum(t["latency_ms"] > rnd["deadline_ms"] for t in trains) == res["systems"]["late"]["count"]


def test_bundle_fills_markers_once_and_escapes_the_data(static):
    data = {"format": "switchyard-replay", "x": "</script><b>"}
    html = replay.bundle(data, static)
    assert "body{color:red}" in html and "<\\/script><b>" in html and "</script><b>" not in html
    # replaced text is never rescanned: app.js keeps its literal marker strings
    assert 'const raw = "__SWITCHYARD_DATA__"; /*__SWITCHYARD_CSS__*/' in html
    payload = html.split('id="switchyard-data">', 1)[1].split("</script>", 1)[0]
    assert json.loads(payload.replace("<\\/", "</")) == data


def test_bundle_refuses_a_broken_template(static):
    (static / "index.html").write_text("<html>/*__SWITCHYARD_CSS__*/ /*__SWITCHYARD_JS__*/</html>")
    with pytest.raises(ValueError, match="__SWITCHYARD_DATA__"):
        replay.bundle({}, static)


def test_bundle_refuses_js_that_closes_its_script(static):
    (static / "app.js").write_text("document.write('</script>')")
    with pytest.raises(ValueError, match="closes"):
        replay.bundle({}, static)


# ----------------------------------------------------------------------------- CLI


@pytest.fixture
def cli_env(fake, static, monkeypatch):
    from laya_apple.demos.switchyard import cli

    real = replay.static_files
    monkeypatch.setattr(replay, "static_files", lambda static_dir=None: real(static_dir or static))
    monkeypatch.setattr(result, "machine", lambda: MACHINE)
    monkeypatch.setattr(result, "model_info", lambda artifacts: MODEL)
    opened = []
    monkeypatch.setattr(cli.webbrowser, "open", opened.append)
    monkeypatch.setattr(world, "whole_burst_cycles", lambda d: True)  # the fake rounds are shorter than one cycle
    return SimpleNamespace(opened=opened, fake=fake)


def test_cli_parses_exactly_the_contract_flags():
    from laya_apple.cli import build_parser

    a = build_parser().parse_args(["switchyard"])
    assert (a.seed, a.duration, a.out, a.no_open, a.setup_ane, a.replay) == (11, 60.0, None, False, False, None)
    a = build_parser().parse_args(
        ["switchyard", "--seed", "12", "--duration", "5", "--out", "d", "--no-open", "--setup-ane", "--replay", "r"]
    )
    assert (a.seed, a.duration, a.out, a.no_open, a.setup_ane, a.replay) == (12, 5.0, "d", True, True, "r")
    with pytest.raises(SystemExit):
        build_parser().parse_args(["switchyard", "--duration", "0"])


def test_cli_missing_ane_exits_zero_and_prints_the_setup_command(cli_env, monkeypatch, tmp_path, capsys):
    from laya_apple.cli import main

    monkeypatch.setattr(
        ane_state, "detect", lambda spec: ane_state.AneStatus("setup_available", ane_state.NO_COREMLTOOLS)
    )
    assert main(["switchyard", "--duration", str(DURATION), "--out", str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out
    assert "The Neural Engine runtime (Core ML Tools) is not installed" in out and ane_state.SETUP_COMMAND in out
    assert "GPU only (GPU + ANE did not run: The Neural Engine runtime (Core ML Tools) is not installed)." in out
    assert "the end of the run says how to add GPU + ANE" in out and "trains late" in out and "P99" in out
    assert {p.name for p in (tmp_path / "out").iterdir()} == {"result.json", "trace.jsonl", "replay.html"}
    assert cli_env.opened == [(tmp_path / "out" / "replay.html").resolve().as_uri()]
    r = json.loads((tmp_path / "out" / "result.json").read_text())
    assert r["ane"]["setup_command"] == ane_state.SETUP_COMMAND and list(r["configs"]) == ["gpu_only"]
    assert (r["ane"]["reason"], r["ane"]["reason_text"]) == (
        "ane_runtime_unavailable",
        "The Neural Engine runtime (Core ML Tools) is not installed",
    )


def test_cli_ready_runs_both_rounds_and_replay_rebuilds(cli_env, monkeypatch, tmp_path, capsys):
    from laya_apple.cli import main

    monkeypatch.setattr(ane_state, "detect", lambda spec: READY)
    out_dir = tmp_path / "out"
    assert main(["switchyard", "--duration", str(DURATION), "--out", str(out_dir), "--no-open"]) == 0
    text = capsys.readouterr().out
    assert "Round 1 of 2: GPU + ANE" in text and "Round 2 of 2: GPU only" in text
    summary = text[text.index("Same timetable. GPU only vs GPU + ANE.") :]  # GPU only first, whatever the run order
    assert summary.index("GPU only ") < summary.index("GPU + ANE ") and "Late = decision more than 100 ms" in summary
    assert ane_state.SETUP_COMMAND not in text and cli_env.opened == []
    first = (out_dir / "replay.html").read_text()
    (out_dir / "replay.html").unlink()
    assert main(["switchyard", "--replay", str(out_dir), "--no-open"]) == 0
    assert (out_dir / "replay.html").read_text() == first
    assert main(["switchyard", "--replay", str(tmp_path / "nothing"), "--no-open"]) == 2  # usage error
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "result.json").write_text("{not json")
    assert main(["switchyard", "--replay", str(bad), "--no-open"]) == 2
    (bad / "result.json").write_text('{"schema": "something-else"}')
    assert main(["switchyard", "--replay", str(bad), "--no-open"]) == 2


def test_cli_messages_are_all_used_and_every_reason_has_text():
    import inspect

    from laya_apple.demos.switchyard import cli, messages

    source = inspect.getsource(cli)
    for key, text in messages.MESSAGES.items():
        assert f'"{key}"' in source or key.startswith("setup_")
        assert text.strip()
    assert set(messages.REASON_TEXT) == set(ane_state.REASONS)
    assert (ane_state.NO_COREMLTOOLS, ane_state.NO_ARTIFACTS, ane_state.NOT_CALIBRATED) == (
        "ane_runtime_unavailable",
        "ane_artifact_unavailable",
        "platform_not_validated",
    )


def test_cli_without_mlx_prints_one_line_and_exits_2(monkeypatch, capsys):
    from laya_apple.cli import main
    from laya_apple.demos.switchyard import cli, messages

    monkeypatch.setattr(cli, "mlx_available", lambda: False)
    monkeypatch.setattr(ane_state, "detect", lambda spec: pytest.fail("must not probe anything"))
    assert main(["switchyard", "--no-open"]) == 2
    captured = capsys.readouterr()
    assert captured.out == messages.MESSAGES["needs_apple_silicon"] + "\n" and "Traceback" not in captured.err


def test_default_out_dir_names_time_and_soc():
    from datetime import datetime, timezone

    from laya_apple.demos.switchyard import cli

    p = cli.default_out(datetime(2026, 9, 24, 13, 5, 9, tzinfo=timezone.utc))
    assert p.parent.name == "switchyard-results" and re.fullmatch(r"20260924T130509Z-[a-z0-9-]+", p.name)
    assert driver.soc_slug("Apple M4 Max") == "apple-m4-max" and driver.soc_slug("") == "unknown-soc"


# ----------------------------------------------------------------------------- failure paths


def _warm_trains():
    return sum(a.cls == world.TRAIN for a in world.warmup_schedule(11))


@pytest.mark.parametrize("served", ["none", "some"])
def test_hybrid_round_that_stops_using_the_ane_is_not_a_comparison(fake, served):
    """The warmup reaches the ANE, the round does not (or only partly): kept, but demoted."""
    fake.opts["ane_budget"] = _warm_trains() + (0 if served == "none" else 5)
    steps = []
    run = driver.run(11, DURATION, READY, step=lambda *a: steps.append(a[0]))
    hybrid = next(r for r in run.rounds if r.config == "hybrid")
    assert [r.config for r in run.rounds] == ["hybrid", "gpu_only"] and hybrid.ane_verified is False
    assert (run.ane.state, run.ane.reason) == ("unavailable", ane_state.ANE_LOST) and "ane_failed" in steps
    r = build(run)
    assert result.validate(r) == []
    assert r["comparison"]["available"] is False and r["rounds"][0]["ane_verified"] is False
    assert r["rounds"][1]["ane_verified"] is None and "hybrid" in r["configs"]  # data kept
    assert r["ane"]["reason_text"] == "The Neural Engine stopped being used during the round"


def test_cli_does_not_present_an_unverified_hybrid_round(cli_env, monkeypatch, tmp_path, capsys):
    from laya_apple.cli import main

    monkeypatch.setattr(ane_state, "detect", lambda spec: READY)
    cli_env.fake.opts["ane_budget"] = _warm_trains()
    assert main(["switchyard", "--duration", str(DURATION), "--out", str(tmp_path / "o"), "--no-open"]) == 0
    out = capsys.readouterr().out
    assert "GPU only (the GPU + ANE round is not counted: The Neural Engine stopped being used" in out
    assert "Same timetable" not in out and "GPU + ANE       " not in out


def test_ane_load_failure_exits_zero_with_the_gpu_only_result(cli_env, monkeypatch, tmp_path, capsys):
    from laya_apple.cli import main

    monkeypatch.setattr(ane_state, "detect", lambda spec: READY)
    cli_env.fake.opts["fail_load"] = "hybrid"
    assert main(["switchyard", "--duration", str(DURATION), "--out", str(tmp_path / "o"), "--no-open"]) == 0
    out = capsys.readouterr().out
    assert "GPU + ANE round not counted: The GPU + ANE round failed on this Mac." in out
    r = json.loads((tmp_path / "o" / "result.json").read_text())
    assert list(r["configs"]) == ["gpu_only"] and r["design"]["sequence"] == ["gpu_only"]
    assert (r["ane"]["state"], r["ane"]["reason"]) == ("unavailable", ane_state.ANE_FAILED)
    assert r["ane"]["detail"] == "RuntimeError: cannot load hybrid" and r["comparison"]["available"] is False


def test_gpu_round_failure_saves_the_finished_rounds_and_exits_1(cli_env, monkeypatch, tmp_path, capsys):
    from laya_apple.cli import main

    monkeypatch.setattr(ane_state, "detect", lambda spec: READY)
    cli_env.fake.opts["fail_at"] = {"gpu_only": _warm_trains() + 20}  # a round request, after the warmup
    assert main(["switchyard", "--duration", str(DURATION), "--out", str(tmp_path / "o"), "--no-open"]) == 1
    out = capsys.readouterr().out
    assert "The GPU-only round could not finish (RoundError: request" in out and "Traceback" not in out
    assert "Saved the rounds that finished in" in out
    back, _ = result.read(tmp_path / "o")
    assert result.validate(back) == [] and list(back["configs"]) == ["hybrid"]


def test_gpu_failure_before_any_round_writes_nothing(cli_env, monkeypatch, tmp_path, capsys):
    from laya_apple.cli import main

    monkeypatch.setattr(
        ane_state, "detect", lambda spec: ane_state.AneStatus("setup_available", ane_state.NO_COREMLTOOLS)
    )
    cli_env.fake.opts["fail_at"] = {"gpu_only": 1}
    assert main(["switchyard", "--duration", str(DURATION), "--out", str(tmp_path / "o"), "--no-open"]) == 1
    assert "warmup request 0" in capsys.readouterr().out and not (tmp_path / "o").exists()


def test_warmup_failures_carry_context(fake):
    laya = FakeLaya("gpu_only", lambda t: None, fail=2)
    items = world.warmup_schedule(11)
    reqs = [world.train_prompt(a.train) if a.train else ("x", {"q0": {}}) for a in items]
    with pytest.raises(driver.RoundError, match=r"warmup request 1 \(\w+\) failed: RuntimeError: device error"):
        driver.warmup(laya, SimpleQueue(), items, reqs)


@pytest.mark.parametrize("duration", ["2.5", "1", "4", "61"])
def test_duration_must_be_whole_burst_cycles_and_is_checked_before_loading(fake, capsys, duration):
    from laya_apple.cli import main

    assert main(["switchyard", "--duration", duration, "--no-open"]) == 2
    assert "--duration must be a whole number of 3 s burst cycles" in capsys.readouterr().out
    assert fake.events == []


def test_whole_burst_cycles():
    assert world.MIN_DURATION_S == world.BURST_PERIOD_S == 3.0
    assert all(world.whole_burst_cycles(d) for d in (3, 6.0, 9, 60, 60.0, 600))
    assert not any(world.whole_burst_cycles(d) for d in (0, 1, 2.9, 3.5, 5, 59, 61))


def test_offered_rate_equals_the_nominal_mean_only_for_whole_cycles():
    """L8: offered.req_s is the realised rate over the window."""
    for d in (3, 6, 60):
        assert world.offered(world.schedule(11, d), d)["req_s"] == pytest.approx(world.RATE_REQ_S, rel=0.25)
    assert world.offered(world.schedule(11, 1), 1)["req_s"] > 2 * world.RATE_REQ_S  # 1 s is all burst


def test_single_round_has_no_schedule_comparison_and_submit_lag_covers_every_request(fake):
    r = build(driver.run(11, DURATION, ane_state.AneStatus("setup_available", ane_state.NO_COREMLTOOLS)))
    assert r["comparison"] == {"available": False, "same_schedule_sha256": None, "decision_disagreements": 0}
    sec = r["rounds"][0]["systems"]["secondary"]
    assert sec["submit_lag_all_requests"]["n"] == r["workload"]["offered"]["requests"]
    assert sec["submit_lag"]["n"] == r["workload"]["offered"]["trains"]


def test_burst_shape_is_recorded_from_the_arrival_constants():
    from laya_apple import workload

    r = build(cached_run())
    assert (r["workload"]["burst_period_s"], r["workload"]["burst_on_s"]) == (3.0, 1.0)
    assert (workload.BURST_PERIOD_S, workload.BURST_ON_S) == (3.0, 1.0)


def test_setup_failure_is_reported_and_the_gpu_round_still_runs(cli_env, monkeypatch, tmp_path, capsys):
    from laya_apple.cli import main

    failed = ane_state.AneStatus("setup_available", ane_state.NO_CONVERT, detail="L64: ArtifactError: needs torch")
    monkeypatch.setattr(ane_state, "setup", lambda spec, **kw: failed)
    monkeypatch.setattr(ane_state, "detect", lambda spec: pytest.fail("the setup failure is the state"))
    args = ["switchyard", "--setup-ane", "--duration", str(DURATION), "--out", str(tmp_path / "o"), "--no-open"]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "Neural Engine setup did not finish: The Neural Engine build tools (laya-apple[convert])" in out
    assert json.loads((tmp_path / "o" / "result.json").read_text())["ane"]["reason"] == ane_state.NO_CONVERT
