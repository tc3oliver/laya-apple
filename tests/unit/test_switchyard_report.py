"""Unit tests for scripts/switchyard_report.py, using small synthetic result dicts built
in-test (not fixtures owned by other agents). Every fixture is validated against the real
laya_apple/data/switchyard-result.schema.json, since the report script now rejects anything
that does not pass."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_switchyard_report():
    spec = importlib.util.spec_from_file_location("switchyard_report", ROOT / "scripts" / "switchyard_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sr():
    return _load_switchyard_report()


def _pct(n=10, p50=10.0, p95=30.0, p99=42.0, mean=15.0, max_=90.0):
    return {"n": n, "p50_ms": p50, "p95_ms": p95, "p99_ms": p99, "mean_ms": mean, "max_ms": max_}


def _secondary():
    bg = {
        cls: {"decision_latency": {"n": 1}, "queue_wait": {"n": 1}, "devices": {}}
        for cls in ("medium_1q", "long_1q", "short_4q")
    }
    return {
        "decisions_per_s": 1.0,
        "completed_req_s": 1.0,
        "makespan_s": 60.0,
        "submit_lag": _pct(),
        "inference": _pct(),
        "overhead": _pct(),
        "devices": {},
        "routing_reasons": {},
        "background": bg,
        "submit_lag_all_requests": _pct(),
    }


def _systems(late_count=1, p99_decision=42.0, p99_queue=3.0, p50=10.0, p95=30.0, n=10):
    return {
        "late": {"count": late_count, "rate": late_count / n, "deadline_ms": 100},
        "decision_latency": _pct(n=n, p50=p50, p95=p95, p99=p99_decision),
        "queue_wait": _pct(n=n, p50=1.0, p95=2.5, p99=p99_queue, mean=1.2, max_=4.0),
        "miss_rate_at_ms": {"25": 0.5, "50": 0.3, "100": late_count / n, "250": 0.0, "500": 0.0},
        "secondary": _secondary(),
    }


def _round(index, config, delivered=8, late=1, misrouted=1, trains=10, p99_decision=42.0):
    return {
        "index": index,
        "config": config,
        "started_utc": "2026-01-01T00:00:00Z",
        "round_start_ns": 0,
        "timetable_sha256": "0" * 64,
        "ane_verified": True,
        "conditions": {
            "start": {"utc": "2026-01-01T00:00:00Z", "loadavg": [0.1, 0.1, 0.1], "top_cpu": [], "pmset_therm": []},
            "end": {"utc": "2026-01-01T00:01:00Z", "loadavg": [0.1, 0.1, 0.1], "top_cpu": [], "pmset_therm": []},
            "other_laya_processes": [],
        },
        "game": {"trains": trains, "delivered": delivered, "late": late, "misrouted": misrouted, "route_accuracy": 0.9},
        "systems": _systems(late_count=late, p99_decision=p99_decision, n=trains),
    }


def _config(device, ane_placement, rounds, delivered=8, late=1, misrouted=1, trains=10, p99_decision=42.0):
    return {
        "device": device,
        "execution": "workers",
        "ane_placement": ane_placement,
        "rounds": rounds,
        "game": {"trains": trains, "delivered": delivered, "late": late, "misrouted": misrouted, "route_accuracy": 0.9},
        "summary": _systems(late_count=late, p99_decision=p99_decision, n=trains),
    }


def _result(
    *,
    hybrid=True,
    standard=True,
    laya_apple_version="1.1.0",
    model_revision="a" * 40,
    schedule_sha256="b" * 64,
    ane_state="ready",
    late_gpu=1,
    late_hybrid=0,
    p99_gpu=42.0,
    p99_hybrid=20.0,
):
    rounds = [_round(0, "gpu_only", late=late_gpu, p99_decision=p99_gpu)]
    configs = {"gpu_only": _config("gpu", None, [0], late=late_gpu, p99_decision=p99_gpu)}
    if hybrid:
        rounds.append(_round(1, "hybrid", delivered=9, late=late_hybrid, misrouted=1, p99_decision=p99_hybrid))
        configs["hybrid"] = _config(
            "auto", "process", [1], delivered=9, late=late_hybrid, misrouted=1, p99_decision=p99_hybrid
        )
    return {
        "schema": "laya-apple/switchyard-result",
        "schema_version": 1,
        "tool": "laya-apple switchyard",
        "laya_apple": laya_apple_version,
        "created_utc": "2026-01-01T00:00:00Z",
        "mode": "standard",
        "standard": standard,
        "submission_eligible": standard,
        "workload": {
            "id": "switchyard-v1",
            "version": 1,
            "seed": 11,
            "duration_s": 60,
            "warmup_s": 5,
            "arrivals": "bursty",
            "nominal_rate_req_s": 40,
            "burst_period_s": 3,
            "burst_on_s": 1,
            "deadline_ms": 100,
            "thresholds_ms": [25, 50, 100, 250, 500],
            "mix": {"train": 0.6, "medium_1q": 0.2, "long_1q": 0.1, "short_4q": 0.1},
            "lengths": {"train": 128, "medium_1q": 512, "long_1q": 1024, "short_4q": 128},
            "prompt_template_sha256": "c" * 64,
            "schedule_sha256": schedule_sha256,
            "offered": {"requests": 100, "trains": 60, "req_s": 40.0},
        },
        "machine": {
            "platform": {},
            "memory_gb": 64,
            "python": "3.12",
            "mlx": "0.1",
            "routing_profile_validated": True,
            "calibrated_profile": True,
        },
        "model": {
            "name": "laya-typed-decisions",
            "revision": model_revision,
            "weights_sha256": "d" * 64,
            "ane_artifacts": {},
        },
        "ane": {
            "state": ane_state,
            "reason": None if ane_state == "ready" else "no calibrated profile",
            "reason_text": None,
            "setup_command": None,
            "warmup_ane_requests": 3 if ane_state == "ready" else 0,
        },
        "design": {
            "ordering": "seed_parity",
            "counterbalance": "none",
            "sequence": ["hybrid", "gpu_only"] if hybrid else ["gpu_only"],
        },
        "rounds": rounds,
        "configs": configs,
        "comparison": {
            "available": hybrid,
            "same_schedule_sha256": hybrid,
            "decision_disagreements": 2 if hybrid else 0,
        },
        "trace_file": "trace.jsonl",
    }


def test_render_reports_metrics_in_frozen_order(sr, tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(_result()))
    report = sr.render([path])
    header = report.index("late (count/rate)")
    assert (
        header
        < report.index("P99 decision")
        < report.index("P99 queue")
        < report.index("delivered")
        < report.index("P50 / P95")
    )
    assert "gpu_only" in report and "hybrid" in report


def test_render_handles_single_config_run(sr, tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(_result(hybrid=False)))
    report = sr.render([path])
    assert "gpu_only" in report
    assert "hybrid" not in report


def test_load_run_rejects_schema_violation(sr, tmp_path):
    data = _result()
    del data["ane"]["state"]  # required field, but also violates the state enum implicitly
    del data["configs"]["gpu_only"]["summary"]  # required
    path = tmp_path / "result.json"
    path.write_text(json.dumps(data))
    with pytest.raises(SystemExit, match="schema violation"):
        sr.load_run(path)
    with pytest.raises(SystemExit, match="schema violation"):
        sr.render([path])


def test_load_run_rejects_bad_revision_pattern(sr, tmp_path):
    data = _result(model_revision="not-hex")
    path = tmp_path / "result.json"
    path.write_text(json.dumps(data))
    with pytest.raises(SystemExit, match="schema violation"):
        sr.load_run(path)


def test_spread_table_reports_min_max_across_runs(sr, tmp_path):
    paths = []
    for i, late in enumerate((1, 3)):
        d = _result(late_gpu=late)
        p = tmp_path / f"result-{i}.json"
        p.write_text(json.dumps(d))
        paths.append(p)
    report = sr.render(paths)
    assert "Cross-run spread" in report
    assert "1–3" in report  # min-max late count across the two runs


def test_spread_table_excludes_non_standard_runs(sr, tmp_path):
    paths = []
    for i, standard in enumerate((True, False)):
        d = _result(standard=standard)
        p = tmp_path / f"result-{i}.json"
        p.write_text(json.dumps(d))
        paths.append(p)
    report = sr.render(paths)
    assert "Excluded from cross-run spread" in report
    assert "standard: false" in report


def test_spread_table_excludes_runs_with_different_fingerprint(sr, tmp_path):
    paths = []
    for i, rev in enumerate(("a" * 40, "e" * 40)):
        d = _result(model_revision=rev)
        p = tmp_path / f"result-{i}.json"
        p.write_text(json.dumps(d))
        paths.append(p)
    report = sr.render(paths)
    assert "Excluded from cross-run spread" in report
    assert "model.revision" in report


def test_single_non_standard_run_still_reports_exclusion_reason(sr, tmp_path):
    """A single-run report (no spread table at all) must still say why that run doesn't count."""
    path = tmp_path / "result.json"
    path.write_text(json.dumps(_result(standard=False)))
    report = sr.render([path])
    assert "Cross-run spread" not in report  # only one run: no spread table
    assert "Excluded from cross-run spread" in report
    assert "standard: false" in report


def test_spread_section_reports_when_every_run_is_excluded(sr, tmp_path):
    paths = []
    for i in range(2):
        d = _result(standard=False)
        p = tmp_path / f"result-{i}.json"
        p.write_text(json.dumps(d))
        paths.append(p)
    report = sr.render(paths)
    assert "No runs eligible for the cross-run spread." in report
    assert "| config | runs |" not in report  # no empty table header
    assert "Excluded from cross-run spread" in report


def test_discover_runs_finds_run_subdirectories(sr, tmp_path):
    for i in (1, 2):
        d = tmp_path / f"run-00{i}"
        d.mkdir()
        (d / "result.json").write_text(json.dumps(_result()))
    found = sr.discover_runs(tmp_path)
    assert len(found) == 2
    assert all(p.name == "result.json" for p in found)


def test_discover_runs_returns_empty_for_no_results(sr, tmp_path):
    assert sr.discover_runs(tmp_path) == []


def test_render_uses_paths_relative_to_root(sr, tmp_path):
    raw = tmp_path / "raw" / "run-001"
    raw.mkdir(parents=True)
    (raw / "result.json").write_text(json.dumps(_result()))
    report = sr.render([raw / "result.json"], root=tmp_path)
    assert "### raw/run-001/result.json" in report
    assert str(tmp_path) not in report


def test_check_passes_when_readme_matches_regenerated_report(sr, tmp_path):
    bench_dir = tmp_path / "bench"
    campaign_dir = bench_dir / "v1-test"
    raw = campaign_dir / "raw" / "run-001"
    raw.mkdir(parents=True)
    (raw / "result.json").write_text(json.dumps(_result()))
    report = sr.render([raw / "result.json"], root=campaign_dir)
    begin, end = sr.markers("v1-test")
    readme = bench_dir / "README.md"
    readme.write_text(f"# Title\n\n## Results\n\n### v1-test\n\n{begin}\n\n{report.strip()}\n\n{end}\n")
    sr.check(campaign_dir)  # must not raise


def test_check_fails_when_readme_is_stale(sr, tmp_path):
    bench_dir = tmp_path / "bench"
    campaign_dir = bench_dir / "v1-test"
    raw = campaign_dir / "raw" / "run-001"
    raw.mkdir(parents=True)
    (raw / "result.json").write_text(json.dumps(_result()))
    begin, end = sr.markers("v1-test")
    readme = bench_dir / "README.md"
    readme.write_text(f"# Title\n\n{begin}\n\nstale content\n\n{end}\n")
    with pytest.raises(SystemExit, match="stale"):
        sr.check(campaign_dir)


def test_check_fails_without_markers(sr, tmp_path):
    bench_dir = tmp_path / "bench"
    campaign_dir = bench_dir / "v1-test"
    raw = campaign_dir / "raw" / "run-001"
    raw.mkdir(parents=True)
    (raw / "result.json").write_text(json.dumps(_result()))
    (bench_dir / "README.md").write_text("# Title\n\nno markers here\n")
    with pytest.raises(SystemExit, match="markers"):
        sr.check(campaign_dir)


def test_check_is_scoped_to_its_own_campaign_label(sr, tmp_path):
    """A second campaign's section must not be disturbed by, or satisfy, --check for another."""
    bench_dir = tmp_path / "bench"
    campaign_dir = bench_dir / "v1-test"
    raw = campaign_dir / "raw" / "run-001"
    raw.mkdir(parents=True)
    (raw / "result.json").write_text(json.dumps(_result()))
    other_begin, other_end = sr.markers("v0-other")
    readme = bench_dir / "README.md"
    readme.write_text(f"# Title\n\n### v0-other\n\n{other_begin}\n\nunrelated\n\n{other_end}\n")
    with pytest.raises(SystemExit, match="markers"):
        sr.check(campaign_dir)


def test_update_readme_replaces_only_its_own_labelled_block(sr, tmp_path):
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()
    campaign_dir = bench_dir / "v1-test"
    begin, end = sr.markers("v1-test")
    other_begin, other_end = sr.markers("v0-other")
    readme = bench_dir / "README.md"
    readme.write_text(
        f"# Title\nbefore\n{other_begin}\nother campaign, untouched\n{other_end}\n{begin}\nold\n{end}\nafter\n"
    )
    sr.update_readme(campaign_dir, "new report")
    text = readme.read_text()
    assert "before" in text and "after" in text
    assert "new report" in text
    assert "old" not in text
    assert "other campaign, untouched" in text
