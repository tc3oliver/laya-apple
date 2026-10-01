"""cli.py: prune's reported count, `artifacts import` argument validation, that main()
does not swallow unrelated ValueErrors as clean CLI errors (code review finding 11), and
`artifacts list --capabilities` provenance output."""

from __future__ import annotations

import json

import pytest

from laya_apple import cli
from laya_apple.artifacts import COMPILED, artifact_dir, tree_sha256
from laya_apple.registry import models


def test_cmd_artifacts_prune_prints_the_actual_removed_count(monkeypatch, capsys):
    plan = [{"path": "/x/a", "reason": "r", "bytes": 10}, {"path": "/x/b", "reason": "r", "bytes": 20}]
    monkeypatch.setattr("laya_apple.lifecycle.plan_prune", lambda: plan)
    monkeypatch.setattr("laya_apple.lifecycle.stranded_artifacts", lambda: [])
    # prune() removes fewer than planned (e.g. another process raced it); the printed count
    # must reflect what prune() actually reports, not len(plan).
    monkeypatch.setattr("laya_apple.lifecycle.prune", lambda p: p[:1])
    a = cli.build_parser().parse_args(["artifacts", "prune", "--yes"])
    assert cli.cmd_artifacts(a) == 0
    out = capsys.readouterr().out
    assert "removed 1 entries" in out


def test_cmd_artifacts_prune_reports_a_stranded_previous_artifact(monkeypatch, capsys):
    report = {"path": "/x/a.old-1", "registered": "/x/a", "message": "previous artifact left after a failed restore"}
    monkeypatch.setattr("laya_apple.lifecycle.stranded_artifacts", lambda: [report])
    monkeypatch.setattr("laya_apple.lifecycle.plan_prune", lambda: [])
    a = cli.build_parser().parse_args(["artifacts", "prune", "--yes"])
    assert cli.cmd_artifacts(a) == 0
    out = capsys.readouterr().out
    assert "keep  previous artifact left after a failed restore" in out and "nothing to prune" in out


def test_cmd_artifacts_import_without_path_raises_clear_system_exit():
    a = cli.build_parser().parse_args(["artifacts", "import"])
    with pytest.raises(SystemExit, match="archive path"):
        cli.cmd_artifacts(a)


def test_main_wraps_argument_value_error_from_from_pretrained(monkeypatch):
    def boom(*a, **k):
        raise ValueError("bad dtype")

    monkeypatch.setattr("laya_apple.Laya.from_pretrained", boom)
    rc = cli.main(["predict", "laya", "--context", "hi", "--questions", "{}"])
    assert rc == 2


def test_main_does_not_catch_unrelated_value_errors(monkeypatch):
    """Only ValueErrors from the from_pretrained call sites are converted to clean CLI
    errors; a bug elsewhere must keep its traceback (main only catches LayaAppleError)."""

    def boom(a):
        raise ValueError("some internal bug, not user input")

    monkeypatch.setattr(cli, "cmd_info", boom)
    with pytest.raises(ValueError, match="internal bug"):
        cli.main(["info"])


def _build_synthetic_artifact(spec, bucket, manifest_factory, tmp_path, monkeypatch, **overrides):
    monkeypatch.setenv("LAYA_APPLE_CACHE", str(tmp_path))
    d = artifact_dir(spec, bucket)
    (d / COMPILED).mkdir(parents=True)
    (d / COMPILED / "weights.bin").write_bytes(b"synthetic")
    data = manifest_factory(spec, bucket, **overrides)
    data["integrity"]["artifact_sha256"] = tree_sha256(d / COMPILED)
    (d / "manifest.json").write_text(json.dumps(data))
    return d


def test_artifacts_list_capabilities_reports_provenance_fields(monkeypatch, capsys, tmp_path, manifest_factory):
    spec = models()["laya-typed-decisions"]
    bucket = spec.ane_buckets[0]
    _build_synthetic_artifact(
        spec,
        bucket,
        manifest_factory,
        tmp_path,
        monkeypatch,
        **{
            "conversion": {
                "laya_apple_version": "9.9.9",
                "git_revision": "deadbeef",
                "code_sha256": "c" * 64,
                "packages": {"coremltools": "9.0"},
            },
            "placement": {"compute_units": "CPU_AND_NE", "ops": {"ane": 5}, "transitions": 0},
            "parity": {
                "passed": True,
                "tolerance": 0.01,
                "prob_max_abs": 0.002,
                "hard_mismatches": 0,
                "near_tie_flips": [{"case": 1}, {"case": 2}],
            },
        },
    )
    a = cli.build_parser().parse_args(["artifacts", "list", "--capabilities"])
    assert cli.cmd_artifacts(a) == 0
    (record,) = json.loads(capsys.readouterr().out)

    assert record["model"] == spec.name
    assert record["repo"] == spec.repo
    assert record["revision"] == spec.revision
    assert record["source_weights_sha256"] == spec.weights_sha256
    assert record["conversion_revision"] == {
        "laya_apple_version": "9.9.9",
        "git_revision": "deadbeef",
        "code_sha256": "c" * 64,
    }
    assert record["graph"] and record["bucket"] == bucket and record["batch"] == 1
    assert record["compute_target"] == {"compute_units": "CPU_AND_NE", "ops": {"ane": 5}, "transitions": 0}
    assert record["precision"]
    assert record["platform"]["soc"] == "test"
    assert record["parity"] == {
        "passed": True,
        "tolerance": 0.01,
        "prob_max_abs": 0.002,
        "hard_mismatches": 0,
        "near_tie_flips": 2,
    }
    assert record["artifact_sha256"]
    assert record["offered_by_auto"] == (bucket in spec.auto_ane_buckets)
    assert record["offered_explicit"] is True


def test_artifacts_list_capabilities_missing_fields_are_null_not_crash(monkeypatch, capsys, tmp_path, manifest_factory):
    spec = models()["laya-typed-decisions"]
    bucket = spec.ane_buckets[0]
    _build_synthetic_artifact(
        spec,
        bucket,
        manifest_factory,
        tmp_path,
        monkeypatch,
        **{"conversion": {}, "parity": {"passed": True}},
    )
    a = cli.build_parser().parse_args(["artifacts", "list", "--capabilities"])
    assert cli.cmd_artifacts(a) == 0
    (record,) = json.loads(capsys.readouterr().out)

    assert record["conversion_revision"] == {"laya_apple_version": None, "git_revision": None, "code_sha256": None}
    assert record["parity"]["tolerance"] is None
    assert record["parity"]["near_tie_flips"] is None
    assert record["compute_target"]["ops"] is None


def test_artifacts_list_capabilities_default_output_unchanged(monkeypatch, capsys, tmp_path, manifest_factory):
    """Backward compatible: plain `artifacts list` keeps its existing keys."""
    spec = models()["laya-typed-decisions"]
    bucket = spec.ane_buckets[0]
    _build_synthetic_artifact(spec, bucket, manifest_factory, tmp_path, monkeypatch)
    a = cli.build_parser().parse_args(["artifacts", "list"])
    assert cli.cmd_artifacts(a) == 0
    (record,) = json.loads(capsys.readouterr().out)
    assert set(record) == {"path", "status", "source", "artifact", "parity_passed", "sha256"}
