"""Unit tests for the pure parts of scripts/hardware_report.py (no model loading)."""

from __future__ import annotations

import importlib.util
import json
import os
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def hr():
    spec = importlib.util.spec_from_file_location("hardware_report", ROOT / "scripts" / "hardware_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------- slug and output dir


@pytest.mark.parametrize(
    "soc, macos, expected",
    [
        ("Apple M4 Max", "26.6.2", "apple-m4-max-macos26"),
        ("Apple M1", "15.5", "apple-m1-macos15"),
        ("Apple M2 Ultra", "14", "apple-m2-ultra-macos14"),
        ("", "", "unknown-soc-macosunknown"),
        (None, None, "unknown-soc-macosunknown"),
    ],
)
def test_slug(hr, soc, macos, expected):
    assert hr.slug(soc, macos) == expected


def test_output_dir_never_overwrites(tmp_path, hr):
    now = datetime(2026, 9, 23, 21, 5, 7)
    first = hr.output_dir(tmp_path, "apple-m4-max-macos26", now)
    assert first == tmp_path / "apple-m4-max-macos26"
    first.mkdir()
    second = hr.output_dir(tmp_path, "apple-m4-max-macos26", now)
    assert second == tmp_path / "apple-m4-max-macos26-20260923-210507"


# --------------------------------------------------------------------- sanitization


def test_sanitize_replaces_home_everywhere(hr):
    home = "/Users/someone"
    data = {
        "path": f"{home}/.cache/laya-apple",
        "nested": {"list": [f"error in {home}/x.py", 3, None], f"{home}/key": (f"{home}", 1.5)},
        "other": "/opt/data/cache",
    }
    out = hr.sanitize(data, home)
    assert out == {
        "path": "~/.cache/laya-apple",
        "nested": {"list": ["error in ~/x.py", 3, None], "~/key": ["~", 1.5]},
        "other": "/opt/data/cache",
    }
    assert home not in repr(out)


def test_sanitize_replaces_checkout_root_before_home(hr):
    home = "/Users/someone"
    root = f"{home}/src/laya-apple"
    out = hr.sanitize({"source": f"{root}/laya_apple", "cache": f"{home}/.cache"}, home, root)
    assert out == {"source": "./laya_apple", "cache": "~/.cache"}


def test_sanitize_defaults_to_real_home(hr):
    assert hr.sanitize(str(Path.home()) + "/a") == "~/a"


# --------------------------------------------------------------------- environment


def test_collect_environment_shape(monkeypatch, hr):
    answers = {
        ("sysctl", "-n", "machdep.cpu.brand_string"): "Apple M9 Test",
        ("sysctl", "-n", "hw.model"): "Mac99,1",
        ("sysctl", "-n", "hw.memsize"): str(64 * 2**30),
        ("sw_vers", "-productVersion"): "27.1",
        ("sw_vers", "-buildVersion"): "27A100",
    }

    def fake_check_output(cmd, **kwargs):
        cmd = tuple(cmd)
        if cmd[0] == "git":
            if "rev-parse" in cmd:
                return "0123456789abcdef0123456789abcdef01234567\n"
            return " M laya_apple/model.py\n"
        if cmd in answers:
            return answers[cmd] + "\n"
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(hr.subprocess, "check_output", fake_check_output)
    env = hr.collect_environment()
    assert env["soc"] == "Apple M9 Test"
    assert env["hw_model"] == "Mac99,1"
    assert env["memory_bytes"] == 64 * 2**30
    assert env["memory_gb"] == 64
    assert (env["macos"], env["macos_build"]) == ("27.1", "27A100")
    assert set(env["packages"]) == {"mlx", "coremltools", "numpy"}
    assert isinstance(env["python"], str)
    assert set(env["laya_apple"]) == {"version", "source", "git"}
    git = env["laya_apple"]["git"]
    if git is not None:  # only when laya_apple is imported from this checkout
        assert git == {"revision": "0123456789abcdef0123456789abcdef01234567", "dirty": True}
    assert hr.slug(env["soc"], env["macos"]) == "apple-m9-test-macos27"


def test_collect_environment_tolerates_missing_tools(monkeypatch, hr):
    def failing(cmd, **kwargs):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(hr.subprocess, "check_output", failing)
    env = hr.collect_environment()
    assert env["soc"] is None and env["memory_bytes"] is None and env["macos"] is None
    assert env["laya_apple"]["git"] is None


def test_git_info_outside_checkout(tmp_path, hr):
    assert hr.git_info(tmp_path) is None


# --------------------------------------------------------------------- matrix


def _model(mlx=True, ane=True, uses_ane=True, het=True):
    def section(v):
        if v is None:
            return {"status": "unavailable", "reason": "ArtifactMissingError"}
        return {"status": "ok", "passed": v}

    return {
        "mlx_parity": section(mlx),
        "ane_parity": section(ane),
        "routing": {"status": "ok", "uses_ane": uses_ane},
        "heterogeneous": section(het) if het is not None else {"status": "skipped", "reason": "no ANE"},
    }


def test_matrix_all_pass(hr):
    cells = hr.matrix_cells({"a": _model(), "b": _model()})
    assert cells == {"mlx": "✓", "ane": "✓", "auto_uses_ane": "yes", "heterogeneous": "✓"}


def test_matrix_mlx_only_machine(hr):
    cells = hr.matrix_cells({"a": _model(ane=None, uses_ane=False, het=None)})
    assert cells == {"mlx": "✓", "ane": "untested", "auto_uses_ane": "no", "heterogeneous": "untested"}


def test_matrix_any_failure_is_a_failure(hr):
    cells = hr.matrix_cells({"a": _model(), "b": _model(mlx=False, ane=False, het=False)})
    assert cells["mlx"] == "✗" and cells["ane"] == "✗" and cells["heterogeneous"] == "✗"


def test_matrix_errors_are_failures(hr):
    model = _model()
    model["mlx_parity"] = {"status": "error", "error": "RuntimeError: boom"}
    assert hr.matrix_cells({"a": model})["mlx"] == "✗"


def test_matrix_row_format(hr):
    env = {"soc": "Apple M4 Max", "memory_gb": 64, "macos": "26.6.2"}
    row = hr.matrix_row(env, {"a": _model(ane=None, uses_ane=False, het=None)})
    assert row == "| Apple M4 Max (64 GB, macOS 26.6.2) | ✓ | untested | no | untested |"


def test_sanitize_replaces_the_cache_directory(hr):
    out = hr.sanitize(
        {"p": "/data/cache/laya-apple/profiles/x.json"}, home="/Users/someone", cache="/data/cache/laya-apple"
    )
    assert out == {"p": "<cache>/profiles/x.json"}


# --------------------------------------------------------------------- JSON summary (--json)

FORWARD_SAMPLES = {
    ("gpu", 64, 1): [16.1, 15.9, 16.0, 16.4, 15.8],
    ("gpu", 128, 1): [28.0, 28.2, 28.1],
    ("gpu", 128, 4): [90.0, 91.0, 89.5, 90.5],
    ("ane", 64, 1): [7.5, 7.4, 7.6],
    ("ane", 128, 1): [9.3, 9.4, 9.2],
}


def _latency(device, length, questions):
    from laya_apple.benchmark import stats

    samples = FORWARD_SAMPLES[(device, length, questions)]
    return {
        "status": "ok",
        "device": device,
        "device_requested": device,
        "length": length,
        "questions": questions,
        "forward": stats(samples),
        "predict": stats([s + 0.5 for s in samples]),
        "samples": {"forward_ms": samples, "predict_ms": [s + 0.5 for s in samples]},
    }


def _bundle(**overrides):
    """The parts of a bundle.json that `build_summary` reads, from a machine with a working ANE."""
    from laya_apple.artifacts import platform_profile

    typed = {
        "model": "laya-typed-decisions",
        "revision": "f9ab0b228f0f",
        "mlx_parity": {"status": "ok", "passed": True},
        "ane_parity": {"status": "ok", "passed": True},
        "mlx_latency": [_latency("gpu", 64, 1), _latency("gpu", 128, 1), _latency("gpu", 128, 4)],
        "ane_latency": [_latency("ane", 64, 1), _latency("ane", 128, 1)],
    }
    bundle = {
        "format": "laya-apple-hardware-report",
        "format_version": 1,
        "started_at": "2026-10-01T12:00:00+00:00",
        "environment": {
            "soc": "ignored: the summary takes the platform profile",
            "hw_model": "Mac99,1",
            "memory_gb": 64,
            "laya_apple": {"version": "9.9.9", "source": ".", "git": None},
        },
        "platform": {"profile": platform_profile(), "shipped_profile_matches": False, "local_profile": None},
        "configuration": {"quick": True, "warmup": 3, "iters": 20},
        "models": {"laya-typed-decisions": typed},
    }
    return bundle | overrides


def test_summary_schema_is_stable(hr):
    """The exact output for a fixed bundle. A failure here means the schema changed: a consumer of
    summary.json would break, so change the expectation and bump SUMMARY_SCHEMA_VERSION
    (and docs/community-benchmarks.md) in the same change."""
    profile = {"soc": "Apple M9 Test", "macos": "27.1", "macos_build": "27A100", "coremltools": "9.0"}
    bundle = _bundle(platform={"profile": profile, "shipped_profile_matches": False, "local_profile": None})
    assert hr.build_summary(bundle) == {
        "format": "laya-apple-hardware-summary",
        "schema_version": 1,
        "laya_apple": "9.9.9",
        "started_at": "2026-10-01T12:00:00+00:00",
        "quick": True,
        "platform": profile,
        "hardware": {"hw_model": "Mac99,1", "memory_gb": 64},
        "models": {
            "laya-typed-decisions": {
                "revision": "f9ab0b228f0f",
                "parity": {"mlx": True, "ane": True},
                "forward_p50_ms": [
                    {"device": "gpu", "length": 64, "questions": 1, "p50_ms": 16.0},
                    {"device": "gpu", "length": 128, "questions": 1, "p50_ms": 28.1},
                    {"device": "gpu", "length": 128, "questions": 4, "p50_ms": 90.25},
                    {"device": "ane", "length": 64, "questions": 1, "p50_ms": 7.5},
                    {"device": "ane", "length": 128, "questions": 1, "p50_ms": 9.3},
                ],
            }
        },
    }
    assert hr.SUMMARY_SCHEMA_VERSION == 1


def test_summary_platform_identifiers_match_platform_profile(hr):
    from laya_apple.artifacts import platform_profile

    summary = hr.build_summary(_bundle())
    assert summary["platform"] == platform_profile()
    assert set(summary["platform"]) == {"soc", "macos", "macos_build", "coremltools"}


def test_summary_forward_p50s_are_the_raw_samples_median(hr):
    summary = hr.build_summary(_bundle())
    p50 = {
        (r["device"], r["length"], r["questions"]): r["p50_ms"]
        for r in summary["models"]["laya-typed-decisions"]["forward_p50_ms"]
    }
    assert p50.keys() == FORWARD_SAMPLES.keys()
    for key, samples in FORWARD_SAMPLES.items():
        assert p50[key] == pytest.approx(statistics.median(samples))


def test_summary_parity_is_true_false_or_null_per_model(hr):
    bundle = _bundle()
    models = bundle["models"]
    models["failed"] = {**models["laya-typed-decisions"], "mlx_parity": {"status": "ok", "passed": False}}
    models["errored"] = {**models["laya-typed-decisions"], "mlx_parity": {"status": "error", "error": "boom"}}
    models["mlx-only"] = {
        **models["laya-typed-decisions"],
        "ane_parity": {"status": "unavailable", "reason": "ArtifactMissingError", "build_command": "x"},
    }
    parity = {name: m["parity"] for name, m in hr.build_summary(bundle)["models"].items()}
    assert parity == {
        "laya-typed-decisions": {"mlx": True, "ane": True},
        "failed": {"mlx": False, "ane": True},
        "errored": {"mlx": False, "ane": True},
        "mlx-only": {"mlx": True, "ane": None},
    }


def test_summary_leaves_out_latency_that_was_not_measured(hr):
    bundle = _bundle()
    typed = bundle["models"]["laya-typed-decisions"]
    typed["mlx_latency"] = [
        _latency("gpu", 64, 1),
        {"status": "skipped: exceeds checkpoint max_len", "length": 512, "questions": 1, "device_requested": "gpu"},
        {"status": "error: UnsupportedShapeError: x", "length": 1024, "questions": 1, "device_requested": "gpu"},
    ]
    typed["ane_latency"] = {"status": "error", "error": "RuntimeError: boom"}  # the step failed as a whole
    bundle["models"]["no-weights"] = {
        "model": "no-weights",
        "revision": "abc",
        "weights": {"status": "error", "error": "OSError: offline"},
    }
    models = hr.build_summary(bundle)["models"]
    assert models["laya-typed-decisions"]["forward_p50_ms"] == [
        {"device": "gpu", "length": 64, "questions": 1, "p50_ms": 16.0}
    ]
    assert models["no-weights"] == {"revision": "abc", "parity": {"mlx": None, "ane": None}, "forward_p50_ms": []}


def test_summary_is_plain_json_and_does_not_mutate_the_bundle(hr):
    bundle = _bundle()
    before = json.dumps(bundle, sort_keys=True)
    summary = hr.build_summary(bundle)
    assert json.loads(json.dumps(summary)) == summary
    assert json.dumps(bundle, sort_keys=True) == before
    summary["platform"]["soc"] = "changed"
    assert bundle["platform"]["profile"]["soc"] != "changed"


def test_write_summary_writes_the_summary_as_json(hr, tmp_path):
    out = tmp_path / "summary.json"
    hr.write_summary(_bundle(), out)
    assert out.read_text().endswith("}\n")
    assert json.loads(out.read_text()) == hr.build_summary(_bundle())


COMMITTED_BUNDLES = sorted((ROOT / "hardware-results").glob("*/bundle.json"))


def test_there_are_committed_bundles_to_summarise():
    assert COMMITTED_BUNDLES  # otherwise the parametrised test below would silently run nothing


@pytest.mark.parametrize("path", COMMITTED_BUNDLES, ids=lambda p: p.parent.name)
def test_summary_of_every_committed_bundle(hr, path):
    bundle = json.loads(path.read_text())
    summary = hr.build_summary(bundle)
    assert set(summary) == {
        "format",
        "schema_version",
        "laya_apple",
        "started_at",
        "quick",
        "platform",
        "hardware",
        "models",
    }
    assert summary["platform"] == bundle["platform"]["profile"]
    assert set(summary["models"]) == set(bundle["models"])
    for name, model in summary["models"].items():
        assert set(model) == {"revision", "parity", "forward_p50_ms"}
        assert model["forward_p50_ms"], f"{name} has measured latency in the bundle"
        raw = {
            (r["device"], r["length"], r["questions"]): r["forward"]["p50_ms"]
            for key in ("mlx_latency", "ane_latency")
            for r in bundle["models"][name].get(key, [])
            if r.get("status") == "ok"
        }
        assert {(r["device"], r["length"], r["questions"]): r["p50_ms"] for r in model["forward_p50_ms"]} == raw


def test_the_documented_example_is_the_summary_of_the_bundle_it_names(hr):
    doc = (ROOT / "docs" / "community-benchmarks.md").read_text()
    example = json.loads(doc.split("## Summary JSON")[1].split("```json\n")[1].split("```")[0])
    path = ROOT / "hardware-results" / "apple-m2-pro-macos26" / "bundle.json"
    assert example == hr.build_summary(json.loads(path.read_text()))


def _refuse_to_measure(hr, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("this must not measure anything")

    monkeypatch.setattr(hr, "collect_environment", refuse)
    monkeypatch.setattr(hr, "measure_model", refuse)


def test_render_writes_the_json_summary_without_measuring(hr, tmp_path, capsys, monkeypatch):
    _refuse_to_measure(hr, monkeypatch)
    bundle_path = COMMITTED_BUNDLES[0]
    out = tmp_path / "summary.json"
    assert hr.main(["--render", str(bundle_path), "--json", str(out)]) == 0
    assert capsys.readouterr().out.startswith("# Hardware report:")
    assert json.loads(out.read_text()) == hr.build_summary(json.loads(bundle_path.read_text()))


def _rejected(hr, monkeypatch, capsys, *argv):
    """main() exits 2 with an `--json` error, and nothing was measured (the stubs would raise)."""
    _refuse_to_measure(hr, monkeypatch)
    with pytest.raises(SystemExit) as e:
        hr.main(list(argv))
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "--json" in err
    return err


@pytest.mark.parametrize(
    ("where", "reason"),
    [("missing-dir/summary.json", "directory does not exist"), (".", "is a directory")],
)
def test_a_bad_json_path_fails_before_anything_is_measured(hr, tmp_path, capsys, monkeypatch, where, reason):
    assert reason in _rejected(hr, monkeypatch, capsys, "--json", str(tmp_path / where))


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write anywhere")
def test_an_unwritable_json_directory_fails_before_anything_is_measured(hr, tmp_path, capsys, monkeypatch):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        err = _rejected(hr, monkeypatch, capsys, "--json", str(locked / "summary.json"))
    finally:
        locked.chmod(0o700)
    assert "not writable" in err


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write anywhere")
def test_an_unwritable_existing_json_file_fails_before_anything_is_measured(hr, tmp_path, capsys, monkeypatch):
    out = tmp_path / "summary.json"
    out.write_text("{}")
    out.chmod(0o400)
    assert "not writable" in _rejected(hr, monkeypatch, capsys, "--json", str(out))


def test_render_refuses_to_overwrite_its_own_bundle(hr, tmp_path, capsys, monkeypatch):
    bundle = tmp_path / "bundle.json"
    bundle.write_text(COMMITTED_BUNDLES[0].read_text())
    link = tmp_path / "link.json"
    link.symlink_to(bundle)
    for same in (bundle, tmp_path / "." / "bundle.json", link):
        err = _rejected(hr, monkeypatch, capsys, "--render", str(bundle), "--json", str(same))
        assert "--render bundle" in err
    assert bundle.read_text() == COMMITTED_BUNDLES[0].read_text()  # untouched
    # a different file next to it is fine
    assert hr.main(["--render", str(bundle), "--json", str(tmp_path / "summary.json")]) == 0


def test_the_live_run_writes_the_json_summary_of_the_bundle_it_writes(hr, tmp_path, capsys, monkeypatch):
    """The measuring branch of main(), with every measurement stubbed out."""
    from laya_apple.artifacts import platform_profile

    monkeypatch.setenv("LAYA_APPLE_CACHE", str(tmp_path / "cache"))
    env = {
        "soc": "Apple M9 Test",
        "hw_model": "Mac99,1",
        "memory_bytes": 64 * 2**30,
        "memory_gb": 64,
        "macos": "27.1",
        "macos_build": "27A100",
        "python": "3.12.0",
        "packages": {"mlx": "0.1", "coremltools": None, "numpy": "2.0"},
        "laya_apple": {"version": "9.9.9", "source": "./laya_apple", "git": None},
    }
    measured = []

    def fake_model(name, *, quick, local_files_only, log):
        measured.append(name)
        return {
            "model": name,
            "repo": "org/repo",
            "revision": "rev1",
            "weights_sha256_pinned": "0" * 64,
            "mlx_parity": {"status": "ok", "passed": True},
            "mlx_latency": [_latency("gpu", 64, 1), _latency("gpu", 128, 4)],
            "ane_parity": {"status": "unavailable", "reason": "no coremltools", "build_command": "build it"},
            "routing": {"status": "skipped", "reason": "stubbed"},
            "heterogeneous": {"status": "skipped", "reason": "stubbed"},
        }

    monkeypatch.setattr(hr, "collect_environment", lambda: env)
    monkeypatch.setattr(hr, "measure_model", fake_model)
    results, summary_path = tmp_path / "results", tmp_path / "summary.json"

    assert hr.main(["--models", "m1", "--out-root", str(results), "--json", str(summary_path)]) == 0

    assert measured == ["m1"]
    (bundle_dir,) = results.iterdir()
    assert bundle_dir.name == "apple-m9-test-macos27"
    assert sorted(p.name for p in bundle_dir.iterdir()) == ["bundle.json", "summary.md"]  # the summary is elsewhere
    bundle = json.loads((bundle_dir / "bundle.json").read_text())
    summary = json.loads(summary_path.read_text())
    assert summary == hr.build_summary(bundle)
    assert summary["platform"] == platform_profile()
    assert summary["models"]["m1"]["parity"] == {"mlx": True, "ane": None}
    assert [(r["device"], r["length"], r["questions"]) for r in summary["models"]["m1"]["forward_p50_ms"]] == [
        ("gpu", 64, 1),
        ("gpu", 128, 4),
    ]
    assert f"wrote {summary_path} (schema_version 1)" in capsys.readouterr().out
