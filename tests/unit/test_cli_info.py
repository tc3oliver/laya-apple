"""`laya-apple info`: the JSON keys that predate the `environment` block are unchanged (the
output is under SemVer, docs/api.md), and `environment` reports Python, MLX and offline mode."""

from __future__ import annotations

import json
import platform
import sys

import pytest
from huggingface_hub import constants

from laya_apple import __version__, cli
from laya_apple.artifacts import platform_profile
from laya_apple.registry import models

TOP_LEVEL_KEYS = {"laya_apple", "platform", "platform_validated_for_auto_ane", "coremltools_available", "models"}
PLATFORM_KEYS = {"soc", "macos", "macos_build", "coremltools"}
MODEL_KEYS = {"repo", "revision", "encoder", "max_len", "mlx_dtypes", "ane_buckets", "auto_ane_buckets", "artifacts"}


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("LAYA_APPLE_CACHE", str(tmp_path))


def run_info(capsys, *argv):
    assert cli.main([*argv, "info"]) == 0
    return json.loads(capsys.readouterr().out)


def test_info_keeps_every_existing_key(capsys):
    out = run_info(capsys)
    assert TOP_LEVEL_KEYS <= set(out)
    assert out["laya_apple"] == __version__
    assert out["platform"] == platform_profile()
    assert set(out["platform"]) == PLATFORM_KEYS
    assert isinstance(out["platform_validated_for_auto_ane"], bool)
    assert isinstance(out["coremltools_available"], bool)
    assert set(out["models"]) == set(models())
    for name, entry in out["models"].items():
        spec = models()[name]
        assert set(entry) == MODEL_KEYS
        assert entry == {
            "repo": spec.repo,
            "revision": spec.revision,
            "encoder": spec.encoder,
            "max_len": spec.max_len,
            "mlx_dtypes": list(spec.mlx_dtypes),
            "ane_buckets": list(spec.ane_buckets),
            "auto_ane_buckets": list(spec.auto_ane_buckets),
            "artifacts": {str(b): "missing" for b in spec.ane_buckets},  # nothing built in the empty cache
        }


def test_info_adds_only_the_environment_block(capsys):
    out = run_info(capsys)
    assert set(out) - TOP_LEVEL_KEYS == {"environment"}
    env = out["environment"]
    assert set(env) == {"python", "mlx", "offline"}
    assert set(env["mlx"]) == {"available", "version"}
    assert set(env["offline"]) == {"local_files_only", "hf_hub_offline", "effective"}


def test_info_for_one_model_keeps_the_shape(capsys):
    assert cli.main(["info", "laya-typed-decisions"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert set(out) == TOP_LEVEL_KEYS | {"environment"}
    assert set(out["models"]) == {"laya-typed-decisions"}


def test_environment_reports_the_python_version(capsys):
    assert run_info(capsys)["environment"]["python"] == platform.python_version()


def test_environment_reports_an_importable_mlx(capsys):
    mx = pytest.importorskip("mlx.core")
    assert run_info(capsys)["environment"]["mlx"] == {"available": True, "version": mx.__version__}


def test_environment_reports_a_missing_mlx_without_failing(capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx.core", None)  # `import mlx.core` raises ImportError
    out = run_info(capsys)
    assert out["environment"]["mlx"] == {"available": False, "version": None}
    assert TOP_LEVEL_KEYS <= set(out)


class _FailingMlxFinder:
    """Meta-path finder that makes `import mlx.core` raise something other than ImportError."""

    @staticmethod
    def find_spec(name, path=None, target=None):
        if name == "mlx.core":
            raise RuntimeError("mlx.core failed to initialise")
        return None


def test_environment_reports_an_mlx_that_fails_to_import_with_another_error(capsys, monkeypatch):
    pytest.importorskip("mlx.core")  # installed, so the finder below is what fails the import
    monkeypatch.delitem(sys.modules, "mlx.core")
    monkeypatch.setattr(sys, "meta_path", [_FailingMlxFinder, *sys.meta_path])
    out = run_info(capsys)  # run_info asserts the exit code is 0
    assert out["environment"]["mlx"] == {"available": False, "version": None}
    assert TOP_LEVEL_KEYS <= set(out)


@pytest.mark.parametrize(
    ("hf_hub_offline", "argv", "local_files_only", "effective"),
    [
        (False, (), False, False),
        (True, (), False, True),
        (False, ("--offline",), True, True),
        (True, ("--offline",), True, True),
    ],
)
def test_environment_offline_follows_the_flag_and_the_hub_offline_mode(
    capsys, monkeypatch, hf_hub_offline, argv, local_files_only, effective
):
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", hf_hub_offline)
    assert run_info(capsys, *argv)["environment"]["offline"] == {
        "local_files_only": local_files_only,
        "hf_hub_offline": hf_hub_offline,
        "effective": effective,
    }
