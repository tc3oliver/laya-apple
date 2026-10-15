"""Opt-in W8 (`ane_quantization="w8"`): naming, validation, fail-closed loading, prune safety.

Everything here is metadata and control flow; the artifact itself is covered by the `ane` tests.
"""

from __future__ import annotations

import json
import os

import pytest

from laya_apple import Laya, lifecycle
from laya_apple.artifacts import (
    COMPILED,
    artifact_dir,
    artifacts_root,
    check_quantization_supported,
    normalize_quantization,
    quantized_buckets,
    tree_sha256,
    verify_manifest,
)
from laya_apple.backends import coreml_ane
from laya_apple.errors import (
    ArtifactMissingError,
    ArtifactRevisionError,
    BackendUnavailableError,
    UnsupportedShapeError,
)
from laya_apple.registry import ANE_QUANTIZATIONS, models

SPEC = models()["laya-typed-decisions"]
M4_MAX_27 = {"soc": "Apple M4 Max", "macos": "27.0", "macos_build": "26A428", "coremltools": "9.0"}
W8 = {"scheme": "w8-pt", **ANE_QUANTIZATIONS["w8-pt"]}


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("LAYA_APPLE_CACHE", str(tmp_path))
    return tmp_path


def test_names_and_paths():
    assert normalize_quantization(None) is None
    assert normalize_quantization("w8") == normalize_quantization("w8-pt") == "w8-pt"
    with pytest.raises(ValueError, match="ane_quantization"):
        normalize_quantization("w4")
    assert artifact_dir(SPEC, 64).name == "bc1s-masked-L64-B1"  # FP16 path is unchanged
    assert artifact_dir(SPEC, 64, "w8").name == "bc1s-masked-w8-pt-L64-B1"
    assert quantized_buckets(SPEC, "w8") == (64,)
    assert quantized_buckets(SPEC, None) == ()
    assert quantized_buckets(models()["laya"], "w8") == ()


def test_supported_only_on_the_validated_model_and_machine():
    check_quantization_supported(SPEC, None, {"soc": "anything"})  # FP16 is never restricted
    check_quantization_supported(SPEC, "w8", M4_MAX_27)
    with pytest.raises(UnsupportedShapeError, match="not validated"):
        check_quantization_supported(models()["laya"], "w8", M4_MAX_27)
    for profile in (
        {**M4_MAX_27, "soc": "Apple M3 Max"},
        {**M4_MAX_27, "macos": "26.6.2"},
        {**M4_MAX_27, "macos": "28.0"},
    ):
        with pytest.raises(BackendUnavailableError, match="validated only on"):
            check_quantization_supported(SPEC, "w8", profile)


def test_manifest_quantization_is_matched_exactly(manifest_factory):
    verify_manifest(SPEC, 64, manifest_factory(SPEC, 64, **{"artifact.quantization": W8}), quantization="w8")
    with pytest.raises(ArtifactRevisionError, match="FP16 artifact was requested"):
        verify_manifest(SPEC, 64, manifest_factory(SPEC, 64, **{"artifact.quantization": W8}))
    with pytest.raises(ArtifactRevisionError, match="does not match the validated"):
        verify_manifest(SPEC, 64, manifest_factory(SPEC, 64), quantization="w8")
    for field, value in (("nbits", 4), ("granularity", "per_grouped_channel"), ("scheme", "w8-gc32")):
        bad = {**W8, field: value}
        with pytest.raises(ArtifactRevisionError, match="does not match the validated"):
            verify_manifest(SPEC, 64, manifest_factory(SPEC, 64, **{"artifact.quantization": bad}), quantization="w8")
    with pytest.raises(ArtifactRevisionError, match="not validated"):
        verify_manifest(
            SPEC, 96, manifest_factory(SPEC, 96, **{"artifact.quantization": W8}), quantization="w8"
        )  # a quantized artifact only exists for the validated buckets


def _artifact(spec, bucket, manifest_factory, where, **overrides):
    (where / COMPILED).mkdir(parents=True)
    (where / COMPILED / "weights.bin").write_bytes(os.urandom(64))
    data = manifest_factory(spec, bucket, **{"platform": lifecycle.platform_profile(), **overrides})
    data["integrity"]["artifact_sha256"] = tree_sha256(where / COMPILED)
    (where / "manifest.json").write_text(json.dumps(data))
    return where


def test_prune_keeps_a_quantized_artifact_at_its_own_path_only(cache, manifest_factory):
    quant = {"artifact.quantization": W8}
    good = _artifact(SPEC, 64, manifest_factory, artifact_dir(SPEC, 64, "w8"), **quant)
    fp16 = _artifact(SPEC, 64, manifest_factory, artifact_dir(SPEC, 64))
    in_fp16_slot = _artifact(
        SPEC, 96, manifest_factory, artifact_dir(SPEC, 96), **{"artifact.quantization": W8}
    )  # W8 manifest at the FP16 path, and a bucket W8 was not validated for
    unknown = _artifact(
        SPEC,
        128,
        manifest_factory,
        artifacts_root() / SPEC.name / SPEC.revision[:12] / "bc1s-masked-w4-L128-B1",
        **{"artifact.quantization": {"scheme": "w4"}},
    )
    listed = {item["path"]: item["reason"] for item in lifecycle.plan_prune()}
    assert str(good) not in listed and str(fp16) not in listed
    assert "quantization" in listed[str(in_fp16_slot)]
    assert "quantization" in listed[str(unknown)]


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        (dict(device="gpu"), ValueError, "device='gpu'"),
        (dict(device="auto", ane_startup="background", execution="workers"), ValueError, 'ane_startup="wait"'),
    ],
)
def test_an_unusable_request_raises_before_anything_loads(kwargs, error, match):
    with pytest.raises(error, match=match):
        Laya.from_pretrained("laya-typed-decisions", ane_quantization="w8", local_files_only=True, **kwargs)


def test_an_unsupported_model_raises_and_language_routing_does_not_take_it():
    with pytest.raises(UnsupportedShapeError, match="not validated"):
        Laya.from_pretrained("laya", ane_quantization="w8", local_files_only=True)
    with pytest.raises(UnsupportedShapeError, match="not validated"):
        Laya.from_pretrained("auto", ane_quantization="w8", local_files_only=True)


class _Manifest:
    artifact_sha256 = "f" * 64


def _backend(monkeypatch, loader, buckets, *, strict, quantization="w8"):
    monkeypatch.setattr(coreml_ane, "HostWeights", lambda *a, **k: object())
    monkeypatch.setattr(coreml_ane, "check_quantization_supported", lambda *a, **k: None)
    monkeypatch.setattr(coreml_ane, "load_verified", loader)
    monkeypatch.setattr(coreml_ane, "probe_placement", lambda *a, **k: {"ratio": 0.3})
    monkeypatch.setattr(coreml_ane.ANEBackend, "_probe_features", lambda self, b: {})
    return coreml_ane.ANEBackend(SPEC, None, 0, 128, buckets, strict=strict, quantization=quantization)


@pytest.mark.parametrize("strict", [True, False])
def test_a_missing_quantized_artifact_raises_instead_of_degrading(monkeypatch, strict):
    seen = []

    def loader(spec, bucket, *, quantization=None):
        seen.append((bucket, quantization))
        if quantization:
            raise ArtifactMissingError("no W8 artifact")
        return object(), _Manifest()

    with pytest.raises(ArtifactMissingError, match="no W8"):
        _backend(monkeypatch, loader, (64, 96), strict=strict)
    assert (64, "w8-pt") in seen


def test_only_the_validated_bucket_is_quantized_and_the_others_stay_fp16(monkeypatch):
    seen = []

    def loader(spec, bucket, *, quantization=None):
        seen.append((bucket, quantization))
        return object(), _Manifest()

    backend = _backend(monkeypatch, loader, (64, 96, 128), strict=False)
    assert sorted(seen) == [(64, "w8-pt"), (96, None), (128, None)]
    assert backend.quantized == (64,) and backend.buckets == (64, 96, 128)


def test_a_quantization_request_with_no_quantized_bucket_offered_raises(monkeypatch):
    with pytest.raises(UnsupportedShapeError, match="none of which is offered"):
        _backend(monkeypatch, lambda *a, **k: (object(), _Manifest()), (96, 128), strict=False)


def test_fp16_load_is_unchanged_by_the_new_argument(monkeypatch):
    seen = []

    def loader(spec, bucket, *, quantization=None):
        seen.append((bucket, quantization))
        return object(), _Manifest()

    _backend(monkeypatch, loader, (64, 96), strict=False, quantization=None)
    assert seen == [(64, None), (96, None)]


def test_table_marks_quantized_rows_and_leaves_fp16_rows_alone():
    from laya_apple.cli import _artifact_table

    base = {"model": "m", "status": "validated", "parity": {"passed": True}, "compute_target": None}
    table = _artifact_table(
        [
            {**base, "bucket": 64, "quantization": None},
            {**base, "bucket": 64, "quantization": {"scheme": "w8-pt"}},
        ]
    )
    rows = [line.split() for line in table.splitlines()[1:]]
    assert rows[0][1] == "64" and rows[1][1:3] == ["64", "w8-pt"]


@pytest.mark.parametrize("model, args", [("laya", ["verify"]), ("laya-typed-decisions", ["list"]), (None, ["build"])])
def test_cli_quantize_refuses_what_it_does_not_cover(model, args):
    from laya_apple.cli import main

    argv = ["artifacts", *args, *([model] if model else []), "--quantize", "w8"]
    with pytest.raises(SystemExit):
        main(argv)
