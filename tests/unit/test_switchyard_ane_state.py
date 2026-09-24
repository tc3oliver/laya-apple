"""The ANE state decision table (laya_apple/demos/switchyard/ane_state.py), probes mocked."""

from __future__ import annotations

import pytest

from laya_apple.demos.switchyard import ane_state
from laya_apple.errors import (
    ArtifactIntegrityError,
    ArtifactMissingError,
    ArtifactParityError,
    ArtifactRevisionError,
    BackendUnavailableError,
    ComputeUnitMismatchError,
)
from laya_apple.registry import resolve

SPEC = resolve("laya-typed-decisions")


class Manifest:
    def __init__(self, bucket):
        self.artifact_sha256 = f"{bucket:064d}"


def loader(errors=None):
    errors = errors or {}

    def load(spec, bucket):
        if bucket in errors:
            raise errors[bucket]
        return object(), Manifest(bucket)

    return load


def detect(*, apple=True, ct=True, validated=True, calibrated=False, errors=None):
    return ane_state.detect(
        SPEC,
        apple_silicon=lambda: apple,
        coremltools=lambda: ct,
        validated=lambda: validated,
        calibrated=lambda model: calibrated,
        load=loader(errors),
    )


def test_ready_with_validated_profile_and_verified_artifacts():
    s = detect()
    assert s.state == "ready" and s.reason is None and s.setup_command is None
    assert s.artifacts == {str(b): f"{b:064d}" for b in SPEC.auto_ane_buckets}


def test_ready_with_a_local_calibration(monkeypatch):
    from laya_apple import profiles

    monkeypatch.setattr(profiles, "load_local", lambda model: {"auto_ane_buckets": [64, 96]})
    s = detect(validated=False, calibrated=True)
    assert s.state == "ready" and set(s.artifacts) == {"64", "96"}


@pytest.mark.parametrize(
    ("kwargs", "state", "reason"),
    [
        ({"apple": False}, "unavailable", ane_state.NOT_APPLE_SILICON),
        ({"ct": False}, "setup_available", ane_state.NO_COREMLTOOLS),
        ({"errors": {64: ArtifactMissingError("x")}}, "setup_available", ane_state.NO_ARTIFACTS),
        ({"errors": {96: ArtifactRevisionError("x")}}, "setup_available", ane_state.REBUILD_ARTIFACTS),
        ({"errors": {96: ArtifactIntegrityError("x")}}, "setup_available", ane_state.REBUILD_ARTIFACTS),
        ({"errors": {64: BackendUnavailableError("x")}}, "setup_available", ane_state.NO_COREMLTOOLS),
        ({"errors": {128: ArtifactParityError("x")}}, "unavailable", ane_state.PARITY_FAILED),
        ({"errors": {128: ComputeUnitMismatchError("x")}}, "unavailable", ane_state.NOT_ON_ANE),
        # an unusable ANE outranks a fixable one, whatever the bucket order
        (
            {"errors": {64: ArtifactMissingError("x"), 128: ArtifactParityError("x")}},
            "unavailable",
            ane_state.PARITY_FAILED,
        ),
    ],
)
def test_decision_table(kwargs, state, reason):
    s = detect(**kwargs)
    assert (s.state, s.reason) == (state, reason)
    assert s.setup_command == (ane_state.SETUP_COMMAND if state == "setup_available" else None)
    assert s.reason_text and s.reason_text[0].isupper()
    assert s.detail is None or s.detail.startswith("L")


def test_unknown_load_error_is_unavailable_with_its_message():
    s = detect(errors={64: RuntimeError("boom")})
    assert (s.state, s.reason) == ("unavailable", ane_state.NOT_ON_ANE) and "RuntimeError: boom" in s.detail


def test_not_calibrated(monkeypatch):
    from laya_apple import profiles

    monkeypatch.setattr(profiles, "load_local", lambda model: None)
    s = detect(validated=False, calibrated=False)
    assert (s.state, s.reason) == ("setup_available", ane_state.NOT_CALIBRATED)


def test_warmup_confirms_or_demotes_ready():
    ready = detect()
    ok = ane_state.after_warmup(ready, 12)
    assert ok.state == "ready" and ok.warmup_ane_requests == 12
    none = ane_state.after_warmup(ready, 0)
    assert (none.state, none.reason, none.warmup_ane_requests) == ("setup_available", ane_state.NO_ANE_WARMUP, 0)
    assert none.setup_command == ane_state.SETUP_COMMAND
    missing = detect(ct=False)
    assert ane_state.after_warmup(missing, 0) is missing


def test_setup_command_is_exact():
    assert ane_state.SETUP_COMMAND == 'uvx --from "laya-apple[convert]" laya-apple switchyard --setup-ane'
    d = detect(ct=False).to_dict()
    assert d == {
        "state": "setup_available",
        "reason": "ane_runtime_unavailable",
        "reason_text": "The Neural Engine runtime (Core ML Tools) is not installed",
        "detail": None,
        "setup_command": ane_state.SETUP_COMMAND,
        "warmup_ane_requests": None,
    }


def test_setup_builds_only_what_does_not_verify_then_calibrates(monkeypatch, tmp_path):
    from laya_apple import artifacts, model, profiles
    from laya_apple.conversion import build as build_mod

    built, calibrated, steps = [], [], []

    def load_verified(spec, b):
        if b == 96:
            raise ArtifactMissingError("x")
        if b == 128:
            raise ArtifactRevisionError("x")
        return object(), Manifest(b)

    monkeypatch.setattr(artifacts, "load_verified", load_verified)
    monkeypatch.setattr(artifacts, "artifact_dir", lambda spec, b: tmp_path / ("exists" if b == 128 else "missing"))
    (tmp_path / "exists").mkdir()
    monkeypatch.setattr(build_mod, "build", lambda spec, b, **kw: built.append((b, kw)))
    monkeypatch.setattr(model, "platform_validated", lambda: False)
    monkeypatch.setattr(profiles, "calibrate", lambda spec, **kw: calibrated.append(spec.name))
    assert (
        ane_state.setup(SPEC, local_files_only=True, step=lambda kind, bucket=None: steps.append((kind, bucket)))
        is None
    )
    assert built == [(96, {"local_files_only": True, "force": False}), (128, {"local_files_only": True, "force": True})]
    assert SPEC.ane_buckets == (64, 96, 128)  # every offered bucket, which calibrate needs
    assert calibrated == [SPEC.name]
    assert steps == [("verified", 64), ("building", 96), ("building", 128), ("calibrating", None)]


def _setup_with(monkeypatch, build_error=None, calibrate_error=None):
    from laya_apple import artifacts, model, profiles
    from laya_apple.conversion import build as build_mod

    def fail(e):
        def f(*a, **kw):
            if e is not None:
                raise e

        return f

    monkeypatch.setattr(artifacts, "load_verified", fail(ArtifactMissingError("x")))
    monkeypatch.setattr(build_mod, "build", fail(build_error))
    monkeypatch.setattr(model, "platform_validated", lambda: False)
    monkeypatch.setattr(profiles, "calibrate", fail(calibrate_error))
    return ane_state.setup(SPEC)


def test_setup_classifies_build_and_calibration_failures(monkeypatch):
    from laya_apple.errors import ArtifactError

    missing = ArtifactError("building artifacts needs: the [ane] and [convert] extras")
    missing.__cause__ = ImportError("torch")
    s = _setup_with(monkeypatch, build_error=missing)
    assert (s.state, s.reason) == ("setup_available", ane_state.NO_CONVERT) and s.detail.startswith(
        "L64: ArtifactError"
    )
    s = _setup_with(monkeypatch, build_error=ArtifactParityError("3 hard mismatches"))
    assert (s.state, s.reason) == ("unavailable", ane_state.PARITY_FAILED)
    s = _setup_with(monkeypatch, calibrate_error=RuntimeError("no"))
    assert (s.state, s.reason, s.detail) == ("unavailable", ane_state.CALIBRATION_FAILED, "RuntimeError: no")
    assert _setup_with(monkeypatch) is None
