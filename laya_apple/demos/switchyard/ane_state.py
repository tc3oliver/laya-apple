"""Whether this Mac can run the GPU + ANE round, from the runtime's own checks only.

  ready            coremltools, verified auto-bucket artifacts and a validated or calibrated
                   routing profile; confirmed only once the warmup routed >= 1 train to the ANE
  setup_available  Apple silicon, but coremltools, artifacts or calibration are missing (or
                   the warmup sent no train to the ANE): `--setup-ane` can fix it
  unavailable      an artifact failed parity here, Core ML cannot run the ANE path, or the
                   GPU + ANE round failed or stopped using the ANE (then its data is kept but
                   it is not presented as a comparison)

Anything but `ready` runs the GPU-only round alone. It is never an error.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field, replace

from ... import routing
from ...errors import (
    ArtifactIntegrityError,
    ArtifactMissingError,
    ArtifactParityError,
    ArtifactRevisionError,
    BackendUnavailableError,
    ComputeUnitMismatchError,
)

READY, SETUP_AVAILABLE, UNAVAILABLE = "ready", "setup_available", "unavailable"
STATES = (READY, SETUP_AVAILABLE, UNAVAILABLE)
SETUP_COMMAND = 'uvx --from "laya-apple[convert]" laya-apple switchyard --setup-ane'

# Reason codes, recorded raw in result.json; messages.REASON_TEXT has the plain-language
# text for each. The first three are the runtime's own routing reasons.
NO_COREMLTOOLS = routing.RUNTIME_UNAVAILABLE
NO_ARTIFACTS = routing.ARTIFACT_UNAVAILABLE
NOT_CALIBRATED = routing.PLATFORM_NOT_VALIDATED
NOT_APPLE_SILICON = "not_apple_silicon"
REBUILD_ARTIFACTS = "ane_artifact_stale"
PARITY_FAILED = "ane_parity_failed"
NOT_ON_ANE = "ane_placement_failed"
NO_ANE_WARMUP = "ane_unused_in_warmup"
ANE_LOST = "ane_lost_during_round"  # the round's trains stopped reaching the ANE
ANE_FAILED = "ane_round_failed"  # loading, warming up or running the GPU + ANE round raised
NO_CONVERT = "ane_convert_unavailable"  # --setup-ane without the [convert] extra
CALIBRATION_FAILED = "ane_calibration_failed"
REASONS = (
    NO_COREMLTOOLS,
    NO_ARTIFACTS,
    NOT_CALIBRATED,
    NOT_APPLE_SILICON,
    REBUILD_ARTIFACTS,
    PARITY_FAILED,
    NOT_ON_ANE,
    NO_ANE_WARMUP,
    ANE_LOST,
    ANE_FAILED,
    NO_CONVERT,
    CALIBRATION_FAILED,
)


@dataclass(frozen=True)
class AneStatus:
    state: str
    reason: str | None = None  # one of REASONS
    detail: str | None = None  # the underlying error, when there is one
    artifacts: dict = field(default_factory=dict)  # bucket (str) -> artifact sha256
    warmup_ane_requests: int | None = None

    @property
    def setup_command(self) -> str | None:
        return SETUP_COMMAND if self.state == SETUP_AVAILABLE else None

    @property
    def reason_text(self) -> str | None:
        from .messages import REASON_TEXT

        return REASON_TEXT[self.reason] if self.reason else None

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "reason": self.reason,
            "reason_text": self.reason_text,
            "detail": self.detail,
            "setup_command": self.setup_command,
            "warmup_ane_requests": self.warmup_ane_requests,
        }


def _apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _calibrated(model: str) -> bool:
    from ...profiles import load_local

    return load_local(model) is not None


def _auto_buckets(spec, validated: bool) -> tuple:
    """The buckets auto routes to the ANE here: shipped, or the local calibration's."""
    if not validated:
        from ...profiles import load_local

        local = load_local(spec.name)
        if local is not None:
            return tuple(local.get("auto_ane_buckets") or ())
    return tuple(spec.auto_ane_buckets)


def classify_artifact_error(e: Exception) -> tuple[str, str]:
    """(state, reason) for an error from artifacts.load_verified."""
    if isinstance(e, ArtifactParityError):
        return UNAVAILABLE, PARITY_FAILED
    if isinstance(e, ComputeUnitMismatchError):
        return UNAVAILABLE, NOT_ON_ANE
    if isinstance(e, ArtifactMissingError):
        return SETUP_AVAILABLE, NO_ARTIFACTS
    if isinstance(e, (ArtifactRevisionError, ArtifactIntegrityError)):
        return SETUP_AVAILABLE, REBUILD_ARTIFACTS
    if isinstance(e, BackendUnavailableError):
        return SETUP_AVAILABLE, NO_COREMLTOOLS
    return UNAVAILABLE, NOT_ON_ANE


def detect(spec, *, apple_silicon=None, coremltools=None, validated=None, calibrated=None, load=None) -> AneStatus:
    """The pre-warmup state. The keyword arguments replace the probes (tests)."""
    from ...artifacts import load_verified
    from ...model import _coremltools_available, platform_validated

    apple_silicon = _apple_silicon if apple_silicon is None else apple_silicon
    coremltools = _coremltools_available if coremltools is None else coremltools
    validated = platform_validated if validated is None else validated
    calibrated = _calibrated if calibrated is None else calibrated
    load = load_verified if load is None else load

    if not apple_silicon():
        return AneStatus(UNAVAILABLE, NOT_APPLE_SILICON)
    if not coremltools():
        return AneStatus(SETUP_AVAILABLE, NO_COREMLTOOLS)
    is_validated = bool(validated())
    buckets = _auto_buckets(spec, is_validated)
    if not buckets:
        return AneStatus(SETUP_AVAILABLE, NOT_CALIBRATED)
    shas = {}
    worst = None  # an unavailable finding outranks a fixable one
    for b in buckets:
        try:
            _, manifest = load(spec, b)
            shas[str(b)] = manifest.artifact_sha256
        except Exception as e:
            found = (*classify_artifact_error(e), f"L{b}: {type(e).__name__}: {e}")
            if worst is None or (found[0] == UNAVAILABLE and worst[0] != UNAVAILABLE):
                worst = found
    if worst is not None:
        return AneStatus(*worst, artifacts=shas)
    if not is_validated and not calibrated(spec.name):
        return AneStatus(SETUP_AVAILABLE, NOT_CALIBRATED, artifacts=shas)
    return AneStatus(READY, artifacts=shas)


def after_warmup(status: AneStatus, ane_trains: int) -> AneStatus:
    """`ready` holds only if the warmup actually ran trains on the ANE."""
    if status.state != READY:
        return status
    if ane_trains < 1:
        return replace(status, state=SETUP_AVAILABLE, reason=NO_ANE_WARMUP, warmup_ane_requests=ane_trains)
    return replace(status, warmup_ane_requests=ane_trains)


def setup(spec, *, local_files_only: bool = False, step=lambda kind, bucket=None: None, log=print) -> AneStatus | None:
    """--setup-ane: build (parity-gated) every offered bucket that does not verify here, then
    calibrate when no shipped profile matches this Mac. Rebuilds nothing that verifies.
    `step(kind, bucket)` reports "verified" / "building" / "calibrating"; `log` gets the
    calibration's own measurement lines. Returns None on success, else the AneStatus that
    explains the failure (never raises for a build or calibration failure)."""
    from ...artifacts import artifact_dir, load_verified
    from ...conversion.build import build
    from ...model import platform_validated
    from ...profiles import calibrate

    for b in spec.ane_buckets:  # calibrate needs every offered bucket, not only the auto ones
        try:
            load_verified(spec, b)
            step("verified", b)
            continue
        except Exception:
            pass
        step("building", b)
        try:
            build(spec, b, local_files_only=local_files_only, force=artifact_dir(spec, b).exists())
        except Exception as e:
            detail = f"L{b}: {type(e).__name__}: {e}"
            if isinstance(e.__cause__, ImportError):
                return AneStatus(SETUP_AVAILABLE, NO_CONVERT, detail=detail)
            state, reason = classify_artifact_error(e)
            return AneStatus(state, reason, detail=detail)
    if not platform_validated():
        step("calibrating")
        try:
            calibrate(spec, local_files_only=local_files_only, log=log)
        except Exception as e:
            detail = f"{type(e).__name__}: {e}"
            if isinstance(e, ArtifactParityError):
                return AneStatus(UNAVAILABLE, PARITY_FAILED, detail=detail)
            return AneStatus(UNAVAILABLE, CALIBRATION_FAILED, detail=detail)
    return None
