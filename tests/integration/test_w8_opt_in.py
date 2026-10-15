"""Opt-in W8 artifact (`ane_quantization="w8"`) through the production loader and `Laya`.

Needs the FP16 and the W8 L64 artifacts of laya-typed-decisions in the cache
(`laya-apple artifacts build laya-typed-decisions --length 64 --quantize w8`), on the machine
W8 was validated on (Apple M4 Max, macOS 27); skipped otherwise.
"""

from __future__ import annotations

import os

import pytest

from laya_apple import Laya
from laya_apple.artifacts import check_ane_placement, compute_plan_summary, load_verified, platform_profile
from laya_apple.errors import ArtifactMissingError, BackendUnavailableError, UnsupportedShapeError
from laya_apple.registry import ANE_PRECISION, models
from laya_apple.workload import make_request

pytestmark = [pytest.mark.integration, pytest.mark.ane]
MODEL = "laya-typed-decisions"
SPEC = models()[MODEL]
BUCKET = 64


@pytest.fixture(scope="module", autouse=True)
def _offline():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


@pytest.fixture(scope="module")
def w8_manifest():
    try:
        _, manifest = load_verified(SPEC, BUCKET, quantization="w8")
    except (ArtifactMissingError, BackendUnavailableError) as e:
        pytest.skip(f"no usable W8 L64 artifact here: {e}")
    return manifest


def _choices(result):
    return {k: v.get("choice", v.get("answer")) for k, v in result.answers.items()}


def test_manifest_records_the_quantization_and_the_unchanged_gates(w8_manifest):
    data = w8_manifest.data
    assert data["artifact"]["quantization"]["scheme"] == "w8-pt"
    assert data["artifact"]["precision"] == ANE_PRECISION  # compute and I/O stay FP16
    assert data["conversion"]["palettization"]["conv_palettized"] > 0
    assert data["parity"]["passed"] is True and data["parity"]["hard_mismatches"] == 0
    assert data["parity"]["prob_max_abs"] <= data["parity"]["tolerance"] == 0.02
    assert data["placement"]["transitions"] == 0 and data["placement"]["ops"]["cpu"] == 0
    assert data["platform"]["soc"] == platform_profile()["soc"]
    from laya_apple.artifacts import COMPILED, artifact_dir

    summary = compute_plan_summary(artifact_dir(SPEC, BUCKET, "w8") / COMPILED)
    check_ane_placement(summary)  # 100% ANE, 0 transitions, re-checked now


def test_w8_artifact_is_not_loaded_where_fp16_was_asked(w8_manifest):
    _, fp16 = load_verified(SPEC, BUCKET)
    assert fp16.artifact_sha256 != w8_manifest.artifact_sha256
    assert "quantization" not in fp16.data["artifact"]


def test_device_ane_serves_l64_from_the_w8_artifact_with_the_same_answers(w8_manifest):
    plain = Laya.from_pretrained(MODEL, device="ane", local_files_only=True)
    w8 = Laya.from_pretrained(MODEL, device="ane", local_files_only=True, ane_quantization="w8")
    try:
        assert w8.info()["ane_quantization"] == {"scheme": "w8-pt", "buckets": [64]}
        assert "ane_quantization" not in plain.info()  # opt-in only
        for seed in range(40):
            state, questions = make_request(plain.tokenizer, plain.config, BUCKET, 1, seed=seed)
            a, b = plain.predict(context=state, questions=questions), w8.predict(context=state, questions=questions)
            assert b.runtime.device == "ane" and b.runtime.dtype == ANE_PRECISION
            assert b.runtime.artifact_revision == f"L64:{w8_manifest.artifact_sha256[:12]}"
            assert a.runtime.artifact_revision != b.runtime.artifact_revision
            assert _choices(a) == _choices(b)
    finally:
        plain.close()
        w8.close()


def test_auto_uses_w8_for_l64_and_fp16_for_the_other_buckets(w8_manifest):
    _, fp16_96 = load_verified(SPEC, 96)
    laya = Laya.from_pretrained(MODEL, device="auto", local_files_only=True, ane_quantization="w8")
    try:
        state, questions = make_request(laya.tokenizer, laya.config, 64, 1, seed=3)
        r64 = laya.predict(context=state, questions=questions)
        assert (
            r64.runtime.device == "ane" and r64.runtime.artifact_revision == f"L64:{w8_manifest.artifact_sha256[:12]}"
        )
        state, questions = make_request(laya.tokenizer, laya.config, 96, 1, seed=3)
        r96 = laya.predict(context=state, questions=questions)
        if r96.runtime.device == "ane":  # routing decides; when it takes the ANE it must be the FP16 L96
            assert r96.runtime.artifact_revision == f"L96:{fp16_96.artifact_sha256[:12]}"
    finally:
        laya.close()


def test_w8_requests_that_cannot_be_served_raise(w8_manifest):
    with pytest.raises(UnsupportedShapeError):
        Laya.from_pretrained("laya", device="ane", local_files_only=True, ane_quantization="w8")


@pytest.mark.parametrize("handoff", [None, True], ids=["handoff-auto", "handoff-required"])
def test_workers_and_adaptive_execution_use_the_w8_artifact(w8_manifest, handoff):
    try:
        laya = Laya.from_pretrained(
            MODEL,
            device="auto",
            execution="workers",
            ane_handoff=handoff,
            ane_quantization="w8",
            local_files_only=True,
        )
    except BackendUnavailableError as e:  # ane_handoff=True needs pyobjc-framework-CoreML
        pytest.skip(str(e))
    try:
        state, questions = make_request(laya.tokenizer, laya.config, BUCKET, 1, seed=2)
        result = laya.predict(context=state, questions=questions)
        assert result.runtime.device == "ane"
        assert result.runtime.artifact_revision == f"L64:{w8_manifest.artifact_sha256[:12]}"
        assert laya.info()["ane_quantization"]["scheme"] == "w8-pt"
        if handoff:
            assert laya.info()["ane_handoff"]["enabled"] is True  # the async path loaded the W8 artifact too
    finally:
        laya.close()
