"""The GIL-releasing predict binding (backends/coreml_nogil.py) against coremltools, on real
artifacts: bit-identical Core ML outputs and backend outputs on every golden row, and the
FP16 parity gate through the binding."""

from __future__ import annotations

import json

import numpy as np
import pytest

from laya_apple import Laya
from laya_apple.artifacts import COMPILED, artifact_dir
from laya_apple.backends import coreml_nogil
from laya_apple.backends.coreml_ane import ANEBackend, ane_features
from laya_apple.errors import ArtifactError, UnsupportedShapeError
from laya_apple.parity import evaluate, load_goldens
from laya_apple.registry import ANE_COMPUTE_UNITS, models

pytestmark = [pytest.mark.parity, pytest.mark.ane]


@pytest.fixture
def pair(model_name, monkeypatch):
    """(coremltools ANEBackend, nogil ANEBackend) over the same artifacts, or skip."""
    pytest.importorskip("CoreML")
    monkeypatch.delenv(coreml_nogil.ENV, raising=False)
    spec = models()[model_name]
    if not spec.ane_buckets:
        pytest.skip(f"{model_name} has no ANE buckets")
    try:
        laya = Laya.from_pretrained(model_name, device="ane", local_files_only=True)  # inline: coremltools
    except Exception:
        pytest.skip(f"{model_name}: no validated ANE artifacts present")
    ref = laya.ane
    assert ref.predict_impl == coreml_nogil.COREMLTOOLS
    nogil = ANEBackend(
        laya.spec,
        laya.checkpoint,
        ref.pad_id,
        int(laya._encoder_cfg["local_attention"]),
        ref.buckets,
        strict=True,
        predict="auto",
    )
    assert (nogil.predict_impl, nogil.predict_reason) == (coreml_nogil.NOGIL, coreml_nogil.DEFAULT)
    assert nogil.buckets == ref.buckets and nogil.artifact_sha256 == ref.artifact_sha256
    return laya, ref, nogil


def _rows(model_name):
    for case in load_goldens(model_name)["cases"]:
        for i, it in enumerate(case["items"]):
            yield case["name"], i, it


def _same(a: dict, b: dict, where: str):
    assert set(a) == set(b), where
    for k in a:
        assert a[k].dtype == b[k].dtype and a[k].shape == b[k].shape, f"{where} {k}"
        assert a[k].tobytes() == b[k].tobytes(), f"{where} {k}: outputs differ"


def test_core_ml_outputs_are_bit_identical_on_every_golden_row_and_bucket(model_name, pair):
    """Every golden row, in every bucket that holds it (not only the one forward picks)."""
    laya, ref, nogil = pair
    host, compared = ref.host, 0
    for b in ref.buckets:
        a = ref.models[b]
        n = coreml_nogil.NoGilModel(artifact_dir(laya.spec, b) / COMPILED, ANE_COMPUTE_UNITS)
        for name, i, it in _rows(model_name):
            try:
                ref.check([it])
            except (UnsupportedShapeError, ArtifactError):
                continue
            if len(it["ids"]) > b:
                continue
            feats = ane_features([it], b, 1, host.embedding, host.type_embedding, host.window(b), ref.pad_id)
            _same(a.predict(feats), n.predict(feats), f"{model_name} L{b} {name}[{i}]")
            compared += 1
    assert compared > 0


def test_backend_outputs_are_bit_identical_on_every_golden_row(model_name, pair):
    _, ref, nogil = pair
    compared = 0
    for name, i, it in _rows(model_name):
        try:
            ref.check([it])
        except (UnsupportedShapeError, ArtifactError):
            continue
        (la, aa), (ln, an) = ref.forward([it]), nogil.forward([it])
        assert la.tobytes() == ln.tobytes() and aa.tobytes() == an.tobytes(), f"{model_name} {name}[{i}]"
        compared += 1
    assert compared > 0


def test_nogil_backend_passes_the_parity_gate_for_each_bucket(model_name, pair):
    laya, _, nogil = pair
    for bucket in nogil.buckets:
        summary = evaluate(
            model_name,
            laya.config,
            nogil.forward,
            precision="float16",
            max_len=bucket,
            prepare=lambda s, q: laya.prepare(s, q).items,
        )
        if not summary["passed"]:
            pytest.fail(f"{model_name} nogil L{bucket} failed the parity gate:\n{json.dumps(summary, indent=1)}")


def test_nogil_placement_probe_passed_for_every_bucket(pair):
    _, _, nogil = pair
    from laya_apple.backends.coreml_ane import PROBE_MAX_RATIO

    assert set(nogil.probes) == set(nogil.buckets)
    assert all(np.isfinite(p["ratio"]) and p["ratio"] < PROBE_MAX_RATIO for p in nogil.probes.values())
