"""The asynchronous Core ML path of adaptive execution (laya_apple/backends/coreml_async.py)
against the goldens: every forward it serves must pass the FP16 parity gate, and must equal the
coremltools path's output exactly (same logits, same activations), on every present bucket of
every eligible model."""

from __future__ import annotations

import json

import numpy as np
import pytest

from laya_apple import Laya
from laya_apple.model import ane_placement_for
from laya_apple.parity import evaluate
from laya_apple.registry import models

pytestmark = [pytest.mark.parity, pytest.mark.ane]


class _AlwaysAsync:
    """A handoff that sends every forward to the async path."""

    def decide(self):
        return "async", "async_healthy", 64

    def disable(self, reason):
        raise AssertionError(f"the async path failed: {reason}")


def test_async_path_passes_parity_and_equals_coremltools(model_name, cached_laya):
    spec = models()[model_name]
    if not spec.ane_buckets or ane_placement_for(model_name) != "thread":
        pytest.skip(f"{model_name} does not use adaptive execution")
    pytest.importorskip("CoreML")
    try:
        laya = Laya.from_pretrained(model_name, device="ane", local_files_only=True)
    except Exception:
        pytest.skip(f"{model_name}: no validated ANE artifacts present")
    backend = laya.ane
    backend._load_async(spec)
    assert backend.async_error is None, backend.async_error
    assert set(backend.async_models) == set(backend.models)
    seen = {"n": 0}

    def both(items):
        backend.handoff = None
        sync_logits, sync_act = backend.forward(items)
        backend.handoff = _AlwaysAsync()
        try:
            logits, act = backend.forward(items)
        finally:
            backend.handoff = None
        assert np.array_equal(logits, sync_logits) and np.array_equal(act, sync_act)
        seen["n"] += 1
        return logits, act

    for bucket in backend.buckets:
        summary = evaluate(
            model_name,
            laya.config,
            both,
            precision="float16",
            max_len=bucket,
            prepare=lambda s, q: laya.prepare(s, q).items,
        )
        if not summary["passed"]:
            pytest.fail(f"{model_name} async L{bucket} failed the parity gate:\n{json.dumps(summary, indent=1)}")
    assert seen["n"] > 0
