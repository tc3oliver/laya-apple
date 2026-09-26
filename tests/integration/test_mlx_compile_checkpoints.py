"""The compiled MLX forward (LAYA_APPLE_MLX_COMPILE=1) on the real checkpoints.

It must pass the unchanged parity gate against the goldens and give the same decision as the
eager MLX forward on every golden row. Whether it is also bitwise identical to eager is
printed (run with -s); only a bitwise-identical result would justify turning it on by default.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from laya_apple.parity import evaluate, load_goldens

pytestmark = [pytest.mark.integration, pytest.mark.parity]


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_compiled_forward_passes_parity_and_keeps_eager_decisions(model_name, dtype, cached_laya):
    laya = cached_laya(model_name, device="gpu", dtype=dtype)
    backend = laya.mlx
    before = backend.compile_forward
    try:
        backend.compile_forward = True
        summary = evaluate(
            model_name,
            laya.config,
            backend.forward,
            precision=dtype,
            prepare=lambda s, q: laya.prepare(s, q).items,
        )
        if not summary["passed"]:
            pytest.fail(f"{model_name} compiled MLX {dtype} failed the parity gate:\n{json.dumps(summary, indent=1)}")
        worst, worst_act, bitwise, rows = 0.0, 0.0, True, 0
        for case in load_goldens(model_name)["cases"]:
            items = case["items"]
            backend.compile_forward = False
            eager_logits, eager_act = backend.forward(items)
            backend.compile_forward = True
            logits, act = backend.forward(items)
            bitwise = bitwise and np.array_equal(logits, eager_logits) and np.array_equal(act, eager_act)
            for row, it in enumerate(items):
                k = len(it["markers"])
                assert int(np.argmax(logits[row, :k])) == int(np.argmax(eager_logits[row, :k])), (case["name"], row)
                worst = max(worst, float(np.abs(logits[row, :k] - eager_logits[row, :k]).max()))
                worst_act = max(worst_act, float(np.abs(act[row] - eager_act[row]).max()))
                rows += 1
        print(
            f"\n{model_name} {dtype}: {rows} rows, compiled vs eager max|Δlogit| {worst:.3g}, "
            f"max|Δaction logit| {worst_act:.3g}, bitwise {bitwise}"
        )
    finally:
        backend.compile_forward = before
