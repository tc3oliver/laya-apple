"""Length-bucketed MLX batching (LAYA_APPLE_MLX_LENGTH_BUCKETS=1) on the real checkpoints.

Golden cases have at most 8 rows, fewer than one batch, so each case is run together with
filler rows from the other cases: the request then spans several batches and is reordered by
length. The case's own rows must pass the unchanged parity gate and give the same decisions
as the in-order batching; whether they are also bitwise identical is printed (run with -s).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from laya_apple.backends.mlx import length_order, padded_tokens
from laya_apple.parity import evaluate, load_goldens

pytestmark = [pytest.mark.integration, pytest.mark.parity]


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_bucketed_batches_pass_parity_and_keep_decisions(model_name, dtype, cached_laya):
    laya = cached_laya(model_name, device="gpu", dtype=dtype)
    backend = laya.mlx
    cases = load_goldens(model_name)["cases"]
    pool = [it for case in cases for it in case["items"]]
    filler = pool[:: max(1, len(pool) // 24)][:24]  # 24 rows of mixed lengths
    assert len(filler) + 1 > backend.batch_size

    def run(sub, bucketed):
        backend.length_buckets = bucketed
        rows = sub + filler
        assert not bucketed or length_order(rows, backend.batch_size) is not None
        logits, act = backend.forward(rows)
        return logits[: len(sub)], act[: len(sub)]

    before = backend.length_buckets
    try:
        summary = evaluate(
            model_name,
            laya.config,
            lambda sub: run(sub, True),
            precision=dtype,
            prepare=lambda s, q: laya.prepare(s, q).items,
        )
        if not summary["passed"]:
            pytest.fail(f"{model_name} bucketed MLX {dtype} failed the parity gate:\n{json.dumps(summary, indent=1)}")
        worst, bitwise, saved = 0.0, True, []
        for case in cases:
            items = case["items"]
            plain_logits, plain_act = run(items, False)
            logits, act = run(items, True)
            bitwise = bitwise and np.array_equal(logits, plain_logits) and np.array_equal(act, plain_act)
            for row, it in enumerate(items):
                k = len(it["markers"])
                assert int(np.argmax(logits[row, :k])) == int(np.argmax(plain_logits[row, :k])), (case["name"], row)
                worst = max(worst, float(np.abs(logits[row, :k] - plain_logits[row, :k]).max()))
            rows = items + filler
            saved.append(
                1
                - padded_tokens(rows, backend.batch_size, length_order(rows, backend.batch_size))
                / padded_tokens(rows, backend.batch_size)
            )
        print(
            f"\n{model_name} {dtype}: bucketed vs in-order max|Δlogit| {worst:.3g}, bitwise {bitwise}, "
            f"padded tokens saved {min(saved):.0%}-{max(saved):.0%}"
        )
    finally:
        backend.length_buckets = before
