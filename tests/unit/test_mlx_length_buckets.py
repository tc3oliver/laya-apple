"""Opt-in length-bucketed MLX batching (LAYA_APPLE_MLX_LENGTH_BUCKETS).

The ordering and padding arithmetic is pure Python. The forward tests load a tiny
random-weight model (the real DecisionModel code, 3 layers) through MLXBackend from a
temporary directory; tests/integration/test_mlx_length_buckets_checkpoints.py covers the checkpoints.
"""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

from laya_apple.backends import mlx as mlx_backend
from laya_apple.backends.mlx import length_order, padded_tokens

SPEC = SimpleNamespace(name="tiny", weights_sha256="cd" * 32)
ENCODER = {
    "model_type": "modernbert",
    "vocab_size": 64,
    "hidden_size": 64,
    "intermediate_size": 32,
    "num_hidden_layers": 3,
    "num_attention_heads": 2,
    "local_attention": 8,
}
AGENT = {"head_layers": 1, "act_costs": {"a": 1.0}}


def _items(n, seed=0, lengths=(4, 60), options=(1, 6)):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        length = rng.randint(*lengths)
        k = rng.randint(*options)
        out.append(
            {
                "ids": [1] + [rng.randint(4, 63) for _ in range(length - 1)],
                "markers": sorted(rng.sample(range(1, length), min(k, length - 1))),
                "qtype": rng.randint(0, 2),
            }
        )
    return out


def test_order_only_applies_when_rows_span_several_batches():
    items = _items(16)
    assert length_order(items, 16) is None
    items = _items(17)
    order = length_order(items, 16)
    assert sorted(order) == list(range(17))
    lengths = [len(items[i]["ids"]) for i in order]
    assert lengths == sorted(lengths)
    same = [{"ids": [1] * 5, "markers": [1], "qtype": 0}] * 20
    assert length_order(same, 8) == list(range(20))  # stable: equal lengths keep request order


def test_sorted_batches_never_pad_more():
    for seed in range(20):
        items = _items(40, seed=seed)
        assert padded_tokens(items, 16, length_order(items, 16)) <= padded_tokens(items, 16)
    # alternating long and short rows: in request order every batch pads to the long length
    items = [{"ids": [1] * (100 if i % 2 else 10), "markers": [1], "qtype": 0} for i in range(32)]
    assert padded_tokens(items, 16) == 32 * 100
    assert padded_tokens(items, 16, length_order(items, 16)) == 16 * 10 + 16 * 100


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():  # the backend runs on the GPU stream
        pytest.skip("no Metal device")
    from mlx.utils import tree_flatten

    from laya_apple.models.modernbert_mlx import DecisionModel, EncoderConfig

    path = tmp_path_factory.mktemp("tiny-checkpoint")
    (path / "encoder").mkdir()
    (path / "encoder/config.json").write_text(json.dumps(ENCODER))
    (path / "rl_agent_config.json").write_text(json.dumps(AGENT))
    mx.random.seed(0)
    model = DecisionModel(EncoderConfig.from_dict(ENCODER), AGENT)
    mx.save_safetensors(str(path / "model.safetensors"), dict(tree_flatten(model.parameters())))
    return path


def _backend(checkpoint, dtype="float32", **kw):
    return mlx_backend.MLXBackend(SPEC, checkpoint, 0, dtype=dtype, **kw)


def test_off_by_default_and_follows_the_environment(checkpoint, monkeypatch):
    monkeypatch.delenv(mlx_backend.LENGTH_BUCKETS_ENV, raising=False)
    assert not _backend(checkpoint).length_buckets
    monkeypatch.setenv(mlx_backend.LENGTH_BUCKETS_ENV, "1")
    assert _backend(checkpoint).length_buckets
    assert not _backend(checkpoint, length_buckets=False).length_buckets


@pytest.mark.parametrize("n", [5, 8, 21, 40])
def test_bucketed_forward_returns_rows_in_request_order(checkpoint, n):
    items = _items(n, seed=n)
    b = _backend(checkpoint, batch_size=8, length_buckets=False)
    plain_logits, plain_act = b.forward(items)
    b.length_buckets = True
    logits, act = b.forward(items)
    if n <= 8:  # one batch: the same computation
        assert np.array_equal(logits, plain_logits) and np.array_equal(act, plain_act)
    assert logits.shape == plain_logits.shape and act.shape == plain_act.shape
    np.testing.assert_allclose(logits, plain_logits, rtol=0, atol=1e-4)
    np.testing.assert_allclose(act, plain_act, rtol=0, atol=1e-4)
    for row, it in enumerate(items):
        k = len(it["markers"])
        single_logits, single_act = b.forward([it])  # the row alone: no padding at all
        np.testing.assert_allclose(logits[row, :k], single_logits[0, :k], rtol=0, atol=1e-4)
        np.testing.assert_allclose(act[row], single_act[0], rtol=0, atol=1e-4)
        assert int(np.argmax(logits[row, :k])) == int(np.argmax(plain_logits[row, :k]))
        assert (logits[row, k:] == -1e4).all()
    again = b.forward(items)
    assert np.array_equal(again[0], logits) and np.array_equal(again[1], act)


def test_bucketed_forward_float16_keeps_decisions(checkpoint):
    items = _items(37, seed=3)
    b = _backend(checkpoint, dtype="float16", batch_size=16, length_buckets=False)
    plain_logits, _ = b.forward(items)
    b.length_buckets = True
    logits, _ = b.forward(items)
    np.testing.assert_allclose(logits, plain_logits, rtol=1e-2, atol=2e-2)
    for row, it in enumerate(items):
        k = len(it["markers"])
        assert int(np.argmax(logits[row, :k])) == int(np.argmax(plain_logits[row, :k]))
