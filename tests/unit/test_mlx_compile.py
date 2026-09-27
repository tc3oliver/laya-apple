"""The opt-in compiled MLX forward (LAYA_APPLE_MLX_COMPILE) on a tiny random-weight model.

No checkpoint: a 3-layer ModernBERT with the real DecisionModel code is written to a
temporary directory and loaded through MLXBackend. The real checkpoints are covered by
tests/integration/test_mlx_compile_checkpoints.py (parity gate + decisions identical to eager).
"""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():  # the backend runs on the GPU stream
    pytest.skip("no Metal device", allow_module_level=True)

from laya_apple.backends import mlx as mlx_backend  # noqa: E402
from laya_apple.backends.mlx import MLXBackend, collate, compiled_shape  # noqa: E402

SPEC = SimpleNamespace(name="tiny", weights_sha256="ab" * 32)
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


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
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


def _items(n, seed=0, lengths=(5, 40), options=(1, 6)):
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


def _backend(checkpoint, dtype="float32", **kw):
    return MLXBackend(SPEC, checkpoint, 0, dtype=dtype, **kw)


def test_compiled_shape_bounds_lengths_and_option_counts():
    assert compiled_shape(3, 1, 1) == (3, 32, 2)
    assert compiled_shape(3, 32, 2) == (3, 32, 2)
    assert compiled_shape(3, 33, 3) == (3, 64, 4)
    assert compiled_shape(1, 511, 5) == (1, 512, 8)
    assert compiled_shape(16, 1024, 16) == (16, 1024, 16)
    lengths = {compiled_shape(1, length, 2)[1] for length in range(1, 1025)}
    assert len(lengths) == 1024 // mlx_backend.COMPILE_LENGTH_STEP


def test_collate_minimums_only_add_masked_padding():
    items = _items(3)
    plain = collate(items, 0)
    padded = collate(items, 0, length=64, count=8)
    n, length = plain["input_ids"].shape
    count = plain["marker_pos"].shape[1]
    assert padded["input_ids"].shape == (n, 64) and padded["marker_pos"].shape == (n, 8)
    assert np.array_equal(padded["input_ids"][:, :length], plain["input_ids"])
    assert not padded["input_ids"][:, length:].any() and not padded["attention_mask"][:, length:].any()
    assert np.array_equal(padded["marker_pos"][:, :count], plain["marker_pos"])
    assert not padded["marker_mask"][:, count:].any()
    assert np.array_equal(padded["qtype"], plain["qtype"])


def test_compile_is_off_by_default_and_follows_the_environment(checkpoint, monkeypatch):
    monkeypatch.delenv(mlx_backend.COMPILE_ENV, raising=False)
    b = _backend(checkpoint)
    assert not b.compile_forward
    assert b.artifact_revision([]) == f"mlx:{'ab' * 6}:float32"
    monkeypatch.setenv(mlx_backend.COMPILE_ENV, "1")
    b = _backend(checkpoint)
    assert b.compile_forward
    assert b.artifact_revision([]) == f"mlx:{'ab' * 6}:float32:compiled"
    assert not _backend(checkpoint, compile_forward=False).compile_forward


@pytest.mark.parametrize("n", [1, 5, 21])  # 21 > batch_size: several chunks, different widths
def test_compiled_forward_matches_eager(checkpoint, n):
    items = _items(n, seed=n)
    b = _backend(checkpoint, batch_size=8, compile_forward=False)
    eager_logits, eager_act = b.forward(items)
    b.compile_forward = True
    logits, act = b.forward(items)
    assert logits.shape == eager_logits.shape and act.shape == eager_act.shape
    np.testing.assert_allclose(logits, eager_logits, rtol=0, atol=1e-4)
    np.testing.assert_allclose(act, eager_act, rtol=0, atol=1e-4)
    for row, it in enumerate(items):
        k = len(it["markers"])
        assert int(np.argmax(logits[row, :k])) == int(np.argmax(eager_logits[row, :k]))
        assert (logits[row, k:] == -1e4).all()
    again = b.forward(items)
    assert np.array_equal(again[0], logits) and np.array_equal(again[1], act)


def test_compiled_forward_float16_keeps_decisions(checkpoint):
    items = _items(12, seed=7)
    b = _backend(checkpoint, dtype="float16", compile_forward=False)
    eager_logits, _ = b.forward(items)
    b.compile_forward = True
    logits, _ = b.forward(items)
    np.testing.assert_allclose(logits, eager_logits, rtol=1e-2, atol=2e-2)
    for row, it in enumerate(items):
        k = len(it["markers"])
        assert int(np.argmax(logits[row, :k])) == int(np.argmax(eager_logits[row, :k]))


def test_compiled_graphs_are_bounded_and_reused(checkpoint, monkeypatch):
    monkeypatch.setattr(mlx_backend, "COMPILE_CACHE_SIZE", 2)
    b = _backend(checkpoint, compile_forward=True)
    short = [{"ids": [1, 5, 6, 7], "markers": [1, 2], "qtype": 0}]
    b.forward(short)
    first = next(iter(b._compiled.values()))
    b.forward([{"ids": [1, 5, 6, 8, 9], "markers": [1, 3], "qtype": 1}])  # same padded shape
    assert len(b._compiled) == 1 and next(iter(b._compiled.values())) is first
    for length in (40, 70, 100):  # three more padded lengths
        b.forward([{"ids": [1] + [5] * (length - 1), "markers": [1, 2], "qtype": 2}])
        assert len(b._compiled) <= 2
    assert [key[1] for key in b._compiled] == [compiled_shape(1, 70, 2), compiled_shape(1, 100, 2)]
