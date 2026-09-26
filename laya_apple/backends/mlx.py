"""MLX / Metal GPU backend: the default for every model, length and question count."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from ..errors import BackendUnavailableError
from ..registry import DTYPES, ModelSpec

# Opt-in length-bucketed batching. A request with more rows than batch_size runs as several
# batches, each padded to its longest row; sorting the rows by length first puts rows of
# similar length together, so batches pad less. Off by default: a row padded to a different
# length is expected to give the same decision but is not shown to be bitwise identical
# (tests/integration/test_mlx_length_buckets.py). Requests of at most batch_size rows run
# exactly as before either way.
LENGTH_BUCKETS_ENV = "LAYA_APPLE_MLX_LENGTH_BUCKETS"


def length_buckets_enabled() -> bool:
    return os.environ.get(LENGTH_BUCKETS_ENV, "") == "1"


def length_order(items: list[dict], batch_size: int) -> list[int] | None:
    """The row order to batch in: by length (stable) when the rows span several batches."""
    if len(items) <= batch_size:
        return None
    return sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))


def padded_tokens(items: list[dict], batch_size: int, order: list[int] | None = None) -> int:
    """Tokens the batches hold, padding included, when `items` run in `order`."""
    rows = [len(items[i]["ids"]) for i in (order if order is not None else range(len(items)))]
    return sum(len(c) * max(c) for c in (rows[s : s + batch_size] for s in range(0, len(rows), batch_size)))


def collate(items: list[dict], pad_id: int) -> dict:
    n, length = len(items), max(len(it["ids"]) for it in items)
    count = max(2, max(len(it["markers"]) for it in items))
    batch = {
        "input_ids": np.full((n, length), pad_id, np.int32),
        "attention_mask": np.zeros((n, length), np.bool_),
        "marker_pos": np.zeros((n, count), np.int32),
        "marker_mask": np.zeros((n, count), np.bool_),
        "qtype": np.array([it["qtype"] for it in items], np.int32),
    }
    for i, it in enumerate(items):
        m, k = len(it["ids"]), len(it["markers"])
        batch["input_ids"][i, :m] = it["ids"]
        batch["attention_mask"][i, :m] = True
        batch["marker_pos"][i, :k] = it["markers"]
        batch["marker_mask"][i, :k] = True
    return batch


class MLXBackend:
    name = "mlx"
    device = "gpu"

    def __init__(
        self,
        spec: ModelSpec,
        checkpoint: Path,
        pad_id: int,
        *,
        dtype: str = "float16",
        batch_size: int = 16,
        length_buckets: bool | None = None,
    ):
        if dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        try:
            import mlx.core as mx

            from ..models.modernbert_mlx import DecisionModel, EncoderConfig, sanitize_weights
        except ImportError as e:  # non-Apple platforms
            raise BackendUnavailableError(f"MLX is not available: {e}") from e
        self.mx = mx
        self.spec, self.dtype, self.batch_size, self.pad_id = spec, dtype, batch_size, pad_id
        # None: the LAYA_APPLE_MLX_LENGTH_BUCKETS environment variable decides (it also reaches workers).
        self.length_buckets = length_buckets_enabled() if length_buckets is None else bool(length_buckets)
        agent_cfg = json.loads((checkpoint / "rl_agent_config.json").read_text())
        enc_cfg = EncoderConfig.from_dict(json.loads((checkpoint / "encoder/config.json").read_text()))
        mdtype = {"float16": mx.float16, "float32": mx.float32}[dtype]
        with mx.stream(mx.gpu):
            self.model = DecisionModel(enc_cfg, agent_cfg)
            weights = sanitize_weights(mx.load(str(checkpoint / "model.safetensors")))
            self.model.load_weights([(k, v.astype(mdtype)) for k, v in weights.items()], strict=True)
            self.model.eval()
            mx.eval(self.model.parameters())

    def artifact_revision(self, items) -> str:
        return f"mlx:{self.spec.weights_sha256[:12]}:{self.dtype}"

    def forward(self, items: list[dict]):
        mx = self.mx
        order = length_order(items, self.batch_size) if self.length_buckets else None
        rows = items if order is None else [items[i] for i in order]
        logits, acts = [], []
        for start in range(0, len(rows), self.batch_size):
            batch = collate(rows[start : start + self.batch_size], self.pad_id)
            with mx.stream(mx.gpu):
                lg, ac = self.model(**{k: mx.array(v) for k, v in batch.items()})
                mx.eval(lg, ac)
            logits.append(np.asarray(lg, np.float32))
            acts.append(np.asarray(ac, np.float32))
        # chunks can have different option counts; pad with the same -1e4 the model uses
        width = max(x.shape[1] for x in logits)
        logits = [np.pad(x, ((0, 0), (0, width - x.shape[1])), constant_values=-1e4) for x in logits]
        logits, acts = np.concatenate(logits), np.concatenate(acts)
        if order is None:
            return logits, acts
        back = np.argsort(order)  # row i of the request is row back[i] of the batches
        return logits[back], acts[back]
