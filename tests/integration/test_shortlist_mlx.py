"""predict_shortlist on a real checkpoint with the MLX encoder helper (inline, device="gpu")."""

from __future__ import annotations

import numpy as np
import pytest

from laya_apple import embed_fn_from_laya

pytestmark = pytest.mark.integration

LABELS = {f"label_{i:02d}": f"intent number {i}" for i in range(30)}
LABELS["duplicate_charge"] = "the customer was charged twice for the same transfer"


def test_embed_fn_from_laya_mean_pools_without_padding(laya_gpu):
    embed = embed_fn_from_laya(laya_gpu, batch_size=2)
    short, long_ = "refund", "I was charged twice for a transfer and want the money back"
    both = embed([short, long_, short])
    alone = embed([short])
    assert both.shape[0] == 3 and both.shape[1] == alone.shape[1] > 0
    assert np.isfinite(both).all()
    # Padding is excluded from the mean: the short text embeds the same alone or beside a long one.
    cosine = float(both[0] @ alone[0] / (np.linalg.norm(both[0]) * np.linalg.norm(alone[0])))
    assert cosine > 0.999
    assert embed([]).shape == (0, alone.shape[1])


def test_predict_shortlist_end_to_end(laya_gpu):
    questions = {
        "intent": {"type": "choice", "instructions": "Which banking intent is this?", "criteria": LABELS},
        "urgent": {"type": "noul", "instructions": "Is this urgent?"},
    }
    result = laya_gpu.predict_shortlist(
        "I was charged twice for a transfer", questions, embed_fn=embed_fn_from_laya(laya_gpu), k=5
    )
    meta = result.extra["shortlist"]["intent"]
    assert meta["n"] == len(LABELS) and len(meta["labels"]) == 5 and not meta["passthrough"]
    assert result.answers["intent"]["choice"] in meta["labels"]
    assert set(result.answers) == {"intent", "urgent"}
