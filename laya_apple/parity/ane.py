"""Parity gate for one compiled ANE artifact, run on this machine's Neural Engine.

Shared by `artifacts build` (after compiling) and `artifacts import` (before registering an
artifact built elsewhere). Needs coremltools and the checkpoint; no torch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..prompt import Tokenizer, prepare
from ..registry import ANE_COMPUTE_UNITS, ANE_MAX_OPTIONS, ANE_PRECISION, ModelSpec
from . import evaluate

REFERENCE = "upstream laya 0.3.20 (NandhaKishorM/laya@23a1752), PyTorch CPU FP32"


def ane_parity(spec: ModelSpec, compiled: Path, length: int, checkpoint: Path) -> dict:
    """Run every golden row that fits `length` through the compiled model on CPU_AND_NE."""
    import coremltools as ct

    from ..backends.coreml_ane import HostWeights, ane_features

    cfg = json.loads((checkpoint / "rl_agent_config.json").read_text())
    enc = json.loads((checkpoint / "encoder/config.json").read_text())
    tok = Tokenizer(checkpoint / "tokenizer")
    host = HostWeights(checkpoint, int(enc["local_attention"]))
    model = ct.models.CompiledMLModel(str(compiled), compute_units=getattr(ct.ComputeUnit, ANE_COMPUTE_UNITS))

    def forward(items):
        logits = np.full((len(items), ANE_MAX_OPTIONS), -1e4, np.float32)
        acts = []
        for r, it in enumerate(items):
            feats = ane_features(
                [it], length, 1, host.embedding, host.type_embedding, host.window(length), tok.pad_token_id
            )
            lg, ac = host.tail(model.predict(feats), [it])
            logits[r] = lg[0]
            acts.append(ac[0])
        return logits, np.stack(acts)

    summary = evaluate(
        spec.name,
        cfg,
        forward,
        precision=ANE_PRECISION,
        max_len=length,
        prepare=lambda s, q: prepare(tok, cfg, s, q).items,
    )
    summary["reference"] = REFERENCE
    return summary
