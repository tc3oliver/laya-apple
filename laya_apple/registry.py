"""Supported checkpoints, pinned revisions, and validated backend capabilities.

Revisions and weight hashes are the ones every Phase -1 measurement used
(research/phase-0-feasibility/environment.md). ANE buckets and auto-routing thresholds
come from laya_apple/data/routing.json, which scripts/derive_routing.py generates from the
committed Phase -1 evidence. They are not hand-written.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cache
from importlib import resources

from .errors import UnsupportedModelError

ANE_GRAPH = "bc1s-masked"  # the only Core ML graph that passed parity on the Neural Engine
ANE_COMPUTE_UNITS = "CPU_AND_NE"  # the only compute-unit setting the ANE artifacts are validated on
ANE_PRECISION = "float16"
DTYPES = ("float16", "float32")  # MLX; the ANE runs ANE_PRECISION only
ANE_MAX_OPTIONS = 32

# Opt-in weight palettization of the ANE artifacts. Only the configuration measured in
# research/ane-w8-l64-gate and research/ane-w8-l64-s5-open-loop exists: 8-bit k-means look-up
# tables per tensor on the conv weights (FP16 look-up values, FP16 compute and I/O, same graph).
# It is never the default: it needs `ane_quantization="w8"` and its own artifact.
ANE_QUANTIZATIONS = {
    "w8-pt": {"mode": "kmeans", "nbits": 8, "granularity": "per_tensor", "weight_threshold": 65536, "op_type": "conv"},
}
QUANTIZATION_ALIASES = {"w8": "w8-pt"}
# Where each quantization was validated: (model, bucket) pairs, and (SoC, macOS major) pairs.
QUANTIZATION_BUCKETS = {"w8-pt": {"laya-typed-decisions": (64,)}}
QUANTIZATION_PLATFORMS = {"w8-pt": (("Apple M4 Max", "27"),)}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo: str
    revision: str
    weights_sha256: str
    encoder: str
    max_len: int
    mlx_dtypes: tuple = ("float16", "float32")
    ane_buckets: tuple = ()
    auto_ane_buckets: tuple = ()
    auto_ane_max_len: int = 0
    auto_ane_max_questions: int = 1
    aliases: tuple = field(default=())


_BASE = {
    "laya": dict(
        repo="convaiinnovations/laya",
        # Same revision as the MLX/Core ML ports; model.safetensors is byte-identical to the
        # later head 1c5edc17.
        revision="c5d78730f3493e4fe16d61507ef4b78eef7318cf",
        weights_sha256="891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c",
        encoder="ModernBERT-large",
        max_len=512,
    ),
    "laya-multilingual": dict(
        repo="convaiinnovations/laya-multilingual",
        revision="052592a15d198d9ad47da779604259b10b47b7aa",
        weights_sha256="9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204",
        encoder="mmBERT-base",
        max_len=1024,
    ),
    "laya-typed-decisions": dict(
        repo="convaiinnovations/laya-typed-decisions",
        revision="f9ab0b228f0fc0f14d873dbc99038f135c2da1b2",
        weights_sha256="4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e",
        encoder="ModernBERT-large",
        max_len=1024,
    ),
}


@cache
def routing_table() -> dict:
    return json.loads(resources.files("laya_apple.data").joinpath("routing.json").read_text())


@cache
def models() -> dict[str, ModelSpec]:
    table = routing_table()["models"]
    out = {}
    for name, base in _BASE.items():
        r = table[name]
        out[name] = ModelSpec(
            name=name,
            ane_buckets=tuple(r["ane_buckets"]),
            auto_ane_buckets=tuple(r["auto_ane_buckets"]),
            auto_ane_max_len=int(r["auto_ane_max_len"]),
            auto_ane_max_questions=int(r["auto_ane_max_questions"]),
            aliases=(base["repo"],),
            **base,
        )
    return out


def resolve(model_id: str) -> ModelSpec:
    """Map a Hugging Face id or short name to its pinned spec, or raise."""
    for spec in models().values():
        if model_id == spec.name or model_id in spec.aliases:
            return spec
    known = ", ".join(s.repo for s in models().values())
    raise UnsupportedModelError(f"{model_id!r} is not a supported checkpoint. Supported: {known}")
