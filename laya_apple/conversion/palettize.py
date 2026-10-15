"""Opt-in 8-bit weight palettization of the converted ANE model (`w8-pt`).

The configuration is the one measured in research/ane-w8-l64-gate: k-means look-up tables,
8 bits, one table per tensor, on the conv weights with more than 65,536 elements (every
projection and MLP matrix; no bias, norm, RoPE table or scorer vector). Compute and I/O stay
FP16, so the graph, inputs and outputs are the production ones and only the stored weights shrink.
"""

from __future__ import annotations

from ..errors import ArtifactError
from ..registry import ANE_QUANTIZATIONS


def palettize(mlmodel, quantization: str):
    """Return (palettized MLModel, inventory) for the registered `quantization` config."""
    from coremltools._deps import _HAS_KMEANS1D
    from coremltools.optimize.coreml import (
        OpPalettizerConfig,
        OptimizationConfig,
        _quantization_passes,
        palettize_weights,
    )

    # FP16 weights with >= 10,000 elements take coremltools' exact 1-D k-means (kmeans1d), which is
    # deterministic. Without it coremltools silently falls back to scikit-learn's KMeans.
    if not _HAS_KMEANS1D:
        raise ArtifactError("coremltools' bundled kmeans1d is not importable; refusing the scikit-learn fallback")
    cfg = ANE_QUANTIZATIONS[quantization]
    # coremltools 9.0 keeps one module-level k-means Pool that is closed after a model's pass, so a
    # second model in the same process would fail with "Pool not running".
    _quantization_passes.palettize_weights._compress_pool = None
    op = OpPalettizerConfig(
        mode=cfg["mode"],
        nbits=cfg["nbits"],
        granularity=cfg["granularity"],
        weight_threshold=cfg["weight_threshold"],
        num_kmeans_workers=1,
    )
    out = palettize_weights(mlmodel, OptimizationConfig(op_type_configs={cfg["op_type"]: op}))
    inventory = conv_inventory(out)
    if not inventory["conv_palettized"]:
        raise ArtifactError("palettization changed no conv weight")
    return out, inventory


def conv_inventory(mlmodel) -> dict:
    """Counts of conv weights stored as look-up tables and as dense tensors."""
    prog = getattr(mlmodel, "_mil_program", None)
    if prog is None:
        raise ArtifactError("the palettized model has no MIL program to inspect")
    lut = dense = 0
    for fn in prog.functions.values():
        for op in fn.operations:
            if op.op_type == "conv":
                src = op.weight.op.op_type if op.weight.op is not None else "input"
                if src.startswith("constexpr_lut_to_dense"):
                    lut += 1
                else:
                    dense += 1
    return {"conv_palettized": lut, "conv_dense": dense}
