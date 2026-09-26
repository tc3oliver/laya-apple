"""Build and gate one W8-palettized BC1S artifact per preregistered cell (criteria.md).

    uv run --extra ane --extra convert python research/ane-w8/scripts/build_w8.py \
        --config w8-pt [--model laya] [--length 64] [--root <research-root>]

Per cell, in one process (CPU-heavy conversion, then the ANE for parity; not timing-sensitive):

  1. the production pipeline up to conversion, unchanged: pinned checkpoint + weight hash,
     PyTorch FP32 reference, production `ConvBody`, the FP32 layout check (< 1e-3), trace, and
     `ct.convert` with the production arguments (FP16 compute and I/O, macOS 15 target);
  2. the only change: `palettize_weights` with the config's `OpPalettizerConfig` on the conv
     weights (see CONFIGS), before saving;
  3. save, compile to model.mlmodelc (the .mlpackage is removed), sizes;
  4. the unchanged placement gate: `compute_plan_summary` + `check_ane_placement`
     (100% ANE, 0 transitions), plus the per-op-type plan as a diagnostic;
  5. the unchanged parity gate: production `ane_parity` at this bucket, plus a recording pass
     that keeps every golden row's raw outputs.

Nothing is registered: the artifact stays under the research root, which must not overlap
LAYA_APPLE_CACHE, and the runtime never loads it. The record goes to raw/<model>/<cell>/build.json.
A cell that fails a step is recorded with the step and the reason and the run moves on.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CELLS,
    RAW,
    cell_name,
    environment,
    log,
    now,
    parity,
    plan_detail,
    research_root,
    tree_bytes,
    weight_bytes,
    write_json,
)

# Preregistered configurations (criteria.md, "Compression"). Only conv weights with more than
# WEIGHT_THRESHOLD elements are palettized: every projection/MLP matrix (>= 768 x 768) and no
# bias, norm, RoPE table or scorer output vector (all <= 4,096 elements).
WEIGHT_THRESHOLD = 65_536
KMEANS_WORKERS = int(os.environ.get("W8_KMEANS_WORKERS", "1"))
CONFIGS = {
    "w8-pt": dict(mode="kmeans", nbits=8, granularity="per_tensor"),
    "w8-gc32": dict(mode="kmeans", nbits=8, granularity="per_grouped_channel", group_size=32),
}


def palettize(mlmodel, config: str):
    from coremltools._deps import _HAS_KMEANS1D
    from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig, palettize_weights

    # FP16 weights with >= 10,000 elements take coremltools' exact 1-D k-means (kmeans1d), which
    # is deterministic. Without it coremltools silently falls back to scikit-learn's KMeans.
    if not _HAS_KMEANS1D:
        raise RuntimeError("coremltools' bundled kmeans1d is not importable; refusing the sklearn fallback")
    # Worker processes only parallelise the per-group k-means; each group's exact 1-D k-means is
    # deterministic, so the result does not depend on this (execution detail, not a config change).
    # coremltools 9.0 keeps one module-level k-means Pool and it is closed after a model's pass, so
    # a second model in the same process fails with "Pool not running". Start each cell fresh.
    from coremltools.optimize.coreml import _quantization_passes

    _quantization_passes.palettize_weights._compress_pool = None
    op = OpPalettizerConfig(weight_threshold=WEIGHT_THRESHOLD, num_kmeans_workers=KMEANS_WORKERS, **CONFIGS[config])
    return palettize_weights(mlmodel, OptimizationConfig(op_type_configs={"conv": op}))


def program_inventory(mlmodel) -> dict:
    """Which conv weights were palettized and which stayed dense (from the MIL program)."""
    prog = getattr(mlmodel, "_mil_program", None)
    if prog is None:
        return {"error": "no MIL program on the palettized model"}
    lut, dense = [], []
    for fn in prog.functions.values():
        for op in fn.operations:
            if op.op_type == "conv":
                w = op.weight
                src = w.op.op_type if w.op is not None else "input"
                entry = {"conv": op.name, "shape": [int(d) for d in w.shape], "source_op": src}
                (lut if src.startswith("constexpr_lut_to_dense") else dense).append(entry)
    return {
        "conv_palettized": len(lut),
        "conv_dense": len(dense),
        "dense": dense,
        "palettized_shapes": sorted({str(e["shape"]) for e in lut}),
    }


def build_cell(spec, length: int, config: str, root: Path) -> dict:
    import coremltools as ct
    import torch

    from laya_apple.artifacts import COMPILED, check_ane_placement, compute_plan_summary
    from laya_apple.backends.coreml_ane import HostWeights
    from laya_apple.conversion.bc1s import ConvBody
    from laya_apple.conversion.build import DEPLOYMENT_TARGET, INPUTS, _layout_check
    from laya_apple.conversion.torch_reference import load_model
    from laya_apple.hub import checkpoint_path, verify_weights
    from laya_apple.prompt import Tokenizer
    from laya_apple.registry import ANE_COMPUTE_UNITS, ANE_MAX_OPTIONS

    name = cell_name(length, config)
    out = root / spec.name / spec.revision[:12] / f"bc1s-masked-{config}-L{length}-B1"
    rec = {
        "experiment": "ane-w8/build",
        "model": spec.name,
        "revision": spec.revision,
        "length": length,
        "config": config,
        "palettizer": CONFIGS[config]
        | {"weight_threshold": WEIGHT_THRESHOLD, "op_type": "conv", "num_kmeans_workers": KMEANS_WORKERS},
        "started_at": now(),
        "environment": environment(),
        "artifact_dir": "<research-root>/" + str(out.relative_to(root)),
        "timings": {},
        "step": "start",
    }
    raw = RAW / spec.name / name
    try:
        rec["step"] = "checkpoint"
        ckpt = checkpoint_path(spec, local_files_only=True)
        verify_weights(spec, ckpt)
        cfg = json.loads((ckpt / "rl_agent_config.json").read_text())
        enc_cfg = json.loads((ckpt / "encoder/config.json").read_text())
        tok = Tokenizer(ckpt / "tokenizer")
        host = HostWeights(ckpt, int(enc_cfg["local_attention"]))

        rec["step"] = "layout"
        torch.set_num_threads(max(1, (os.cpu_count() or 8) // 2))
        source = load_model(ckpt, length, attention_implementation="explicit")
        body = ConvBody(source, length).eval()
        rec["layout_check_max_abs_logit_fp32"] = layout = _layout_check(spec, source, body, tok, cfg, length, host)
        if not layout < 1e-3:
            raise RuntimeError(f"layout check failed: {layout}")

        rec["step"] = "convert"
        width = body.embedding_norm.weight.shape[1]
        example = (
            torch.randn(1, width, 1, length),
            torch.zeros(1, length, 1, length),
            torch.zeros(1, length, 1, length),
            torch.zeros(1, width, 1, 1),
            torch.zeros(1, length, 1, ANE_MAX_OPTIONS),
        )
        t = time.perf_counter()
        with torch.inference_mode():
            traced = torch.jit.trace(body, example, strict=True, check_trace=False)
        mlmodel = ct.convert(
            traced,
            source="pytorch",
            convert_to="mlprogram",
            inputs=[ct.TensorType(name=n, shape=tuple(v.shape), dtype=np.float16) for n, v in zip(INPUTS, example)],
            outputs=[ct.TensorType(name="logits"), ct.TensorType(name="cls")],
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=getattr(ct.target, DEPLOYMENT_TARGET),
            skip_model_load=True,
        )
        rec["timings"]["convert_s"] = time.perf_counter() - t
        del traced, source, body

        rec["step"] = "palettize"
        t = time.perf_counter()
        mlmodel = palettize(mlmodel, config)
        rec["timings"]["palettize_s"] = time.perf_counter() - t
        try:
            rec["inventory"] = program_inventory(mlmodel)
        except Exception as e:  # diagnostic only
            rec["inventory"] = {"error": f"{type(e).__name__}: {e}"}

        rec["step"] = "compile"
        if out.exists():
            shutil.rmtree(out)  # our own research output from an earlier attempt of this cell
        out.mkdir(parents=True)
        package = out / "model.mlpackage"
        t = time.perf_counter()
        mlmodel.save(str(package))
        del mlmodel
        rec["package_bytes"] = tree_bytes(package)
        compiled_tmp = ct.utils.compile_model(str(package))
        shutil.move(str(compiled_tmp), str(out / COMPILED))
        shutil.rmtree(package)
        rec["timings"]["save_compile_s"] = time.perf_counter() - t
        compiled = out / COMPILED
        rec["compiled_bytes"] = tree_bytes(compiled)
        rec["weight_bin_bytes"] = weight_bytes(compiled)

        rec["step"] = "placement"
        t = time.perf_counter()
        rec["placement"] = placement = compute_plan_summary(compiled, ANE_COMPUTE_UNITS)
        rec["plan_by_op_type"] = plan_detail(compiled, ANE_COMPUTE_UNITS)
        rec["timings"]["plan_s"] = time.perf_counter() - t
        try:
            check_ane_placement(placement)
            rec["placement_gate"] = "PASS"
        except Exception as e:  # ComputeUnitMismatchError: recorded, parity still measured as a diagnostic
            rec["placement_gate"] = "FAIL"
            rec["placement_error"] = str(e)

        rec["step"] = "parity"
        rec["parity"] = parity(spec, compiled, length, ckpt, raw / "parity_rows.jsonl")
        rec["timings"]["parity_s"] = rec["parity"]["seconds"]
        rec["parity_gate"] = "PASS" if rec["parity"]["passed"] else "FAIL"
        rec["step"] = "done"
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()
        log(f"{spec.name} {name}: FAILED at {rec['step']}: {rec['error']}")
    rec["finished_at"] = now()
    write_json(raw / "build.json", rec)
    return rec


def baseline_sizes(spec, length: int) -> dict:
    """Sizes of the shipped FP16 artifact (production cache, read only)."""
    from laya_apple.artifacts import COMPILED, artifact_dir, read_manifest

    d = artifact_dir(spec, length)
    m = read_manifest(spec, length)
    return {
        "experiment": "ane-w8/baseline-size",
        "model": spec.name,
        "length": length,
        "artifact_sha256": m["integrity"]["artifact_sha256"],
        "laya_apple_version": m["conversion"]["laya_apple_version"],
        "compiled_bytes": tree_bytes(d / COMPILED),
        "weight_bin_bytes": weight_bytes(d / COMPILED),
    }


def main() -> None:
    from laya_apple.registry import resolve

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", choices=sorted(CONFIGS), required=True)
    ap.add_argument("--model", choices=sorted(CELLS), action="append")
    ap.add_argument("--length", type=int, action="append")
    ap.add_argument("--root", help="research artifact root (default /Volumes/Data/cache/laya-apple-research/w8)")
    ap.add_argument("--only-failed-primary", action="store_true", help="w8-gc32 only: cells whose w8-pt parity failed")
    args = ap.parse_args()
    root = research_root(args.root)
    for model in args.model or list(CELLS):
        spec = resolve(model)
        for length in CELLS[model]:
            if args.length and length not in args.length:
                continue
            write_json(RAW / model / f"L{length}-fp16" / "baseline_size.json", baseline_sizes(spec, length))
            if args.only_failed_primary:
                prim = RAW / model / cell_name(length, "w8-pt") / "build.json"
                if not prim.exists():
                    log(f"{model} L{length}: no w8-pt record; skipped")
                    continue
                p = json.loads(prim.read_text())
                if p.get("parity_gate") != "FAIL":
                    log(f"{model} L{length}: w8-pt parity {p.get('parity_gate')}; secondary not run")
                    continue
            done = RAW / model / cell_name(length, args.config) / "build.json"
            if done.exists() and json.loads(done.read_text()).get("step") == "done":
                log(f"{model} L{length} {args.config}: already built; not rebuilt")
                continue
            log(f"{model} L{length} {args.config}: building")
            rec = build_cell(spec, length, args.config, root)
            log(
                f"{model} L{length} {args.config}: step={rec['step']} placement={rec.get('placement_gate')} "
                f"parity={rec.get('parity_gate')} weights={rec.get('weight_bin_bytes')}"
            )


if __name__ == "__main__":
    main()
