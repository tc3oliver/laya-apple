"""Build and gate one BC1S artifact per preregistered cell and variant (criteria.md).

    uv run --extra ane --extra convert python research/ane-windowed-attention/scripts/build_windowed.py \
        --variant windowed|masked [--model laya] [--length 256] [--root <research-root>]

Variants:
  windowed  the production ConvBody with the local layers' attention replaced by the exact
            block-local rewrite (windowed_body.py), at every cell (64/96/128/256/512);
  masked    the production ConvBody unchanged, at 256/512 only: the paired baseline for the
            long cells, built by the same script so the two arms differ only in the rewrite.
            (At 64/96/128 the baseline is the shipped artifact in LAYA_APPLE_CACHE.)

Per cell, in one process (CPU-heavy conversion, then the ANE for parity; not timing-sensitive):
pinned checkpoint + weight hash, PyTorch FP32 reference, body, the production FP32 layout check
(< 1e-3, against the PyTorch reference), an FP32 windowed-vs-masked exactness diagnostic, trace
and `ct.convert` with the production arguments, compile, the unchanged placement gate
(compute_plan_summary + check_ane_placement), the first CPU_AND_NE load time (the on-device ANE
compile), and the unchanged parity gate (ane_parity) plus a recording pass. Nothing is
registered. Record: raw/<model>/L<L>-<variant>/build.json.
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
    BLOCK,
    CELLS,
    LONG,
    RAW,
    artifact_path,
    cell_name,
    environment,
    log,
    now,
    parity,
    plan_detail,
    research_root,
    tree_bytes,
    write_json,
)
from windowed_body import dense_score_fraction, windowed_body  # noqa: E402

VARIANTS = ("windowed", "masked")


def _body_logits(body, feats: dict, inputs, k: int):
    import torch

    with torch.inference_mode():
        got, cls = body(*[torch.from_numpy(np.asarray(feats[n], np.float32)) for n in inputs])
    return got.reshape(-1).numpy()[:k], cls.reshape(-1).numpy()


def exactness(spec, masked, windowed, tok, cfg, host, length: int) -> dict:
    """FP32 windowed vs FP32 masked body: max |d logit| and |d CLS| on an exact-length row and on
    the shortest golden row padded to L (the padded-query path). Diagnostic; the gate is layout +
    parity."""
    from laya_apple.backends.coreml_ane import ane_features
    from laya_apple.conversion.build import INPUTS
    from laya_apple.parity import load_goldens
    from laya_apple.prompt import prepare
    from laya_apple.workload import make_request

    state, questions = make_request(tok, cfg, length, n_questions=1, seed=1)
    exact = prepare(tok, cfg, state, questions).items[0]
    shortest = min((it for c in load_goldens(spec.name)["cases"] for it in c["items"]), key=lambda it: len(it["ids"]))
    out = {}
    for tag, it in (("exact", exact), ("padded", shortest)):
        feats = ane_features(
            [it],
            length,
            1,
            host.embedding.astype(np.float32),
            host.type_embedding.astype(np.float32),
            host.window(length),
            tok.pad_token_id,
        )
        k = len(it["markers"])
        a, ca = _body_logits(masked, feats, INPUTS, k)
        b, cb = _body_logits(windowed, feats, INPUTS, k)
        out[tag] = {
            "tokens": len(it["ids"]),
            "max_abs_logit": float(np.abs(a - b).max()),
            "max_abs_cls": float(np.abs(ca - cb).max()),
        }
    return out


def build_cell(spec, length: int, variant: str, root: Path) -> dict:
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

    name = cell_name(length, variant)
    out = artifact_path(root, spec, variant, length)
    rec = {
        "experiment": "ane-windowed-attention/build",
        "model": spec.name,
        "revision": spec.revision,
        "length": length,
        "variant": variant,
        "block": BLOCK if variant == "windowed" else None,
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
        radius = int(enc_cfg["local_attention"]) // 2
        if radius != host.radius:
            raise RuntimeError(f"window radius {radius} != host mask radius {host.radius}")

        rec["step"] = "layout"
        torch.set_num_threads(max(1, (os.cpu_count() or 8) // 2))
        source = load_model(ckpt, length, attention_implementation="explicit")
        body = ConvBody(source, length).eval()
        if variant == "windowed":
            reference = ConvBody(source, length).eval()
            body, n_local = windowed_body(body, length=length, radius=radius, block=BLOCK)
            rec["windowed_layers"] = n_local
            rec["radius"] = radius
            rec["dense_score_fraction"] = dense_score_fraction(length, radius, BLOCK)
            rec["exactness_fp32_vs_masked"] = exactness(spec, reference, body, tok, cfg, host, length)
            del reference
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

        rec["step"] = "compile"
        if out.exists():
            shutil.rmtree(out)  # our own research output from an earlier attempt of this cell
        out.mkdir(parents=True)
        package = out / "model.mlpackage"
        t = time.perf_counter()
        mlmodel.save(str(package))
        del mlmodel
        compiled_tmp = ct.utils.compile_model(str(package))
        shutil.move(str(compiled_tmp), str(out / COMPILED))
        shutil.rmtree(package)
        rec["timings"]["save_compile_s"] = time.perf_counter() - t
        compiled = out / COMPILED
        rec["compiled_bytes"] = tree_bytes(compiled)

        rec["step"] = "placement"
        t = time.perf_counter()
        rec["placement"] = placement = compute_plan_summary(compiled, ANE_COMPUTE_UNITS)
        rec["plan_by_op_type"] = detail = plan_detail(compiled, ANE_COMPUTE_UNITS)
        rec["mil_ops"] = sum(sum(v.values()) for k, v in detail.items() if k != "const")
        rec["timings"]["plan_s"] = time.perf_counter() - t
        try:
            check_ane_placement(placement)
            rec["placement_gate"] = "PASS"
        except Exception as e:  # ComputeUnitMismatchError: recorded, parity still measured as a diagnostic
            rec["placement_gate"] = "FAIL"
            rec["placement_error"] = str(e)

        rec["step"] = "first_load"
        t = time.perf_counter()
        first = ct.models.CompiledMLModel(str(compiled), compute_units=getattr(ct.ComputeUnit, ANE_COMPUTE_UNITS))
        rec["timings"]["first_load_s"] = time.perf_counter() - t
        del first

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


def main() -> None:
    from laya_apple.registry import resolve

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--model", choices=sorted(CELLS), action="append")
    ap.add_argument("--length", type=int, action="append")
    ap.add_argument("--root", help="research artifact root (default /Volumes/Data/cache/laya-apple-research/windowed)")
    args = ap.parse_args()
    root = research_root(args.root)
    for model in args.model or list(CELLS):
        spec = resolve(model)
        for length in CELLS[model]:
            if args.length and length not in args.length:
                continue
            if args.variant == "masked" and length not in LONG:
                continue  # the shipped artifact is the baseline at 64/96/128
            log(f"{model} L{length} {args.variant}: building")
            rec = build_cell(spec, length, args.variant, root)
            log(
                f"{model} L{length} {args.variant}: step={rec['step']} placement={rec.get('placement_gate')} "
                f"parity={rec.get('parity_gate')} first_load={rec['timings'].get('first_load_s')}"
            )


if __name__ == "__main__":
    main()
