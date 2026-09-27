"""Check that the k-means Pool reset in build_w8.palettize is result-neutral.

    W8_KMEANS_WORKERS=8 uv run --extra ane --extra convert python research/ane-w8/scripts/check_pool_neutrality.py

Converts laya L64 with the production pipeline, then palettizes it with w8-gc32 twice in
this one process: the first call creates coremltools' module-level Pool, and the second call
runs only because palettize() resets it (without the reset it fails with "Pool not running").
Both results are compiled and their weight.bin SHA-256 is compared with the laya L64 w8-gc32
artifact from the build run. Nothing is gated or registered. The result goes to
raw/pool_neutrality.json.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_w8 import KMEANS_WORKERS, palettize  # noqa: E402
from common import RAW, now, research_root, write_json  # noqa: E402

MODEL, LENGTH, CONFIG = "laya", 64, "w8-gc32"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def convert(spec, ckpt):
    import coremltools as ct
    import torch

    from laya_apple.conversion.bc1s import ConvBody
    from laya_apple.conversion.build import DEPLOYMENT_TARGET, INPUTS
    from laya_apple.conversion.torch_reference import load_model
    from laya_apple.registry import ANE_MAX_OPTIONS

    source = load_model(ckpt, LENGTH, attention_implementation="explicit")
    body = ConvBody(source, LENGTH).eval()
    width = body.embedding_norm.weight.shape[1]
    example = (
        torch.randn(1, width, 1, LENGTH),
        torch.zeros(1, LENGTH, 1, LENGTH),
        torch.zeros(1, LENGTH, 1, LENGTH),
        torch.zeros(1, width, 1, 1),
        torch.zeros(1, LENGTH, 1, ANE_MAX_OPTIONS),
    )
    with torch.inference_mode():
        traced = torch.jit.trace(body, example, strict=True, check_trace=False)
    return ct.convert(
        traced,
        source="pytorch",
        convert_to="mlprogram",
        inputs=[ct.TensorType(name=n, shape=tuple(v.shape), dtype=np.float16) for n, v in zip(INPUTS, example)],
        outputs=[ct.TensorType(name="logits"), ct.TensorType(name="cls")],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=getattr(ct.target, DEPLOYMENT_TARGET),
        skip_model_load=True,
    )


def main() -> None:
    import coremltools as ct

    from laya_apple.artifacts import COMPILED
    from laya_apple.hub import checkpoint_path
    from laya_apple.registry import resolve

    root = research_root(None)
    spec = resolve(MODEL)
    ckpt = checkpoint_path(spec, local_files_only=True)
    reference = root / spec.name / spec.revision[:12] / f"bc1s-masked-{CONFIG}-L{LENGTH}-B1" / COMPILED
    work = root / "_pool_neutrality"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    rec = {
        "started_at": now(),
        "cell": f"{MODEL} L{LENGTH} {CONFIG}",
        "num_kmeans_workers": KMEANS_WORKERS,
        "reference_weight_bin_sha256": sha(reference / "weights" / "weight.bin"),
        "calls": [],
    }
    for call in (1, 2):
        compressed = palettize(convert(spec, ckpt), CONFIG)
        package = work / f"call{call}.mlpackage"
        compressed.save(str(package))
        compiled = Path(ct.utils.compile_model(str(package)))
        rec["calls"].append(
            {
                "call": call,
                "package_weight_bin_sha256": sha(package / "Data/com.apple.CoreML/weights/weight.bin"),
                "compiled_weight_bin_sha256": sha(compiled / "weights" / "weight.bin"),
            }
        )
        shutil.rmtree(package)
        shutil.rmtree(compiled)
    rec["identical_to_reference"] = all(
        c["compiled_weight_bin_sha256"] == rec["reference_weight_bin_sha256"] for c in rec["calls"]
    )
    rec["finished_at"] = now()
    shutil.rmtree(work)
    write_json(RAW / "pool_neutrality.json", rec)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
