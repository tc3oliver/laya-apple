"""Build, validate and atomically register one fixed-shape ANE artifact.

    laya-apple artifacts build laya-typed-decisions --length 128

Pipeline, all inside a private staging directory on the cache filesystem:

  1. load the original checkpoint in PyTorch FP32 (verified weights) and build the BC1S body
  2. layout check: FP32 BC1S body vs the FP32 PyTorch DecisionModel on an exact-length row
  3. trace and convert: FP16 compute, FP16 I/O, fixed B=1 x L, macOS 15 target
  4. compile to model.mlmodelc (the .mlpackage is not kept)
  5. placement: Core ML compute plan on CPU_AND_NE must be 100% ANE with 0 transitions
  6. parity: every golden row that fits the bucket, against the upstream PyTorch FP32
     reference, with the unchanged Phase -1 gate
  7. write the manifest with full provenance, then rename into place atomically

A failure at any step leaves nothing registered. A parity or placement failure keeps its
manifest under artifacts/rejected/ as evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .. import __version__
from ..artifacts import (
    COMPILED,
    MANIFEST_FORMAT,
    MANIFEST_VERSION,
    artifact_dir,
    artifacts_root,
    check_ane_placement,
    compute_plan_summary,
    platform_profile,
    tree_sha256,
)
from ..errors import ArtifactError, ArtifactParityError, ComputeUnitMismatchError, UnsupportedShapeError
from ..hub import checkpoint_path, verify_weights
from ..lifecycle import BUILDING, build_lock
from ..prompt import Tokenizer, prepare
from ..registry import ANE_COMPUTE_UNITS, ANE_GRAPH, ANE_MAX_OPTIONS, ANE_PRECISION, ModelSpec
from ..workload import make_request

DEPLOYMENT_TARGET = "macOS15"
INPUTS = ("embeddings", "full_mask", "local_mask", "type_vectors", "marker_map")


def _log(msg):
    print(f"[build] {msg}", file=sys.stderr, flush=True)


def _package_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for name in ("coremltools", "torch", "numpy", "mlx", "tokenizers", "safetensors"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def conversion_code_sha256() -> dict:
    here = Path(__file__).parent
    files = [here / "bc1s.py", here / "torch_reference.py", here / "build.py", here.parent / "backends/coreml_ane.py"]
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def _git_revision() -> str | None:
    import subprocess

    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return None
    try:
        rev = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", "laya_apple"], check=False
        ).returncode
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return None


def _layout_check(spec, source, body, tok, cfg, length, host) -> float:
    """Max |logit| difference, FP32 BC1S body vs FP32 DecisionModel, one exact-length row."""
    import torch

    from ..backends.coreml_ane import ane_features

    state, questions = make_request(tok, cfg, length, n_questions=1, seed=1)
    items = prepare(tok, cfg, state, questions).items
    feats = ane_features(
        items,
        length,
        1,
        host.embedding.astype(np.float32),
        host.type_embedding.astype(np.float32),
        host.window(length),
        tok.pad_token_id,
    )
    it = items[0]
    with torch.inference_mode():
        got, _ = body(*[torch.from_numpy(np.asarray(feats[k], np.float32)) for k in INPUTS])
        got = got.reshape(-1).numpy()
        m, k = len(it["ids"]), len(it["markers"])
        ids = torch.full((1, length), tok.pad_token_id, dtype=torch.long)
        ids[0, :m] = torch.tensor(it["ids"])
        att = torch.zeros((1, length), dtype=torch.long)
        att[0, :m] = 1
        mpos = torch.zeros((1, ANE_MAX_OPTIONS), dtype=torch.long)
        mpos[0, :k] = torch.tensor(it["markers"])
        mmask = torch.zeros((1, ANE_MAX_OPTIONS), dtype=torch.bool)
        mmask[0, :k] = True
        ref, _ = source(ids, att, mpos, mmask, torch.tensor([it["qtype"]]))
    return float(np.abs(got[:k] - ref.numpy()[0, :k]).max())


def build(spec: ModelSpec, length: int, *, local_files_only: bool = False, force: bool = False) -> Path:
    """Build and register spec@length; return the artifact directory, or raise."""
    if length not in spec.ane_buckets:
        raise UnsupportedShapeError(
            f"{spec.name} L{length} is not a validated ANE configuration; offered buckets: {list(spec.ane_buckets)}"
        )
    try:
        import coremltools  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        raise ArtifactError(
            "building artifacts needs: the [ane] and [convert] extras (uv sync --extra ane --extra convert)"
        ) from e

    final = artifact_dir(spec, length)
    if final.exists() and not force:
        raise ArtifactError(f"{final} already exists; pass force=True (--force) to rebuild")
    with build_lock(spec, length, log=_log):
        if final.exists() and not force:  # another process registered it while we waited
            _log(f"{final} was built by another process")
            return final
        return _build_locked(spec, length, final, local_files_only=local_files_only, force=force)


def _build_locked(spec: ModelSpec, length: int, final: Path, *, local_files_only: bool, force: bool) -> Path:
    import coremltools as ct
    import torch

    from ..backends.coreml_ane import HostWeights
    from ..parity.ane import ane_parity
    from .bc1s import ConvBody
    from .torch_reference import load_model

    t_start = time.perf_counter()
    ckpt = checkpoint_path(spec, local_files_only=local_files_only)
    verify_weights(spec, ckpt)
    cfg = json.loads((ckpt / "rl_agent_config.json").read_text())
    enc_cfg = json.loads((ckpt / "encoder/config.json").read_text())
    tok = Tokenizer(ckpt / "tokenizer")
    host = HostWeights(ckpt, int(enc_cfg["local_attention"]))
    timings = {}

    staging_root = artifacts_root() / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f"{spec.name}-L{length}-", dir=staging_root))
    (stage / BUILDING).write_text(json.dumps({"pid": os.getpid()}))
    try:
        _log(f"{spec.name} L{length}: loading PyTorch FP32 reference")
        torch.set_num_threads(max(1, (os.cpu_count() or 8) // 2))
        source = load_model(ckpt, length, attention_implementation="explicit")
        body = ConvBody(source, length).eval()
        layout = _layout_check(spec, source, body, tok, cfg, length, host)
        _log(f"layout check (FP32 BC1S vs FP32 reference): max |dlogit| = {layout:.3g}")
        if not layout < 1e-3:
            raise ArtifactParityError(f"BC1S layout check failed: max |dlogit| {layout} >= 1e-3")

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
        timings["convert_s"] = time.perf_counter() - t
        del traced, source, body
        package = stage / "model.mlpackage"
        mlmodel.save(str(package))
        del mlmodel
        _log(f"converted in {timings['convert_s']:.1f} s; compiling")
        t = time.perf_counter()
        compiled_tmp = ct.utils.compile_model(str(package))
        shutil.move(str(compiled_tmp), str(stage / COMPILED))
        shutil.rmtree(package)
        timings["compile_s"] = time.perf_counter() - t

        placement = compute_plan_summary(stage / COMPILED, ANE_COMPUTE_UNITS)
        _log(f"placement: {placement['ops']} transitions={placement['transitions']}")
        manifest = {
            "format": MANIFEST_FORMAT,
            "format_version": MANIFEST_VERSION,
            "status": "building",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "model": spec.name,
                "repo": spec.repo,
                "revision": spec.revision,
                "weights_sha256": spec.weights_sha256,
            },
            "conversion": {
                "laya_apple_version": __version__,
                "git_revision": _git_revision(),
                "code_sha256": conversion_code_sha256(),
                "packages": _package_versions(),
                "minimum_deployment_target": DEPLOYMENT_TARGET,
                "compute_precision": "FLOAT16",
                "io_dtype": "float16",
                "layout_check_max_abs_logit_fp32": layout,
            },
            "artifact": {
                "graph": ANE_GRAPH,
                "length": length,
                "batch": 1,
                "precision": ANE_PRECISION,
                "max_options": ANE_MAX_OPTIONS,
                "inputs": list(INPUTS),
                "outputs": ["logits", "cls"],
            },
            "placement": {"compute_units": ANE_COMPUTE_UNITS, **placement},
            "platform": platform_profile(),
            "parity": None,
            "integrity": {},
            "timings": timings,
        }
        try:
            check_ane_placement(placement)
        except ComputeUnitMismatchError as e:
            _reject(spec, length, manifest, f"placement: {e}")
            raise

        _log("parity gate against the PyTorch FP32 goldens")
        t = time.perf_counter()
        parity = ane_parity(spec, stage / COMPILED, length, ckpt)
        timings["parity_s"] = time.perf_counter() - t
        manifest["parity"] = parity
        _log(
            f"parity rows={parity['rows']} prob={parity['prob_max_abs']:.4f} act={parity['action_prob_max_abs']:.4f} "
            f"hard={parity['hard_mismatches']} passed={parity['passed']}"
        )
        if not parity["passed"]:
            _reject(spec, length, manifest, "parity gate failed")
            raise ArtifactParityError(f"{spec.name} L{length} failed the parity gate: {json.dumps(parity)[:800]}")

        manifest["integrity"] = {"artifact_sha256": tree_sha256(stage / COMPILED), "tree_hash": "laya-apple tree v1"}
        manifest["status"] = "validated"
        timings["total_s"] = time.perf_counter() - t_start
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

        final.parent.mkdir(parents=True, exist_ok=True)
        old = None
        if final.exists():  # force: move the old one aside first, then swap
            old = final.with_name(final.name + f".old-{os.getpid()}")
            os.rename(final, old)
        try:
            # The marker only guards the staging directory; a registered artifact must not carry it.
            (stage / BUILDING).unlink(missing_ok=True)
            os.rename(stage, final)
        except BaseException:
            # The rename into place is the only way the staging directory leaves, so its presence
            # says whether the new artifact got registered. If it did (an interrupt right after
            # the rename), it stays registered and the previous one is kept beside it; otherwise
            # the previous artifact is put back. The original error is raised either way.
            if old is not None and stage.exists():
                try:
                    os.rename(old, final)
                except OSError as e:
                    _log(f"{spec.name} L{length}: could not restore the previous artifact; it was left at {old}: {e}")
            elif old is not None:
                _log(f"{spec.name} L{length}: interrupted after registering {final}; the previous artifact is at {old}")
            raise
        if old is not None:
            try:
                shutil.rmtree(old)
            except OSError as e:  # the new artifact is registered; only cleanup failed
                _log(
                    f"{spec.name} L{length}: registered, but could not remove the replaced copy {old}: {e}. "
                    "Remove it by hand once the new one works; `laya-apple artifacts prune` lists it but keeps it."
                )
        _log(f"registered {final} ({timings['total_s']:.0f} s)")
        # Core ML's on-device ANE compile is cached per model location, so the load at the
        # staging path does not cover the registered path. Pay that compile now, through the
        # same verified loader the runtime uses, instead of on the first user request.
        t = time.perf_counter()
        from ..artifacts import load_verified

        load_verified(spec, length, full=True)
        _log(f"pre-warmed the ANE compile at the registered path in {time.perf_counter() - t:.1f} s")
        return final
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def _reject(spec, length, manifest, why):
    manifest = dict(manifest, status="rejected", rejected_because=why)
    d = artifacts_root() / "rejected"
    d.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (d / f"{spec.name}-{spec.revision[:12]}-L{length}-{stamp}.json").write_text(json.dumps(manifest, indent=1) + "\n")
