"""Shared pieces of the windowed-attention track: environment guard, cell paths, sizes, plan
detail, parity. (Same helpers as research/ane-w8/scripts/common.py; only the track constants
differ.)

Research only: `laya_apple` never imports this. Everything here calls the production code
paths unchanged (conversion.bc1s, parity.evaluate, parity.ane.ane_parity, artifacts.*,
backends.coreml_ane.*); nothing is re-implemented except the recording wrapper around the
parity forward, which copies ane_parity's forward and is cross-checked against it.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

TRACK = "windowed"
DEFAULT_ROOT = "/Volumes/Data/cache/laya-apple-research/windowed"
REPO = Path(__file__).resolve().parents[3]
RAW = Path(__file__).resolve().parents[1] / "raw"

# Preregistered cells (criteria.md, "Cells"). Not read from the registry, so a routing or bucket
# change elsewhere cannot change what this experiment measures.
SHIPPED = (64, 96, 128)  # #15: baseline = the shipped artifact in LAYA_APPLE_CACHE
LONG = (256, 512)  # #14: baseline = a masked artifact built here by the same script
CELLS = {
    "laya": SHIPPED + LONG,
    "laya-typed-decisions": SHIPPED + LONG,
}
BLOCK = 64  # query block of the exact block-local rewrite (the Phase -1 value; not tuned)


def log(msg: str) -> None:
    print(f"[{TRACK}] {msg}", file=sys.stderr, flush=True)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def research_root(arg: str | None) -> Path:
    """The research artifact root; refuses anything inside the production cache."""
    cache = os.environ.get("LAYA_APPLE_CACHE")
    if not cache:
        sys.exit("set LAYA_APPLE_CACHE to the production cache (baselines are read from it; never the default)")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        sys.exit("set HF_HUB_OFFLINE=1 (the pinned checkpoints must already be in HF_HOME)")
    root = Path(arg or DEFAULT_ROOT).expanduser().resolve()
    prod = Path(cache).expanduser().resolve()
    if root == prod or prod in root.parents or root in prod.parents:
        sys.exit(f"research root {root} overlaps the production cache {prod}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def cell_name(length: int, config: str) -> str:
    return f"L{length}-{config}"


def artifact_path(root: Path, spec, variant: str, length: int) -> Path:
    return root / spec.name / spec.revision[:12] / f"bc1s-{variant}-L{length}-B1"


def tree_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def weight_bytes(compiled: Path) -> int:
    return sum(f.stat().st_size for f in Path(compiled).rglob("weight.bin"))


def git_revision() -> str | None:
    try:
        rev = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD"], check=False).returncode
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return None


def environment() -> dict:
    from laya_apple import __version__
    from laya_apple.artifacts import platform_profile
    from laya_apple.conversion.build import _package_versions

    return {
        "laya_apple": __version__,
        "git_revision": git_revision(),
        "python": platform.python_version(),
        "packages": _package_versions(),
        "platform": platform_profile(),
        "loadavg": os.getloadavg(),
    }


def plan_detail(compiled: Path, compute_units: str = "CPU_AND_NE") -> dict:
    """Per-operator-type device counts from the Core ML compute plan (diagnostic only; the gate
    is laya_apple.artifacts.compute_plan_summary + check_ane_placement, unchanged)."""
    import coremltools as ct

    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        str(compiled), compute_units=getattr(ct.ComputeUnit, compute_units)
    )
    out: dict[str, dict[str, int]] = {}

    def visit(block):
        for op in block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            dev = type(usage.preferred_compute_device).__name__ if usage is not None else "none"
            row = out.setdefault(op.operator_name, {})
            row[dev] = row.get(dev, 0) + 1
            for nested in op.blocks:
                visit(nested)

    for fn in plan.model_structure.program.functions.values():
        visit(fn.block)
    return out


def item_key(it: dict) -> str:
    blob = json.dumps([list(map(int, it["ids"])), list(map(int, it["markers"])), int(it["qtype"])])
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def parity(spec, compiled: Path, length: int, ckpt: Path, raw_out: Path) -> dict:
    """The verdict of record is production `ane_parity` (unchanged gate). A second pass with the
    same forward records every golden row's raw outputs to `raw_out` (JSONL) for the per-row
    tables; analyze.py recomputes the per-row metrics from it and checks they reproduce the
    summary of record."""
    import coremltools as ct

    from laya_apple.backends.coreml_ane import HostWeights, ane_features
    from laya_apple.parity import evaluate
    from laya_apple.parity.ane import ane_parity
    from laya_apple.prompt import Tokenizer, prepare
    from laya_apple.registry import ANE_COMPUTE_UNITS, ANE_MAX_OPTIONS, ANE_PRECISION

    t = time.perf_counter()
    summary = ane_parity(spec, compiled, length, ckpt)
    summary["seconds"] = time.perf_counter() - t

    cfg = json.loads((ckpt / "rl_agent_config.json").read_text())
    enc = json.loads((ckpt / "encoder/config.json").read_text())
    tok = Tokenizer(ckpt / "tokenizer")
    host = HostWeights(ckpt, int(enc["local_attention"]))
    model = ct.models.CompiledMLModel(str(compiled), compute_units=getattr(ct.ComputeUnit, ANE_COMPUTE_UNITS))
    seen: dict[str, dict] = {}

    def forward(items):  # identical to laya_apple.parity.ane.ane_parity's forward, plus recording
        logits = np.full((len(items), ANE_MAX_OPTIONS), -1e4, np.float32)
        acts = []
        for r, it in enumerate(items):
            feats = ane_features(
                [it], length, 1, host.embedding, host.type_embedding, host.window(length), tok.pad_token_id
            )
            lg, ac = host.tail(model.predict(feats), [it])
            logits[r] = lg[0]
            acts.append(ac[0])
            seen.setdefault(
                item_key(it),
                {
                    "key": item_key(it),
                    "tokens": len(it["ids"]),
                    "logits": lg[0].tolist(),
                    "action_logits": ac[0].tolist(),
                },
            )
        return logits, np.stack(acts)

    second = evaluate(
        spec.name,
        cfg,
        forward,
        precision=ANE_PRECISION,
        max_len=length,
        prepare=lambda s, q: prepare(tok, cfg, s, q).items,
    )
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    with raw_out.open("w") as f:
        for rec in seen.values():
            f.write(json.dumps(rec) + "\n")
    summary["recording_pass"] = {
        k: second[k] for k in ("rows", "prob_max_abs", "action_prob_max_abs", "hard_mismatches", "passed")
    }
    summary["recording_pass_agrees"] = all(summary[k] == second[k] for k in ("rows", "hard_mismatches", "passed"))
    return summary


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str) + "\n")
    os.replace(tmp, path)
