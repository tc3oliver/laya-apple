"""Shared definitions for research/intra-request-split/ (research only; never imported by laya_apple).

Workloads, fixtures, how the harness opens the product instance, and the FP16 answer comparison.
The numbers that gate the verdict are in ../criteria.json, not here.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRACK = HERE.parent
ROOT = TRACK.parents[1]
RAW = TRACK / "raw"
CRITERIA = TRACK / "criteria.json"

# name -> (questions, tokens): the longest prompt row is exactly `tokens` long
# (laya_apple.workload.make_request), the questions cycle through its QUESTION_POOL.
WORKLOADS = {"8x128": (8, 128), "8x512": (8, 512), "1x512": (1, 512), "32x64": (32, 64)}
SEEDS = (0, 1, 2, 3)  # request variants per workload, cycled request by request
ARMS = ("split", "gpu", "ane")  # candidate, MLX-only (today's Laya.submit), ANE-only

# The auto tie band laya-apple 1.5.0 loads (explicit-only buckets that queue-aware auto may use):
# pinned so that loading an extra bucket for the split never changes single-question routing.
TIE_BUCKETS_1_5 = {"laya": (), "laya-typed-decisions": (), "laya-multilingual": (256,)}


def criteria() -> dict:
    return json.loads(CRITERIA.read_text())


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def bench_serve():
    """scripts/bench_serve.py (its timetable, request builder and FP16 answer comparison)."""
    return load_by_path("split_bench_serve", ROOT / "scripts" / "bench_serve.py")


def git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def environment() -> dict:
    import laya_apple
    from laya_apple.artifacts import platform_profile

    env = {
        "laya_apple": laya_apple.__version__,
        "git": git_revision(),
        "python": platform.python_version(),
        "platform": platform_profile(),
        "loadavg": list(os.getloadavg()),
    }
    for pkg in ("mlx", "coremltools", "numpy", "torch"):
        try:
            from importlib.metadata import version

            env[pkg] = version(pkg)
        except Exception:
            env[pkg] = None
    try:
        env["pmset_therm"] = subprocess.check_output(["pmset", "-g", "therm"], text=True).strip().splitlines()
    except Exception:
        env["pmset_therm"] = None
    return env


def open_laya(model: str, extra_buckets=()):
    """Laya(device="auto", execution="workers") exactly as the product builds it, with `extra_buckets`
    added to the model's explicit ANE buckets in this process only (so its ANE worker loads them),
    and the auto tie band pinned to laya-apple 1.5.0's. Auto routing of every single-question
    request is therefore unchanged. The registry entry is replaced in memory; nothing on disk
    changes. A bucket already offered by the installed laya-apple is left as it is."""
    from laya_apple import registry
    from laya_apple.model import Laya

    table = registry.models()  # functools.cache: the dict every resolve() reads
    spec = table[model]
    add = tuple(b for b in extra_buckets if b not in spec.ane_buckets)
    if add:
        table[model] = dataclasses.replace(spec, ane_buckets=tuple(sorted(spec.ane_buckets + add)))
    laya = Laya.from_pretrained(model, device="auto", execution="workers", local_files_only=True)
    laya._tie_buckets = tuple(b for b in laya._tie_buckets if b in TIE_BUCKETS_1_5[model])
    info = laya.info()
    if not info["ane_ready"]:
        laya.close()
        raise SystemExit(f"{model}: the ANE is not ready ({info['auto_ane']}); refusing to measure a fallback")
    missing = [b for b in extra_buckets if b not in info["ane_buckets"]]
    if missing:
        laya.close()
        raise SystemExit(
            f"{model}: ANE buckets {missing} did not load ({info['ane_load_errors']}); build them with "
            f"`laya-apple artifacts build {model} --length L` first"
        )
    return laya


def fixtures_path(model: str) -> Path:
    return RAW / f"fixtures-{model}.json"


def make_fixtures(model: str) -> dict:
    """Every workload request: state, questions and this runtime's token lengths per row."""
    from laya_apple.hub import checkpoint_path
    from laya_apple.prompt import Tokenizer, prepare
    from laya_apple.registry import resolve
    from laya_apple.workload import make_request

    spec = resolve(model)
    ckpt = checkpoint_path(spec, local_files_only=True)
    tok = Tokenizer(ckpt / "tokenizer")
    cfg = json.loads((ckpt / "rl_agent_config.json").read_text())
    out = {"model": model, "revision": spec.revision, "workloads": {}}
    for name, (q, length) in WORKLOADS.items():
        if length > spec.max_len:
            continue
        reqs = []
        for seed in SEEDS:
            state, questions = make_request(tok, cfg, length, n_questions=q, seed=seed)
            prep = prepare(tok, cfg, state, questions)
            reqs.append(
                {
                    "seed": seed,
                    "state": state,
                    "questions": questions,
                    "question_ids": list(prep.question_ids),
                    "lengths": [len(it["ids"]) for it in prep.items],
                    "options": [len(it["markers"]) for it in prep.items],
                }
            )
        out["workloads"][name] = {"questions": q, "tokens": length, "requests": reqs}
    return out


def load_fixtures(model: str) -> dict:
    path = fixtures_path(model)
    if not path.exists():
        raise SystemExit(f"{path} is missing: run scripts/fixtures.py --model {model} first")
    return json.loads(path.read_text())


def compare_answers(ref: dict, got: dict) -> dict:
    """The FP16 gate between two answers objects (scripts/bench_serve.compare_answers)."""
    return bench_serve().compare_answers(ref, got)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_default) + "\n")
    tmp.replace(path)  # written once, when complete


def _default(o):
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serialisable")
