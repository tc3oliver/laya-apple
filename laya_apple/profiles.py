"""Local capability profiles (v0.3).

The shipped routing table (laya_apple/data/routing.json) is valid only for the profile it
was measured on. On any other profile (SoC, macOS major, coremltools version), `auto`
uses MLX only, unless the user calibrates that machine:

    laya-apple artifacts build MODEL      # builds and parity-validates the ANE artifacts here
    laya-apple calibrate MODEL            # measures MLX and ANE latency here

`calibrate` applies the same rule as the shipped table (laya_apple.derivation) to local
measurements and writes `<cache>/profiles/<profile-key>.json`. The ANE artifacts
themselves already passed the parity gate on this machine when they were built. A local
profile is used only when no shipped profile matches the machine, and `Laya.info()`
reports which one is in effect.

Profiles validated after routing.json was derived ship as calibrate output in
`laya_apple/data/profiles/<profile-key>.json` (Apple M4 Max, macOS 27, coremltools 9.0).
They count as shipped, not local.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import time
import warnings
from datetime import datetime, timezone
from functools import cache
from importlib import resources
from pathlib import Path

import numpy as np

from . import __version__
from .artifacts import artifact_dir, platform_profile, profile_matches
from .derivation import derive_model
from .errors import ArtifactParityError
from .hub import cache_root
from .registry import ModelSpec, models, routing_table

PROFILE_FORMAT = "laya-apple-profile"
PROFILE_VERSION = 1
QUESTION_COUNTS = (1, 4, 8)


def profile_key(profile: dict) -> str:
    raw = f"{profile.get('soc')}-macos{str(profile.get('macos') or '').split('.')[0]}-coremltools{profile.get('coremltools')}"
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", raw)


def local_profile_path(profile: dict | None = None) -> Path:
    return cache_root() / "profiles" / f"{profile_key(profile or platform_profile())}.json"


def shipped_profile_matches(profile: dict | None = None) -> bool:
    """The shipped table (routing.json) or a shipped calibrated profile covers this machine."""
    profile = profile or platform_profile()
    if any(profile_matches(profile, p) for p in routing_table().get("validated_profiles", [])):
        return True
    return _shipped_profile(profile) is not None


def _shipped_profile(profile: dict) -> dict | None:
    """The calibrated profile shipped in laya_apple/data/profiles/ for this profile, if any.
    It is the unedited output of `laya-apple calibrate` on a release-validated machine."""
    data = _read_shipped(profile_key(profile))
    if data is None or not profile_matches(data.get("platform") or {}, profile):
        return None
    return data


@cache
def _read_shipped(key: str) -> dict | None:
    """Parsed shipped profile file for a profile key (package data, read once per process)."""
    path = resources.files("laya_apple.data").joinpath("profiles", f"{key}.json")
    if not path.is_file():
        return None
    return dict(json.loads(path.read_text()), source=f"laya_apple/data/profiles/{path.name}")


def load_shipped(model: str, profile: dict | None = None) -> dict | None:
    """The shipped calibrated profile's entry for `model` on this profile, if one exists.
    Unlike a local profile it is not tied to this cache's artifact hashes: like routing.json,
    it applies to any artifact that passed parity and placement on this profile."""
    data = _shipped_profile(profile or platform_profile())
    if data is None or model not in data.get("models", {}):
        return None
    return dict(data["models"][model], source=data["source"])


def load_local(model: str, profile: dict | None = None) -> dict | None:
    """The local calibration for `model` on this profile, if one exists and matches."""
    profile = profile or platform_profile()
    path = local_profile_path(profile)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except ValueError as e:
        warnings.warn(
            f"laya-apple: ignoring unreadable local profile {path} ({e}); re-run laya-apple calibrate",
            RuntimeWarning,
            stacklevel=3,
        )
        return None
    if (
        not isinstance(data, dict)
        or data.get("format") != PROFILE_FORMAT
        or data.get("format_version") != PROFILE_VERSION
    ):
        warnings.warn(
            f"laya-apple: ignoring local profile {path} in an unknown format; re-run laya-apple calibrate",
            RuntimeWarning,
            stacklevel=3,
        )
        return None
    if not profile_matches(data.get("platform") or {}, profile):
        return None
    entry = data.get("models", {}).get(model)
    if entry is None:
        return None
    spec = models().get(model)
    if spec is not None and _stale(spec, entry):
        warnings.warn(
            f"laya-apple: stale local profile {path} for {model} (revision or artifacts changed); "
            "re-run laya-apple calibrate",
            RuntimeWarning,
            stacklevel=3,
        )
        return None
    return dict(entry, source=str(path))


def _stale(spec: ModelSpec, entry: dict) -> bool:
    """The profile no longer matches the pinned revision or the artifacts it measured."""
    if entry.get("revision") != spec.revision:
        return True
    for bucket, expected_sha in (entry.get("artifacts") or {}).items():
        manifest_path = artifact_dir(spec, int(bucket)) / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            actual_sha = json.loads(manifest_path.read_text()).get("integrity", {}).get("artifact_sha256")
        except ValueError:
            continue
        if actual_sha and actual_sha != expected_sha:
            return True
    return False


def _p50(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t) * 1e3)
    return float(np.percentile(samples, 50))


def _parity_records(model: str, manifests: dict) -> dict:
    """bucket -> (passed, prob_max_abs) from each manifest's parity record; raises when a
    manifest was built without one (should not happen for a load_verified'd artifact)."""
    parity = {}
    for b, m in manifests.items():
        rec = m.data.get("parity")
        if not rec or "passed" not in rec or "prob_max_abs" not in rec:
            raise ArtifactParityError(f"{model} L{b} manifest has no parity record")
        parity[b] = (bool(rec["passed"]), rec["prob_max_abs"])
    return parity


def calibrate(spec: ModelSpec, *, warmup: int = 5, iters: int = 30, local_files_only: bool = False, log=print) -> dict:
    """Measure this machine and write/update its local profile for `spec`. Needs every
    offered bucket built and validated here (laya-apple artifacts build)."""
    from .artifacts import load_verified
    from .model import Laya
    from .workload import make_request

    for b in spec.ane_buckets:
        load_verified(spec, b)  # raises with the build command if an artifact is missing or invalid
    gpu = Laya.from_pretrained(spec.name, device="gpu", local_files_only=local_files_only)
    ane = Laya.from_pretrained(spec.name, device="ane", local_files_only=local_files_only)
    try:
        lengths = sorted(
            {
                *spec.ane_buckets,
                *[L for L in (64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 768, 1024) if L <= spec.max_len],
            }
        )
        service_gpu: dict = {}
        for q in QUESTION_COUNTS:
            for L in lengths if q == 1 else [L for L in lengths if L in (64, 128, 256, 512, 1024)]:
                state, qs = make_request(gpu.tokenizer, gpu.config, L, n_questions=q)
                items = gpu.prepare(state, qs).items
                service_gpu.setdefault(str(q), {})[str(L)] = round(
                    _p50(lambda: gpu.mlx.forward(items), warmup, iters), 4
                )
                log(f"{spec.name} MLX q{q} L{L}: {service_gpu[str(q)][str(L)]:.2f} ms")
        service_ane: dict = {"1": {}}
        for b in spec.ane_buckets:
            state, qs = make_request(ane.tokenizer, ane.config, b, n_questions=1)
            items = ane.prepare(state, qs).items
            service_ane["1"][str(b)] = round(_p50(lambda: ane.ane.forward(items), warmup, iters), 4)
            log(f"{spec.name} ANE L{b}: {service_ane['1'][str(b)]:.2f} ms")
        parity = _parity_records(spec.name, ane.ane.manifests)
        derived = derive_model(
            spec.ane_buckets,
            parity,
            {b: service_ane["1"][str(b)] for b in spec.ane_buckets},
            {int(L): v for L, v in service_gpu["1"].items()},
        )
        entry = {
            **derived,
            "revision": spec.revision,
            "service_ms": {"gpu": service_gpu, "ane": service_ane},
            "artifacts": {str(b): m.artifact_sha256 for b, m in ane.ane.manifests.items()},
            "measured_at": datetime.now(timezone.utc).isoformat(),
            "method": {"warmup": warmup, "iters": iters, "boundary": "backend forward, synchronised, P50"},
        }
    finally:
        gpu.close()
        ane.close()
    profile = platform_profile()
    path = local_profile_path(profile)
    _save_profile(path, profile, spec.name, entry)
    return dict(entry, path=str(path))


def _save_profile(path: Path, profile: dict, model: str, entry: dict) -> None:
    """Merge `entry` into the profile file at `path`, locked against concurrent writers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    with open(lock_path, "a+") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            data = {}
            if path.exists():
                try:
                    data = json.loads(path.read_text())
                except ValueError:
                    data = {}
            if data.get("format") != PROFILE_FORMAT:
                data = {}
            data.update(format=PROFILE_FORMAT, format_version=PROFILE_VERSION, platform=profile, laya_apple=__version__)
            data.setdefault("models", {})[model] = entry
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(data, indent=1) + "\n")
            os.replace(tmp, path)
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
