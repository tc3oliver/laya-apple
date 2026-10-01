"""Pinned checkpoint download, offline resolution and weight verification."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .errors import ArtifactRevisionError, BackendUnavailableError
from .registry import ModelSpec

CHECKPOINT_FILES = ["model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/*"]


def cache_root() -> Path:
    """laya-apple's own cache (artifacts, verification stamps). HF weights stay in HF's cache."""
    env = os.environ.get("LAYA_APPLE_CACHE")
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "laya-apple"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def offline_mode(local_files_only: bool = False) -> str | None:
    """Why Hugging Face downloads are off, or None when they are on: `local_files_only`
    (--offline), else HF_HUB_OFFLINE as huggingface_hub read it at import."""
    from huggingface_hub import constants

    if local_files_only:
        return "offline (local_files_only)"
    if constants.HF_HUB_OFFLINE:
        return "offline (HF_HUB_OFFLINE)"
    return None


def checkpoint_path(spec: ModelSpec, *, local_files_only: bool = False) -> Path:
    """Local directory of the pinned checkpoint revision (downloads unless offline)."""
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(
            spec.repo,
            revision=spec.revision,
            allow_patterns=CHECKPOINT_FILES,
            local_files_only=local_files_only,
        )
    except Exception as e:  # offline and not cached, network errors
        mode = offline_mode(local_files_only) or "online"
        raise BackendUnavailableError(
            f"cannot resolve {spec.repo}@{spec.revision[:12]} {mode}: {e}. "
            f"Run once online, or `laya-apple download {spec.name}`."
        ) from e
    return Path(path)


def verify_weights(spec: ModelSpec, path: Path) -> str:
    """Check model.safetensors against the pinned SHA-256; cached by size+mtime."""
    weights = path / "model.safetensors"
    st = weights.stat()
    stamp_dir = cache_root() / "verified"
    stamp = stamp_dir / f"{spec.name}-{spec.revision[:12]}.json"
    key = {"path": str(weights.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    if stamp.exists():
        try:
            data = json.loads(stamp.read_text())
            if data.get("key") == key and data.get("sha256") == spec.weights_sha256:
                return spec.weights_sha256
        except (OSError, ValueError):
            pass
    digest = sha256_file(weights)
    if digest != spec.weights_sha256:
        raise ArtifactRevisionError(
            f"{spec.repo}@{spec.revision[:12]} weights hash {digest[:16]}… does not match the pinned "
            f"{spec.weights_sha256[:16]}…"
        )
    stamp_dir.mkdir(parents=True, exist_ok=True)
    stamp.write_text(json.dumps({"key": key, "sha256": digest}))
    return digest
