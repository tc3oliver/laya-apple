"""Research prototype for #162, mechanism A: validate a first install at the registered path.

`import_artifact` on main (laya_apple/lifecycle.py at 484a645) runs the compute plan, the parity
gate and the staged placement probe on the staging copy, renames it into place, then loads and
probes it again at the registered path. Core ML's on-device ANE compile is tied to the path, so
the import pays it twice (#121, Phase B). This prototype runs the same checks once, on the model
at the registered path, while `manifest.json` is withheld, and publishes the manifest last:

1. extract into staging and run the static checks there (archive members, manifest, `expect`,
   platform profile, file hash), exactly as main does;
2. under `build_lock`, rename the staging directory into the registered path with the manifest
   kept as `manifest.pending.json` and a `PENDING.json` marker holding this pid; the runtime's
   `read_manifest` sees no `manifest.json` and raises ArtifactMissingError;
3. at the registered path: the compute plan, the weight check, the full parity gate and one
   placement probe (`registered_probe` when given, else `probe`);
4. publish with `os.replace(manifest.pending.json -> manifest.json)`, then unlink the marker and
   write the verification stamp `load_verified` would have written for the same files.

Any failure in 2-4 renames the directory to `.failed-<pid>` and removes it, so the registered
path is left absent. A pending directory left by an import that died (SIGKILL) is found under
the build lock, which no live import of the same key can hold, and removed the same way before
the new install. `force=True` (replace) is not part of mechanism A and delegates to main's
`import_artifact` unchanged.

Research only: this imports laya_apple, never the reverse. Differences from main are listed in
README.md ("Differences from `import_artifact`").
"""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

PENDING = "PENDING.json"
PENDING_MANIFEST = "manifest.pending.json"


def _abandoned_pending(final: Path) -> bool:
    """A registered path holding an unpublished install (marker, no manifest). Only meaningful
    under build_lock: the import that wrote it cannot still be running."""
    return final.is_dir() and not (final / "manifest.json").exists() and (final / PENDING).exists()


def _remove_aside(final: Path, log) -> None:
    """Free the registered path with one rename, then remove the renamed copy."""
    failed = final.with_name(final.name + f".failed-{os.getpid()}")
    if final.exists():
        try:
            os.rename(final, failed)
        except OSError as e:
            log(f"could not move {final} aside: {e}")
    shutil.rmtree(failed, ignore_errors=True)


def proto_import_artifact(
    archive: Path,
    *,
    local_files_only: bool = False,
    force: bool = False,
    log=print,
    source: str | None = None,
    probe=None,
    registered_probe=None,
    expect: tuple[str, int] | None = None,
) -> Path:
    """Same signature and checks as laya_apple.lifecycle.import_artifact; see the module doc."""
    import tarfile
    import tempfile

    from laya_apple.lifecycle import BUILDING, MAX_IMPORT_BYTES, build_lock, import_artifact

    if force:  # replace stays on main's path (#162, decision 2)
        return import_artifact(
            archive,
            local_files_only=local_files_only,
            force=True,
            log=log,
            source=source,
            probe=probe,
            registered_probe=registered_probe,
            expect=expect,
        )

    # Looked up at call time, as main does, so a harness can time or wrap the same functions.
    from laya_apple.artifacts import (
        COMPILED,
        _stamp_key,
        _stamp_path,
        artifact_dir,
        artifacts_root,
        check_ane_placement,
        compute_plan_summary,
        platform_profile,
        tree_sha256,
        verify_files,
        verify_manifest,
        verify_profile,
    )
    from laya_apple.errors import ArtifactError, ArtifactIntegrityError, ArtifactParityError, BackendUnavailableError
    from laya_apple.hub import checkpoint_path, verify_weights
    from laya_apple.parity.ane import ane_parity
    from laya_apple.registry import ANE_COMPUTE_UNITS, resolve

    staging_root = artifacts_root() / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="import-", dir=staging_root))
    (stage / BUILDING).write_text(json.dumps({"pid": os.getpid()}))
    try:
        if not hasattr(tarfile, "data_filter"):
            raise BackendUnavailableError(
                "artifacts import needs Python >= 3.11.4, for tarfile's 'data' extraction filter"
            )
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            names = {m.name for m in members}
            if "manifest.json" not in names or not any(n == COMPILED or n.startswith(COMPILED + "/") for n in names):
                raise ArtifactError(f"{archive} is not a laya-apple artifact export")
            total = 0
            for m in members:
                if m.name != "manifest.json" and m.name != COMPILED and not m.name.startswith(COMPILED + "/"):
                    raise ArtifactError(f"{archive} contains an unexpected member: {m.name}")
                if m.issym() or m.islnk():
                    raise ArtifactError(f"{archive} contains a link member: {m.name}")
                if m.isfile():
                    total += m.size
            if total > MAX_IMPORT_BYTES:
                raise ArtifactError(f"{archive} is larger than the {MAX_IMPORT_BYTES}-byte import limit")
            tar.extractall(stage, filter="data")
        data = json.loads((stage / "manifest.json").read_text())
        src, art = data.get("source", {}), data.get("artifact", {})
        spec = resolve(src.get("model", "?"))
        try:
            bucket = int(art.get("length", -1))
        except (TypeError, ValueError) as e:
            raise ArtifactIntegrityError(f"manifest has a malformed artifact.length: {art.get('length')!r}") from e
        manifest = verify_manifest(spec, bucket, data)
        if expect is not None and (spec.name, bucket) != tuple(expect):
            raise ArtifactIntegrityError(f"{archive} holds {spec.name} L{bucket}, not {expect[0]} L{expect[1]}")
        verify_profile(data)
        verify_files(manifest, stage)
        sha_check = tree_sha256(stage / COMPILED)  # main computes this on the staged copy too
        final = artifact_dir(spec, bucket)
        with build_lock(spec, bucket, log=log):
            if _abandoned_pending(final):
                log(f"{spec.name} L{bucket}: removing an unpublished install left at {final}")
                _remove_aside(final, log)
            if final.exists():
                raise ArtifactError(f"{final} already exists; pass force=True (--force) to replace it")
            # Withhold the manifest: the runtime treats the path as missing until it is published.
            os.rename(stage / "manifest.json", stage / PENDING_MANIFEST)
            (stage / PENDING).write_text(
                json.dumps({"pid": os.getpid(), "since": datetime.now(timezone.utc).isoformat()})
            )
            final.parent.mkdir(parents=True, exist_ok=True)
            (stage / BUILDING).unlink(missing_ok=True)
            os.rename(stage, final)
            try:
                placement = compute_plan_summary(final / COMPILED, ANE_COMPUTE_UNITS)
                check_ane_placement(placement)
                ckpt = checkpoint_path(spec, local_files_only=local_files_only)
                verify_weights(spec, ckpt)
                log(f"parity gate for imported {spec.name} L{bucket} on this machine (at the registered path)")
                t = time.perf_counter()
                parity = ane_parity(spec, final / COMPILED, bucket, ckpt)
                log(f"{spec.name} L{bucket}: parity gate ran in {time.perf_counter() - t:.1f} s (includes the load)")
                if not parity["passed"]:
                    raise ArtifactParityError(f"imported {spec.name} L{bucket} failed the parity gate here: {parity}")
                the_probe = registered_probe if registered_probe is not None else probe
                if the_probe is not None:
                    the_probe(spec, bucket, final / COMPILED)
                data["imported"] = {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "from": source or str(archive),
                    "platform": platform_profile(),
                    "placement": placement,
                    "parity": parity,
                    "artifact_sha256_check": sha_check,
                }
                (final / PENDING_MANIFEST).write_text(json.dumps(data, indent=1) + "\n")
                log(f"{spec.name} L{bucket}: publishing")
                os.replace(final / PENDING_MANIFEST, final / "manifest.json")
            except BaseException:
                _remove_aside(final, log)
                raise
            log(f"{spec.name} L{bucket}: published")
            (final / PENDING).unlink(missing_ok=True)
            try:  # the stamp load_verified(full=True) leaves on main; same files, same checks passed
                profile = platform_profile()
                stamp = _stamp_path(final)
                stamp.parent.mkdir(parents=True, exist_ok=True)
                tmp = stamp.with_suffix(f".{os.getpid()}.tmp")
                tmp.write_text(
                    json.dumps({"key": _stamp_key(manifest, final, ANE_COMPUTE_UNITS, profile), "placement": placement})
                )
                os.replace(tmp, stamp)
            except OSError as e:
                log(f"{spec.name} L{bucket}: registered, but the verification stamp was not written: {e}")
        return final
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
