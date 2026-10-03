"""Artifact lifecycle: build locking, quarantine, pruning, warming.

- **Locking.** One build per artifact key at a time, across processes (`fcntl.flock` on
  `<artifacts>/.locks/<model>-<rev12>-L<bucket>.lock`). A second builder waits, then finds
  the artifact registered and stops.
- **Quarantine.** An artifact whose files no longer match their manifest hash, or whose
  manifest cannot be read, is moved to `<artifacts>/quarantine/` when it is detected. The
  error names the rebuild command. Nothing corrupt stays where the runtime looks for it.
- **Prune.** `plan_prune()` lists removable entries with a reason and never deletes.
  `prune()` deletes only paths inside the artifacts root:
  - artifacts for revisions, weights, models or buckets that are no longer pinned or
    offered;
  - artifacts built on another platform profile;
  - artifacts that are not validated;
  - quarantined and rejected entries;
  - abandoned staging directories;
  - copies an import left beside an artifact, once that import has exited: `.failed-<pid>`,
    and `.old-<pid>` without a manifest;
  - an unpublished install an import left at the registered path (`PENDING.json`, no
    `manifest.json`), once that import has exited and the directory is older than
    STAGING_MAX_AGE_S;
  - orphaned verification stamps.
  A `.old-<pid>` with a manifest is a previous artifact an interrupted or failed replace
  kept, possibly the only good copy: prune never deletes one, even from a hand-written plan,
  and `kept_previous_artifacts()` reports it.
- **Warm.** `warm()` loads each registered artifact once. Core ML's on-device ANE compile
  is cached per location and can be evicted by the system; warming pays that cost ahead of
  the first request.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import (
    COMPILED,
    MANIFEST_FORMAT,
    artifact_dir,
    artifacts_root,
    platform_profile,
    profile_matches,
)
from .errors import ArtifactError, ArtifactIntegrityError, BackendUnavailableError
from .hub import cache_root
from .registry import ANE_GRAPH, ModelSpec, models

STAGING_MAX_AGE_S = 3600
MAX_IMPORT_BYTES = 4 * 1024**3
BUILDING = "BUILDING.json"
# A first install is validated at its registered path with its manifest withheld: the manifest
# waits as PENDING_MANIFEST and PENDING (the import's pid) marks the directory as unpublished.
PENDING = "PENDING.json"
PENDING_MANIFEST = "manifest.pending.json"
_PENDING_REASON = "unpublished install left by an import that exited"
_LEFTOVER = re.compile(r"\.(old|failed)-(\d+)$")


def _pid_alive(pid) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (OSError, OverflowError, ValueError, TypeError):
        return True  # e.g. exists but owned by another user; an impossible pid is never assumed dead
    return True


def _leftover(d: Path):
    """For a `.old-<pid>` / `.failed-<pid>` directory an import left beside an artifact:
    (kind, the registered path it sits beside, pid alive); else None."""
    m = _LEFTOVER.search(d.name)
    if not m or not d.is_dir():
        return None
    return m.group(1), d.with_name(d.name[: m.start()]), _pid_alive(int(m.group(2)))


def _kept_previous(d: Path) -> bool:
    """A `.old-<pid>` that still has its manifest: a previous artifact an interrupted or failed
    replace left behind, which may be the only good copy. prune never deletes one."""
    info = _leftover(d)
    return bool(info) and info[0] == "old" and (d / "manifest.json").exists()


def kept_previous_artifacts() -> list[dict]:
    """Every `.old-<pid>` with a manifest whose import has exited. prune never deletes them;
    the user verifies and removes one by hand, or renames it back to its registered path."""
    root = artifacts_root()
    out = []
    for d in sorted(root.glob("*/*/*")) if root.exists() else []:
        if not _kept_previous(d) or _leftover(d)[2]:
            continue
        registered = _leftover(d)[1]
        missing = not (registered / "manifest.json").exists()
        state = "nothing is registered at" if missing else "an artifact is registered at"
        out.append(
            {
                "path": str(d),
                "registered": str(registered),
                "registered_missing": missing,
                "message": (
                    f"previous artifact kept from an interrupted or failed replace ({state} {registered}); "
                    f"verify and remove it by hand, or rename it back: mv {d} {registered}"
                ),
            }
        )
    return out


def _lock_path(spec: ModelSpec, bucket: int) -> Path:
    return artifacts_root() / ".locks" / f"{spec.name}-{spec.revision[:12]}-L{bucket}.lock"


@contextlib.contextmanager
def build_lock(spec: ModelSpec, bucket: int, *, timeout: float = 3600.0, log=None):
    """Exclusive, cross-process lock for building one artifact. Waits up to `timeout`."""
    path = _lock_path(spec, bucket)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as fh:
        deadline = time.monotonic() + timeout
        announced = False
        while True:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise ArtifactError(
                        f"another process has been building {spec.name} L{bucket} for over {timeout:.0f} s ({path})"
                    ) from None
                if log and not announced:
                    log(f"waiting for another build of {spec.name} L{bucket} to finish")
                    announced = True
                time.sleep(0.5)
        fh.seek(0)
        fh.truncate()
        fh.write(json.dumps({"pid": os.getpid(), "since": datetime.now(timezone.utc).isoformat()}))
        fh.flush()
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _inside_root(path: Path) -> bool:
    root = artifacts_root().resolve()
    try:
        return path.resolve().is_relative_to(root) and path.resolve() != root
    except OSError:
        return False


def quarantine(directory: Path, reason: str) -> Path | None:
    """Move a corrupt artifact directory out of the lookup path. Returns the new location,
    or None if another process moved it first."""
    if not _inside_root(directory) or not directory.exists():
        return None
    rel = directory.relative_to(artifacts_root())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = artifacts_root() / "quarantine" / f"{str(rel).replace('/', '__')}-{stamp}-{os.getpid()}"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(directory, target)
    except FileNotFoundError:
        return None
    (target / "QUARANTINED.json").write_text(json.dumps({"reason": reason, "at": stamp, "from": str(directory)}))
    return target


def _dir_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _abandoned_staging(p: Path) -> bool:
    """True when nothing is building in staging directory `p` and it has been untouched for
    STAGING_MAX_AGE_S. Raises FileNotFoundError if `p` vanishes meanwhile.

    The marker is read before the age, never after. A finishing import unlinks the marker just
    before it renames the directory into place, and the unlink refreshes the directory's mtime.
    Reading the age first could pair the age from before that with a marker that is already
    gone, and call a live import's directory old and unmarked."""
    building = p / BUILDING
    if building.exists():
        try:
            pid = json.loads(building.read_text()).get("pid")
        except ValueError:
            pid = None
        if pid is not None and _pid_alive(pid):
            return False  # build still in progress; not abandoned regardless of age
    return time.time() - p.stat().st_mtime > STAGING_MAX_AGE_S


def unpublished(d: Path) -> bool:
    """`d` holds a first install whose manifest was never published: `PENDING.json` and no
    `manifest.json`. The runtime treats such a path as missing (ArtifactMissingError)."""
    return (d / PENDING).exists() and not (d / "manifest.json").exists()


def _abandoned_pending(d: Path) -> bool:
    """True when `d` is unpublished, the import that wrote it has exited (or its marker is
    unreadable), and it has been untouched for STAGING_MAX_AGE_S. Raises FileNotFoundError if
    `d` or its marker vanishes meanwhile.

    Read in this order: the marker, the manifest, then the age. An import that is still running
    is kept by its pid whatever it does next; a dead one can no longer publish."""
    try:
        marker = json.loads((d / PENDING).read_text())
    except ValueError:
        marker = None
    pid = marker.get("pid") if isinstance(marker, dict) else None
    if pid is not None and _pid_alive(pid):
        return False  # its import is still validating it
    if (d / "manifest.json").exists():
        return False  # published; the marker is only a leftover
    return time.time() - d.stat().st_mtime > STAGING_MAX_AGE_S


def _remove_aside(final: Path, log) -> None:
    """Free the registered path with one rename to `.failed-<pid>`, then remove that copy; prune
    removes whatever the removal leaves once this process has exited."""
    failed = final.with_name(final.name + f".failed-{os.getpid()}")
    try:
        os.rename(final, failed)
    except FileNotFoundError:
        return
    except OSError as e:
        log(f"could not move {final} aside: {e}")
        return
    shutil.rmtree(failed, ignore_errors=True)


def plan_prune(profile: dict | None = None) -> list[dict]:
    """Everything prune() would delete, with the reason. Deletes nothing."""
    root = artifacts_root()
    if not root.exists():
        return []
    profile = profile or platform_profile()
    specs = models()
    out: list[dict] = []

    def add(path: Path, reason: str):
        out.append({"path": str(path), "reason": reason, "bytes": _dir_bytes(path)})

    for d in sorted(root.glob("*/*/*")):
        # `.old-<pid>` / `.failed-<pid>`: a replaced or failed copy an import left beside an
        # artifact, possibly partly removed (no manifest). Kept while that import still runs.
        # A `.old-<pid>` with a manifest is never listed (kept_previous_artifacts reports it).
        info = _leftover(d)
        if info:
            if not info[2] and not _kept_previous(d):
                add(d, f"leftover {info[0]} copy from an earlier import")
        elif (d / PENDING).exists():
            # An install whose manifest was never published (manifest.json absent): kept while
            # its import runs; listed once that import has exited and the directory is old.
            try:
                if _abandoned_pending(d):
                    add(d, _PENDING_REASON)
            except FileNotFoundError:
                continue  # published, removed or recovered by an import meanwhile
    for mf in sorted(root.glob("*/*/*/manifest.json")):
        d = mf.parent
        if _LEFTOVER.search(d.name):
            continue  # handled above
        try:
            data = json.loads(mf.read_text())
        except ValueError:
            add(d, "unreadable manifest")
            continue
        src, art = data.get("source", {}), data.get("artifact", {})
        spec = specs.get(src.get("model"))
        if data.get("format") != MANIFEST_FORMAT:
            add(d, f"unknown manifest format {data.get('format')!r}")
        elif spec is None:
            add(d, f"model {src.get('model')!r} is not registered")
        elif src.get("revision") != spec.revision or src.get("weights_sha256") != spec.weights_sha256:
            add(d, f"built from revision {str(src.get('revision'))[:12]}, pinned is {spec.revision[:12]}")
        elif art.get("graph") != ANE_GRAPH or int(art.get("length", -1)) not in spec.ane_buckets:
            add(d, f"{art.get('graph')} L{art.get('length')} is no longer an offered configuration")
        elif d != artifact_dir(spec, int(art["length"])):
            add(d, "not at the path the runtime uses for this configuration")
        elif not profile_matches(data.get("platform") or {}, profile):
            add(d, "built on another platform profile (SoC / macOS major / coremltools)")
        elif data.get("status") != "validated" or not (d / COMPILED).exists():
            add(d, f"status {data.get('status')!r}")
    for sub, reason in (("quarantine", "quarantined"), ("rejected", "rejected build record")):
        for p in sorted((root / sub).glob("*")) if (root / sub).exists() else []:
            add(p, reason)
    staging = root / ".staging"
    if staging.exists():
        for p in sorted(staging.glob("*")):
            try:
                if _abandoned_staging(p):
                    add(p, "abandoned staging directory")
            except FileNotFoundError:
                continue  # renamed into place, or removed, by the build that owned it
    stamps = cache_root() / "verified" / "artifacts"
    if stamps.exists():
        for s in sorted(stamps.glob("*.json")):
            target = root / s.stem.replace("__", "/")
            if not target.exists():
                out.append({"path": str(s), "reason": "orphaned verification stamp", "bytes": s.stat().st_size})
    return out


def prune(plan: list[dict] | None = None) -> list[dict]:
    """Delete what plan_prune() lists (or the given plan). Returns what was removed."""
    plan = plan_prune() if plan is None else plan
    stamps_root = (cache_root() / "verified" / "artifacts").resolve()
    staging_root = (artifacts_root() / ".staging").resolve()
    removed = []
    for item in plan:
        p = Path(item["path"])
        info = _leftover(p)
        if info and (info[2] or _kept_previous(p)):
            continue  # its import is running, or it is a kept previous artifact (never deleted here)
        if p.resolve().parent == staging_root and _inside_root(p):  # resolved: a plan may spell the cache differently
            try:
                if _abandoned_staging(p):  # checked again: its build may have started or finished since the plan
                    shutil.rmtree(p) if p.is_dir() else p.unlink()
                    removed.append(item)
            except FileNotFoundError:
                pass  # renamed into place, or removed, since the plan was made
            continue
        if _inside_root(p) and not info and (item.get("reason") == _PENDING_REASON or unpublished(p)):
            # Only ever deleted as an unpublished install: by now an import may have recovered and
            # published it, or be validating a new one there (and a plan may be hand-written).
            try:
                if _abandoned_pending(p):  # checked again
                    shutil.rmtree(p)
                    removed.append(item)
            except FileNotFoundError:
                pass  # published, removed or recovered by an import since the plan was made
            continue
        if p.is_file() and p.resolve().parent == stamps_root:
            p.unlink()
        elif _inside_root(p):
            shutil.rmtree(p) if p.is_dir() else p.unlink()
        else:
            continue  # never delete outside the cache
        removed.append(item)
    return removed


def warm(spec: ModelSpec, buckets=None) -> dict:
    """Load each artifact once (paying any evicted ANE compile now); seconds per bucket."""
    from .artifacts import load_verified

    out = {}
    for b in buckets or spec.ane_buckets:
        t = time.perf_counter()
        load_verified(spec, b)
        out[b] = time.perf_counter() - t
    return out


# ----------------------------------------------------------------------------- export / import


def export_artifact(spec: ModelSpec, bucket: int, dest: Path) -> Path:
    """Write a registered artifact (manifest + compiled model) to a .tar.gz for another machine."""
    import tarfile

    from .artifacts import load_verified

    load_verified(spec, bucket, full=True)  # only export what verifies here
    src = artifact_dir(spec, bucket)
    dest = Path(dest)
    if dest.suffixes[-2:] != [".tar", ".gz"]:
        dest = dest.with_name(dest.name + ".tar.gz")
    with tarfile.open(dest, "w:gz") as tar:
        tar.add(src / "manifest.json", arcname="manifest.json")
        tar.add(src / COMPILED, arcname=COMPILED)
    return dest


def import_artifact(
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
    """Register an artifact built elsewhere, only after validating it here.

    Checked on this machine before registration:
    - the manifest against the pinned checkpoint;
    - the build platform profile against this one;
    - the file hash;
    - the compute plan (100% ANE, no transitions);
    - the full parity gate against the shipped goldens.
    The original build provenance is kept, and the local results are recorded under
    `imported` (`from` is `source` when given, such as the repository an archive was fetched
    from, else the archive's path).

    `expect`, when given, is the (model name, bucket) the archive must hold. It is checked
    right after the manifest is verified, before any other check; anything else is refused
    with ArtifactIntegrityError (`artifacts fetch` passes the index entry's).

    **First install** (`force=False`; #162). Core ML's on-device ANE compile is tied to the
    model's path, so the checks that load the model run once, at the registered path, before
    the artifact is registered there. The archive, manifest, `expect`, profile and file hash
    are checked in a staging directory. Then, under the build lock, the staging directory is
    renamed into the registered path with its manifest withheld (`manifest.pending.json`, plus
    a `PENDING.json` marker naming this pid): until it is published the runtime finds no
    artifact there and raises ArtifactMissingError, and nothing quarantines it. At the
    registered path run the compute plan, the weight check, the full parity gate and exactly
    one placement probe (`registered_probe` if given, else `probe`, called with the registered
    compiled path); the CLI passes neither, so only the parity gate loads the model. Then the
    manifest is published with one `os.replace`, the marker is removed and the verification
    stamp `load_verified` would write is written. If anything fails or the import is
    interrupted, the directory is renamed to `.failed-<pid>` and removed, so nothing is
    registered. An unpublished directory an import left at the registered path when it died
    is removed (under the build lock, so its import cannot be running) before installing.
    `artifacts prune` lists one only once its import has exited and it is older than
    STAGING_MAX_AGE_S. A registered artifact (with a manifest) is refused without `force`.

    **Replace** (`force=True`) keeps the path below: every check on the staged copy, the move
    into place, then a load and the probe at the registered path.

    `probe`, when given, is called as probe(spec, bucket, staged_compiled_path) after every
    check above and before registration (`artifacts fetch` runs the runtime placement probe
    there). If any check or the probe raises, nothing is registered and an artifact already
    at the registered path, which `force` would replace, is left as it was; so is it if moving
    the new artifact into place fails.

    `registered_probe`, when given, makes registration all-or-nothing: after the move, still
    under the build lock, the artifact is loaded at its registered path (load_verified,
    full) and registered_probe(spec, bucket, registered_compiled_path) is called
    (`artifacts fetch` runs the placement probe again there, since Core ML compiles per
    path). If either raises (or the import is interrupted), the new artifact is moved aside
    with one rename and removed, the previous one is put back, and the error is raised. The
    guarantee is about what is left on disk, not isolation from concurrent readers: a
    runtime loading the bucket meanwhile can see the new artifact, and verifies it on each
    load. Without the hook, the pre-warm load runs after registration, and its failure
    raises with the new artifact already registered (the runtime checks it again on every
    load). Either way, a failure to remove the replaced copy is only logged.
    """
    import tarfile
    import tempfile

    from .artifacts import (
        _stamp_key,
        _stamp_path,
        check_ane_placement,
        compute_plan_summary,
        load_verified,
        tree_sha256,
        verify_files,
        verify_manifest,
        verify_profile,
    )
    from .errors import ArtifactParityError
    from .hub import checkpoint_path, verify_weights
    from .parity.ane import ane_parity
    from .registry import ANE_COMPUTE_UNITS, resolve

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
            tar.extractall(stage, filter="data")  # rejects absolute paths, '..', links outside
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
        final = artifact_dir(spec, bucket)
        with build_lock(spec, bucket, log=log):
            if not force:  # first install: validated at the registered path (see the docstring)
                if final.is_dir() and unpublished(final):  # the lock means its import has exited
                    log(f"{spec.name} L{bucket}: removing an unpublished install left at {final}")
                    _remove_aside(final, log)
                if final.exists():
                    raise ArtifactError(f"{final} already exists; pass force=True (--force) to replace it")
                sha_check = tree_sha256(stage / COMPILED)
                os.rename(stage / "manifest.json", stage / PENDING_MANIFEST)
                (stage / PENDING).write_text(
                    json.dumps({"pid": os.getpid(), "since": datetime.now(timezone.utc).isoformat()})
                )
                final.parent.mkdir(parents=True, exist_ok=True)
                (stage / BUILDING).unlink(missing_ok=True)  # PENDING guards it from here on
                os.rename(stage, final)
                try:
                    placement = compute_plan_summary(final / COMPILED, ANE_COMPUTE_UNITS)
                    check_ane_placement(placement)
                    ckpt = checkpoint_path(spec, local_files_only=local_files_only)
                    verify_weights(spec, ckpt)
                    log(f"parity gate for imported {spec.name} L{bucket} on this machine, at its registered path")
                    t = time.perf_counter()
                    parity = ane_parity(spec, final / COMPILED, bucket, ckpt)
                    took = time.perf_counter() - t
                    log(f"{spec.name} L{bucket}: parity gate ran in {took:.1f} s (includes the load)")
                    if not parity["passed"]:
                        raise ArtifactParityError(
                            f"imported {spec.name} L{bucket} failed the parity gate here: {parity}"
                        )
                    one_probe = registered_probe if registered_probe is not None else probe
                    if one_probe is not None:
                        one_probe(spec, bucket, final / COMPILED)
                    profile = platform_profile()
                    data["imported"] = {
                        "at": datetime.now(timezone.utc).isoformat(),
                        "from": source or str(archive),
                        "platform": profile,
                        "placement": placement,
                        "parity": parity,
                        "artifact_sha256_check": sha_check,
                    }
                    (final / PENDING_MANIFEST).write_text(json.dumps(data, indent=1) + "\n")
                    os.replace(final / PENDING_MANIFEST, final / "manifest.json")  # registers it
                except BaseException:
                    _remove_aside(final, log)
                    raise
                (final / PENDING).unlink(missing_ok=True)
                try:  # the stamp load_verified writes after the same checks on the same files
                    stamp = _stamp_path(final)
                    stamp.parent.mkdir(parents=True, exist_ok=True)
                    tmp = stamp.with_suffix(f".{os.getpid()}.tmp")
                    key = _stamp_key(manifest, final, ANE_COMPUTE_UNITS, profile)
                    tmp.write_text(json.dumps({"key": key, "placement": placement}))
                    os.replace(tmp, stamp)
                except OSError as e:  # registered; the first load re-hashes and stamps it instead
                    log(f"{spec.name} L{bucket}: registered, but the verification stamp was not written: {e}")
                log(f"{spec.name} L{bucket}: registered after validation at its registered path")
                return final
            placement = compute_plan_summary(stage / COMPILED, ANE_COMPUTE_UNITS)
            check_ane_placement(placement)
            ckpt = checkpoint_path(spec, local_files_only=local_files_only)
            verify_weights(spec, ckpt)
            log(f"parity gate for imported {spec.name} L{bucket} on this machine")
            t = time.perf_counter()
            parity = ane_parity(spec, stage / COMPILED, bucket, ckpt)
            log(f"{spec.name} L{bucket}: parity gate ran in {time.perf_counter() - t:.1f} s (includes the staged load)")
            if not parity["passed"]:
                raise ArtifactParityError(f"imported {spec.name} L{bucket} failed the parity gate here: {parity}")
            if probe is not None:
                probe(spec, bucket, stage / COMPILED)
            data["imported"] = {
                "at": datetime.now(timezone.utc).isoformat(),
                "from": source or str(archive),
                "platform": platform_profile(),
                "placement": placement,
                "parity": parity,
                "artifact_sha256_check": tree_sha256(stage / COMPILED),
            }
            (stage / "manifest.json").write_text(json.dumps(data, indent=1) + "\n")
            final.parent.mkdir(parents=True, exist_ok=True)
            old = None
            if final.exists():
                old = final.with_name(final.name + f".old-{os.getpid()}")
                os.rename(final, old)

            def restore():
                """Put the previous artifact back, if there was one; the caller re-raises."""
                if old is None:
                    return
                try:
                    os.rename(old, final)
                except OSError as e:
                    log(f"{spec.name} L{bucket}: could not restore the previous artifact; it was left at {old}: {e}")

            try:
                # The marker only guards the staging directory; unlinking it also refreshes that
                # directory's mtime, so `prune` cannot see it unmarked and old before the rename.
                (stage / BUILDING).unlink(missing_ok=True)
                os.rename(stage, final)
            except BaseException:
                restore()
                raise
            if registered_probe is not None:
                t = time.perf_counter()
                try:
                    load_verified(spec, bucket, full=True)  # pre-warm the ANE compile at the registered path
                    registered_probe(spec, bucket, final / COMPILED)
                except BaseException:
                    # Unregister the new artifact with one rename, so the path is free for the
                    # previous one whatever happens to the removal; prune removes any leftover.
                    failed = final.with_name(final.name + f".failed-{os.getpid()}")
                    if final.exists():  # load_verified may already have quarantined it
                        try:
                            os.rename(final, failed)
                        except OSError as e:
                            kept = f"; the previous artifact is at {old}" if old is not None else ""
                            log(f"{spec.name} L{bucket}: could not move the failed artifact out of {final}: {e}{kept}")
                    if not final.exists():
                        restore()
                    shutil.rmtree(failed, ignore_errors=True)
                    raise
                log(f"{spec.name} L{bucket}: registered, loaded and probed in {time.perf_counter() - t:.1f} s")
            if old is not None:
                try:
                    shutil.rmtree(old)
                except OSError as e:  # the new artifact is registered; only cleanup failed
                    log(
                        f"{spec.name} L{bucket}: registered, but could not remove the replaced copy {old}: {e}. "
                        "Remove it by hand once the new one works; `laya-apple artifacts prune` lists it but keeps it."
                    )
        if registered_probe is None:
            t = time.perf_counter()
            load_verified(spec, bucket, full=True)  # pre-warm the ANE compile at the registered path
            log(f"{spec.name} L{bucket}: registered and loaded in {time.perf_counter() - t:.1f} s")
        return final
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
