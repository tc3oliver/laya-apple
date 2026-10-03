"""Prebuilt ANE artifacts from a Hugging Face repository, re-validated on this machine.

    laya-apple artifacts fetch laya-typed-decisions            # every offered bucket
    laya-apple artifacts fetch laya --length 64 --repo OWNER/REPO

Building an artifact needs the `[convert]` extra (PyTorch, coremltools conversion) and minutes
per bucket. `fetch` downloads an `artifacts export` archive instead and registers it through
`lifecycle.import_artifact`, the same path as `laya-apple artifacts import`. Nothing is
trusted because it was downloaded; before an artifact is registered this machine checks:
- the manifest against the pinned checkpoint revision and weight hash;
- the build platform profile against this one (same SoC, macOS major and coremltools);
- the archive's SHA-256 against the repository index, and every file against the manifest;
- the compute plan (100% Neural Engine, no transitions);
- the full parity gate against the shipped goldens;
- the runtime placement probe (`backends.coreml_ane.probe_placement`).
Core ML's on-device compile is tied to the model's path, so on a first install the compute
plan, the parity gate and the probe run at the path the runtime loads from, with the manifest
withheld until all of them pass; the runtime sees no artifact there until then (#162). With
`force` (a replace) they run on a staged copy, and after the move into place the artifact is
loaded and probed again there. The reported probe result is the one at the registered path. A
bucket that fails any check is not registered, and with `force` the artifact it would replace
is kept or put back. The runtime repeats the probe on every load.

A bucket counts as already registered only if its artifact passes `artifacts.load_verified`,
the runtime's own checks. One that does not is fetched again and replaced (a corrupt one is
quarantined first, by `load_verified`); it is never reported as registered.

The Core ML on-device ANE compile still happens on this machine, when the imported artifact is
first loaded at its registered location. A first install pays that per-path compile at one
path instead of two (research/import-compile-once/); a replace pays it at the staged path and
again at the registered path. Prebuilt artifacts remove the build (and the PyTorch dependency),
not that compile; see research/coreml-compile-cache/ for the measurement plan on the
remaining cost.

Repository layout (written by scripts/publish_prebuilt.py):

    index.json                                            # INDEX_FORMAT, the entries below
    <model>/<revision[:12]>/<profile-slug>/<model>-L<bucket>.tar.gz

An index entry names the archive's path, model, revision, weights_sha256, graph, length, the
build platform profile, the archive's SHA-256 and size, and the artifact tree hash. Entries
are selected with the same profile rule the runtime uses (`artifacts.profile_matches`).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from .artifacts import artifact_dir, artifacts_root, load_verified, platform_profile, profile_matches
from .errors import ArtifactError, ArtifactIntegrityError, ArtifactMissingError, BackendUnavailableError
from .hub import sha256_file
from .registry import ANE_GRAPH, ModelSpec

PREBUILT_REPO_ENV = "LAYA_APPLE_PREBUILT_REPO"
# The published repository (Apple M4 Max with coremltools 9.0: macOS 26 since 1.6.0, macOS 27
# since 1.6.2). While
# DEFAULT_REPO_PUBLISHED is False, fetching without --repo / LAYA_APPLE_PREBUILT_REPO raises
# instead of trying a repository that does not exist.
DEFAULT_PREBUILT_REPO = "tc3oliver/laya-apple-artifacts"
DEFAULT_REPO_PUBLISHED = True
# The immutable repository commit this release was qualified against. A fetch from the default
# repository without an explicit revision reads this commit, never the mutable `main`, so one
# laya-apple version always resolves to the same index and archives. Another repository
# (--repo / LAYA_APPLE_PREBUILT_REPO) defaults to CUSTOM_REPO_REVISION instead.
# 1.6.2: the commit that added the macOS 27 archives; the macOS 26 archives are the ones 1.6.0
# and 1.6.1 read at 93181067cfee9c6117a7919321eb303ec36fcbd4, unchanged.
DEFAULT_PREBUILT_REVISION = "44a54765f89d434e9d85f401126cfee7d2059e7d"
CUSTOM_REPO_REVISION = "main"

INDEX = "index.json"
INDEX_FORMAT = "laya-apple-prebuilt-index"
INDEX_VERSION = 1
_ENTRY_KEYS = (
    "path",
    "model",
    "revision",
    "weights_sha256",
    "graph",
    "length",
    "platform",
    "archive_sha256",
    "bytes",
    "artifact_sha256",
)


def resolve_repo(repo: str | None = None) -> str:
    """The repository to fetch from: `repo`, else $LAYA_APPLE_PREBUILT_REPO, else the default
    (refused while it is still a placeholder)."""
    repo = (repo or os.environ.get(PREBUILT_REPO_ENV) or "").strip()
    if repo:
        return repo
    if not DEFAULT_REPO_PUBLISHED:
        raise BackendUnavailableError(
            "no prebuilt artifact repository is published yet. Pass --repo OWNER/NAME (or set "
            f"{PREBUILT_REPO_ENV}), or build the artifacts here: laya-apple artifacts build MODEL"
        )
    return DEFAULT_PREBUILT_REPO


def resolve_revision(repo: str, revision: str | None = None) -> str:
    """The repository revision to read: `revision` when given; else the pinned
    DEFAULT_PREBUILT_REVISION for the default repository, and CUSTOM_REPO_REVISION for any
    other repository."""
    revision = (revision or "").strip()
    if revision:
        return revision
    return DEFAULT_PREBUILT_REVISION if repo == DEFAULT_PREBUILT_REPO else CUSTOM_REPO_REVISION


def profile_slug(profile: dict) -> str:
    """A readable directory name for a platform profile (selection uses the index, not this)."""
    soc = re.sub(r"[^a-z0-9]+", "-", str(profile.get("soc") or "unknown").lower()).strip("-")
    macos = str(profile.get("macos") or "0").split(".")[0]
    ct = re.sub(r"[^0-9a-z.]+", "-", str(profile.get("coremltools") or "none").lower())
    return f"{soc}-macos{macos}-coremltools{ct}"


def archive_path(spec: ModelSpec, bucket: int, profile: dict) -> str:
    return f"{spec.name}/{spec.revision[:12]}/{profile_slug(profile)}/{spec.name}-L{bucket}.tar.gz"


# ----------------------------------------------------------------------------- index


def make_index_entry(manifest: dict, archive: Path, path: str) -> dict:
    """The index entry for one exported archive, from its manifest (publisher side)."""
    src, art = manifest["source"], manifest["artifact"]
    return {
        "path": path,
        "model": src["model"],
        "revision": src["revision"],
        "weights_sha256": src["weights_sha256"],
        "graph": art["graph"],
        "length": int(art["length"]),
        "platform": manifest["platform"],
        "archive_sha256": sha256_file(archive),
        "bytes": Path(archive).stat().st_size,
        "artifact_sha256": manifest["integrity"]["artifact_sha256"],
    }


def merge_index(index: dict | None, entries: list[dict]) -> dict:
    """Add or replace entries by path; returns a new index with entries sorted by path."""
    by_path = {e["path"]: e for e in (validate_index(index)["artifacts"] if index else [])}
    for e in entries:
        by_path[e["path"]] = e
    return {"format": INDEX_FORMAT, "format_version": INDEX_VERSION, "artifacts": sorted(by_path.values(), key=_path)}


def _path(entry: dict) -> str:
    return entry["path"]


def validate_index(index) -> dict:
    if not isinstance(index, dict) or index.get("format") != INDEX_FORMAT:
        raise ArtifactIntegrityError(f"not a laya-apple prebuilt index (format {INDEX_FORMAT!r} expected)")
    if index.get("format_version") != INDEX_VERSION:
        raise ArtifactIntegrityError(
            f"prebuilt index format_version {index.get('format_version')!r} is not supported "
            f"(this laya-apple reads {INDEX_VERSION}); upgrade laya-apple"
        )
    entries = index.get("artifacts")
    if not isinstance(entries, list):
        raise ArtifactIntegrityError("prebuilt index has no 'artifacts' list")
    for e in entries:
        missing = [k for k in _ENTRY_KEYS if not isinstance(e, dict) or k not in e]
        if missing:
            raise ArtifactIntegrityError(f"prebuilt index entry is missing {missing}: {e!r}")
        p = e["path"]
        if not isinstance(p, str) or p.startswith("/") or ".." in p.split("/") or not p.endswith(".tar.gz"):
            raise ArtifactIntegrityError(f"prebuilt index entry has an unsafe path: {p!r}")
    return index


def select(index: dict, spec: ModelSpec, buckets, profile: dict) -> tuple[dict, dict]:
    """({bucket: entry} for this pinned checkpoint and platform profile, {bucket: why not})."""
    found, why = {}, {}
    entries = validate_index(index)["artifacts"]
    for b in buckets:
        candidates = [
            e
            for e in entries
            if e["model"] == spec.name
            and e["revision"] == spec.revision
            and e["weights_sha256"] == spec.weights_sha256
            and e["graph"] == ANE_GRAPH
            and int(e["length"]) == b
        ]
        matching = [e for e in candidates if profile_matches(e["platform"] or {}, profile)]
        if matching:
            found[b] = matching[0]
        elif candidates:
            built = sorted({profile_slug(e["platform"] or {}) for e in candidates})
            why[b] = f"built only for {', '.join(built)}; this machine is {profile_slug(profile)}"
        else:
            why[b] = f"no archive for {spec.name}@{spec.revision[:12]} L{b}"
    return found, why


# ----------------------------------------------------------------------------- fetch


def _download(repo: str, filename: str, revision: str, local_dir: Path, *, local_files_only: bool = False) -> Path:
    """One file from the model repository. Online it is downloaded into `local_dir` (staging,
    not kept). Offline (`local_files_only` or HF_HUB_OFFLINE) it is read from the Hugging Face
    cache only, never the network, and is returned in place."""
    from huggingface_hub import hf_hub_download

    from .hub import offline_mode

    mode = offline_mode(local_files_only)
    try:
        if mode:
            return Path(hf_hub_download(repo, filename, revision=revision, local_files_only=True))
        return Path(hf_hub_download(repo, filename, revision=revision, local_dir=str(local_dir)))
    except Exception as e:
        if mode:
            raise BackendUnavailableError(
                f"cannot read {filename} from {repo}@{revision} {mode}: it is not in the Hugging Face cache ({e}). "
                f"Fetch without --offline, or cache it first: hf download {repo} {filename} --revision {revision}"
            ) from e
        raise BackendUnavailableError(f"cannot download {filename} from {repo}@{revision}: {e}") from e


def _probe(spec: ModelSpec, bucket: int, compiled: Path, *, local_files_only: bool) -> dict:
    """The runtime placement probe on one compiled model (the staged copy, or the registered
    one before its registration is kept): loaded on CPU_AND_NE as the runtime loads it, with
    the runtime's probe input. A failure raises ComputeUnitMismatchError."""
    import coremltools as ct

    from .backends.coreml_ane import HostWeights, probe_features, probe_placement
    from .hub import checkpoint_path
    from .prompt import Tokenizer
    from .registry import ANE_COMPUTE_UNITS

    ckpt = checkpoint_path(spec, local_files_only=local_files_only)
    local_attention = int(json.loads((ckpt / "encoder/config.json").read_text())["local_attention"])
    pad_id = Tokenizer(ckpt / "tokenizer").pad_token_id
    feats = probe_features(HostWeights(ckpt, local_attention), pad_id, bucket)
    model = ct.models.CompiledMLModel(str(compiled), compute_units=getattr(ct.ComputeUnit, ANE_COMPUTE_UNITS))
    return probe_placement(spec, bucket, model, compiled, feats)


def fetch(
    spec: ModelSpec,
    buckets=None,
    *,
    repo: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
    force: bool = False,
    log=print,
) -> dict:
    """Download, validate here and register prebuilt artifacts for `spec`.

    Returns {bucket: {"path", "source", "probe"}} for the fetched buckets. A bucket whose
    registered artifact passes `load_verified` is skipped unless `force`; one that does not is
    fetched again and replaced. Raises ArtifactMissingError when a requested bucket has no
    archive for this checkpoint and platform profile (nothing is imported then),
    ArtifactIntegrityError on a hash mismatch, and whatever the import checks or the placement
    probe raise. On a first install the probe runs once, at the registered path, before the
    manifest is published there; on a replace it runs on the staged copy before the move into
    place, then again at the registered path (Core ML compiles per path). "probe" in the result
    is the one at the registered path. An unpublished install an earlier import left behind
    (it died before publishing) is not a registered artifact: it is installed over, not
    replaced.
    If anything fails, the new artifact is not kept and an artifact it would replace is put
    back, so registration is all-or-nothing.

    Offline (`local_files_only`, or HF_HUB_OFFLINE) the index and archives are read from the
    Hugging Face cache only, and BackendUnavailableError is raised when they are not there;
    the network is never used. Every check above still runs."""
    from .lifecycle import BUILDING, import_artifact, unpublished

    repo = resolve_repo(repo)
    revision = resolve_revision(repo, revision)
    buckets = [int(b) for b in (buckets or spec.ane_buckets)]
    unoffered = [b for b in buckets if b not in spec.ane_buckets]
    if unoffered:
        raise ArtifactError(f"{spec.name} does not offer ANE buckets {unoffered}; offered: {list(spec.ane_buckets)}")

    def occupied(b: int) -> bool:
        """Something other than an unpublished install (which the import recovers) is there."""
        d = artifact_dir(spec, b)
        return d.exists() and not unpublished(d)

    todo, replace = [], set()
    for b in buckets:
        if not force:
            try:
                load_verified(spec, b)
            except Exception as e:  # any failure to load it here, not only ArtifactError, means not usable
                if not isinstance(e, ArtifactMissingError) or occupied(b):
                    log(
                        f"{spec.name} L{b}: the registered artifact is not usable "
                        f"({type(e).__name__}: {e}); fetching a replacement"
                    )
            else:
                log(f"{spec.name} L{b}: already registered and verified, skipped (--force replaces it)")
                continue
        todo.append(b)
        if occupied(b):  # replaced only after its replacement passes every check
            replace.add(b)
    if not todo:
        return {}
    profile = platform_profile()
    staging_root = artifacts_root() / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="fetch-", dir=staging_root))
    (tmp / BUILDING).write_text(json.dumps({"pid": os.getpid()}))  # `artifacts prune` leaves it alone
    out: dict = {}
    probes: dict = {}

    def probe(spec_: ModelSpec, bucket: int, compiled: Path) -> None:
        """Run inside import_artifact on the staged copy, before it is moved into place."""
        _probe(spec_, bucket, compiled, local_files_only=local_files_only)

    def registered_probe(spec_: ModelSpec, bucket: int, compiled: Path) -> None:
        """Run inside import_artifact at the registered path, before the move is kept; its
        result is the one fetch reports."""
        probes[bucket] = _probe(spec_, bucket, compiled, local_files_only=local_files_only)

    try:
        index_file = _download(repo, INDEX, revision, tmp, local_files_only=local_files_only)
        try:
            index = json.loads(Path(index_file).read_text())
        except ValueError as e:
            raise ArtifactIntegrityError(f"{repo}@{revision}/{INDEX} is not valid JSON: {e}") from e
        found, why = select(index, spec, todo, profile)
        if why:
            detail = "; ".join(f"L{b}: {w}" for b, w in sorted(why.items()))
            raise ArtifactMissingError(
                f"{repo}@{revision} has no usable prebuilt artifact for {spec.name} ({detail}). "
                f"Build it here: laya-apple artifacts build {spec.name}"
            )
        for b in todo:
            entry = found[b]
            log(f"{spec.name} L{b}: downloading {entry['path']} ({entry['bytes'] / 1e6:.0f} MB) from {repo}")
            archive = _download(repo, entry["path"], revision, tmp, local_files_only=local_files_only)
            digest = sha256_file(archive)
            if digest != entry["archive_sha256"]:
                raise ArtifactIntegrityError(
                    f"{repo}@{revision}/{entry['path']}: SHA-256 {digest[:16]}… does not match the index "
                    f"({entry['archive_sha256'][:16]}…)"
                )
            source = f"hf://{repo}@{revision}/{entry['path']}#sha256={digest}"
            path = import_artifact(
                archive,
                local_files_only=local_files_only,
                force=force or b in replace,
                log=log,
                source=source,
                probe=probe,
                registered_probe=registered_probe,
                expect=(spec.name, b),
            )
            if archive.is_relative_to(tmp):  # an offline read from the Hugging Face cache stays there
                archive.unlink()  # the registered copy is the only one kept
            p = probes[b]
            out[b] = {"path": path, "source": source, "probe": p}
            log(f"{spec.name} L{b}: placement probe ratio {p['ratio']} (ANE {p['ane_ms']} ms, CPU {p['cpu_ms']} ms)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out
