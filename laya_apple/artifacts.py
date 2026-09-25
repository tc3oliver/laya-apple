"""Fixed-shape Core ML artifacts: cache layout, manifest, strict verification, placement.

An artifact is usable only if every check below passes:
manifest schema, model revision and weight hash, graph variant, bucket, compute units,
parity status, file integrity, and - at load - a Core ML compute plan that is 100% Neural
Engine with no device transitions. Any failure raises a specific error; nothing falls back.

Compute-plan inspection adapted from mizorewww/laya-coreml@4619e04
benchmarks/common.py::compute_plan (Apache-2.0; see NOTICE).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from .errors import (
    ArtifactError,
    ArtifactIntegrityError,
    ArtifactMissingError,
    ArtifactParityError,
    ArtifactRevisionError,
    BackendUnavailableError,
    ComputeUnitMismatchError,
)
from .hub import cache_root
from .registry import ANE_COMPUTE_UNITS, ANE_GRAPH, ANE_MAX_OPTIONS, ANE_PRECISION, ModelSpec
from .schema import manifest_errors

MANIFEST_FORMAT = "laya-apple-artifact"
MANIFEST_VERSION = 1
COMPILED = "model.mlmodelc"


def artifacts_root() -> Path:
    return cache_root() / "artifacts"


def artifact_dir(spec: ModelSpec, bucket: int) -> Path:
    return artifacts_root() / spec.name / spec.revision[:12] / f"{ANE_GRAPH}-L{bucket}-B1"


def tree_sha256(path: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(Path(path).rglob("*")):
        if f.is_file():
            h.update(str(f.relative_to(path)).encode())
            fh = hashlib.sha256()
            with open(f, "rb") as stream:
                for block in iter(lambda: stream.read(8 << 20), b""):
                    fh.update(block)
            h.update(fh.digest())
    return h.hexdigest()


# ----------------------------------------------------------------------------- profile


def _sh(*cmd) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


@cache
def _profile() -> tuple:
    try:
        import coremltools as ct

        ct_version = ct.__version__
    except Exception:
        ct_version = None
    return tuple(
        {
            "soc": _sh("sysctl", "-n", "machdep.cpu.brand_string"),
            "macos": platform.mac_ver()[0],
            "macos_build": _sh("sw_vers", "-buildVersion"),
            "coremltools": ct_version,
        }.items()
    )


def platform_profile() -> dict:
    """SoC, macOS version/build and coremltools version of this process (cached)."""
    return dict(_profile())


def _major(version: str | None) -> str:
    return (version or "").split(".")[0]


def profile_matches(a: dict, b: dict) -> bool:
    """Same SoC, same macOS major, same coremltools version."""
    return (
        a.get("soc") == b.get("soc")
        and _major(a.get("macos")) == _major(b.get("macos"))
        and a.get("coremltools") == b.get("coremltools")
    )


# ----------------------------------------------------------------------------- manifest


@dataclass(frozen=True)
class Manifest:
    data: dict

    @property
    def artifact_sha256(self) -> str:
        return self.data["integrity"]["artifact_sha256"]

    @property
    def bucket(self) -> int:
        return int(self.data["artifact"]["length"])


def verify_manifest(spec: ModelSpec, bucket: int, data: dict, *, compute_units: str = ANE_COMPUTE_UNITS) -> Manifest:
    """Pure validation of a manifest against what is requested. Raises on any mismatch."""
    if not isinstance(data, dict):
        raise ArtifactIntegrityError(f"manifest is not a JSON object (got {type(data).__name__})")
    if data.get("format") != MANIFEST_FORMAT or data.get("format_version") != MANIFEST_VERSION:
        raise ArtifactIntegrityError(
            f"unsupported manifest format {data.get('format')!r}/{data.get('format_version')!r}"
        )
    problems = manifest_errors(data)
    try:
        manifest = _verify_fields(spec, bucket, data, compute_units)
    except (TypeError, ValueError, AttributeError):
        manifest = None  # malformed values; the schema errors below say which
    if problems or manifest is None:
        raise ArtifactIntegrityError("manifest does not match the format_version 1 schema: " + "; ".join(problems[:5]))
    return manifest


def _verify_fields(spec: ModelSpec, bucket: int, data: dict, compute_units: str) -> Manifest:
    src, art = data.get("source", {}), data.get("artifact", {})
    if src.get("model") != spec.name or src.get("repo") != spec.repo:
        raise ArtifactRevisionError(f"artifact is for {src.get('repo')!r}, not {spec.repo!r}")
    if src.get("revision") != spec.revision:
        raise ArtifactRevisionError(
            f"artifact was built from revision {str(src.get('revision'))[:12]}, pinned is {spec.revision[:12]}"
        )
    if src.get("weights_sha256") != spec.weights_sha256:
        raise ArtifactRevisionError("artifact was built from different model weights")
    if art.get("graph") != ANE_GRAPH:
        raise ArtifactRevisionError(f"artifact graph {art.get('graph')!r} is not the validated {ANE_GRAPH!r}")
    if int(art.get("length", -1)) != bucket or int(art.get("batch", -1)) != 1:
        raise ArtifactRevisionError(
            f"artifact shape L{art.get('length')}/B{art.get('batch')} != requested L{bucket}/B1"
        )
    if art.get("precision") != ANE_PRECISION or int(art.get("max_options", -1)) != ANE_MAX_OPTIONS:
        raise ArtifactRevisionError("artifact precision/options do not match the validated configuration")
    placement = data.get("placement", {})
    if compute_units != ANE_COMPUTE_UNITS:
        raise ComputeUnitMismatchError(
            f"ANE artifacts are validated only on {ANE_COMPUTE_UNITS}; {compute_units} was requested"
        )
    if placement.get("compute_units") != ANE_COMPUTE_UNITS:
        raise ComputeUnitMismatchError(
            f"artifact declares {placement.get('compute_units')!r}, expected {ANE_COMPUTE_UNITS}"
        )
    parity = data.get("parity") or {}
    if parity.get("passed") is not True or data.get("status") != "validated":
        raise ArtifactParityError(
            f"artifact has no passing parity record (status={data.get('status')!r}, passed={parity.get('passed')!r})"
        )
    if not data.get("integrity", {}).get("artifact_sha256"):
        raise ArtifactIntegrityError("manifest has no artifact hash")
    return Manifest(data)


def verify_profile(data: dict, current: dict | None = None) -> None:
    """Parity and placement were validated on the build profile; refuse any other."""
    built = data.get("platform") or {}
    current = current or platform_profile()
    if not profile_matches(built, current):
        raise ArtifactRevisionError(
            f"artifact was validated on {built.get('soc')!r} macOS {built.get('macos')!r} "
            f"coremltools {built.get('coremltools')!r}; this machine is {current.get('soc')!r} "
            f"macOS {current.get('macos')!r} coremltools {current.get('coremltools')!r}. Rebuild it here."
        )


def read_manifest(spec: ModelSpec, bucket: int) -> dict:
    d = artifact_dir(spec, bucket)
    path = d / "manifest.json"
    if not path.exists() or not (d / COMPILED).exists():
        raise ArtifactMissingError(
            f"no ANE artifact for {spec.name}@{spec.revision[:12]} L{bucket} in {d.parent}. "
            f"Build it with: laya-apple artifacts build {spec.name} --length {bucket}"
        )
    try:
        return json.loads(path.read_text())
    except ValueError as e:
        raise ArtifactIntegrityError(f"unreadable manifest {path}: {e}") from e


def verify_files(manifest: Manifest, directory: Path) -> None:
    actual = tree_sha256(directory / COMPILED)
    if actual != manifest.artifact_sha256:
        raise ArtifactIntegrityError(
            f"{directory / COMPILED} does not match its manifest hash ({actual[:12]} != {manifest.artifact_sha256[:12]})"
        )


# ----------------------------------------------------------------------------- placement


def compute_plan_summary(compiled_path: Path, compute_units: str = ANE_COMPUTE_UNITS) -> dict:
    """Core ML's anticipated placement: op counts per device, cost share, transitions."""
    import coremltools as ct

    short = {"MLCPUComputeDevice": "cpu", "MLGPUComputeDevice": "gpu", "MLNeuralEngineComputeDevice": "ane"}
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        str(compiled_path), compute_units=getattr(ct.ComputeUnit, compute_units)
    )
    counts, cost, seq = {"ane": 0, "gpu": 0, "cpu": 0}, {"ane": 0.0, "gpu": 0.0, "cpu": 0.0}, []

    def visit(block):
        for op in block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if usage is not None and op.operator_name != "const":
                dev = short.get(type(usage.preferred_compute_device).__name__)
                if dev:
                    counts[dev] += 1
                    seq.append(dev)
                    est = plan.get_estimated_cost_for_mlprogram_operation(op)
                    if est:
                        cost[dev] += est.weight
            for nested in op.blocks:
                visit(nested)

    for fn in plan.model_structure.program.functions.values():
        visit(fn.block)
    total = sum(cost.values()) or 1.0
    return {
        "compute_units": compute_units,
        "ops": counts,
        "cost_share": {k: v / total for k, v in cost.items()},
        "transitions": sum(1 for a, b in zip(seq, seq[1:]) if a != b),
    }


def check_ane_placement(summary: dict) -> None:
    ops = summary["ops"]
    if ops["ane"] == 0 or ops["cpu"] or ops["gpu"] or summary["transitions"]:
        raise ComputeUnitMismatchError(
            "Core ML did not place the artifact entirely on the Neural Engine "
            f"(ops ane={ops['ane']} cpu={ops['cpu']} gpu={ops['gpu']}, transitions={summary['transitions']}). "
            "This happens when ANE compilation fails; refusing to run it elsewhere."
        )


def _stamp_key(manifest: Manifest, directory: Path, compute_units: str, profile: dict) -> dict:
    files = sorted(f for f in (directory / COMPILED).rglob("*") if f.is_file())
    stat = hashlib.sha256()
    for f in files:
        st = f.stat()
        stat.update(f"{f.relative_to(directory)}:{st.st_size}:{st.st_mtime_ns}:{st.st_ino}".encode())
    return {
        "artifact_sha256": manifest.artifact_sha256,
        "files_stat": stat.hexdigest(),
        "compute_units": compute_units,
        "macos_build": profile.get("macos_build"),
        "coremltools": profile.get("coremltools"),
    }


def _stamp_path(directory: Path) -> Path:
    rel = directory.relative_to(artifacts_root())
    return cache_root() / "verified" / "artifacts" / (str(rel).replace("/", "__") + ".json")


def load_verified(
    spec: ModelSpec, bucket: int, *, compute_units: str = ANE_COMPUTE_UNITS, full: bool = False, open_model=None
):
    """Return (model, Manifest) after every check, or raise.

    The model is coremltools' CompiledMLModel, or `open_model(compiled_path, compute_units)`
    if given (the GIL-releasing binding, backends/coreml_nogil.py). Every check below runs
    before either opens the artifact.

    Manifest, profile and compute-unit checks run on every load. The file hash and the
    Core ML compute-plan placement check (together ~1.1 s per bucket) run on the first load
    and again whenever the artifact's files (size, mtime, inode), the macOS build or the
    coremltools version change; the passing result is stamped in the cache. `full=True`
    (`laya-apple artifacts verify`) always runs both.
    """
    try:
        import coremltools as ct
    except ImportError as e:
        raise BackendUnavailableError(
            "the ANE path needs coremltools: install the [ane] extra (uv sync --extra ane)"
        ) from e
    d = artifact_dir(spec, bucket)
    try:
        data = read_manifest(spec, bucket)
    except ArtifactIntegrityError as e:
        raise _quarantined(spec, bucket, d, e) from e
    try:
        manifest = verify_manifest(spec, bucket, data, compute_units=compute_units)
    except ArtifactIntegrityError as e:
        raise _quarantined(spec, bucket, d, e) from e
    profile = platform_profile()
    verify_profile(manifest.data, profile)
    key, stamp = _stamp_key(manifest, d, compute_units, profile), _stamp_path(d)
    cached = False
    if not full and stamp.exists():
        try:
            cached = json.loads(stamp.read_text()).get("key") == key
        except (OSError, ValueError):
            cached = False
    if not cached:
        try:
            verify_files(manifest, d)
        except ArtifactIntegrityError as e:
            # A hash mismatch can be a race: --force or import can swap the directory's
            # contents between our manifest read and our hash. Re-read and re-hash once
            # before quarantining a freshly (re)built artifact.
            try:
                data2 = read_manifest(spec, bucket)
                manifest2 = verify_manifest(spec, bucket, data2, compute_units=compute_units)
                verify_files(manifest2, d)
            except ArtifactError:
                raise _quarantined(spec, bucket, d, e) from e
            manifest = manifest2
        placement = compute_plan_summary(d / COMPILED, compute_units)
        check_ane_placement(placement)
        stamp.parent.mkdir(parents=True, exist_ok=True)
        tmp = stamp.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"key": key, "placement": placement}))
        os.replace(tmp, stamp)
    if open_model is not None:
        return open_model(d / COMPILED, compute_units), manifest
    model = ct.models.CompiledMLModel(str(d / COMPILED), compute_units=getattr(ct.ComputeUnit, compute_units))
    return model, manifest


def _quarantined(spec: ModelSpec, bucket: int, directory: Path, error: ArtifactIntegrityError):
    """Move a corrupt artifact aside and return an error that says where and how to rebuild."""
    from .lifecycle import quarantine

    moved = quarantine(directory, str(error))
    where = f"; moved to {moved}" if moved else ""
    return ArtifactIntegrityError(
        f"{error}{where}. Rebuild it with: laya-apple artifacts build {spec.name} --length {bucket}"
    )


def list_artifacts() -> list[dict]:
    out = []
    root = artifacts_root()
    if not root.exists():
        return out
    for m in sorted(root.glob("*/*/*/manifest.json")):
        try:
            data = json.loads(m.read_text())
        except ValueError:
            data = {"status": "unreadable"}
        out.append({"path": str(m.parent), "manifest": data})
    return out


def artifact_capabilities() -> list[dict]:
    """One provenance record per registered artifact (see docs/guide.md: Artifact
    provenance). Missing manifest fields become `null`; a manifest never crashes this."""
    from .registry import models

    specs = models()
    out = []
    for x in list_artifacts():
        m = x["manifest"] if isinstance(x["manifest"], dict) else {}
        src = m.get("source") or {}
        conv = m.get("conversion") or {}
        art = m.get("artifact") or {}
        placement = m.get("placement") or {}
        parity = m.get("parity") or {}
        integrity = m.get("integrity") or {}
        near_tie = parity.get("near_tie_flips")
        bucket = art.get("length")
        spec = specs.get(src.get("model"))
        out.append(
            {
                "path": x["path"],
                "status": m.get("status"),
                "model": src.get("model"),
                "repo": src.get("repo"),
                "revision": src.get("revision"),
                "source_weights_sha256": src.get("weights_sha256"),
                "conversion_revision": {
                    "laya_apple_version": conv.get("laya_apple_version"),
                    "git_revision": conv.get("git_revision"),
                    "code_sha256": conv.get("code_sha256"),
                },
                "graph": art.get("graph"),
                "bucket": bucket,
                "batch": art.get("batch"),
                "compute_target": {
                    "compute_units": placement.get("compute_units"),
                    "ops": placement.get("ops"),
                    "transitions": placement.get("transitions"),
                },
                "precision": art.get("precision"),
                "platform": m.get("platform"),
                "parity": {
                    "passed": parity.get("passed"),
                    "tolerance": parity.get("tolerance"),
                    "prob_max_abs": parity.get("prob_max_abs"),
                    "hard_mismatches": parity.get("hard_mismatches"),
                    "near_tie_flips": len(near_tie) if isinstance(near_tie, list) else None,
                },
                "artifact_sha256": integrity.get("artifact_sha256"),
                "offered_by_auto": bool(spec) and bucket in spec.auto_ane_buckets,
                "offered_explicit": bool(spec) and bucket in spec.ane_buckets,
            }
        )
    return out
