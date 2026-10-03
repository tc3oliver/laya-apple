"""laya_apple/prebuilt.py without the network or Core ML: repository resolution, the index
format, per-profile selection, that `fetch` skips only buckets whose registered artifact passes
load_verified, and that it hands every archive to import_artifact (which re-validates it) only
after its SHA-256 matches, with the placement probe run on the staged copy before registration.

coremltools is replaced by a stand-in module and the compute plan by a clean ANE summary, so
the real load_verified and import_artifact checks (manifest, profile, file hash) run here."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
import tarfile
import types
from pathlib import Path

import pytest

import laya_apple.artifacts as A
import laya_apple.hub as hub
import laya_apple.parity.ane as parity_ane
from laya_apple import cli, lifecycle, prebuilt
from laya_apple.artifacts import COMPILED, artifact_dir, artifacts_root, tree_sha256
from laya_apple.errors import (
    ArtifactError,
    ArtifactIntegrityError,
    ArtifactMissingError,
    ArtifactParityError,
    BackendUnavailableError,
    ComputeUnitMismatchError,
)
from laya_apple.registry import ANE_COMPUTE_UNITS, ANE_GRAPH, models

SPEC = models()["laya"]
HERE = {"soc": "Apple M4 Max", "macos": "26.0.1", "macos_build": "25A1", "coremltools": "9.0"}
OTHER = {"soc": "Apple M1", "macos": "15.6", "macos_build": "24G1", "coremltools": "9.0"}
REAL_IMPORT = lifecycle.import_artifact
REAL_DOWNLOAD = prebuilt._download
REAL_PROBE = prebuilt._probe
STAGED_RATIO, REGISTERED_RATIO = 0.333, 0.25
CLEAN_PLAN = {"compute_units": "CPU_AND_NE", "ops": {"ane": 1, "cpu": 0, "gpu": 0}, "transitions": 0}


def entry(bucket, platform=HERE, payload=b"archive", **overrides):
    e = {
        "path": prebuilt.archive_path(SPEC, bucket, platform),
        "model": SPEC.name,
        "revision": SPEC.revision,
        "weights_sha256": SPEC.weights_sha256,
        "graph": ANE_GRAPH,
        "length": bucket,
        "platform": platform,
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "artifact_sha256": "a" * 64,
    }
    e.update(overrides)
    return e


def index(*entries):
    return {"format": prebuilt.INDEX_FORMAT, "format_version": prebuilt.INDEX_VERSION, "artifacts": list(entries)}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A fake repository: {filename: bytes}; records downloads, imports and probes. The probe
    raises ComputeUnitMismatchError for the buckets in calls["probe_fails"] on the staged copy
    and in calls["registered_fails"] at the registered path; its ratio says which it probed."""
    monkeypatch.setenv("LAYA_APPLE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(prebuilt, "platform_profile", lambda: dict(HERE))
    monkeypatch.setattr(A, "platform_profile", lambda: dict(HERE))
    ct = types.ModuleType("coremltools")
    ct.ComputeUnit = types.SimpleNamespace(CPU_AND_NE="CPU_AND_NE", CPU_ONLY="CPU_ONLY")
    ct.models = types.SimpleNamespace(CompiledMLModel=lambda *a, **k: object())
    monkeypatch.setitem(sys.modules, "coremltools", ct)
    monkeypatch.setattr(A, "compute_plan_summary", lambda *a, **k: dict(CLEAN_PLAN))
    files: dict = {}
    calls = {"download": [], "import": [], "probe": [], "probe_fails": set(), "registered_fails": set()}

    def download(repo_id, filename, revision, local_dir, *, local_files_only=False):
        calls["download"].append((repo_id, filename, revision, local_files_only))
        if filename not in files:
            raise BackendUnavailableError(f"404 {filename}")
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(files[filename])
        return path

    def import_artifact(archive, *, local_files_only, force, log, source, probe, registered_probe, expect):
        assert Path(archive).exists()
        calls["import"].append({"archive": Path(archive).name, "source": source, "force": force, "expect": expect})
        bucket = int(re.search(r"-L(\d+)\.tar\.gz$", Path(archive).name).group(1))
        probe(SPEC, bucket, Path("/staged") / COMPILED)
        registered_probe(SPEC, bucket, Path("/registered") / COMPILED)
        return Path("/registered") / Path(archive).name

    def probe(spec, bucket, compiled, *, local_files_only):
        compiled = Path(compiled)
        staged = compiled.parts[1] == "staged" or ".staging" in compiled.parts
        calls["probe"].append(bucket)
        calls.setdefault("probe_paths", []).append(compiled)
        if bucket in (calls["probe_fails"] if staged else calls["registered_fails"]):
            where = "staged" if staged else "registered"
            raise ComputeUnitMismatchError(f"{spec.name} L{bucket}: simulated {where} probe failure")
        return {"ane_ms": 1.0, "cpu_ms": 3.0, "ratio": STAGED_RATIO if staged else REGISTERED_RATIO}

    monkeypatch.setattr(prebuilt, "_download", download)
    monkeypatch.setattr(lifecycle, "import_artifact", import_artifact)
    monkeypatch.setattr(prebuilt, "_probe", probe)
    return files, calls


def put(files, *entries, payloads=None):
    files[prebuilt.INDEX] = json.dumps(index(*entries)).encode()
    for e in entries:
        files[e["path"]] = (payloads or {}).get(e["path"], b"archive")


def registered(manifest_factory, bucket, weights=b"old weights"):
    """A valid artifact at the path the runtime loads `bucket` from."""
    d = artifact_dir(SPEC, bucket)
    (d / COMPILED).mkdir(parents=True)
    (d / COMPILED / "weights.bin").write_bytes(weights)
    data = manifest_factory(SPEC, bucket, platform=HERE)
    data["integrity"]["artifact_sha256"] = tree_sha256(d / COMPILED)
    (d / "manifest.json").write_text(json.dumps(data))
    return d


def snapshot(d: Path) -> dict:
    return {str(f.relative_to(d)): f.read_bytes() for f in sorted(d.rglob("*")) if f.is_file()}


def export_archive(manifest_factory, bucket, weights=b"new weights") -> bytes:
    """An `artifacts export` .tar.gz for `bucket`, as bytes."""
    data = manifest_factory(SPEC, bucket, platform=HERE)
    h = hashlib.sha256()
    h.update(b"weights.bin")
    h.update(hashlib.sha256(weights).digest())
    data["integrity"]["artifact_sha256"] = h.hexdigest()  # tree_sha256 of the one-file model
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, payload in (("manifest.json", json.dumps(data).encode()), (f"{COMPILED}/weights.bin", weights)):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


@pytest.fixture
def real_import(repo, tmp_path, monkeypatch):
    """The real import_artifact, with the checkpoint, weight check and parity gate stubbed
    (parity passes); the placement probe is the `repo` fixture's."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    monkeypatch.setattr(lifecycle, "import_artifact", REAL_IMPORT)
    monkeypatch.setattr(hub, "checkpoint_path", lambda spec, local_files_only=False: ckpt)
    monkeypatch.setattr(hub, "verify_weights", lambda spec, path: "ok")
    monkeypatch.setattr(parity_ane, "ane_parity", lambda *a, **k: {"passed": True})
    return repo


# ----------------------------------------------------------------------------- repository


def test_the_default_repository_is_used_once_published(monkeypatch):
    monkeypatch.delenv(prebuilt.PREBUILT_REPO_ENV, raising=False)
    assert prebuilt.DEFAULT_REPO_PUBLISHED is True
    assert prebuilt.resolve_repo() == prebuilt.DEFAULT_PREBUILT_REPO


def test_an_unpublished_default_repository_is_refused(monkeypatch):
    monkeypatch.delenv(prebuilt.PREBUILT_REPO_ENV, raising=False)
    monkeypatch.setattr(prebuilt, "DEFAULT_REPO_PUBLISHED", False)
    with pytest.raises(BackendUnavailableError, match="no prebuilt artifact repository is published"):
        prebuilt.resolve_repo()
    monkeypatch.setenv(prebuilt.PREBUILT_REPO_ENV, "someone/artifacts")
    assert prebuilt.resolve_repo() == "someone/artifacts"
    assert prebuilt.resolve_repo("explicit/repo") == "explicit/repo"


def test_the_default_repository_is_pinned_to_an_immutable_commit():
    pin = prebuilt.DEFAULT_PREBUILT_REVISION
    assert re.fullmatch(r"[0-9a-f]{40}", pin)
    assert prebuilt.resolve_revision(prebuilt.DEFAULT_PREBUILT_REPO) == pin
    assert prebuilt.resolve_revision(prebuilt.DEFAULT_PREBUILT_REPO, "") == pin
    # an explicit revision still wins, also on the default repository
    assert prebuilt.resolve_revision(prebuilt.DEFAULT_PREBUILT_REPO, "main") == "main"
    assert prebuilt.resolve_revision(prebuilt.DEFAULT_PREBUILT_REPO, "abc123") == "abc123"


def test_another_repository_is_not_bound_to_the_default_pin():
    assert prebuilt.resolve_revision("someone/artifacts") == prebuilt.CUSTOM_REPO_REVISION == "main"
    assert prebuilt.resolve_revision("someone/artifacts", "v2") == "v2"


def test_archive_path_names_model_revision_and_profile():
    path = prebuilt.archive_path(SPEC, 64, HERE)
    assert path == f"laya/{SPEC.revision[:12]}/apple-m4-max-macos26-coremltools9.0/laya-L64.tar.gz"


# ----------------------------------------------------------------------------- index


@pytest.mark.parametrize(
    "bad, message",
    [
        ({"format": "other"}, "not a laya-apple prebuilt index"),
        ({"format": prebuilt.INDEX_FORMAT, "format_version": 2, "artifacts": []}, "format_version"),
        ({"format": prebuilt.INDEX_FORMAT, "format_version": 1}, "no 'artifacts' list"),
        (index({"path": "x.tar.gz"}), "missing"),
        (index(entry(64, path="../escape.tar.gz")), "unsafe path"),
        (index(entry(64, path="/abs/laya-L64.tar.gz")), "unsafe path"),
        (index(entry(64, path="laya/model.mlmodelc")), "unsafe path"),
    ],
)
def test_malformed_index_is_rejected(bad, message):
    with pytest.raises(ArtifactIntegrityError, match=message):
        prebuilt.validate_index(bad)


def test_merge_replaces_by_path_and_keeps_other_entries():
    old = index(entry(64, platform=OTHER), entry(64, artifact_sha256="b" * 64))
    merged = prebuilt.merge_index(old, [entry(64, artifact_sha256="c" * 64), entry(96)])
    by_path = {e["path"]: e for e in merged["artifacts"]}
    assert len(by_path) == 3
    assert by_path[prebuilt.archive_path(SPEC, 64, HERE)]["artifact_sha256"] == "c" * 64
    assert [e["path"] for e in merged["artifacts"]] == sorted(by_path)
    assert prebuilt.merge_index(None, [entry(64)])["artifacts"] == [entry(64)]


def test_make_index_entry_hashes_the_archive(tmp_path):
    archive = tmp_path / "laya-L64.tar.gz"
    archive.write_bytes(b"archive")
    manifest = {
        "source": {"model": SPEC.name, "revision": SPEC.revision, "weights_sha256": SPEC.weights_sha256},
        "artifact": {"graph": ANE_GRAPH, "length": 64},
        "platform": HERE,
        "integrity": {"artifact_sha256": "a" * 64},
    }
    assert prebuilt.make_index_entry(manifest, archive, entry(64)["path"]) == entry(64)


def test_select_matches_checkpoint_bucket_and_profile():
    idx = index(
        entry(64),
        entry(96, platform=OTHER),
        entry(128, revision="0" * 40),
        entry(64, weights_sha256="0" * 64, path="laya/x/y/laya-L64.tar.gz"),
    )
    found, why = prebuilt.select(idx, SPEC, [64, 96, 128], HERE)
    assert list(found) == [64] and found[64]["path"] == entry(64)["path"]
    assert "built only for apple-m1-macos15" in why[96]
    assert "no archive" in why[128]


# ----------------------------------------------------------------------------- fetch


def test_fetch_validates_each_archive_through_import_then_probes(repo):
    files, calls = repo
    put(files, entry(64), entry(96))
    out = prebuilt.fetch(SPEC, [64, 96], repo="o/r", revision="abc", log=lambda m: None)
    assert [c["archive"] for c in calls["import"]] == ["laya-L64.tar.gz", "laya-L96.tar.gz"]
    digest = hashlib.sha256(b"archive").hexdigest()
    assert calls["import"][0]["source"] == f"hf://o/r@abc/{entry(64)['path']}#sha256={digest}"
    assert calls["probe"] == [64, 64, 96, 96]  # staged, then registered, per bucket
    assert out[64]["probe"]["ratio"] == REGISTERED_RATIO and out[96]["probe"]["ratio"] == REGISTERED_RATIO
    assert all(d[0] == "o/r" and d[2] == "abc" for d in calls["download"])
    staging = artifacts_root() / ".staging"
    assert not any(staging.iterdir())  # downloads are not kept beside the registered copy


def test_hash_mismatch_is_refused_before_import(repo):
    files, calls = repo
    put(files, entry(64), payloads={entry(64)["path"]: b"tampered"})
    with pytest.raises(ArtifactIntegrityError, match="does not match the index"):
        prebuilt.fetch(SPEC, [64], repo="o/r", log=lambda m: None)
    assert calls["import"] == [] and calls["probe"] == []


def test_no_archive_for_this_profile_downloads_nothing_else(repo):
    files, calls = repo
    put(files, entry(64, platform=OTHER))
    with pytest.raises(ArtifactMissingError, match="artifacts build laya"):
        prebuilt.fetch(SPEC, [64], repo="o/r", log=lambda m: None)
    assert [d[1] for d in calls["download"]] == [prebuilt.INDEX]
    assert calls["import"] == []


def test_already_registered_bucket_is_skipped_unless_forced(repo, manifest_factory):
    files, calls = repo
    put(files, entry(64))
    registered(manifest_factory, 64)
    logs = []
    assert prebuilt.fetch(SPEC, [64], repo="o/r", log=logs.append) == {}
    assert calls["download"] == [] and calls["import"] == []
    assert any("already registered and verified, skipped" in m for m in logs)
    prebuilt.fetch(SPEC, [64], repo="o/r", force=True, log=lambda m: None)
    assert calls["import"][0]["force"] is True


@pytest.mark.parametrize("with_model", [True, False], ids=["manifest-and-model", "manifest-only"])
def test_an_empty_manifest_is_not_accepted_as_registered(repo, with_model):
    files, calls = repo
    put(files, entry(64))
    d = artifact_dir(SPEC, 64)
    d.mkdir(parents=True)
    if with_model:
        (d / COMPILED).mkdir()
    (d / "manifest.json").write_text("{}")
    logs = []
    out = prebuilt.fetch(SPEC, [64], repo="o/r", log=logs.append)
    assert list(out) == [64] and len(calls["import"]) == 1
    assert any("the registered artifact is not usable" in m for m in logs)
    if with_model:  # load_verified quarantined it, so the import needs no force
        assert not d.exists() and len(list((artifacts_root() / "quarantine").iterdir())) == 1
        assert calls["import"][0]["force"] is False
    else:  # incomplete, left where it is: replaced only by an import that passes every check
        assert calls["import"][0]["force"] is True


def test_a_file_integrity_failure_is_not_accepted_as_registered(repo, manifest_factory):
    files, calls = repo
    put(files, entry(64))
    d = registered(manifest_factory, 64)
    (d / COMPILED / "weights.bin").write_bytes(b"tampered")
    logs = []
    out = prebuilt.fetch(SPEC, [64], repo="o/r", log=logs.append)
    assert list(out) == [64] and len(calls["import"]) == 1
    assert any("does not match its manifest hash" in m for m in logs)
    assert not d.exists()  # quarantined by load_verified, not reported as registered
    moved = list((artifacts_root() / "quarantine").iterdir())
    assert len(moved) == 1 and (moved[0] / "QUARANTINED.json").exists()


def assert_probed(calls, *, registered: bool):
    """The first probe ran on the staged copy; the second, if any, at the registered path."""
    staged, *rest = calls["probe_paths"]
    assert staged.is_relative_to(artifacts_root() / ".staging") and staged.name == COMPILED
    assert not staged.is_relative_to(artifact_dir(SPEC, 64))
    assert rest == ([artifact_dir(SPEC, 64) / COMPILED] if registered else [])


def test_a_failed_placement_probe_registers_nothing(real_import, manifest_factory):
    """With `force` (the replace path, here with nothing to replace) the staged copy is probed first."""
    files, calls = real_import
    payload = export_archive(manifest_factory, 64)
    put(files, entry(64, payload=payload), payloads={entry(64)["path"]: payload})
    calls["probe_fails"].add(64)
    with pytest.raises(ComputeUnitMismatchError, match="simulated staged probe failure"):
        prebuilt.fetch(SPEC, [64], repo="o/r", force=True, log=lambda m: None)
    assert calls["probe"] == [64]
    assert_probed(calls, registered=False)
    assert not artifact_dir(SPEC, 64).exists()
    assert not any((artifacts_root() / ".staging").iterdir())


def test_a_failed_registered_path_probe_unregisters_the_new_artifact(real_import, manifest_factory):
    """A first install: its one probe runs at the registered path, before the manifest is published."""
    files, calls = real_import
    payload = export_archive(manifest_factory, 64)
    put(files, entry(64, payload=payload), payloads={entry(64)["path"]: payload})
    calls["registered_fails"].add(64)
    with pytest.raises(ComputeUnitMismatchError, match="simulated registered probe failure"):
        prebuilt.fetch(SPEC, [64], repo="o/r", log=lambda m: None)
    assert calls["probe_paths"] == [artifact_dir(SPEC, 64) / COMPILED]
    assert not (artifacts_root() / "quarantine").exists()
    assert not artifact_dir(SPEC, 64).exists()
    assert not any(artifact_dir(SPEC, 64).parent.iterdir())  # no .old-* or partial copy either
    assert not any((artifacts_root() / ".staging").iterdir())


def test_a_failed_registered_path_probe_restores_the_previous_artifact(real_import, manifest_factory):
    files, calls = real_import
    old = registered(manifest_factory, 64)
    before = snapshot(old)
    payload = export_archive(manifest_factory, 64)
    put(files, entry(64, payload=payload), payloads={entry(64)["path"]: payload})
    calls["registered_fails"].add(64)
    with pytest.raises(ComputeUnitMismatchError, match="simulated registered probe failure"):
        prebuilt.fetch(SPEC, [64], repo="o/r", force=True, log=lambda m: None)
    assert_probed(calls, registered=True)
    assert snapshot(old) == before
    assert [p.name for p in old.parent.iterdir()] == [old.name]
    A.load_verified(SPEC, 64, full=True)


def test_a_failed_restore_says_where_the_previous_artifact_is(real_import, manifest_factory, monkeypatch):
    files, calls = real_import
    old = registered(manifest_factory, 64)
    payload = export_archive(manifest_factory, 64)
    put(files, entry(64, payload=payload), payloads={entry(64)["path"]: payload})
    calls["registered_fails"].add(64)
    real_rename = os.rename

    def rename(src, dst):
        if ".old-" in Path(src).name:  # putting the previous artifact back
            raise OSError("simulated restore failure")
        return real_rename(src, dst)

    monkeypatch.setattr(lifecycle.os, "rename", rename)
    logs = []
    with pytest.raises(ComputeUnitMismatchError):  # the original error, not the restore's
        prebuilt.fetch(SPEC, [64], repo="o/r", force=True, log=logs.append)
    monkeypatch.setattr(lifecycle.os, "rename", real_rename)
    left = [p for p in old.parent.iterdir() if ".old-" in p.name]
    assert len(left) == 1 and not old.exists()
    assert any(f"could not restore the previous artifact; it was left at {left[0]}" in m for m in logs)


def test_a_failed_move_aside_keeps_the_previous_artifact_out_of_prune(
    real_import, manifest_factory, monkeypatch, capsys
):
    """The registered-path probe fails and the failed artifact cannot be moved aside: the
    failed artifact (with its manifest) stays registered and the previous one at .old-<pid>.
    prune must never delete that .old-<pid>, and `artifacts prune` reports it."""
    from laya_apple import cli

    files, calls = real_import
    old = registered(manifest_factory, 64)
    before = snapshot(old)
    payload = export_archive(manifest_factory, 64)
    put(files, entry(64, payload=payload), payloads={entry(64)["path"]: payload})
    calls["registered_fails"].add(64)
    real_rename = os.rename

    def rename(src, dst):
        if ".failed-" in Path(dst).name:  # moving the failed artifact aside
            raise OSError("simulated move-aside failure")
        return real_rename(src, dst)

    monkeypatch.setattr(lifecycle.os, "rename", rename)
    logs = []
    with pytest.raises(ComputeUnitMismatchError):
        prebuilt.fetch(SPEC, [64], repo="o/r", force=True, log=logs.append)
    monkeypatch.setattr(lifecycle.os, "rename", real_rename)
    (kept,) = [p for p in old.parent.iterdir() if ".old-" in p.name]
    assert snapshot(kept) == before and (old / "manifest.json").exists()  # the failed one is still registered
    assert any(f"the previous artifact is at {kept}" in m for m in logs)
    monkeypatch.setattr(lifecycle, "_pid_alive", lambda pid: False)  # as after this process exits
    assert all(item["path"] != str(kept) for item in lifecycle.plan_prune(dict(HERE)))
    a = cli.build_parser().parse_args(["artifacts", "prune", "--yes"])
    assert cli.cmd_artifacts(a) == 0
    out = capsys.readouterr().out
    assert (
        f"keep  previous artifact kept from an interrupted or failed replace (an artifact is registered at {old})"
        in out
    )
    assert snapshot(kept) == before


def test_a_partly_removed_failed_artifact_does_not_strand_the_previous_one(real_import, manifest_factory, monkeypatch):
    files, calls = real_import
    old = registered(manifest_factory, 64)
    before = snapshot(old)
    payload = export_archive(manifest_factory, 64)
    put(files, entry(64, payload=payload), payloads={entry(64)["path"]: payload})
    calls["registered_fails"].add(64)
    real_rmtree = lifecycle.shutil.rmtree

    def rmtree(path, *a, **k):
        if ".failed-" in Path(path).name:  # removes the manifest, then gives up
            (Path(path) / "manifest.json").unlink()
            return None
        return real_rmtree(path, *a, **k)

    monkeypatch.setattr(lifecycle.shutil, "rmtree", rmtree)
    with pytest.raises(ComputeUnitMismatchError):
        prebuilt.fetch(SPEC, [64], repo="o/r", force=True, log=lambda m: None)
    monkeypatch.setattr(lifecycle.shutil, "rmtree", real_rmtree)
    assert snapshot(old) == before  # the previous artifact is back where the runtime looks
    A.load_verified(SPEC, 64, full=True)
    (leftover,) = [p for p in old.parent.iterdir() if p != old]
    assert ".failed-" in leftover.name and not (leftover / "manifest.json").exists()
    assert lifecycle.plan_prune(dict(HERE)) == []  # kept while the process that left it is alive
    monkeypatch.setattr(lifecycle, "_pid_alive", lambda pid: False)  # as after this process exits
    plan = lifecycle.plan_prune(dict(HERE))
    assert [(item["path"], item["reason"]) for item in plan] == [
        (str(leftover), "leftover failed copy from an earlier import")
    ]
    lifecycle.prune(plan)
    assert [p.name for p in old.parent.iterdir()] == [old.name]


def test_a_load_failure_at_the_registered_path_restores_the_previous_artifact(
    real_import, manifest_factory, monkeypatch
):
    files, calls = real_import
    old = registered(manifest_factory, 64)
    before = snapshot(old)
    payload = export_archive(manifest_factory, 64)
    put(files, entry(64, payload=payload), payloads={entry(64)["path"]: payload})
    real_load = A.load_verified

    def load_verified(spec, bucket, *, full=False, **k):
        if full:  # the import's load at the registered path
            raise RuntimeError("Core ML could not load the registered model")
        return real_load(spec, bucket, full=full, **k)

    monkeypatch.setattr(A, "load_verified", load_verified)
    with pytest.raises(RuntimeError, match="could not load the registered model"):
        prebuilt.fetch(SPEC, [64], repo="o/r", force=True, log=lambda m: None)
    monkeypatch.setattr(A, "load_verified", real_load)
    assert_probed(calls, registered=False)  # it never got to the registered-path probe
    assert snapshot(old) == before
    assert [p.name for p in old.parent.iterdir()] == [old.name]


def test_an_interrupt_at_the_registered_path_restores_the_previous_artifact(real_import, manifest_factory, monkeypatch):
    files, calls = real_import
    old = registered(manifest_factory, 64)
    before = snapshot(old)
    payload = export_archive(manifest_factory, 64)
    put(files, entry(64, payload=payload), payloads={entry(64)["path"]: payload})
    staged_probe = prebuilt._probe

    def probe(spec, bucket, compiled, *, local_files_only):
        if ".staging" not in Path(compiled).parts:
            raise KeyboardInterrupt
        return staged_probe(spec, bucket, compiled, local_files_only=local_files_only)

    monkeypatch.setattr(prebuilt, "_probe", probe)
    with pytest.raises(KeyboardInterrupt):
        prebuilt.fetch(SPEC, [64], repo="o/r", force=True, log=lambda m: None)
    assert snapshot(old) == before
    assert [p.name for p in old.parent.iterdir()] == [old.name]
    assert not any((artifacts_root() / ".staging").iterdir())


def test_a_failed_forced_replacement_keeps_the_previous_artifact(real_import, manifest_factory):
    files, calls = real_import
    old = registered(manifest_factory, 64)
    before = snapshot(old)
    payload = export_archive(manifest_factory, 64)
    put(files, entry(64, payload=payload), payloads={entry(64)["path"]: payload})
    calls["probe_fails"].add(64)
    with pytest.raises(ComputeUnitMismatchError):
        prebuilt.fetch(SPEC, [64], repo="o/r", force=True, log=lambda m: None)
    assert_probed(calls, registered=False)
    assert snapshot(old) == before
    A.load_verified(SPEC, 64, full=True)  # still a valid registered artifact
    assert [p.name for p in old.parent.iterdir()] == [old.name]  # no leftover .old-* directory


def test_a_successful_fetch_registers_with_provenance_and_probe(real_import, manifest_factory):
    files, calls = real_import
    old = registered(manifest_factory, 64)
    payload = export_archive(manifest_factory, 64)
    put(files, entry(64, payload=payload), payloads={entry(64)["path"]: payload})
    out = prebuilt.fetch(SPEC, [64], repo="o/r", revision="abc", force=True, log=lambda m: None)
    assert out[64]["path"] == old and out[64]["probe"]["ratio"] == REGISTERED_RATIO  # the registered-path probe
    assert_probed(calls, registered=True)
    assert (old / COMPILED / "weights.bin").read_bytes() == b"new weights"
    data = json.loads((old / "manifest.json").read_text())
    assert data["imported"]["from"] == out[64]["source"]
    assert out[64]["source"] == f"hf://o/r@abc/{entry(64)['path']}#sha256={hashlib.sha256(payload).hexdigest()}"
    assert data["imported"]["parity"]["passed"] is True
    assert not any((artifacts_root() / ".staging").iterdir())
    assert not list(old.rglob(lifecycle.BUILDING))


@pytest.mark.parametrize("force", [False, True], ids=["new", "forced-replacement"])
def test_a_registered_artifact_never_carries_the_build_marker(real_import, manifest_factory, tmp_path, force):
    """The marker guards the staging directory while it is validated and is gone once the
    directory is renamed into place; a replaced artifact that carried one leaves none behind."""
    if force:
        stray = registered(manifest_factory, 64)
        (stray / lifecycle.BUILDING).write_text(json.dumps({"pid": os.getpid()}))
    archive = tmp_path / "laya-L64.tar.gz"
    archive.write_bytes(export_archive(manifest_factory, 64))
    staged = []

    def probe(spec, bucket, compiled):
        staged.append((compiled.parent / lifecycle.BUILDING).exists())

    final = lifecycle.import_artifact(archive, local_files_only=True, force=force, log=lambda m: None, probe=probe)
    # a replace probes the staged copy, which carries the marker; a first install probes once at
    # the registered path, after the marker is gone
    assert staged == [force]
    assert final == artifact_dir(SPEC, 64)
    assert not list(final.rglob(lifecycle.BUILDING))
    assert sorted(p.name for p in final.iterdir()) == sorted(["manifest.json", COMPILED])
    assert (final / COMPILED / "weights.bin").read_bytes() == b"new weights"
    assert [p.name for p in final.parent.iterdir()] == [final.name]
    assert not any((artifacts_root() / ".staging").iterdir())
    A.load_verified(SPEC, 64, full=True)


def test_a_registered_artifact_with_a_stray_build_marker_still_verifies(repo, manifest_factory):
    """Artifacts registered before the marker was kept out of them carry one; neither the file
    hash, the manifest nor `artifacts prune` looks at it."""
    d = registered(manifest_factory, 64)
    (d / lifecycle.BUILDING).write_text(json.dumps({"pid": os.getpid()}))
    _, manifest = A.load_verified(SPEC, 64, full=True)
    assert manifest.artifact_sha256 == tree_sha256(d / COMPILED)
    A.load_verified(SPEC, 64)
    assert str(d) not in {x["path"] for x in lifecycle.plan_prune(HERE)}
    assert (d / lifecycle.BUILDING).exists()


def test_an_archive_for_another_bucket_is_not_registered(real_import, manifest_factory):
    files, calls = real_import
    payload = export_archive(manifest_factory, 96)  # the index says L64
    put(files, entry(64, payload=payload), payloads={entry(64)["path"]: payload})
    with pytest.raises(ArtifactIntegrityError, match="holds laya L96, not laya L64"):
        prebuilt.fetch(SPEC, [64], repo="o/r", log=lambda m: None)
    assert not artifact_dir(SPEC, 64).exists() and not artifact_dir(SPEC, 96).exists()
    assert calls["probe"] == []


def test_the_real_probe_loads_the_staged_model_on_the_ane_compute_units(real_import, manifest_factory, monkeypatch):
    import laya_apple.backends.coreml_ane as coreml_ane
    import laya_apple.prompt as prompt
    from laya_apple.registry import ANE_COMPUTE_UNITS

    files, calls = real_import
    monkeypatch.setattr(prebuilt, "_probe", REAL_PROBE)
    ckpt = hub.checkpoint_path(SPEC)  # the real_import stand-in
    (ckpt / "encoder").mkdir()
    (ckpt / "encoder/config.json").write_text(json.dumps({"local_attention": 128}))
    monkeypatch.setattr(prompt, "Tokenizer", lambda path: types.SimpleNamespace(pad_token_id=0))
    monkeypatch.setattr(coreml_ane, "HostWeights", lambda ckpt, local_attention: "host")
    monkeypatch.setattr(coreml_ane, "probe_features", lambda host, pad_id, bucket: {"bucket": bucket})
    loaded, probed = [], []

    def compiled_model(path, compute_units):
        loaded.append((Path(path), compute_units))
        return ("model", Path(path))

    def probe_placement(spec, bucket, model, compiled, feats):
        probed.append((model, Path(compiled), feats))
        return {
            "ane_ms": 1.0,
            "cpu_ms": 3.0,
            "ratio": STAGED_RATIO if ".staging" in Path(compiled).parts else REGISTERED_RATIO,
        }

    monkeypatch.setattr(sys.modules["coremltools"].models, "CompiledMLModel", compiled_model)
    monkeypatch.setattr(coreml_ane, "probe_placement", probe_placement)
    payload = export_archive(manifest_factory, 64)
    put(files, entry(64, payload=payload), payloads={entry(64)["path"]: payload})
    out = prebuilt.fetch(SPEC, [64], repo="o/r", force=True, log=lambda m: None)  # the replace path
    final = artifact_dir(SPEC, 64) / COMPILED
    # the staged probe's load, the pre-warm load at the registered path, the registered probe's load
    (staged, units), (prewarm, _), (registered_load, registered_units) = loaded
    assert staged.is_relative_to(artifacts_root() / ".staging") and staged.name == COMPILED
    assert not staged.is_relative_to(artifact_dir(SPEC, 64))
    assert prewarm == registered_load == final
    assert units == registered_units == ANE_COMPUTE_UNITS
    assert probed == [(("model", staged), staged, {"bucket": 64}), (("model", final), final, {"bucket": 64})]
    assert out[64]["probe"]["ratio"] == REGISTERED_RATIO  # from the registered-path probe


def test_a_failed_move_into_place_restores_the_previous_artifact(real_import, manifest_factory, monkeypatch):
    files, _ = real_import
    old = registered(manifest_factory, 64)
    before = snapshot(old)
    payload = export_archive(manifest_factory, 64)
    put(files, entry(64, payload=payload), payloads={entry(64)["path"]: payload})
    real_rename = os.rename

    def rename(src, dst):
        if Path(src).name.startswith("import-"):  # the staged artifact moving into place
            raise OSError("simulated rename failure")
        return real_rename(src, dst)

    monkeypatch.setattr(lifecycle.os, "rename", rename)
    with pytest.raises(OSError, match="simulated rename failure"):
        prebuilt.fetch(SPEC, [64], repo="o/r", force=True, log=lambda m: None)
    monkeypatch.setattr(lifecycle.os, "rename", real_rename)
    assert snapshot(old) == before
    assert [p.name for p in old.parent.iterdir()] == [old.name]
    A.load_verified(SPEC, 64, full=True)


def test_a_failed_cleanup_of_the_replaced_copy_is_only_logged(real_import, manifest_factory, monkeypatch):
    files, _ = real_import
    old = registered(manifest_factory, 64)
    payload = export_archive(manifest_factory, 64)
    put(files, entry(64, payload=payload), payloads={entry(64)["path"]: payload})
    real_rmtree = lifecycle.shutil.rmtree

    def rmtree(path, *a, **k):
        if ".old-" in Path(path).name:
            raise OSError("simulated rmtree failure")
        return real_rmtree(path, *a, **k)

    monkeypatch.setattr(lifecycle.shutil, "rmtree", rmtree)
    logs = []
    out = prebuilt.fetch(SPEC, [64], repo="o/r", force=True, log=logs.append)
    monkeypatch.setattr(lifecycle.shutil, "rmtree", real_rmtree)
    assert out[64]["path"] == old and (old / COMPILED / "weights.bin").read_bytes() == b"new weights"
    assert any("could not remove the replaced copy" in m and "Remove it by hand" in m for m in logs)


def test_a_registered_artifact_that_fails_to_load_is_replaced(repo, manifest_factory, monkeypatch):
    files, calls = repo
    put(files, entry(64))
    registered(manifest_factory, 64)

    def load_verified(spec, bucket):
        raise RuntimeError("Core ML could not load the model")

    monkeypatch.setattr(prebuilt, "load_verified", load_verified)
    logs = []
    out = prebuilt.fetch(SPEC, [64], repo="o/r", log=logs.append)
    assert list(out) == [64] and calls["import"][0]["force"] is True
    assert calls["import"][0]["expect"] == (SPEC.name, 64)
    assert any("not usable (RuntimeError: Core ML could not load the model)" in m for m in logs)


def test_fetch_from_the_default_repository_reads_the_pinned_commit(repo, monkeypatch):
    files, calls = repo
    monkeypatch.delenv(prebuilt.PREBUILT_REPO_ENV, raising=False)
    put(files, entry(64))
    prebuilt.fetch(SPEC, [64], log=lambda m: None)
    pin = prebuilt.DEFAULT_PREBUILT_REVISION
    assert calls["download"] and all(
        d[:1] == (prebuilt.DEFAULT_PREBUILT_REPO,) and d[2] == pin for d in calls["download"]
    )
    assert calls["import"][0]["source"].startswith(f"hf://{prebuilt.DEFAULT_PREBUILT_REPO}@{pin}/")


def test_fetch_from_another_repository_defaults_to_main(repo, monkeypatch):
    files, calls = repo
    put(files, entry(64))
    monkeypatch.setenv(prebuilt.PREBUILT_REPO_ENV, "someone/artifacts")
    prebuilt.fetch(SPEC, [64], log=lambda m: None)
    prebuilt.fetch(SPEC, [64], repo="o/r", force=True, log=lambda m: None)
    assert {(d[0], d[2]) for d in calls["download"]} == {("someone/artifacts", "main"), ("o/r", "main")}


def test_unoffered_bucket_is_refused(repo):
    with pytest.raises(ArtifactError, match="does not offer"):
        prebuilt.fetch(SPEC, [999], repo="o/r", log=lambda m: None)


# ----------------------------------------------------------------------------- offline


@pytest.fixture
def hf_cache(tmp_path, monkeypatch):
    """huggingface_hub.hf_hub_download over a fake cache {filename: bytes}; any call that is
    not local_files_only counts as a network request."""
    import huggingface_hub
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    cached: dict = {}
    calls: list = []
    root = tmp_path / "hf-cache"

    def hf_hub_download(repo_id, filename, *, revision, local_dir=None, local_files_only=False):
        calls.append({"filename": filename, "local_dir": local_dir, "local_files_only": local_files_only})
        if not local_files_only:
            raise AssertionError(f"network request for {filename}")
        if filename not in cached:
            raise FileNotFoundError(f"{filename} is not in the cache")
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(cached[filename])
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    return cached, calls, root


def test_fetch_passes_offline_to_every_download(repo):
    files, calls = repo
    put(files, entry(64))
    prebuilt.fetch(SPEC, [64], repo="o/r", local_files_only=True, log=lambda m: None)
    assert [d[1] for d in calls["download"]] == [prebuilt.INDEX, entry(64)["path"]]
    assert all(d[3] is True for d in calls["download"])


def test_download_offline_reads_only_the_hugging_face_cache(hf_cache, tmp_path):
    cached, calls, root = hf_cache
    cached[prebuilt.INDEX] = b"{}"
    path = REAL_DOWNLOAD("o/r", prebuilt.INDEX, "abc", tmp_path / "staging", local_files_only=True)
    assert path == root / prebuilt.INDEX
    assert calls == [{"filename": prebuilt.INDEX, "local_dir": None, "local_files_only": True}]


@pytest.mark.parametrize("flag, env, mode", [(True, False, "local_files_only"), (False, True, "HF_HUB_OFFLINE")])
def test_download_offline_and_not_cached_fails_without_the_network(hf_cache, tmp_path, monkeypatch, flag, env, mode):
    from huggingface_hub import constants

    _, calls, _ = hf_cache
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", env)
    with pytest.raises(BackendUnavailableError, match=rf"offline \({mode}\): it is not in the Hugging Face cache"):
        REAL_DOWNLOAD("o/r", prebuilt.INDEX, "abc", tmp_path / "staging", local_files_only=flag)
    assert [c["local_files_only"] for c in calls] == [True]


def test_download_online_stages_the_file(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    seen = {}

    def hf_hub_download(repo_id, filename, *, revision, local_dir=None, local_files_only=False):
        seen.update(local_dir=local_dir, local_files_only=local_files_only)
        return str(Path(local_dir) / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    REAL_DOWNLOAD("o/r", prebuilt.INDEX, "abc", tmp_path / "staging")
    assert seen == {"local_dir": str(tmp_path / "staging"), "local_files_only": False}


def test_offline_fetch_from_an_empty_cache_fails_and_registers_nothing(real_import, hf_cache, monkeypatch):
    _, calls, _ = hf_cache
    monkeypatch.setattr(prebuilt, "_download", REAL_DOWNLOAD)
    with pytest.raises(BackendUnavailableError, match="not in the Hugging Face cache"):
        prebuilt.fetch(SPEC, [64], repo="o/r", local_files_only=True, log=lambda m: None)
    assert all(c["local_files_only"] for c in calls)
    assert not artifact_dir(SPEC, 64).exists()


def test_offline_fetch_from_the_cache_runs_the_full_import(real_import, hf_cache, monkeypatch, manifest_factory):
    cached, calls, root = hf_cache
    _, repo_calls = real_import
    monkeypatch.setattr(prebuilt, "_download", REAL_DOWNLOAD)
    payload = export_archive(manifest_factory, 64)
    cached[prebuilt.INDEX] = json.dumps(index(entry(64, payload=payload))).encode()
    cached[entry(64)["path"]] = payload
    out = prebuilt.fetch(SPEC, [64], repo="o/r", local_files_only=True, log=lambda m: None)
    assert all(c["local_files_only"] for c in calls) and len(calls) == 2
    assert out[64]["path"] == artifact_dir(SPEC, 64) and repo_calls["probe"] == [64]  # once, at the registered path
    data = json.loads((artifact_dir(SPEC, 64) / "manifest.json").read_text())
    assert data["imported"]["parity"]["passed"] is True and data["imported"]["from"] == out[64]["source"]
    assert (root / entry(64)["path"]).exists()  # the cached archive is left in the cache


def test_checkpoint_error_names_the_effective_offline_mode(monkeypatch):
    import huggingface_hub
    from huggingface_hub import constants

    def snapshot_download(*a, **k):
        raise FileNotFoundError("not cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    with pytest.raises(BackendUnavailableError, match=r"offline \(HF_HUB_OFFLINE\)"):
        hub.checkpoint_path(SPEC)
    with pytest.raises(BackendUnavailableError, match=r"offline \(local_files_only\)"):
        hub.checkpoint_path(SPEC, local_files_only=True)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    with pytest.raises(BackendUnavailableError, match=r" online: "):
        hub.checkpoint_path(SPEC)


def test_cli_parses_fetch():
    a = cli.build_parser().parse_args(["artifacts", "fetch", "laya", "--length", "64", "--repo", "o/r"])
    assert (a.action, a.model, a.length, a.repo, a.revision) == ("fetch", "laya", [64], "o/r", None)
    a = cli.build_parser().parse_args(["artifacts", "fetch", "laya", "--revision", "abc"])
    assert (a.repo, a.revision) == (None, "abc")


# ----------------------------------------------------------------------------- first install (#162)

PARITY = {"passed": True, "hard_mismatches": 0, "prob_max_abs": 0.004, "rows": 40, "near_tie_flips": []}


class Killed(BaseException):
    """Stands in for SIGKILL: nothing after it runs, including the import's own cleanup."""


def reader_view(final: Path) -> dict:
    """What a runtime loading the bucket sees at the registered path right now."""
    try:
        A.load_verified(SPEC, 64)
        outcome = "model"
    except ArtifactMissingError:
        outcome = "missing"
    marker = final / lifecycle.PENDING
    return {
        "outcome": outcome,
        "files": sorted(p.name for p in final.iterdir()) if final.exists() else None,
        "pid": json.loads(marker.read_text())["pid"] if marker.exists() else None,
        "quarantine": (artifacts_root() / "quarantine").exists(),
    }


@pytest.fixture
def at_final(real_import, manifest_factory, monkeypatch):
    """real_import with an L64 archive in the repository, recording where the compute plan and
    the parity gate ran, what a reader of the registered path saw during the gate, and how many
    full loads ran. The gate returns state["parity"], or raises it when it is an exception."""
    files, calls = real_import
    final = artifact_dir(SPEC, 64)
    state = {"plan": [], "parity_at": [], "seen": [], "full_loads": 0, "parity": dict(PARITY)}

    def plan(path, *a, **k):
        state["plan"].append(Path(path))
        return dict(CLEAN_PLAN)

    def parity(spec, compiled, bucket, ckpt):
        state["parity_at"].append(Path(compiled))
        state["seen"].append(reader_view(final))
        if isinstance(state["parity"], BaseException):
            raise state["parity"]
        return dict(state["parity"])

    real_load = A.load_verified

    def load_verified(spec, bucket, *, full=False, **k):
        state["full_loads"] += bool(full)
        return real_load(spec, bucket, full=full, **k)

    monkeypatch.setattr(A, "compute_plan_summary", plan)
    monkeypatch.setattr(parity_ane, "ane_parity", parity)
    monkeypatch.setattr(A, "load_verified", load_verified)
    payload = export_archive(manifest_factory, 64)
    put(files, entry(64, payload=payload), payloads={entry(64)["path"]: payload})
    return calls, state


def assert_nothing_left(final: Path) -> None:
    """Nothing registered, no `.failed-<pid>` or partial copy, no staging, no quarantine, no stamp."""
    assert not final.exists()
    assert list(final.parent.iterdir()) == []
    assert not any((artifacts_root() / ".staging").iterdir())
    assert not (artifacts_root() / "quarantine").exists()
    assert not A._stamp_path(final).exists()


def test_a_first_install_is_validated_once_at_its_registered_path(at_final):
    calls, state = at_final
    out = prebuilt.fetch(SPEC, [64], repo="o/r", log=lambda m: None)
    final = artifact_dir(SPEC, 64)
    assert state["plan"] == [final / COMPILED] and state["parity_at"] == [final / COMPILED]
    assert calls["probe_paths"] == [final / COMPILED]  # exactly one probe: registered_probe
    assert state["full_loads"] == 0  # no second load at the registered path
    assert out[64]["path"] == final and out[64]["probe"]["ratio"] == REGISTERED_RATIO
    assert sorted(p.name for p in final.iterdir()) == sorted(["manifest.json", COMPILED])
    assert [p.name for p in final.parent.iterdir()] == [final.name]
    assert not any((artifacts_root() / ".staging").iterdir())
    data = json.loads((final / "manifest.json").read_text())
    assert data["imported"]["parity"] == PARITY and data["imported"]["placement"] == CLEAN_PLAN
    assert data["imported"]["artifact_sha256_check"] == tree_sha256(final / COMPILED)
    A.load_verified(SPEC, 64, full=True)


def test_the_runtime_sees_no_artifact_until_the_manifest_is_published(at_final):
    _, state = at_final
    prebuilt.fetch(SPEC, [64], repo="o/r", log=lambda m: None)
    (seen,) = state["seen"]  # during the parity gate at the registered path
    assert seen["outcome"] == "missing" and seen["quarantine"] is False
    assert seen["files"] == sorted([COMPILED, lifecycle.PENDING, lifecycle.PENDING_MANIFEST])
    assert seen["pid"] == os.getpid()
    assert reader_view(artifact_dir(SPEC, 64))["outcome"] == "model"  # once published


@pytest.mark.parametrize("full", [False, True], ids=["load", "full-verify"])
def test_load_verified_never_quarantines_an_unpublished_install(repo, manifest_factory, full):
    d = registered(manifest_factory, 64)
    os.rename(d / "manifest.json", d / lifecycle.PENDING_MANIFEST)
    (d / lifecycle.PENDING).write_text(json.dumps({"pid": os.getpid()}))
    before = snapshot(d)
    with pytest.raises(ArtifactMissingError):
        A.load_verified(SPEC, 64, full=full)
    assert snapshot(d) == before
    assert not (artifacts_root() / "quarantine").exists()
    assert A.list_artifacts() == []


def test_the_first_install_records_what_the_replace_path_records(at_final, manifest_factory, tmp_path, monkeypatch):
    """Same checks, same results: only where and in what order they run differ."""
    archive = tmp_path / "laya-L64.tar.gz"
    archive.write_bytes(export_archive(manifest_factory, 64))
    manifests = {}
    for force in (False, True):  # nothing is registered in either cache
        monkeypatch.setenv("LAYA_APPLE_CACHE", str(tmp_path / f"cache-{force}"))
        final = lifecycle.import_artifact(archive, local_files_only=True, force=force, log=lambda m: None)
        manifests[force] = json.loads((final / "manifest.json").read_text())
        manifests[force]["imported"].pop("at")
    assert manifests[False] == manifests[True]
    assert manifests[False]["imported"]["parity"] == PARITY


def test_cli_import_runs_only_the_parity_gate_at_the_registered_path(at_final, manifest_factory, tmp_path):
    _, state = at_final
    archive = tmp_path / "laya-L64.tar.gz"
    archive.write_bytes(export_archive(manifest_factory, 64))
    assert cli.main(["--offline", "artifacts", "import", str(archive)]) == 0
    final = artifact_dir(SPEC, 64)
    assert state["parity_at"] == [final / COMPILED] and state["full_loads"] == 0
    assert (final / "manifest.json").exists() and not (final / lifecycle.PENDING).exists()


@pytest.mark.parametrize("failure", ["parity", "probe", "interrupt"])
def test_a_failure_at_the_registered_path_registers_nothing_and_leaves_nothing(at_final, monkeypatch, failure):
    calls, state = at_final

    def interrupted(spec, bucket, compiled, *, local_files_only):
        raise KeyboardInterrupt

    if failure == "parity":
        state["parity"]["passed"] = False
    elif failure == "probe":
        calls["registered_fails"].add(64)
    else:
        monkeypatch.setattr(prebuilt, "_probe", interrupted)
    expected = {"parity": ArtifactParityError, "probe": ComputeUnitMismatchError, "interrupt": KeyboardInterrupt}
    with pytest.raises(expected[failure]):
        prebuilt.fetch(SPEC, [64], repo="o/r", log=lambda m: None)
    final = artifact_dir(SPEC, 64)
    assert [s["outcome"] for s in state["seen"]] == ["missing"]
    assert_nothing_left(final)
    with pytest.raises(ArtifactMissingError):
        A.load_verified(SPEC, 64)


def test_an_unpublished_install_left_by_a_killed_import_is_recovered_by_the_next_import(at_final, monkeypatch):
    import subprocess

    _, state = at_final
    final = artifact_dir(SPEC, 64)
    real_remove = lifecycle._remove_aside
    monkeypatch.setattr(lifecycle, "_remove_aside", lambda final, log: None)  # killed: no cleanup runs
    state["parity"] = Killed()
    with pytest.raises(Killed):
        prebuilt.fetch(SPEC, [64], repo="o/r", log=lambda m: None)
    monkeypatch.setattr(lifecycle, "_remove_aside", real_remove)
    assert lifecycle.unpublished(final)
    proc = subprocess.Popen(["true"])
    proc.wait()  # a real pid that has exited, as the killed import's has
    (final / lifecycle.PENDING).write_text(json.dumps({"pid": proc.pid}))
    with pytest.raises(ArtifactMissingError):
        A.load_verified(SPEC, 64)
    assert not (artifacts_root() / "quarantine").exists()

    state["parity"] = dict(PARITY)
    logs = []
    out = prebuilt.fetch(SPEC, [64], repo="o/r", log=logs.append)
    assert out[64]["path"] == final
    assert any("removing an unpublished install" in m for m in logs)
    assert not any("not usable" in m for m in logs)  # it was never a registered artifact
    assert sorted(p.name for p in final.iterdir()) == sorted(["manifest.json", COMPILED])
    assert [p.name for p in final.parent.iterdir()] == [final.name]  # no `.failed-<pid>` left
    assert not any((artifacts_root() / ".staging").iterdir())
    A.load_verified(SPEC, 64, full=True)


def test_the_first_install_writes_the_stamp_load_verified_would(at_final, monkeypatch):
    _, state = at_final
    monkeypatch.setattr(lifecycle, "platform_profile", lambda: dict(HERE))
    prebuilt.fetch(SPEC, [64], repo="o/r", log=lambda m: None)
    final = artifact_dir(SPEC, 64)
    manifest = A.verify_manifest(SPEC, 64, json.loads((final / "manifest.json").read_text()))
    stamp = json.loads(A._stamp_path(final).read_text())
    assert stamp == {"key": A._stamp_key(manifest, final, ANE_COMPUTE_UNITS, dict(HERE)), "placement": CLEAN_PLAN}
    hashed, plans = [], len(state["plan"])
    monkeypatch.setattr(A, "verify_files", lambda *a: hashed.append(a))
    A.load_verified(SPEC, 64)
    assert hashed == [] and len(state["plan"]) == plans  # the stamp is honoured: no re-hash, no compute plan


def test_no_stamp_is_written_when_the_files_change_after_they_were_hashed(
    at_final, manifest_factory, tmp_path, monkeypatch
):
    archive = tmp_path / "laya-L64.tar.gz"
    archive.write_bytes(export_archive(manifest_factory, 64))
    final = artifact_dir(SPEC, 64)

    def touch(spec, bucket, compiled):  # runs at the registered path, after the files were stat'ed
        os.utime(Path(compiled) / "weights.bin", ns=(1_000_000_000, 1_000_000_000))

    logs = []
    lifecycle.import_artifact(archive, local_files_only=True, log=logs.append, probe=touch)
    assert (final / "manifest.json").exists() and not (final / lifecycle.PENDING).exists()
    assert not A._stamp_path(final).exists()
    assert any("files changed after they were hashed" in m for m in logs)
    real_verify, hashed = A.verify_files, []

    def verify_files(manifest, directory):
        hashed.append(directory)
        return real_verify(manifest, directory)

    monkeypatch.setattr(A, "verify_files", verify_files)
    A.load_verified(SPEC, 64)
    assert hashed == [final]  # the first load re-hashes, then stamps
    assert A._stamp_path(final).exists()


def test_an_interrupt_right_after_the_move_into_place_leaves_nothing(at_final, manifest_factory, tmp_path, monkeypatch):
    _, state = at_final
    archive = tmp_path / "laya-L64.tar.gz"
    archive.write_bytes(export_archive(manifest_factory, 64))
    real_rename = os.rename

    def rename(src, dst):
        real_rename(src, dst)
        if Path(src).name.startswith("import-"):  # the staged directory has just moved into place
            raise KeyboardInterrupt

    monkeypatch.setattr(lifecycle.os, "rename", rename)
    with pytest.raises(KeyboardInterrupt):
        lifecycle.import_artifact(archive, local_files_only=True, log=lambda m: None)
    monkeypatch.setattr(lifecycle.os, "rename", real_rename)
    assert state["parity_at"] == []
    assert_nothing_left(artifact_dir(SPEC, 64))


def test_force_keeps_the_replace_path(at_final, manifest_factory):
    calls, state = at_final
    old = registered(manifest_factory, 64)
    out = prebuilt.fetch(SPEC, [64], repo="o/r", force=True, log=lambda m: None)
    (staged,) = state["parity_at"]
    assert staged.is_relative_to(artifacts_root() / ".staging") and staged.name == COMPILED
    assert_probed(calls, registered=True)  # staged, then at the registered path
    assert state["full_loads"] == 1  # the load at the registered path after the move
    assert out[64]["path"] == old and not (old / lifecycle.PENDING).exists()
    assert (old / COMPILED / "weights.bin").read_bytes() == b"new weights"
