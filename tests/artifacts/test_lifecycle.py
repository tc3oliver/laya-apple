"""Artifact lifecycle (v0.3): prune plan and safety, quarantine on corruption, build locking."""

from __future__ import annotations

import json
import os
import threading
import time

import pytest

from laya_apple import lifecycle
from laya_apple.artifacts import COMPILED, artifact_dir, artifacts_root, load_verified, tree_sha256
from laya_apple.errors import ArtifactIntegrityError, ArtifactMissingError
from laya_apple.registry import models

PROFILE = {"soc": "Test SoC", "macos": "26.1", "macos_build": "X", "coremltools": "9.0"}


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("LAYA_APPLE_CACHE", str(tmp_path))
    return tmp_path


def _artifact(spec, bucket, manifest_factory, *, where=None, **overrides):
    d = where or artifact_dir(spec, bucket)
    (d / COMPILED).mkdir(parents=True)
    (d / COMPILED / "weights.bin").write_bytes(os.urandom(64))
    data = manifest_factory(spec, bucket, **{"platform": PROFILE, **overrides})
    data["integrity"]["artifact_sha256"] = tree_sha256(d / COMPILED)
    (d / "manifest.json").write_text(json.dumps(data))
    return d


def test_prune_keeps_current_artifacts_and_lists_everything_stale(cache, manifest_factory):
    spec = models()["laya-typed-decisions"]
    good = _artifact(spec, 64, manifest_factory)
    old_rev = _artifact(
        spec,
        96,
        manifest_factory,
        where=artifacts_root() / spec.name / "aaaaaaaaaaaa" / "bc1s-masked-L96-B1",
        **{"source.revision": "a" * 40},
    )
    other_profile = _artifact(spec, 128, manifest_factory, platform={**PROFILE, "coremltools": "8.0"})
    unregistered = artifacts_root() / "no-such-model" / "rev" / "bc1s-masked-L64-B1"
    unregistered.mkdir(parents=True)
    (unregistered / "manifest.json").write_text(json.dumps({"format": "laya-apple-artifact", "source": {"model": "x"}}))
    (artifacts_root() / "quarantine" / "old").mkdir(parents=True)
    staging = artifacts_root() / ".staging" / "abandoned"
    staging.mkdir(parents=True)
    old = time.time() - 2 * lifecycle.STAGING_MAX_AGE_S
    os.utime(staging, (old, old))
    fresh_staging = artifacts_root() / ".staging" / "in-progress"
    fresh_staging.mkdir(parents=True)
    stamp = cache / "verified" / "artifacts" / "gone__rev__bc1s-masked-L64-B1.json"
    stamp.parent.mkdir(parents=True)
    stamp.write_text("{}")

    plan = lifecycle.plan_prune(PROFILE)
    reasons = {os.path.basename(x["path"]): x["reason"] for x in plan}
    listed = {x["path"] for x in plan}
    assert str(good) not in listed and str(fresh_staging) not in listed
    assert str(old_rev) in listed and "pinned is" in reasons[old_rev.name]
    assert str(other_profile) in listed and "platform profile" in reasons[other_profile.name]
    assert str(unregistered) in listed
    assert any("quarantined" == x["reason"] for x in plan)
    assert str(staging) in listed and str(stamp) in listed

    assert all(os.path.exists(x["path"]) for x in plan)  # planning deletes nothing
    lifecycle.prune(plan)
    assert not any(os.path.exists(x["path"]) for x in plan)
    assert good.exists() and fresh_staging.exists()


def test_prune_never_deletes_outside_the_cache(cache, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    (outside / "keep.txt").write_text("x")
    removed = lifecycle.prune([{"path": str(outside), "reason": "forged", "bytes": 0}])
    assert removed == [] and (outside / "keep.txt").exists()


def test_corrupt_artifact_is_quarantined_with_rebuild_hint(cache, manifest_factory, monkeypatch):
    pytest.importorskip("coremltools")
    import laya_apple.artifacts as A

    spec = models()["laya-typed-decisions"]
    monkeypatch.setattr(A, "platform_profile", lambda: dict(PROFILE))
    d = _artifact(spec, 64, manifest_factory)
    (d / COMPILED / "weights.bin").write_bytes(b"tampered")
    with pytest.raises(ArtifactIntegrityError) as e:
        load_verified(spec, 64)
    assert "laya-apple artifacts build laya-typed-decisions --length 64" in str(e.value)
    assert not d.exists()
    moved = list((artifacts_root() / "quarantine").glob("*"))
    assert len(moved) == 1 and (moved[0] / "QUARANTINED.json").exists()
    with pytest.raises(ArtifactMissingError):  # nothing corrupt is left where the runtime looks
        load_verified(spec, 64)


def test_unreadable_manifest_is_quarantined(cache, monkeypatch):
    pytest.importorskip("coremltools")
    spec = models()["laya"]
    d = artifact_dir(spec, 64)
    (d / COMPILED).mkdir(parents=True)
    (d / "manifest.json").write_text("{not json")
    with pytest.raises(ArtifactIntegrityError):
        load_verified(spec, 64)
    assert not d.exists()


def test_hash_mismatch_race_is_rechecked_before_quarantine(cache, manifest_factory, monkeypatch):
    """A --force rebuild/import can swap the directory between the manifest read and the
    hash check; load_verified must re-read and re-hash once before quarantining."""
    pytest.importorskip("coremltools")
    import laya_apple.artifacts as A

    spec = models()["laya-typed-decisions"]
    monkeypatch.setattr(A, "platform_profile", lambda: dict(PROFILE))
    d = _artifact(spec, 64, manifest_factory)

    real_verify_files = A.verify_files
    calls = []

    def flaky_verify_files(manifest, directory):
        calls.append(1)
        if len(calls) == 1:
            raise ArtifactIntegrityError("stale hash (racing rebuild)")
        return real_verify_files(manifest, directory)

    monkeypatch.setattr(A, "verify_files", flaky_verify_files)
    # The fake artifact isn't a real compiled Core ML model; stub out the placement check
    # and model load so this test stays focused on the hash-mismatch recheck.
    import coremltools as ct

    monkeypatch.setattr(
        A, "compute_plan_summary", lambda *a, **k: {"ops": {"ane": 1, "cpu": 0, "gpu": 0}, "transitions": 0}
    )
    monkeypatch.setattr(ct.models, "CompiledMLModel", lambda *a, **k: object())
    model, m = load_verified(spec, 64)
    assert d.exists()  # not quarantined: the second check passed
    assert len(calls) == 2


def test_hash_mismatch_still_quarantined_when_it_persists(cache, manifest_factory, monkeypatch):
    pytest.importorskip("coremltools")
    import laya_apple.artifacts as A

    spec = models()["laya-typed-decisions"]
    monkeypatch.setattr(A, "platform_profile", lambda: dict(PROFILE))
    d = _artifact(spec, 64, manifest_factory)
    (d / COMPILED / "weights.bin").write_bytes(b"tampered")
    with pytest.raises(ArtifactIntegrityError):
        load_verified(spec, 64)
    assert not d.exists()


def test_schema_invalid_manifest_is_quarantined(cache, manifest_factory, monkeypatch):
    pytest.importorskip("coremltools")
    import laya_apple.artifacts as A

    spec = models()["laya-typed-decisions"]
    monkeypatch.setattr(A, "platform_profile", lambda: dict(PROFILE))
    # a wrong-typed "artifact" fails the schema and makes the field checks raise
    # AttributeError, which verify_manifest turns into a plain ArtifactIntegrityError.
    d = _artifact(spec, 64, manifest_factory, artifact="not-an-object")
    with pytest.raises(ArtifactIntegrityError):
        load_verified(spec, 64)
    assert not d.exists()
    assert list((artifacts_root() / "quarantine").glob("*"))


def test_valid_artifact_for_another_revision_is_not_quarantined(cache, manifest_factory, monkeypatch):
    """A manifest that is schema-valid but for another revision/config is a valid artifact
    for a different configuration; it must not be quarantined."""
    pytest.importorskip("coremltools")
    import laya_apple.artifacts as A
    from laya_apple.errors import ArtifactRevisionError

    spec = models()["laya-typed-decisions"]
    monkeypatch.setattr(A, "platform_profile", lambda: dict(PROFILE))
    d = _artifact(spec, 64, manifest_factory, **{"source.revision": "0" * 40})
    with pytest.raises(ArtifactRevisionError):
        load_verified(spec, 64)
    assert d.exists()  # left in place; nothing corrupt about it


def test_staging_still_building_is_not_pruned_even_when_old(cache):
    staging = artifacts_root() / ".staging" / "in-progress"
    staging.mkdir(parents=True)
    (staging / lifecycle.BUILDING).write_text(json.dumps({"pid": os.getpid()}))
    old = time.time() - 2 * lifecycle.STAGING_MAX_AGE_S
    os.utime(staging, (old, old))  # after writing the pid file: that write bumps the dir mtime
    plan = lifecycle.plan_prune(PROFILE)
    assert str(staging) not in {x["path"] for x in plan}


def test_staging_with_dead_pid_is_pruned_once_old(cache):
    import subprocess

    staging = artifacts_root() / ".staging" / "crashed"
    staging.mkdir(parents=True)
    proc = subprocess.Popen(["true"])
    proc.wait()  # a real, valid pid that has definitely exited
    (staging / lifecycle.BUILDING).write_text(json.dumps({"pid": proc.pid}))
    old = time.time() - 2 * lifecycle.STAGING_MAX_AGE_S
    os.utime(staging, (old, old))
    plan = lifecycle.plan_prune(PROFILE)
    assert str(staging) in {x["path"] for x in plan}


def _make_export(tmp_path, *, extra_members=(), length=64, model="laya"):
    """A minimal, valid-shaped .tar.gz export, optionally with extra tar members."""
    import tarfile

    manifest = {
        "format": "laya-apple-artifact",
        "format_version": 1,
        "source": {"model": model},
        "artifact": {"length": length},
    }
    archive = tmp_path / "export.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        mpath = tmp_path / "manifest.json"
        mpath.write_text(json.dumps(manifest))
        tar.add(mpath, arcname="manifest.json")
        model_file = tmp_path / "model.bin"
        model_file.write_bytes(b"x")
        tar.add(model_file, arcname=f"{COMPILED}/weights.bin")
        for arcname, add in extra_members:
            add(tar, arcname)
    return archive


def test_import_rejects_unexpected_archive_member(cache, tmp_path):
    from laya_apple.lifecycle import import_artifact

    def add_extra(tar, arcname):
        p = tmp_path / "evil.txt"
        p.write_text("x")
        tar.add(p, arcname=arcname)

    archive = _make_export(tmp_path, extra_members=[("../evil.txt", add_extra)])
    with pytest.raises(Exception, match="unexpected member|is not a laya-apple"):
        import_artifact(archive)


def test_import_rejects_symlink_member(cache, tmp_path):
    import tarfile

    from laya_apple.lifecycle import import_artifact

    archive = tmp_path / "export.tar.gz"
    manifest = {
        "format": "laya-apple-artifact",
        "format_version": 1,
        "source": {"model": "laya"},
        "artifact": {"length": 64},
    }
    with tarfile.open(archive, "w:gz") as tar:
        mpath = tmp_path / "manifest.json"
        mpath.write_text(json.dumps(manifest))
        tar.add(mpath, arcname="manifest.json")
        link = tarfile.TarInfo(name=f"{COMPILED}/evil-link")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tar.addfile(link)
    with pytest.raises(Exception, match="link member"):
        import_artifact(archive)


def test_import_rejects_oversized_archive(cache, tmp_path, monkeypatch):
    from laya_apple import lifecycle
    from laya_apple.lifecycle import import_artifact

    monkeypatch.setattr(lifecycle, "MAX_IMPORT_BYTES", 1)
    archive = _make_export(tmp_path)
    with pytest.raises(Exception, match="larger than"):
        import_artifact(archive)


def test_import_rejects_missing_data_filter(cache, tmp_path, monkeypatch):
    import tarfile

    from laya_apple.errors import BackendUnavailableError
    from laya_apple.lifecycle import import_artifact

    monkeypatch.delattr(tarfile, "data_filter", raising=False)
    archive = _make_export(tmp_path)
    with pytest.raises(BackendUnavailableError, match="3.11.4"):
        import_artifact(archive)


def test_import_malformed_length_raises_integrity_error(cache, tmp_path):
    import json as _json
    import tarfile

    from laya_apple.lifecycle import import_artifact

    manifest = {
        "format": "laya-apple-artifact",
        "format_version": 1,
        "source": {"model": "laya"},
        "artifact": {"length": "not-an-int"},
    }
    archive = tmp_path / "export.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        mpath = tmp_path / "manifest.json"
        mpath.write_text(_json.dumps(manifest))
        tar.add(mpath, arcname="manifest.json")
        model_file = tmp_path / "model.bin"
        model_file.write_bytes(b"x")
        tar.add(model_file, arcname=f"{COMPILED}/weights.bin")
    with pytest.raises(ArtifactIntegrityError, match="malformed"):
        import_artifact(archive)


def test_build_lock_is_exclusive(cache):
    spec = models()["laya"]
    events = []

    def hold(name, secs):
        with lifecycle.build_lock(spec, 64, timeout=10):
            events.append((name, "in", time.monotonic()))
            time.sleep(secs)
            events.append((name, "out", time.monotonic()))

    a = threading.Thread(target=hold, args=("a", 0.4))
    a.start()
    time.sleep(0.1)
    b = threading.Thread(target=hold, args=("b", 0.0))
    b.start()
    a.join()
    b.join()
    order = [(n, k) for n, k, _ in events]
    assert order == [("a", "in"), ("a", "out"), ("b", "in"), ("b", "out")]


def test_build_lock_times_out(cache):
    spec = models()["laya"]
    from laya_apple.errors import ArtifactError

    with lifecycle.build_lock(spec, 96):
        err = []

        def other():
            try:
                with lifecycle.build_lock(spec, 96, timeout=0.3):
                    pass
            except ArtifactError as e:
                err.append(e)

        t = threading.Thread(target=other)
        t.start()
        t.join()
    assert err and "another process" in str(err[0])


def _beside(spec, bucket, suffix, manifest_factory=None):
    """A leftover directory an import could leave beside artifact_dir(spec, bucket)."""
    d = artifact_dir(spec, bucket)
    left = d.with_name(d.name + suffix)
    if manifest_factory:
        _artifact(spec, bucket, manifest_factory, where=left)
    else:
        (left / COMPILED).mkdir(parents=True)
    return left


def test_a_stranded_previous_artifact_is_reported_and_never_pruned(cache, manifest_factory, monkeypatch):
    spec = models()["laya-typed-decisions"]
    monkeypatch.setattr(lifecycle, "_pid_alive", lambda pid: False)
    stranded = _beside(spec, 64, ".old-123", manifest_factory)  # nothing registered beside it
    assert all(item["path"] != str(stranded) for item in lifecycle.plan_prune(PROFILE))
    (report,) = lifecycle.stranded_artifacts()
    assert report["path"] == str(stranded) and report["registered"] == str(artifact_dir(spec, 64))
    assert "nothing is registered at" in report["message"] and "rename it back to recover" in report["message"]
    lifecycle.prune()
    lifecycle.prune([{"path": str(stranded), "reason": "forged", "bytes": 0}])  # a stale or hand-made plan
    assert (stranded / "manifest.json").exists()
    # a registered directory without a manifest does not count as registered either
    artifact_dir(spec, 64).mkdir(parents=True)
    assert [x["path"] for x in lifecycle.stranded_artifacts()] == [str(stranded)]


def test_leftovers_beside_a_registered_artifact_are_pruned_once_their_import_exits(
    cache, manifest_factory, monkeypatch
):
    spec = models()["laya-typed-decisions"]
    _artifact(spec, 64, manifest_factory)
    old = _beside(spec, 64, ".old-123", manifest_factory)
    failed = _beside(spec, 64, ".failed-123")  # partly removed: no manifest
    alive = {"value": True}
    monkeypatch.setattr(lifecycle, "_pid_alive", lambda pid: alive["value"])
    assert lifecycle.plan_prune(PROFILE) == [] and lifecycle.stranded_artifacts() == []
    alive["value"] = False
    plan = lifecycle.plan_prune(PROFILE)
    assert sorted((x["path"], x["reason"]) for x in plan) == [
        (str(failed), "leftover failed copy from an earlier import"),
        (str(old), "leftover old copy from an earlier import"),
    ]
    alive["value"] = True  # the pid is in use again before prune runs
    assert lifecycle.prune(plan) == [] and old.exists() and failed.exists()
    alive["value"] = False
    assert len(lifecycle.prune(plan)) == 2 and not old.exists() and not failed.exists()


def test_an_impossible_pid_in_a_leftover_name_does_not_crash_prune(cache):
    spec = models()["laya-typed-decisions"]
    huge = _beside(spec, 64, ".failed-" + "9" * 30)
    assert lifecycle._pid_alive(int("9" * 30)) is True  # never assumed dead
    assert lifecycle.plan_prune(PROFILE) == [] and lifecycle.stranded_artifacts() == []
    assert huge.exists()
