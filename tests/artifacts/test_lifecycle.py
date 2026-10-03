"""Artifact lifecycle (v0.3): prune plan and safety, quarantine on corruption, build locking."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

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


def _old_staging(name, *, building=False):
    """A staging directory untouched for twice the prune age, with this process's marker if asked."""
    staging = artifacts_root() / ".staging" / name
    staging.mkdir(parents=True)
    if building:
        (staging / lifecycle.BUILDING).write_text(json.dumps({"pid": os.getpid()}))
    old = time.time() - 2 * lifecycle.STAGING_MAX_AGE_S
    os.utime(staging, (old, old))  # after the marker: writing it bumps the directory's mtime
    return staging


def _listed(plan):
    return {x["path"] for x in plan}


def test_unlinking_the_marker_makes_an_old_staging_directory_fresh(cache):
    """What an import does just before it renames its staging directory into place."""
    staging = _old_staging("importing", building=True)
    (staging / lifecycle.BUILDING).unlink()
    assert str(staging) not in _listed(lifecycle.plan_prune(PROFILE))


def test_a_marker_unlinked_between_prune_reading_the_age_and_the_marker_does_not_list_the_directory(cache, monkeypatch):
    """An import unlinks its marker, then renames the directory into place. A prune that read
    the age first and the marker second would see an old directory without a marker."""
    staging = _old_staging("importing", building=True)
    real_stat = Path.stat
    raced = []

    def stat(self, *args, **kwargs):
        result = real_stat(self, *args, **kwargs)
        if self == staging and not raced:  # the directory's age has just been read
            raced.append(True)
            (staging / lifecycle.BUILDING).unlink()
        return result

    monkeypatch.setattr(Path, "stat", stat)
    plan = lifecycle.plan_prune(PROFILE)
    monkeypatch.setattr(Path, "stat", real_stat)
    assert str(staging) not in _listed(plan)


@pytest.mark.parametrize("change", ["a build starts in it", "something touches it"])
def test_prune_rechecks_a_staging_directory_before_deleting_it(cache, change):
    staging = _old_staging("abandoned")
    plan = lifecycle.plan_prune(PROFILE)
    assert str(staging) in _listed(plan)
    if change == "a build starts in it":
        old = staging.stat().st_mtime
        (staging / lifecycle.BUILDING).write_text(json.dumps({"pid": os.getpid()}))
        os.utime(staging, (old, old))
    else:
        os.utime(staging)
    assert lifecycle.prune(plan) == []
    assert staging.exists()


def test_prune_rechecks_a_staging_directory_spelled_through_a_symlinked_cache(tmp_path, monkeypatch):
    """A hand-written plan may name the resolved path while the cache is reached through a symlink."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    monkeypatch.setenv("LAYA_APPLE_CACHE", str(link))
    staging = _old_staging("importing", building=True)
    plan = [{"path": str(staging.resolve()), "reason": "hand-written"}]
    assert lifecycle.prune(plan) == []
    assert staging.exists()


def test_a_staging_directory_that_vanishes_during_the_scan_is_skipped(cache, monkeypatch):
    finished = _old_staging("finished")
    abandoned = _old_staging("zz-abandoned")
    real_stat = Path.stat
    renamed = []

    def stat(self, *args, **kwargs):
        if self == finished and not renamed:  # the import renames it into place
            renamed.append(True)
            os.rename(finished, cache / "registered")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    plan = lifecycle.plan_prune(PROFILE)
    monkeypatch.setattr(Path, "stat", real_stat)
    assert renamed and _listed(plan) == {str(abandoned)}


def test_prune_skips_a_staging_directory_that_is_gone(cache):
    gone = artifacts_root() / ".staging" / "gone"
    gone.parent.mkdir(parents=True)
    assert lifecycle.prune([{"path": str(gone), "reason": "abandoned staging directory", "bytes": 0}]) == []


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


@pytest.mark.parametrize("registered", [False, True], ids=["nothing-registered", "artifact-registered"])
def test_a_previous_artifact_with_a_manifest_is_reported_and_never_pruned(
    cache, manifest_factory, monkeypatch, registered
):
    spec = models()["laya-typed-decisions"]
    monkeypatch.setattr(lifecycle, "_pid_alive", lambda pid: False)
    old = _beside(spec, 64, ".old-123", manifest_factory)
    if registered:
        _artifact(spec, 64, manifest_factory)
    assert all(item["path"] != str(old) for item in lifecycle.plan_prune(PROFILE))
    (report,) = lifecycle.kept_previous_artifacts()
    assert report["path"] == str(old) and report["registered"] == str(artifact_dir(spec, 64))
    assert report["registered_missing"] is not registered
    state = "an artifact is registered at" if registered else "nothing is registered at"
    assert state in report["message"]
    assert f"verify and remove it by hand, or rename it back: mv {old} {artifact_dir(spec, 64)}" in report["message"]
    lifecycle.prune()
    forged = [{"path": str(old), "reason": "forged", "bytes": 0}]  # a stale or hand-written plan
    assert lifecycle.prune(forged) == [] and (old / "manifest.json").exists()


def test_a_plan_listing_the_previous_artifact_and_its_sibling_never_deletes_the_previous_one(
    cache, manifest_factory, monkeypatch
):
    spec = models()["laya-typed-decisions"]
    monkeypatch.setattr(lifecycle, "_pid_alive", lambda pid: False)
    registered = _artifact(spec, 64, manifest_factory)
    old = _beside(spec, 64, ".old-123", manifest_factory)
    plan = [{"path": str(p), "reason": "forged", "bytes": 0} for p in (registered, old)]
    assert [x["path"] for x in lifecycle.prune(plan)] == [str(registered)]
    assert (old / "manifest.json").exists()
    assert lifecycle.kept_previous_artifacts()[0]["registered_missing"] is True


def test_partly_removed_and_failed_leftovers_are_pruned_once_their_import_exits(cache, manifest_factory, monkeypatch):
    spec = models()["laya-typed-decisions"]
    _artifact(spec, 64, manifest_factory)
    old = _beside(spec, 64, ".old-123")  # partly removed: no manifest
    failed = _beside(spec, 64, ".failed-123", manifest_factory)
    alive = {"value": True}
    monkeypatch.setattr(lifecycle, "_pid_alive", lambda pid: alive["value"])
    assert lifecycle.plan_prune(PROFILE) == [] and lifecycle.kept_previous_artifacts() == []
    alive["value"] = False
    plan = lifecycle.plan_prune(PROFILE)
    assert sorted((x["path"], x["reason"]) for x in plan) == [
        (str(failed), "leftover failed copy from an earlier import"),
        (str(old), "leftover old copy from an earlier import"),
    ]
    assert lifecycle.kept_previous_artifacts() == []
    alive["value"] = True  # the pid is in use again before prune runs
    assert lifecycle.prune(plan) == [] and old.exists() and failed.exists()
    alive["value"] = False
    assert len(lifecycle.prune(plan)) == 2 and not old.exists() and not failed.exists()


def test_an_impossible_pid_is_never_assumed_dead(cache):
    spec = models()["laya-typed-decisions"]
    huge = _beside(spec, 64, ".failed-" + "9" * 30)
    assert lifecycle._pid_alive(int("9" * 30)) is True
    assert lifecycle._pid_alive("123") is True  # e.g. a non-int pid read from BUILDING.json (TypeError)
    assert lifecycle._pid_alive(None) is True
    assert lifecycle.plan_prune(PROFILE) == [] and lifecycle.kept_previous_artifacts() == []
    assert huge.exists()


def test_an_abandoned_staging_directory_with_a_non_int_pid_is_kept(cache):
    staging = artifacts_root() / ".staging" / "fetch-x"
    staging.mkdir(parents=True)
    (staging / lifecycle.BUILDING).write_text(json.dumps({"pid": "not-a-pid"}))
    old = time.time() - 2 * lifecycle.STAGING_MAX_AGE_S
    os.utime(staging, (old, old))
    assert lifecycle.plan_prune(PROFILE) == []


# ----------------------------------------------------------------------------- unpublished installs (#162)


def _dead_pid() -> int:
    import subprocess

    proc = subprocess.Popen(["true"])
    proc.wait()  # a real, valid pid that has definitely exited
    return proc.pid


def _unpublished(marker, *, old=True, manifest_factory=None):
    """An install an import left at the registered path with its manifest withheld. `marker` is
    the PENDING.json text; with `manifest_factory` the manifest is published too."""
    spec = models()["laya-typed-decisions"]
    d = artifact_dir(spec, 64)
    if manifest_factory:
        _artifact(spec, 64, manifest_factory)
    else:
        (d / COMPILED).mkdir(parents=True)
        (d / COMPILED / "weights.bin").write_bytes(os.urandom(64))
        (d / lifecycle.PENDING_MANIFEST).write_text("{}")
    (d / lifecycle.PENDING).write_text(marker)
    if old:
        age = time.time() - 2 * lifecycle.STAGING_MAX_AGE_S
        os.utime(d, (age, age))  # after the marker: writing it bumps the directory's mtime
    return d


@pytest.mark.parametrize(
    "marker", ["dead", "unreadable", "no-pid"], ids=["dead-pid", "unreadable-marker", "marker-without-pid"]
)
def test_an_old_unpublished_install_whose_import_exited_is_pruned(cache, marker):
    text = {
        "dead": json.dumps({"pid": _dead_pid()}),
        "unreadable": "{not json",
        "no-pid": json.dumps({"since": "x"}),
    }[marker]
    d = _unpublished(text)
    plan = lifecycle.plan_prune(PROFILE)
    assert [(x["path"], x["reason"]) for x in plan] == [(str(d), "unpublished install left by an import that exited")]
    assert [x["path"] for x in lifecycle.prune(plan)] == [str(d)]
    assert not d.exists()


@pytest.mark.parametrize("pid", [os.getpid(), "not-a-pid"], ids=["live-pid", "non-int-pid"])
def test_an_unpublished_install_whose_import_may_be_running_is_never_pruned(cache, pid):
    d = _unpublished(json.dumps({"pid": pid}))
    assert lifecycle.plan_prune(PROFILE) == []
    forged = [{"path": str(d), "reason": "hand-written", "bytes": 0}]
    assert lifecycle.prune(forged) == [] and d.exists()


def test_a_recent_unpublished_install_is_not_pruned(cache):
    d = _unpublished(json.dumps({"pid": _dead_pid()}), old=False)
    assert lifecycle.plan_prune(PROFILE) == []
    assert lifecycle.prune([{"path": str(d), "reason": "hand-written", "bytes": 0}]) == [] and d.exists()


def test_a_published_artifact_with_a_leftover_marker_is_not_an_unpublished_install(cache, manifest_factory):
    """An import that died between publishing the manifest and removing its marker."""
    d = _unpublished(json.dumps({"pid": _dead_pid()}), manifest_factory=manifest_factory)
    assert not lifecycle.unpublished(d)
    assert lifecycle.plan_prune(PROFILE) == []


@pytest.mark.parametrize(
    "change", ["an import starts validating there", "an import recovers and publishes it", "something touches it"]
)
def test_prune_rechecks_an_unpublished_install_before_deleting_it(cache, change):
    d = _unpublished(json.dumps({"pid": _dead_pid()}))
    plan = lifecycle.plan_prune(PROFILE)
    assert str(d) in _listed(plan)
    age = d.stat().st_mtime
    if change == "an import starts validating there":
        (d / lifecycle.PENDING).write_text(json.dumps({"pid": os.getpid()}))
        os.utime(d, (age, age))
    elif change == "an import recovers and publishes it":
        os.replace(d / lifecycle.PENDING_MANIFEST, d / "manifest.json")
        (d / lifecycle.PENDING).unlink()
        os.utime(d, (age, age))
    else:
        os.utime(d)
    assert lifecycle.prune(plan) == []
    assert d.exists()


def test_prune_skips_an_unpublished_install_that_is_gone(cache):
    import shutil

    d = _unpublished(json.dumps({"pid": _dead_pid()}))
    plan = lifecycle.plan_prune(PROFILE)
    shutil.rmtree(d)  # recovered by an import since the plan was made
    assert lifecycle.prune(plan) == []


def test_an_unpublished_install_that_vanishes_during_the_scan_is_skipped(cache, monkeypatch):
    d = _unpublished(json.dumps({"pid": _dead_pid()}))
    real_stat = Path.stat
    gone = []

    def stat(self, *args, **kwargs):
        if self == d and not gone:  # an import removes it just before prune reads its age
            gone.append(True)
            os.rename(d, cache / "elsewhere")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    plan = lifecycle.plan_prune(PROFILE)
    monkeypatch.setattr(Path, "stat", real_stat)
    assert gone and plan == []


# ----------------------------------------------------------------------------- prune claims before deleting


def _abandoned_entry(kind):
    """An abandoned staging directory or unpublished install, and the check prune applies to it."""
    if kind == "staging":
        return _old_staging("abandoned"), "_abandoned_staging"
    return _unpublished(json.dumps({"pid": _dead_pid()})), "_abandoned_pending"


def _live_entry(path, kind):
    """What a running build or import renames into `path`."""
    path.mkdir()
    marker = lifecycle.BUILDING if kind == "staging" else lifecycle.PENDING
    (path / marker).write_text(json.dumps({"pid": os.getpid()}))
    (path / "live").write_text("in use")


@pytest.mark.parametrize("kind", ["staging", "pending"])
@pytest.mark.parametrize("when", ["after the first check", "after the second check"])
def test_a_directory_renamed_into_the_path_while_prune_checks_it_is_never_deleted(cache, monkeypatch, kind, when):
    p, check = _abandoned_entry(kind)
    plan = lifecycle.plan_prune(PROFILE)
    assert str(p) in _listed(plan)
    real = getattr(lifecycle, check)
    calls = []

    def racing(d):
        result = real(d)
        calls.append(d)
        if len(calls) == (1 if when == "after the first check" else 2):
            if p.exists():  # the abandoned one goes elsewhere first (another prune, or a recovery)
                os.rename(p, cache / "moved-away")
            _live_entry(p, kind)
        return result

    monkeypatch.setattr(lifecycle, check, racing)
    removed = lifecycle.prune(plan)
    monkeypatch.setattr(lifecycle, check, real)
    assert len(calls) == 2
    assert (p / "live").read_text() == "in use"
    assert not [x for x in p.parent.iterdir() if ".pruning-" in x.name]
    assert removed == ([] if when == "after the first check" else [x for x in plan if x["path"] == str(p)])


def _pruning_copy(where, *, pid, marker_pid=None, manifest=False, old=True):
    d = where.with_name(f"{where.name}.pruning-{pid}-0a1b2c3d")
    d.mkdir(parents=True)
    (d / "file").write_text("x")
    if marker_pid is not None:
        (d / lifecycle.BUILDING).write_text(json.dumps({"pid": marker_pid}))
    if manifest:
        (d / "manifest.json").write_text("{}")
    if old:
        age = time.time() - 2 * lifecycle.STAGING_MAX_AGE_S
        os.utime(d, (age, age))
    return d


@pytest.mark.parametrize("where", ["staging", "registered"])
def test_a_copy_left_by_a_prune_that_exited_is_pruned_once_abandoned(cache, where):
    base = artifacts_root() / ".staging" / "x" if where == "staging" else artifact_dir(models()["laya"], 64)
    d = _pruning_copy(base, pid=_dead_pid())
    plan = lifecycle.plan_prune(PROFILE)
    assert [(x["path"], x["reason"]) for x in plan] == [(str(d), "copy left by an interrupted prune")]
    assert [x["path"] for x in lifecycle.prune(plan)] == [str(d)]
    assert not d.exists()


@pytest.mark.parametrize("state", ["prune still running", "build still running", "has a manifest", "recent"])
def test_a_copy_left_by_a_prune_is_kept_while_anything_may_still_need_it(cache, state):
    base = artifact_dir(models()["laya"], 64)
    d = _pruning_copy(
        base,
        pid=os.getpid() if state == "prune still running" else _dead_pid(),
        marker_pid=os.getpid() if state == "build still running" else None,
        manifest=state == "has a manifest",
        old=state != "recent",
    )
    assert str(d) not in _listed(lifecycle.plan_prune(PROFILE))
    assert lifecycle.prune([{"path": str(d), "reason": "hand-written", "bytes": 0}]) == [] and d.exists()


@pytest.mark.parametrize("kind", ["staging", "pending"])
def test_a_link_outside_the_cache_is_never_claimed_or_reported_removed(cache, tmp_path_factory, kind):
    p, _ = _abandoned_entry(kind)
    link = tmp_path_factory.mktemp("outside") / "link"
    link.symlink_to(p)
    reason = "abandoned staging directory" if kind == "staging" else lifecycle._PENDING_REASON
    assert lifecycle.prune([{"path": str(link), "reason": reason, "bytes": 0}]) == []
    assert link.is_symlink() and p.exists()
    assert not [x for x in p.parent.iterdir() if ".pruning-" in x.name]


def test_a_link_to_a_leftover_prune_copy_is_never_removed(cache, tmp_path_factory):
    d = _pruning_copy(artifact_dir(models()["laya"], 64), pid=_dead_pid())
    link = tmp_path_factory.mktemp("outside") / "x.pruning-1-ab"
    link.symlink_to(d)
    assert lifecycle.prune([{"path": str(link), "reason": "hand-written", "bytes": 0}]) == []
    assert link.is_symlink() and d.exists()


def test_a_published_artifact_left_by_a_prune_is_reported_with_how_to_put_it_back(cache):
    registered = artifact_dir(models()["laya"], 64)
    d = _pruning_copy(registered, pid=_dead_pid(), manifest=True)
    [report] = lifecycle.kept_previous_artifacts()
    assert report["path"] == str(d) and report["registered"] == str(registered) and report["registered_missing"]
    assert "artifact left by an interrupted prune" in report["message"]
    assert f"mv {d} {registered}" in report["message"]
    assert str(d) not in _listed(lifecycle.plan_prune(PROFILE))


def test_a_claimed_copy_that_cannot_go_back_is_kept_and_logged(cache, monkeypatch, capsys):
    p, check = _abandoned_entry("pending")
    plan = lifecycle.plan_prune(PROFILE)
    real = getattr(lifecycle, check)
    calls = []

    def racing(d):
        result = real(d)
        calls.append(d)
        if len(calls) == 1:  # an import puts a live install there before the claim...
            os.rename(p, cache / "moved-away")
            _live_entry(p, "pending")
        elif len(calls) == 2:  # ...and something takes the path again after it
            p.mkdir()
            (p / "other").write_text("in use")
        return result

    monkeypatch.setattr(lifecycle, check, racing)
    assert lifecycle.prune(plan) == []
    monkeypatch.setattr(lifecycle, check, real)
    [kept] = [x for x in p.parent.iterdir() if ".pruning-" in x.name]
    assert (kept / "live").read_text() == "in use" and (p / "other").read_text() == "in use"
    assert f"kept {kept}: could not rename it back to {p}" in capsys.readouterr().err
