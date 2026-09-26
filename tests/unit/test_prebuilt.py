"""laya_apple/prebuilt.py without the network or Core ML: repository resolution, the index
format, per-profile selection, and that `fetch` hands every archive to import_artifact (which
re-validates it) only after its SHA-256 matches, then runs the placement probe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from laya_apple import cli, lifecycle, prebuilt
from laya_apple.artifacts import artifact_dir, artifacts_root
from laya_apple.errors import ArtifactError, ArtifactIntegrityError, ArtifactMissingError, BackendUnavailableError
from laya_apple.registry import ANE_GRAPH, models

SPEC = models()["laya"]
HERE = {"soc": "Apple M4 Max", "macos": "26.0.1", "macos_build": "25A1", "coremltools": "9.0"}
OTHER = {"soc": "Apple M1", "macos": "15.6", "macos_build": "24G1", "coremltools": "9.0"}


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
    """A fake repository: {filename: bytes}; records downloads, imports and probes."""
    monkeypatch.setenv("LAYA_APPLE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(prebuilt, "platform_profile", lambda: dict(HERE))
    files: dict = {}
    calls = {"download": [], "import": [], "probe": []}

    def download(repo_id, filename, revision, local_dir):
        calls["download"].append((repo_id, filename, revision))
        if filename not in files:
            raise BackendUnavailableError(f"404 {filename}")
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(files[filename])
        return path

    def import_artifact(archive, *, local_files_only, force, log, source):
        assert Path(archive).exists()
        calls["import"].append({"archive": Path(archive).name, "source": source, "force": force})
        return Path("/registered") / Path(archive).name

    def probe(spec, buckets, *, local_files_only):
        calls["probe"].append(sorted(buckets))
        return {b: {"ane_ms": 1.0, "cpu_ms": 3.0, "ratio": 0.333} for b in buckets}

    monkeypatch.setattr(prebuilt, "_download", download)
    monkeypatch.setattr(lifecycle, "import_artifact", import_artifact)
    monkeypatch.setattr(prebuilt, "_probe", probe)
    return files, calls


def put(files, *entries, payloads=None):
    files[prebuilt.INDEX] = json.dumps(index(*entries)).encode()
    for e in entries:
        files[e["path"]] = (payloads or {}).get(e["path"], b"archive")


# ----------------------------------------------------------------------------- repository


def test_the_placeholder_repository_is_refused_until_published(monkeypatch):
    monkeypatch.delenv(prebuilt.PREBUILT_REPO_ENV, raising=False)
    assert prebuilt.DEFAULT_REPO_PUBLISHED is False
    with pytest.raises(BackendUnavailableError, match="no prebuilt artifact repository is published"):
        prebuilt.resolve_repo()
    monkeypatch.setenv(prebuilt.PREBUILT_REPO_ENV, "someone/artifacts")
    assert prebuilt.resolve_repo() == "someone/artifacts"
    assert prebuilt.resolve_repo("explicit/repo") == "explicit/repo"


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
    assert calls["probe"] == [[64, 96]]
    assert out[64]["probe"]["ratio"] == 0.333
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


def test_already_registered_bucket_is_skipped_unless_forced(repo):
    files, calls = repo
    put(files, entry(64))
    registered = artifact_dir(SPEC, 64)
    registered.mkdir(parents=True)
    (registered / "manifest.json").write_text("{}")
    assert prebuilt.fetch(SPEC, [64], repo="o/r", log=lambda m: None) == {}
    assert calls["download"] == []
    prebuilt.fetch(SPEC, [64], repo="o/r", force=True, log=lambda m: None)
    assert calls["import"][0]["force"] is True


def test_unoffered_bucket_is_refused(repo):
    with pytest.raises(ArtifactError, match="does not offer"):
        prebuilt.fetch(SPEC, [999], repo="o/r", log=lambda m: None)


def test_cli_parses_fetch():
    a = cli.build_parser().parse_args(["artifacts", "fetch", "laya", "--length", "64", "--repo", "o/r"])
    assert (a.action, a.model, a.length, a.repo, a.revision) == ("fetch", "laya", [64], "o/r", "main")
