"""`artifacts fetch` end to end with a local stand-in for the Hugging Face repository: a real
exported archive and index go through import (parity, compute plan) and the placement probe
into an empty cache. Needs the multilingual L64 artifact in the cache, as test_export_import."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from laya_apple import lifecycle, prebuilt
from laya_apple.artifacts import artifact_dir, load_verified, read_manifest
from laya_apple.errors import ArtifactMissingError
from laya_apple.registry import models

pytestmark = [pytest.mark.integration, pytest.mark.ane]
SPEC = models()["laya-multilingual"]
BUCKET = 64


@pytest.fixture(scope="module")
def staged_repo(tmp_path_factory):
    """A directory laid out as the repository: index.json plus one archive."""
    try:
        load_verified(SPEC, BUCKET)
    except ArtifactMissingError:
        pytest.skip("no multilingual L64 artifact in the cache to export")
    root = tmp_path_factory.mktemp("repo")
    manifest = read_manifest(SPEC, BUCKET)
    rel = prebuilt.archive_path(SPEC, BUCKET, manifest["platform"])
    (root / rel).parent.mkdir(parents=True)
    archive = lifecycle.export_artifact(SPEC, BUCKET, root / rel[: -len(".tar.gz")])
    index = prebuilt.merge_index(None, [prebuilt.make_index_entry(manifest, archive, rel)])
    (root / prebuilt.INDEX).write_text(json.dumps(index))
    return root


def test_fetch_registers_only_after_local_validation_and_probe(staged_repo, tmp_path, monkeypatch):
    def download(repo, filename, revision, local_dir):
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staged_repo / filename, dest)
        return dest

    monkeypatch.setattr(prebuilt, "_download", download)
    monkeypatch.setenv("LAYA_APPLE_CACHE", str(tmp_path))
    out = prebuilt.fetch(SPEC, [BUCKET], repo="local/stand-in", local_files_only=True, log=lambda m: None)
    assert out[BUCKET]["path"] == artifact_dir(SPEC, BUCKET)
    assert out[BUCKET]["probe"]["ratio"] <= 0.8
    data = json.loads((artifact_dir(SPEC, BUCKET) / "manifest.json").read_text())
    assert data["imported"]["from"].startswith("hf://local/stand-in@main/")
    assert data["imported"]["parity"]["passed"] is True
    assert not any((tmp_path / "artifacts" / ".staging").iterdir())
