"""Unit tests for profiles.calibrate's internals: parity extraction and the locked,
concurrency-safe profile write (code review finding 12)."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from laya_apple.derivation import derive_model
from laya_apple.errors import ArtifactParityError
from laya_apple.profiles import (
    PROFILE_FORMAT,
    PROFILE_VERSION,
    _parity_records,
    _save_profile,
    load_shipped,
    profile_key,
    shipped_profile_matches,
)
from laya_apple.registry import models


class _Manifest:
    def __init__(self, data):
        self.data = data


def test_parity_records_extracts_passed_and_prob():
    manifests = {64: _Manifest({"parity": {"passed": True, "prob_max_abs": 0.01}})}
    assert _parity_records("m", manifests) == {64: (True, 0.01)}


def test_parity_records_missing_raises_parity_error():
    manifests = {64: _Manifest({"parity": None})}
    with pytest.raises(ArtifactParityError):
        _parity_records("m", manifests)


def test_parity_records_incomplete_raises_parity_error():
    manifests = {64: _Manifest({"parity": {"passed": True}})}  # no prob_max_abs
    with pytest.raises(ArtifactParityError):
        _parity_records("m", manifests)


def test_save_profile_concurrent_writers_do_not_corrupt_the_file(tmp_path):
    path = tmp_path / "profile.json"
    profile = {"soc": "test", "macos": "1.0", "coremltools": "9.0"}
    errors = []

    def write(model, n):
        try:
            for i in range(20):
                _save_profile(path, profile, model, {"i": i, "n": n})
        except Exception as e:  # pragma: no cover - failure path
            errors.append(e)

    threads = [threading.Thread(target=write, args=(f"model-{n}", n)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    data = json.loads(path.read_text())
    assert data["format"] == PROFILE_FORMAT and data["format_version"] == PROFILE_VERSION
    assert set(data["models"]) == {f"model-{n}" for n in range(6)}  # every writer's entry survived


# --------------------------------------------------------------------- shipped calibrated profiles

ROOT = Path(__file__).resolve().parents[2]
SHIPPED = sorted((ROOT / "laya_apple/data/profiles").glob("*.json"))
# Each shipped calibrated profile is the unedited calibrate output committed as evidence.
EVIDENCE = {"Apple_M4_Max-macos27-coremltools9.0.json": ROOT / "benchmarks/routing-macos27/raw/pass1-profile.json"}


def test_every_shipped_profile_has_committed_evidence():
    assert SHIPPED and {p.name for p in SHIPPED} == set(EVIDENCE)


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.stem)
def test_shipped_profile_is_the_committed_calibration(path):
    assert path.read_text() == EVIDENCE[path.name].read_text()


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.stem)
def test_shipped_profile_rederives_and_matches_the_registry(path):
    data = json.loads(path.read_text())
    assert data["format"] == PROFILE_FORMAT and data["format_version"] == PROFILE_VERSION
    assert profile_key(data["platform"]) == path.stem
    assert set(data["models"]) == set(models())
    for name, spec in models().items():
        entry = data["models"][name]
        assert entry["revision"] == spec.revision
        par = {d["bucket"]: (d["parity_pass"], d["parity_prob_max_abs"]) for d in entry["decisions"]}
        ane = {d["bucket"]: d["ane_p50_ms"] for d in entry["decisions"]}
        mlx = {int(L): v for L, v in entry["service_ms"]["gpu"]["1"].items()}
        again = derive_model(spec.ane_buckets, par, ane, mlx)
        assert again["auto_ane_buckets"] == entry["auto_ane_buckets"]
        assert tuple(entry["auto_ane_buckets"]) == spec.ane_buckets[: len(entry["auto_ane_buckets"])]


def test_shipped_profile_lookup_by_platform():
    macos27 = {"soc": "Apple M4 Max", "macos": "27.0", "coremltools": "9.0"}
    entry = load_shipped("laya", macos27)
    assert entry is not None
    assert entry["source"] == "laya_apple/data/profiles/Apple_M4_Max-macos27-coremltools9.0.json"
    assert shipped_profile_matches(macos27)
    assert shipped_profile_matches(dict(macos27, macos="27.1"))  # same macOS major
    assert load_shipped("laya", dict(macos27, coremltools="9.1")) is None
    assert not shipped_profile_matches(dict(macos27, coremltools="9.1"))
    assert not shipped_profile_matches(dict(macos27, soc="Apple M4 Pro"))
    # macOS 26.6.2 is routing.json's profile: shipped, and no calibrated profile overrides it
    macos26 = dict(macos27, macos="26.6.2")
    assert shipped_profile_matches(macos26) and load_shipped("laya", macos26) is None
