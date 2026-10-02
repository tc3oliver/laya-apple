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
# The ship criterion (benchmarks/routing-macos27/PLAN.md): two passes agree on every model's auto buckets.
PASSES = {
    "Apple_M4_Max-macos27-coremltools9.0.json": (
        ROOT / "benchmarks/routing-macos27/raw/pass1-profile.json",
        ROOT / "benchmarks/routing-macos27/raw/pass2-profile.json",
    )
}
MACOS27 = {"soc": "Apple M4 Max", "macos": "27.0", "coremltools": "9.0"}


def test_every_shipped_profile_has_committed_evidence():
    assert SHIPPED and {p.name for p in SHIPPED} == set(EVIDENCE) == set(PASSES)


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.stem)
def test_shipped_profile_passes_agree_on_auto_buckets(path):
    first, second = (json.loads(p.read_text())["models"] for p in PASSES[path.name])
    assert set(first) == set(second) == set(models())
    for name in models():
        assert first[name]["auto_ane_buckets"] == second[name]["auto_ane_buckets"], name


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
        assert entry["auto_ane_max_len"] == max(entry["auto_ane_buckets"])
        for d in entry["decisions"]:
            assert entry["service_ms"]["ane"]["1"][str(d["bucket"])] == d["ane_p50_ms"]


def test_shipped_profile_lookup_by_platform():
    macos27 = dict(MACOS27)
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


def _routing_profile_on(monkeypatch, platform: dict, model="laya-multilingual"):
    """Laya._routing_profile on a stand-in instance, with this process's profile replaced."""
    from types import SimpleNamespace

    from laya_apple import profiles
    from laya_apple.model import Laya

    monkeypatch.setattr(profiles, "platform_profile", lambda: dict(platform))
    obj = SimpleNamespace(spec=models()[model])
    return obj, Laya._routing_profile(obj)


def test_routing_profile_uses_the_shipped_calibration_on_macos27(monkeypatch):
    from laya_apple import profiles

    monkeypatch.setattr(profiles, "load_local", lambda *a, **k: pytest.fail("local profile read on a shipped profile"))
    obj, (name, service) = _routing_profile_on(monkeypatch, MACOS27)
    entry = load_shipped("laya-multilingual", MACOS27)
    assert name == "shipped" and service == entry["service_ms"]
    assert obj.spec.auto_ane_buckets == tuple(entry["auto_ane_buckets"])
    assert obj.spec.auto_ane_max_len == entry["auto_ane_max_len"]


def test_routing_profile_applies_the_shipped_entry_not_routing_json(monkeypatch):
    from laya_apple import profiles

    fake = {"auto_ane_buckets": [64], "auto_ane_max_len": 64, "auto_ane_max_questions": 1, "service_ms": {"x": 1}}
    monkeypatch.setattr(profiles, "load_shipped", lambda model: dict(fake, source="shipped-file"))
    obj, (name, service) = _routing_profile_on(monkeypatch, MACOS27)
    assert (name, service, obj.spec.auto_ane_buckets, obj.spec.auto_ane_max_len) == ("shipped", {"x": 1}, (64,), 64)


def test_routing_profile_on_macos26_is_routing_json(monkeypatch):
    from laya_apple.registry import routing_table

    spec = models()["laya-multilingual"]
    obj, (name, service) = _routing_profile_on(monkeypatch, dict(MACOS27, macos="26.6.2"))
    assert name == "shipped" and service == routing_table()["models"]["laya-multilingual"]["service_ms"]
    assert obj.spec == spec


def test_routing_profile_bad_shipped_buckets_warn_without_calibrate_advice(monkeypatch):
    from laya_apple import profiles

    fake = {"auto_ane_buckets": [96], "auto_ane_max_len": 96, "auto_ane_max_questions": 1, "service_ms": {}}
    monkeypatch.setattr(profiles, "load_shipped", lambda model: dict(fake, source="shipped-file"))
    with pytest.warns(RuntimeWarning, match="not a prefix") as caught:
        _, (name, _) = _routing_profile_on(monkeypatch, MACOS27)
    assert name is None
    message = str(caught[0].message)
    assert "reinstall laya-apple" in message and "calibrate" not in message


def test_switchyard_auto_buckets_follow_the_shipped_calibration(monkeypatch):
    from laya_apple import profiles
    from laya_apple.demos.switchyard import ane_state

    spec = models()["laya-typed-decisions"]
    monkeypatch.setattr(profiles, "platform_profile", lambda: dict(MACOS27))
    assert ane_state._auto_buckets(spec, True) == tuple(load_shipped(spec.name, MACOS27)["auto_ane_buckets"])
    monkeypatch.setattr(profiles, "load_shipped", lambda model: {"auto_ane_buckets": [64]})
    monkeypatch.setattr(profiles, "load_local", lambda model: pytest.fail("local profile read on a shipped profile"))
    assert ane_state._auto_buckets(spec, True) == (64,)
