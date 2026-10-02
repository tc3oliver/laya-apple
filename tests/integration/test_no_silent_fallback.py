"""v1.0 "no silent fallback" audit (docs/no-silent-fallback.md): every failure path that could
change the device, compute units, artifact or dtype either raises or is recorded."""

from __future__ import annotations

import json
import os
import signal
import time
import warnings

import pytest

from laya_apple import Laya, routing
from laya_apple.backends import coreml_ane
from laya_apple.errors import ArtifactIntegrityError, ArtifactMissingError, BackendUnavailableError, LayaAppleError
from laya_apple.workload import make_request

pytestmark = [pytest.mark.integration, pytest.mark.ane]
MODEL = "laya-typed-decisions"


@pytest.fixture(scope="module", autouse=True)
def _offline():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


@pytest.fixture(scope="module")
def gpu():
    return Laya.from_pretrained(MODEL, device="gpu", local_files_only=True)


def _request(laya, length, n=1, seed=5):
    return make_request(laya.tokenizer, laya.config, length, n, seed=seed)


# ----------------------------------------------------------------------------- dtype (audit V2)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_ane_refuses_a_dtype_it_is_not_validated_in(dtype):
    with pytest.raises(ValueError, match="dtype"):
        Laya.from_pretrained(MODEL, device="ane", dtype=dtype, local_files_only=True)


def test_unknown_dtype_is_refused_for_every_device():
    for device in ("gpu", "auto"):
        with pytest.raises(ValueError, match="dtype"):
            Laya.from_pretrained(MODEL, device=device, dtype="int8", local_files_only=True)


# ----------------------------------------------------------------------------- partial artifact cache


def _fail_bucket(monkeypatch, bucket, exc):
    real = coreml_ane.load_verified

    def fake(spec, b, **kw):
        if b == bucket:
            raise exc(f"test: L{b} {exc.__name__}")
        return real(spec, b, **kw)

    monkeypatch.setattr(coreml_ane, "load_verified", fake)


def test_explicit_ane_never_pads_up_to_a_larger_bucket(monkeypatch, gpu):
    _fail_bucket(monkeypatch, 64, ArtifactMissingError)
    laya = Laya.from_pretrained(MODEL, device="ane", local_files_only=True)
    assert laya.ane.buckets == (96, 128)
    state, qs = _request(gpu, 50)
    with pytest.raises(ArtifactMissingError):
        laya.predict(context=state, questions=qs)
    state, qs = _request(gpu, 90)
    assert laya.predict(context=state, questions=qs).runtime.device == "ane"


def test_explicit_ane_refuses_a_corrupt_bucket_at_load(monkeypatch):
    _fail_bucket(monkeypatch, 96, ArtifactIntegrityError)
    with pytest.raises(ArtifactIntegrityError):
        Laya.from_pretrained(MODEL, device="ane", local_files_only=True)


def test_auto_with_a_rejected_bucket_warns_and_records_the_reason(monkeypatch, gpu):
    _fail_bucket(monkeypatch, 64, ArtifactIntegrityError)
    with pytest.warns(RuntimeWarning, match="L64 rejected"):
        laya = Laya.from_pretrained(MODEL, device="auto", local_files_only=True)
    state, qs = _request(gpu, 50)
    r = laya.predict(context=state, questions=qs)
    assert (r.runtime.device, r.runtime.routing_reason) == ("gpu", routing.ARTIFACT_UNAVAILABLE)
    state, qs = _request(gpu, 90)
    r = laya.predict(context=state, questions=qs)
    assert (r.runtime.device, r.runtime.routing_reason) == ("ane", routing.ANE_AUTO)


def test_route_to_a_missing_backend_raises_instead_of_using_the_other(monkeypatch, gpu):
    laya = Laya.from_pretrained(MODEL, device="gpu", local_files_only=True)
    monkeypatch.setattr(laya, "route", lambda *a, **k: routing.Decision("ane", routing.ANE_AUTO, 64))
    called = []
    monkeypatch.setattr(laya.mlx, "forward", lambda *a, **k: called.append(1))
    state, qs = _request(gpu, 50)
    with pytest.raises(LayaAppleError):
        laya.predict(context=state, questions=qs)
    assert not called


# ----------------------------------------------------------------------------- dead or failed ANE worker


def _kill_ane(laya):
    w = laya._workers["ane"]
    assert w.placement == "process"
    os.kill(w.pid, signal.SIGKILL)
    w._proc.wait(timeout=10)


def test_dead_ane_worker_under_auto_is_recorded_and_warned_once(gpu):
    with Laya.from_pretrained(MODEL, execution="workers", ane_placement="process", local_files_only=True) as laya:
        state, qs = _request(gpu, 64)
        assert laya.predict(context=state, questions=qs).runtime.device == "ane"
        _kill_ane(laya)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = [laya.predict(context=state, questions=qs) for _ in range(3)]
        assert [(r.runtime.device, r.runtime.routing_reason) for r in results] == [
            ("gpu", routing.RUNTIME_UNAVAILABLE)
        ] * 3
        assert sum("ANE worker exited" in str(w.message) for w in caught) == 1
        assert results[0].answers == gpu.predict(context=state, questions=qs).answers


def test_dead_ane_worker_under_explicit_ane_keeps_raising(gpu):
    with Laya.from_pretrained(
        MODEL, device="ane", execution="workers", ane_placement="process", local_files_only=True
    ) as laya:
        _kill_ane(laya)
        state, qs = _request(gpu, 64)
        for _ in range(2):
            with pytest.raises(BackendUnavailableError):
                laya.predict(context=state, questions=qs)


def test_background_startup_failure_is_warned_and_recorded(monkeypatch, gpu):
    from laya_apple import executor

    real_warm = executor.warm

    def failing_warm(kind, backend, pad_id):
        if kind == "ane":
            raise RuntimeError("test: ANE compile failed")
        return real_warm(kind, backend, pad_id)

    monkeypatch.setattr(executor, "warm", failing_warm)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with Laya.from_pretrained(
            MODEL, execution="workers", ane_placement="thread", ane_startup="background", local_files_only=True
        ) as laya:
            assert laya.wait_for_ane(60) is False
            state, qs = _request(gpu, 64)
            r = laya.predict(context=state, questions=qs)
            assert (r.runtime.device, r.runtime.routing_reason) == ("gpu", routing.RUNTIME_UNAVAILABLE)
            assert laya.info()["ane_ready"] is False
    assert any("ANE startup failed" in str(w.message) for w in caught)


# ----------------------------------------------------------------------------- verification stamp


def test_changed_artifact_files_invalidate_the_verification_stamp(tmp_path, monkeypatch, manifest_factory):
    """A stamp is valid only for the exact files it was made for: any change re-runs the hash."""
    pytest.importorskip("coremltools")
    import laya_apple.artifacts as A
    from laya_apple.artifacts import COMPILED, artifact_dir, load_verified, tree_sha256
    from laya_apple.registry import models

    monkeypatch.setenv("LAYA_APPLE_CACHE", str(tmp_path))
    profile = {"soc": "Test SoC", "macos": "26.1", "macos_build": "X", "coremltools": "9.0"}
    monkeypatch.setattr(A, "platform_profile", lambda: dict(profile))
    monkeypatch.setattr(A, "compute_plan_summary", lambda *a: {"ops": {"ane": 1, "gpu": 0, "cpu": 0}})
    monkeypatch.setattr(A, "check_ane_placement", lambda p: None)

    class FakeModel:
        def __init__(self, *a, **k):
            pass

    import coremltools as ct

    monkeypatch.setattr(ct.models, "CompiledMLModel", FakeModel)
    spec = models()["laya"]
    d = artifact_dir(spec, 64)
    (d / COMPILED).mkdir(parents=True)
    (d / COMPILED / "weights.bin").write_bytes(b"a" * 64)
    data = manifest_factory(spec, 64, platform=profile)
    data["integrity"]["artifact_sha256"] = tree_sha256(d / COMPILED)
    (d / "manifest.json").write_text(json.dumps(data))
    load_verified(spec, 64)  # writes the stamp
    (d / COMPILED / "weights.bin").write_bytes(b"b" * 64)  # same size, new content and mtime
    with pytest.raises(ArtifactIntegrityError):
        load_verified(spec, 64)


def test_local_profile_with_unoffered_buckets_is_ignored(tmp_path, monkeypatch):
    from laya_apple import model as model_mod
    from laya_apple import profiles
    from laya_apple.artifacts import platform_profile
    from laya_apple.registry import routing_table

    monkeypatch.setattr(profiles, "cache_root", lambda: tmp_path)
    monkeypatch.setattr(model_mod, "platform_validated", lambda profile=None: False)
    path = profiles.local_profile_path()
    path.parent.mkdir(parents=True)
    from laya_apple.registry import models

    entry = {
        **routing_table()["models"][MODEL],
        "auto_ane_buckets": [200],
        "auto_ane_max_len": 200,
        "revision": models()[MODEL].revision,  # current, so only the bucket check can reject it
    }
    path.write_text(
        json.dumps(
            {
                "format": profiles.PROFILE_FORMAT,
                "format_version": profiles.PROFILE_VERSION,
                "platform": platform_profile(),
                "models": {MODEL: entry},
            }
        )
    )
    with pytest.warns(RuntimeWarning, match="not a prefix"):
        laya = Laya.from_pretrained(MODEL, local_files_only=True)
    assert laya.routing_profile is None


def test_unreadable_local_profile_is_warned(tmp_path, monkeypatch):
    from laya_apple import profiles

    monkeypatch.setattr(profiles, "cache_root", lambda: tmp_path)
    path = profiles.local_profile_path()
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    with pytest.warns(RuntimeWarning, match="unreadable local profile"):
        assert profiles.load_local(MODEL) is None


# ----------------------------------------------------------------------------- offline


def test_offline_with_an_uncached_checkpoint_raises_and_never_downloads(tmp_path):
    import subprocess
    import sys

    code = (
        "from laya_apple import Laya\n"
        "from laya_apple.errors import BackendUnavailableError\n"
        "try:\n"
        f"    Laya.from_pretrained({MODEL!r}, device='gpu', local_files_only=True)\n"
        "except BackendUnavailableError as e:\n"
        "    print('raised', 'laya-apple download' in str(e))\n"
    )
    env = dict(os.environ, HF_HOME=str(tmp_path / "hf"), HF_HUB_OFFLINE="1")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert out.stdout.strip() == "raised True", out.stderr[-2000:]
    assert not any((tmp_path / "hf").rglob("*.safetensors"))


# ----------------------------------------------------------------------------- runtime placement probe (audit V1)


class _FakeClock:
    """Stands in for `time.perf_counter`. The fake models advance it by exactly their duration, so the
    probe sees the intended timings whatever `time.sleep` precision the OS has (macOS 27 often sleeps
    10 ms for 15 ms, which made a sleep-based fake accept a model that ran like the CPU)."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class _TimedModel:
    def __init__(self, clock, ms):
        self.clock, self.ms = clock, ms

    def predict(self, feats):
        self.clock.now += self.ms / 1e3


@pytest.mark.parametrize("loaded_ms, cpu_ms, ok", [(4, 10, True), (10, 10, False), (12, 10, False)])
def test_placement_probe_refuses_a_model_that_runs_like_the_cpu(monkeypatch, loaded_ms, cpu_ms, ok, tmp_path):
    ct = pytest.importorskip("coremltools")
    from laya_apple.errors import ComputeUnitMismatchError
    from laya_apple.registry import models

    clock = _FakeClock()
    monkeypatch.setattr(time, "perf_counter", clock)  # the clock coreml_ane._fastest_ms times each predict with
    monkeypatch.setattr(ct.models, "CompiledMLModel", lambda *a, **k: _TimedModel(clock, cpu_ms))
    spec = models()[MODEL]
    if ok:
        r = coreml_ane.probe_placement(spec, 64, _TimedModel(clock, loaded_ms), tmp_path, {})
        assert r["ratio"] == pytest.approx(loaded_ms / cpu_ms)
        assert r["ratio"] < coreml_ane.PROBE_MAX_RATIO
    else:
        with pytest.raises(ComputeUnitMismatchError, match="not running on the Neural Engine"):
            coreml_ane.probe_placement(spec, 64, _TimedModel(clock, loaded_ms), tmp_path, {})


def test_placement_probe_failure_drops_the_bucket_under_auto_and_raises_under_ane(monkeypatch, gpu):
    from laya_apple.errors import ComputeUnitMismatchError

    real = coreml_ane.probe_placement

    def probe(spec, b, *a, **k):
        if b == 96:
            raise ComputeUnitMismatchError(f"test: L{b} is not running on the Neural Engine")
        return real(spec, b, *a, **k)

    monkeypatch.setattr(coreml_ane, "probe_placement", probe)
    with pytest.raises(ComputeUnitMismatchError):
        Laya.from_pretrained(MODEL, device="ane", local_files_only=True)
    with pytest.warns(RuntimeWarning, match="L96 rejected"):
        laya = Laya.from_pretrained(MODEL, device="auto", local_files_only=True)
    state, qs = _request(gpu, 90)
    r = laya.predict(context=state, questions=qs)
    assert (r.runtime.device, r.runtime.routing_reason) == ("gpu", routing.ARTIFACT_UNAVAILABLE)


def test_real_artifacts_pass_the_probe_and_report_it():
    laya = Laya.from_pretrained(MODEL, device="ane", local_files_only=True)
    probes = laya.info()["ane_probes"]
    assert set(probes) == {str(b) for b in laya.ane.buckets}
    assert all(p["ratio"] < coreml_ane.PROBE_MAX_RATIO for p in probes.values())
