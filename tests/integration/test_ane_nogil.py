"""The thread-placed ANE's GIL-releasing predict (backends/coreml_nogil.py) on real artifacts.

- The binding really releases the GIL: another thread's short sleep wakes on time during a
  predict loop through the binding, and late during the same loop through coremltools.
- The thread-placed workers path selects it and records it; LAYA_APPLE_ANE_PREDICT=coremltools
  and process placement keep coremltools. Answers equal the inline coremltools answers.
"""

from __future__ import annotations

import os
import statistics
import threading
import time

import pytest

from laya_apple import Laya
from laya_apple.artifacts import COMPILED, artifact_dir
from laya_apple.backends import coreml_nogil
from laya_apple.backends.coreml_ane import PROBE_MAX_RATIO
from laya_apple.registry import ANE_COMPUTE_UNITS
from laya_apple.workload import make_request

pytestmark = [pytest.mark.integration, pytest.mark.ane]
MODEL = "laya-typed-decisions"
SLEEP_S = 0.001
WAKES = 300


@pytest.fixture(scope="module", autouse=True)
def _offline():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    pytest.importorskip("CoreML")


@pytest.fixture(scope="module")
def inline_ane():
    laya = Laya.from_pretrained(MODEL, device="ane", local_files_only=True)
    assert laya.info()["ane_predict"] == coreml_nogil.COREMLTOOLS
    return laya


def _requests(laya):
    return [make_request(laya.tokenizer, laya.config, n, 1, seed=s) for s, n in enumerate((20, 60, 90, 120))]


def _late_wakes_ms(model, feats) -> float:
    """Median lateness of a 1 ms sleep on this thread while another thread loops `predict`."""
    stop = threading.Event()

    def load():
        while not stop.is_set():
            model.predict(feats)

    t = threading.Thread(target=load, daemon=True)
    t.start()
    time.sleep(0.2)  # the loop is running
    late = []
    try:
        for _ in range(WAKES):
            t0 = time.perf_counter()
            time.sleep(SLEEP_S)
            late.append((time.perf_counter() - t0 - SLEEP_S) * 1e3)
    finally:
        stop.set()
        t.join(30)
    return statistics.median(late)


def test_the_binding_releases_the_gil_during_predict(inline_ane, monkeypatch):
    monkeypatch.delenv(coreml_nogil.ENV, raising=False)
    b = max(inline_ane.ane.buckets)  # the longest predict: the clearest signal
    feats = inline_ane.ane._probe_features(b)
    nogil = coreml_nogil.NoGilModel(artifact_dir(inline_ane.spec, b) / COMPILED, ANE_COMPUTE_UNITS)
    held = _late_wakes_ms(inline_ane.ane.models[b], feats)  # coremltools: holds the GIL
    released = _late_wakes_ms(nogil, feats)
    # #46 measured 8.47 ms (coremltools) against 0.515 ms (PyObjC) at L128, idle ~0.51 ms.
    assert released < 2.0, f"a 1 ms sleep woke {released:.2f} ms late during nogil predicts"
    assert released < 0.5 * held, f"nogil {released:.2f} ms vs coremltools {held:.2f} ms late"


@pytest.mark.parametrize("env", [None, "nogil"])
def test_thread_placed_workers_use_the_selected_binding_and_answer_as_coremltools(inline_ane, monkeypatch, env):
    if env is None:  # the default: DEFAULT_AUTO
        monkeypatch.delenv(coreml_nogil.ENV, raising=False)
        want = (coreml_nogil.DEFAULT_AUTO, coreml_nogil.DEFAULT)
    else:  # forced: a binding failure would raise here instead of falling back
        monkeypatch.setenv(coreml_nogil.ENV, env)
        want = (coreml_nogil.NOGIL, coreml_nogil.ENV_NOGIL)
    with Laya.from_pretrained(
        MODEL, device="ane", execution="workers", ane_placement="thread", local_files_only=True
    ) as laya:
        info = laya.info()
        assert (info["ane_predict"], info["ane_predict_reason"]) == want
        # the load-time placement probe ran through that binding and passed
        assert set(info["ane_probes"]) == {str(b) for b in laya.ane.buckets}
        assert all(p["ratio"] < PROBE_MAX_RATIO for p in info["ane_probes"].values())
        for state, qs in _requests(inline_ane):
            r, ref = laya.predict(context=state, questions=qs), inline_ane.predict(context=state, questions=qs)
            assert (r.runtime.device, r.runtime.compute_units) == ("ane", ANE_COMPUTE_UNITS)
            assert (r.runtime.ane_predict, r.runtime.ane_predict_reason) == want
            assert r.runtime.artifact_revision == ref.runtime.artifact_revision
            assert r.answers == ref.answers


def test_env_override_keeps_the_thread_placed_ane_on_coremltools(inline_ane, monkeypatch):
    monkeypatch.setenv(coreml_nogil.ENV, "coremltools")
    with Laya.from_pretrained(
        MODEL, device="ane", execution="workers", ane_placement="thread", local_files_only=True
    ) as laya:
        info = laya.info()
        assert (info["ane_predict"], info["ane_predict_reason"]) == (
            coreml_nogil.COREMLTOOLS,
            coreml_nogil.ENV_COREMLTOOLS,
        )
        state, qs = _requests(inline_ane)[0]
        r = laya.predict(context=state, questions=qs)
        assert r.runtime.ane_predict == coreml_nogil.COREMLTOOLS
        assert r.answers == inline_ane.predict(context=state, questions=qs).answers


def test_process_placement_keeps_coremltools(inline_ane, monkeypatch):
    monkeypatch.delenv(coreml_nogil.ENV, raising=False)
    with Laya.from_pretrained(
        MODEL, device="ane", execution="workers", ane_placement="process", local_files_only=True
    ) as laya:
        info = laya.info()
        assert (info["ane_predict"], info["ane_predict_reason"]) == (
            coreml_nogil.COREMLTOOLS,
            coreml_nogil.NOT_THREAD_PLACED,
        )
