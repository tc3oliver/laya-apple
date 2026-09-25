"""20,000 predicts through the GIL-releasing binding on a worker thread: flat RSS, flat time.

Guards the leak #46 found (research/coreml-gil-completion-path/): a Python thread has no
autorelease pool, so without one per call every predict leaked Core ML's IOSurface-backed
outputs. The ANE then slowed down within minutes (predict 10.2 -> 11.1 ms) until IOSurface
allocation failed. With the pool, 20,000 calls held predict at 9.85-9.98 ms and RSS flat.
"""

from __future__ import annotations

import gc
import os
import statistics
import threading
import time

import pytest

from laya_apple import Laya
from laya_apple.artifacts import COMPILED, artifact_dir
from laya_apple.backends import coreml_nogil
from laya_apple.registry import ANE_COMPUTE_UNITS

pytestmark = [pytest.mark.ane]
MODEL = "laya-typed-decisions"
CALLS = 20_000
WINDOW = 1_000  # calls per timing window and RSS sample
MAX_RSS_GROWTH = 64 * 1024 * 1024  # after the first window
MAX_SLOWDOWN = 1.10  # last window's median predict against the first's


def test_nogil_predict_soak_has_flat_rss_and_flat_time():
    psutil = pytest.importorskip("psutil")
    pytest.importorskip("CoreML")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    laya = Laya.from_pretrained(MODEL, device="ane", local_files_only=True)
    b = max(laya.ane.buckets)
    feats = laya.ane._probe_features(b)
    model = coreml_nogil.NoGilModel(artifact_dir(laya.spec, b) / COMPILED, ANE_COMPUTE_UNITS)
    proc = psutil.Process(os.getpid())
    windows: list[float] = []
    rss: list[int] = []
    errors: list[BaseException] = []

    def run():  # a non-main Python thread, as the product's ANE dispatcher is
        try:
            for w in range(CALLS // WINDOW):
                times = []
                for _ in range(WINDOW):
                    t = time.perf_counter()
                    model.predict(feats)
                    times.append(time.perf_counter() - t)
                windows.append(statistics.median(times) * 1e3)
                gc.collect()
                rss.append(proc.memory_info().rss)
        except BaseException as e:
            errors.append(e)

    t = threading.Thread(target=run, name="laya-ane-soak")
    t.start()
    t.join()
    assert not errors, errors[0]
    assert len(windows) == CALLS // WINDOW
    growth = rss[-1] - rss[0]
    assert growth < MAX_RSS_GROWTH, f"RSS grew {growth / 1e6:.1f} MB over {CALLS} calls; samples={rss}"
    assert windows[-1] <= MAX_SLOWDOWN * windows[0], f"predict median per {WINDOW} calls (ms): {windows}"
