"""The GIL-releasing Core ML predict for configurations C and D, with its GIL re-acquire wait.

This is #46's binding (research/coreml-gil-completion-path/scripts/nogil_predict.py): the
same verified `model.mlmodelc`, `CPU_AND_NE`, FP16 features copied into MLMultiArrays by
PyObjC, outputs copied back with coremltools' names, dtypes and shapes, every call inside an
autorelease pool. One thing differs: `predictionFromFeatures:error:` is sent from a small
native shim (`predict_stamped.m`) called through `ctypes.CDLL`, not through PyObjC's method
call. Both release the GIL for the call. The shim stamps the clock before and after the
Objective-C message, so the time from its return stamp to the Python stamp taken once ctypes
has re-acquired the GIL is the ANE thread's GIL re-acquire wait. PyObjC's own call has no
point where that could be observed.

Research only: needs pyobjc-framework-CoreML (`uv run --with pyobjc-framework-CoreML==12.2.2`),
which laya-apple does not depend on, and the Xcode command-line tools' clang (the shim is
compiled into a temporary directory when first used; no binary is committed).
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "coreml-gil-completion-path" / "scripts"))

import nogil_predict  # noqa: E402  the #46 binding, reused unchanged

# (python_before_ns, native_before_ns, native_after_ns, python_after_ns) for every predict,
# all time.monotonic_ns(). Appended on the ANE thread; list.append is atomic under the GIL.
STAMPS: list[tuple[int, int, int, int]] = []

_lib = None
_lib_lock = threading.Lock()


def _shim():
    """Compile predict_stamped.m once per process and load it with ctypes.CDLL (GIL released)."""
    global _lib
    with _lib_lock:
        if _lib is None:
            build = Path(tempfile.mkdtemp(prefix="laya-nogil-shim-"))
            try:
                out = build / "libpredict_stamped.dylib"
                subprocess.run(
                    ["xcrun", "clang", "-dynamiclib", "-O2", "-fno-objc-arc", "-framework", "Foundation"]
                    + ["-framework", "CoreML", str(HERE / "predict_stamped.m"), "-o", str(out)],
                    check=True,
                    capture_output=True,
                )
                lib = ctypes.CDLL(str(out))  # CDLL, not PyDLL: ctypes releases the GIL for the call
            finally:
                shutil.rmtree(build, ignore_errors=True)  # the loaded image stays mapped
            fn = lib.laya_predict_stamped
            fn.restype = ctypes.c_void_p
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint64 * 2]
            _lib = fn
    return _lib


class StampedNoGilModel(nogil_predict.NoGilModel):
    """#46's NoGilModel; only the predict message itself goes through the stamping shim."""

    def __init__(self, compiled_path, compute_units: str = "CPU_AND_NE"):
        super().__init__(compiled_path, compute_units)
        self._call = _shim()
        self._model_ptr = self._model.__c_void_p__()

    def _predict(self, feats: dict) -> dict:
        import CoreML
        import objc

        values = {k: CoreML.MLFeatureValue.featureValueWithMultiArray_(self._array(k, v)) for k, v in feats.items()}
        provider, err = CoreML.MLDictionaryFeatureProvider.alloc().initWithDictionary_error_(values, None)
        if provider is None:
            raise RuntimeError(f"feature provider: {err}")
        error, stamps = ctypes.c_void_p(), (ctypes.c_uint64 * 2)()
        t0 = time.monotonic_ns()
        ptr = self._call(self._model_ptr, provider.__c_void_p__(), ctypes.byref(error), stamps)  # GIL released
        t1 = time.monotonic_ns()
        STAMPS.append((t0, stamps[0], stamps[1], t1))
        if not ptr:
            reason = objc.objc_object(c_void_p=error.value) if error.value else None
            raise RuntimeError(f"Core ML predict failed: {reason}")
        out = objc.objc_object(c_void_p=ptr)
        result = {}
        for name in out.featureNames():
            arr = out.featureValueForName_(name).multiArrayValue()
            v = nogil_predict._view(arr, nogil_predict._types()[arr.dataType()])
            result[str(name)] = v.astype(np.float32) if v.dtype == np.float16 else np.array(v)
        return result


def install(backend) -> None:
    """Swap every loaded bucket of one ANEBackend instance to the stamped GIL-releasing path."""
    from laya_apple.artifacts import COMPILED, artifact_dir

    backend.models = {b: StampedNoGilModel(artifact_dir(backend.spec, b) / COMPILED) for b in backend.models}
