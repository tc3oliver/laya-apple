"""PB: the prebound Core ML predict, with one prediction crossing per forward (research only).

At load, for each bucket, once:
  - load the same verified model.mlmodelc with CPU_AND_NE (#46's loader, reused);
  - allocate an MLMultiArray per input (declared shape and data type), take a NumPy view of
    each, and build the MLFeatureValues and one MLDictionaryFeatureProvider over them;
  - allocate an MLMultiArray per output, set them as MLPredictionOptions.outputBackings, and
    take a NumPy view of each;
  - keep strong references to all of it on this object.

Per forward:
  1. write the features into the input views (NumPy only);
  2. push an autorelease pool;
  3. send predictionFromFeatures:options:error: once, through predict_options_stamped.m
     called with ctypes.CDLL (the GIL is released once, for that send only);
  4. pop the pool;
  5. copy the outputs out of the backed views (NumPy only), with coremltools' names, dtypes
     and shapes (FP16 widened to float32, as coremltools and #46 do).

Output backings are checked per output at load (verify_backings): an output whose backing is
filled by the prediction is read from its view ("backed"); any other output is read from the
returned provider with the fewest extra sends, the shape and strides cached ("read"). The mode
of each output is in `self.modes`.

A bucket's buffers are reused, so each bucket has a non-blocking lock: a forward that finds
its bucket already in flight raises instead of overwriting a buffer another forward is using.

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

import nogil_predict  # noqa: E402  #46's loader, dtype table and strided view, reused unchanged

# (python_before_ns, native_before_ns, native_after_ns, python_after_ns) for every predict,
# all time.monotonic_ns(); the same layout as #77's nogil.STAMPS. Appended on the ANE thread.
STAMPS: list[tuple[int, int, int, int]] = []

SENTINEL = 0xFF  # every byte of a backing before verify_backings: an FP16/FP32 NaN pattern

_fn = None
_fn_lock = threading.Lock()


def _shim():
    """Compile predict_options_stamped.m once per process and load it with ctypes.CDLL."""
    global _fn
    with _fn_lock:
        if _fn is None:
            build = Path(tempfile.mkdtemp(prefix="laya-prebind-shim-"))
            try:
                out = build / "libpredict_options_stamped.dylib"
                subprocess.run(
                    ["xcrun", "clang", "-dynamiclib", "-O2", "-fno-objc-arc", "-framework", "Foundation"]
                    + ["-framework", "CoreML", str(HERE / "predict_options_stamped.m"), "-o", str(out)],
                    check=True,
                    capture_output=True,
                )
                # the shim must not call back into Python: then its one call is the one crossing
                undefined = subprocess.run(["nm", "-u", str(out)], check=True, capture_output=True, text=True).stdout
                python_refs = [s for s in undefined.split() if s.startswith("_Py") or s.startswith("__Py")]
                if python_refs:
                    raise RuntimeError(f"predict shim references Python symbols: {python_refs}")
                lib = ctypes.CDLL(str(out))  # CDLL, not PyDLL: ctypes releases the GIL for the call
            finally:
                shutil.rmtree(build, ignore_errors=True)  # the loaded image stays mapped
            fn = lib.laya_predict_options_stamped
            fn.restype = ctypes.c_void_p
            fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint64 * 2]
            _fn = fn
    return _fn


def _address(view: np.ndarray) -> int:
    return int(view.__array_interface__["data"][0])


def _cached_view(arr, np_dtype, shape, strides) -> np.ndarray:
    """nogil_predict._view with the shape and strides known: one dataPointer send, not three."""
    item = np.dtype(np_dtype).itemsize
    span = 1 + sum((n - 1) * s for n, s in zip(shape, strides))
    flat = np.frombuffer(arr.dataPointer().as_buffer(span * item), dtype=np_dtype, count=span)
    return np.lib.stride_tricks.as_strided(flat, shape=shape, strides=[s * item for s in strides])


def _ns_shape(shape):
    """An immutable NSArray of NSNumbers. A Python list handed to Objective-C becomes a
    Python-backed proxy (OC_PythonArray), and Core ML reading it during predict calls back into
    Python, taking the GIL inside the one crossing; a Foundation array never does."""
    import Foundation

    arr = Foundation.NSMutableArray.alloc().initWithCapacity_(len(shape))
    for s in shape:
        arr.addObject_(Foundation.NSNumber.numberWithLongLong_(int(s)))
    return arr.copy()


def _ns_dict(items: dict):
    """An immutable NSDictionary with NSString keys, for the same reason as _ns_shape."""
    import Foundation

    d = Foundation.NSMutableDictionary.alloc().initWithCapacity_(len(items))
    for k, v in items.items():
        d.setObject_forKey_(v, Foundation.NSString.alloc().initWithUTF8String_(k.encode()))
    return d.copy()


def python_backed(obj) -> list[str]:
    """Class names of Python-backed Objective-C objects in a Foundation container tree (load-time
    check: nothing Core ML reads during predict may call back into Python)."""
    import objc

    found, stack = [], [obj]
    while stack:
        o = stack.pop()
        o = getattr(o, "__pyobjc_object__", o)  # a pythonified NSNumber / NSString: its NSObject
        if hasattr(o, "nsstring"):
            o = o.nsstring()
        if not isinstance(o, objc.objc_object):
            found.append(type(o).__name__)  # a plain Python value would be proxied when passed
            continue
        name = str(o.className())
        if name.startswith("OC_Python"):
            found.append(name)
        if o.isKindOfClass_(objc.lookUpClass("NSDictionary")):
            for k in o.allKeys():
                stack += [k, o.objectForKey_(k)]
        elif o.isKindOfClass_(objc.lookUpClass("NSArray")):
            stack += list(o)
    return found


class PrebindModel(nogil_predict.NoGilModel):
    """Duck-types the `predict(feats) -> dict[str, np.ndarray]` of a coremltools model."""

    def __init__(self, compiled_path, compute_units: str = "CPU_AND_NE"):
        import CoreML

        super().__init__(compiled_path, compute_units)  # #46: MLModel + declared inputs
        types = nogil_predict._types()
        self._in_arrays, self._in_views, self._shapes = {}, {}, []
        for name, (code, shape) in self._inputs.items():
            ns_shape = _ns_shape(shape)
            self._shapes.append(ns_shape)
            arr, err = CoreML.MLMultiArray.alloc().initWithShape_dataType_error_(ns_shape, code, None)
            if arr is None:
                raise RuntimeError(f"MLMultiArray for input {name}: {err}")
            self._in_arrays[name] = arr
            self._in_views[name] = nogil_predict._view(arr, types[code])
        self._values = _ns_dict(
            {k: CoreML.MLFeatureValue.featureValueWithMultiArray_(a) for k, a in self._in_arrays.items()}
        )
        provider, err = CoreML.MLDictionaryFeatureProvider.alloc().initWithDictionary_error_(self._values, None)
        if provider is None:
            raise RuntimeError(f"feature provider: {err}")
        self._provider = provider
        self._out_arrays, self._out_views, self._out_meta = {}, {}, {}
        for name, desc in self._model.modelDescription().outputDescriptionsByName().items():
            c = desc.multiArrayConstraint()
            code, shape = c.dataType(), [int(s) for s in c.shape()]
            ns_shape = _ns_shape(shape)
            self._shapes.append(ns_shape)
            arr, err = CoreML.MLMultiArray.alloc().initWithShape_dataType_error_(ns_shape, code, None)
            if arr is None:
                raise RuntimeError(f"MLMultiArray backing for output {name}: {err}")
            self._out_arrays[str(name)] = arr
            self._out_views[str(name)] = nogil_predict._view(arr, types[code])
            self._out_meta[str(name)] = (types[code], shape, [int(s) for s in arr.strides()])
        self._backings = _ns_dict(self._out_arrays)
        options = CoreML.MLPredictionOptions.alloc().init()
        options.setOutputBackings_(self._backings)
        self._options = options
        # everything Core ML reads during predict must be a Foundation object, or it calls back
        # into Python (and takes the GIL) inside the crossing
        self.python_backed = python_backed(provider.dictionary()) + python_backed(options.outputBackings())
        self.python_backed += [
            n for a in (*self._in_arrays.values(), *self._out_arrays.values()) for n in python_backed(a.shape())
        ]
        if self.python_backed:
            raise RuntimeError(f"Python-backed objects reach Core ML: {self.python_backed}")
        # raw pointers for the shim, as ints once; the PyObjC objects above keep them alive
        self._ptrs = tuple(o.__c_void_p__().value for o in (self._model, self._provider, self._options))
        self._err, self._stamps = ctypes.c_void_p(), (ctypes.c_uint64 * 2)()
        self._err_ref = ctypes.byref(self._err)
        self._call = _shim()
        self._busy = threading.Lock()
        self.modes = {n: "unverified" for n in self._out_arrays}
        self.verify_backings()

    # ------------------------------------------------------------------ load time

    def verify_backings(self) -> dict:
        """Decide each output's mode from one prediction: 'backed' if the backing was filled with
        exactly the returned output, else 'read'. Runs at load; its sends are not on the request
        path. Returns the evidence per output."""
        import objc

        for name, view in self._in_views.items():
            view[...] = 0
        for view in self._out_views.values():  # every element set to the all-SENTINEL bit pattern
            view[...] = np.frombuffer(bytes([SENTINEL]) * view.dtype.itemsize, view.dtype)[0]
        evidence = {}
        with objc.autorelease_pool():
            err, stamps = ctypes.c_void_p(), (ctypes.c_uint64 * 2)()
            ptr = self._call(*self._ptrs, ctypes.byref(err), stamps)
            if not ptr:
                reason = objc.objc_object(c_void_p=err.value) if err.value else None
                raise RuntimeError(f"Core ML predict (backing check) failed: {reason}")
            out = objc.objc_object(c_void_p=ptr)
            for name, backing in self._out_views.items():
                arr = out.featureValueForName_(name).multiArrayValue()
                returned = nogil_predict._view(arr, nogil_predict._types()[arr.dataType()])
                written = not np.all(np.ascontiguousarray(backing).view(np.uint8) == SENTINEL)
                equal = returned.dtype == backing.dtype and np.array_equal(
                    np.ascontiguousarray(returned).view(np.uint8), np.ascontiguousarray(backing).view(np.uint8)
                )
                evidence[name] = {
                    "same_address": _address(returned) == _address(backing),
                    "backing_written": bool(written),
                    "equal_to_returned": bool(equal),
                    "dtype": str(backing.dtype),
                    "shape": list(backing.shape),
                }
                self.modes[name] = "backed" if written and equal else "read"
        self.backing_evidence = evidence
        return evidence

    # ------------------------------------------------------------------ request path

    def predict(self, feats: dict) -> dict:
        import objc

        if not self._busy.acquire(blocking=False):
            raise RuntimeError("prebound bucket already in flight: its buffers would be overwritten")
        try:
            for name, view in self._in_views.items():  # the same conversion as #46's _array
                view[...] = np.asarray(feats[name], dtype=view.dtype).reshape(view.shape)
            result = {}
            with objc.autorelease_pool():
                err, stamps = self._err, self._stamps  # this bucket's, reused under self._busy
                err.value = None
                t0 = time.monotonic_ns()
                ptr = self._call(*self._ptrs, self._err_ref, stamps)  # the one crossing
                t1 = time.monotonic_ns()
                STAMPS.append((t0, stamps[0], stamps[1], t1))
                if not ptr:
                    reason = objc.objc_object(c_void_p=err.value) if err.value else None
                    raise RuntimeError(f"Core ML predict failed: {reason}")
                if any(m != "backed" for m in self.modes.values()):
                    out = objc.objc_object(c_void_p=ptr)
                    for name, mode in self.modes.items():
                        if mode != "backed":
                            dtype, shape, strides = self._out_meta[name]
                            arr = out.featureValueForName_(name).multiArrayValue()
                            v = _cached_view(arr, dtype, shape, strides)
                            result[name] = v.astype(np.float32) if v.dtype == np.float16 else np.array(v)
            for name, mode in self.modes.items():
                if mode == "backed":
                    v = self._out_views[name]
                    result[name] = v.astype(np.float32) if v.dtype == np.float16 else np.array(v)
            return result
        finally:
            self._busy.release()


def install(backend) -> None:
    """Swap every loaded bucket of one ANEBackend instance to the prebound path."""
    from laya_apple.artifacts import COMPILED, artifact_dir

    backend.models = {b: PrebindModel(artifact_dir(backend.spec, b) / COMPILED) for b in backend.models}
