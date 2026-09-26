"""Prebound asynchronous Core ML predict for one compiled ANE bucket (PB-ASYNC).

Adaptive execution (laya_apple/handoff.py) runs a forward on this path once a hetero
episode passed its sync guard, until its slow-state breaker opens. The research that measured it is
research/coreml-async-predict/ (#92, #96) and research/coreml-staged-handoff/.

At load, once per bucket:
  - load the verified model.mlmodelc with CPU_AND_NE through Core ML (PyObjC);
  - allocate an MLMultiArray per input with its declared shape and data type, take a NumPy
    view of each, and build one MLDictionaryFeatureProvider over them;
  - allocate an MLMultiArray per output, set them as MLPredictionOptions.outputBackings, and
    take a NumPy view of each;
  - check that everything Core ML reads during predict is a Foundation object (a
    Python-backed proxy would call back into Python, and take the GIL, inside the predict);
  - verify the output backings with one sentinel prediction (verify_backings).

Per predict:
  1. write the features into the input views (NumPy only);
  2. push an autorelease pool;
  3. send predictionFromFeatures:options:completionHandler: as a plain PyObjC send (PyObjC
     releases the GIL around it; it returns once Core ML has queued the work);
  4. wait on the bucket's event (GIL released), with a timeout;
  5. copy the outputs out with coremltools' names, dtypes and shapes (FP16 widened to
     float32, as coremltools does): from the backing when the array the handler returned is
     that backing, otherwise from the returned array itself (its own layout), so a predict
     never reads a backing it did not write; then pop the pool.

The completion handler is one Python callable per bucket, created at load and kept
referenced. Exactly one callback per submit is enforced: a callback with no submit pending
(a duplicate, or one after a timeout) is counted as an anomaly and changes nothing. A
bucket whose predict timed out refuses every later submit: Core ML may still write its
buffers. A bucket's buffers are reused, so a second predict on a bucket already in flight
raises instead of overwriting them.

Needs pyobjc-framework-CoreML (the `ane` extra) and the asynchronous prediction API
(macOS 14 or later); `unavailable_reason()` says why it cannot run here.
"""

from __future__ import annotations

import threading

import numpy as np

from ..errors import BackendUnavailableError, LayaAppleError

TIMEOUT_S = 5.0
SENTINEL = 0xFF  # every byte of a backing before verify_backings: an FP16/FP32 NaN pattern
ASYNC_SELECTOR = b"predictionFromFeatures:options:completionHandler:"

_NP = None  # MLMultiArrayDataType -> NumPy dtype, filled on first use


def unavailable_reason() -> str | None:
    """Why the asynchronous Core ML path cannot run in this interpreter, or None."""
    try:
        import CoreML
        import Foundation  # noqa: F401
        import objc  # noqa: F401
    except ImportError as e:
        return f"pyobjc-framework-CoreML is not importable ({e}); install the [ane] extra (uv sync --extra ane)"
    if not CoreML.MLModel.instancesRespondToSelector_(ASYNC_SELECTOR):
        return "Core ML has no asynchronous prediction API here (it needs macOS 14 or later)"
    return None


def _types():
    global _NP
    if _NP is None:
        import CoreML

        _NP = {
            CoreML.MLMultiArrayDataTypeFloat16: np.float16,
            CoreML.MLMultiArrayDataTypeFloat32: np.float32,
            CoreML.MLMultiArrayDataTypeDouble: np.float64,
            CoreML.MLMultiArrayDataTypeInt32: np.int32,
        }
    return _NP


def _view(arr, np_dtype, shape=None, strides=None) -> np.ndarray:
    """A NumPy view of an MLMultiArray's buffer, honouring its (possibly padded) strides."""
    shape = [int(s) for s in arr.shape()] if shape is None else shape
    strides = [int(s) for s in arr.strides()] if strides is None else strides
    item = np.dtype(np_dtype).itemsize
    span = 1 + sum((n - 1) * s for n, s in zip(shape, strides))
    flat = np.frombuffer(arr.dataPointer().as_buffer(span * item), dtype=np_dtype, count=span)
    return np.lib.stride_tricks.as_strided(flat, shape=shape, strides=[s * item for s in strides])


def _address(view: np.ndarray) -> int:
    return int(view.__array_interface__["data"][0])


def _ns_shape(shape):
    """An immutable NSArray of NSNumbers (a Python list would reach Core ML as a proxy)."""
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
    """Class names of Python-backed Objective-C objects in a Foundation container tree."""
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


class Completion:
    """One bucket's completion: the handler Core ML calls, and the event its submitter waits on.

    States: idle -> pending (arm) -> idle (handler); pending -> timed_out (wait timed out, final).
    Pure Python, so it is unit-tested with a fake send."""

    def __init__(self):
        self.event = threading.Event()
        self._lock = threading.Lock()
        self.state = "idle"
        self.submits = 0
        self.callbacks = 0
        self.anomalies = 0  # callbacks with no submit pending: duplicates, or after a timeout
        self.keep_output = False  # the load-time backing check, or a "read" output
        self.output = None
        self.error = None
        self.handler = self._handler  # one bound method, created once, kept referenced

    def arm(self) -> None:
        with self._lock:
            if self.state == "timed_out":
                raise LayaAppleError(
                    "Core ML async predict timed out earlier on this bucket; it takes no further submits"
                )
            if self.state != "idle":
                raise LayaAppleError("Core ML async predict submitted before the previous one completed")
            self.state = "pending"
            self.submits += 1
            self.output = self.error = None
            self.event.clear()

    def _handler(self, output, error) -> None:
        with self._lock:
            self.callbacks += 1
            if self.state != "pending":
                self.anomalies += 1
                return
            self.state = "idle"
            self.error = error
            if self.keep_output:
                self.output = output
            # Under the lock: a callback can set the event only while its own submit is pending,
            # so a late one (after a timeout) can never wake a later submit.
            self.event.set()

    def wait(self, timeout: float) -> None:
        """Block (GIL released) until the handler ran, or raise after `timeout` seconds."""
        if self.event.wait(timeout):
            return
        with self._lock:
            if self.state == "pending":
                self.state = "timed_out"
                raise LayaAppleError(f"Core ML async predict did not complete within {timeout} s")

    def submit(self, send, timeout: float = TIMEOUT_S) -> None:
        """arm, send(handler), wait. Raises on a Core ML error or a timeout."""
        self.arm()
        try:
            send(self.handler)
        except BaseException:
            with self._lock:  # nothing was queued: no callback will come
                if self.state == "pending":
                    self.state = "idle"
            raise
        self.wait(timeout)
        if self.error is not None:
            raise LayaAppleError(f"Core ML async predict failed: {self.error}")


class AsyncPrebindModel:
    """Duck-types the `predict(feats) -> dict[str, np.ndarray]` of a coremltools model."""

    def __init__(self, compiled_path, *, timeout_s: float = TIMEOUT_S):
        reason = unavailable_reason()
        if reason is not None:
            raise BackendUnavailableError(reason)
        import CoreML
        import Foundation

        self.timeout_s = timeout_s
        config = CoreML.MLModelConfiguration.alloc().init()
        config.setComputeUnits_(CoreML.MLComputeUnitsCPUAndNeuralEngine)
        url = Foundation.NSURL.fileURLWithPath_(str(compiled_path))
        model, err = CoreML.MLModel.modelWithContentsOfURL_configuration_error_(url, config, None)
        if model is None:
            raise BackendUnavailableError(f"Core ML could not load {compiled_path}: {err}")
        self._model = model
        types = _types()
        self._in_arrays, self._in_views, self._shapes = {}, {}, []
        for name, desc in model.modelDescription().inputDescriptionsByName().items():
            c = desc.multiArrayConstraint()
            arr = self._alloc(CoreML, f"input {name}", [int(s) for s in c.shape()], c.dataType())
            self._in_arrays[str(name)] = arr
            self._in_views[str(name)] = _view(arr, types[c.dataType()])
        self._values = _ns_dict(
            {k: CoreML.MLFeatureValue.featureValueWithMultiArray_(a) for k, a in self._in_arrays.items()}
        )
        provider, err = CoreML.MLDictionaryFeatureProvider.alloc().initWithDictionary_error_(self._values, None)
        if provider is None:
            raise BackendUnavailableError(f"Core ML feature provider: {err}")
        self._provider = provider
        self._out_arrays, self._out_views, self._out_meta = {}, {}, {}
        for name, desc in model.modelDescription().outputDescriptionsByName().items():
            c = desc.multiArrayConstraint()
            shape = [int(s) for s in c.shape()]
            arr = self._alloc(CoreML, f"output backing {name}", shape, c.dataType())
            self._out_arrays[str(name)] = arr
            self._out_views[str(name)] = _view(arr, types[c.dataType()])
            self._out_meta[str(name)] = (types[c.dataType()], shape, [int(s) for s in arr.strides()])
        self._backings = _ns_dict(self._out_arrays)
        options = CoreML.MLPredictionOptions.alloc().init()
        options.setOutputBackings_(self._backings)
        self._options = options
        backed = python_backed(provider.dictionary()) + python_backed(options.outputBackings())
        backed += [n for a in (*self._in_arrays.values(), *self._out_arrays.values()) for n in python_backed(a.shape())]
        if backed:
            raise BackendUnavailableError(f"Python-backed objects would reach Core ML during predict: {backed}")
        self._busy = threading.Lock()
        self._completion = Completion()
        self.modes = {n: "unverified" for n in self._out_arrays}
        self.backing_misses = 0  # "backed" outputs a predict returned elsewhere (read from the returned array)
        self.verify_backings()

    def _alloc(self, CoreML, what, shape, code):
        ns_shape = _ns_shape(shape)
        self._shapes.append(ns_shape)  # kept referenced for the model's lifetime
        arr, err = CoreML.MLMultiArray.alloc().initWithShape_dataType_error_(ns_shape, code, None)
        if arr is None:
            raise BackendUnavailableError(f"MLMultiArray for {what}: {err}")
        return arr

    def _send(self, handler) -> None:
        self._model.predictionFromFeatures_options_completionHandler_(self._provider, self._options, handler)

    @property
    def usable(self) -> bool:
        """False once a predict timed out: the bucket refuses every later submit."""
        return self._completion.state != "timed_out"

    def verify_backings(self) -> dict:
        """Decide each output's mode from one prediction through the async call: 'backed' if the
        backing was filled with exactly the output the handler received, else 'read'. Runs at
        load. Returns the evidence per output."""
        import objc

        c = self._completion
        for view in self._in_views.values():
            view[...] = 0
        for view in self._out_views.values():  # every element set to the all-SENTINEL bit pattern
            view[...] = np.frombuffer(bytes([SENTINEL]) * view.dtype.itemsize, view.dtype)[0]
        evidence = {}
        c.keep_output = True
        try:
            with objc.autorelease_pool():
                c.submit(self._send, self.timeout_s)
                out = c.output
                for name, backing in self._out_views.items():
                    arr = out.featureValueForName_(name).multiArrayValue()
                    returned = _view(arr, _types()[arr.dataType()])
                    written = not np.all(np.ascontiguousarray(backing).view(np.uint8) == SENTINEL)
                    equal = returned.dtype == backing.dtype and np.array_equal(
                        np.ascontiguousarray(returned).view(np.uint8), np.ascontiguousarray(backing).view(np.uint8)
                    )
                    evidence[name] = {
                        "same_address": _address(returned) == _address(backing),
                        "backing_written": bool(written),
                        "equal_to_returned": bool(equal),
                    }
                    self.modes[name] = "backed" if written and equal else "read"
                c.output = None
        finally:
            c.keep_output = True  # predict checks every output against what the handler returned
        self.backing_evidence = evidence
        self._backing_address = {n: _address(v) for n, v in self._out_views.items()}
        return evidence

    def predict(self, feats: dict) -> dict:
        import objc

        if not self._busy.acquire(blocking=False):
            raise LayaAppleError("prebound Core ML bucket already in flight: its buffers would be overwritten")
        try:
            for name, view in self._in_views.items():
                view[...] = np.asarray(feats[name], dtype=view.dtype).reshape(view.shape)
            c = self._completion
            result = {}
            with objc.autorelease_pool():
                c.submit(self._send, self.timeout_s)
                for name, mode in self.modes.items():
                    arr = c.output.featureValueForName_(name).multiArrayValue()
                    returned = _view(arr, _types()[arr.dataType()])  # the returned array's own layout
                    if mode == "backed" and _address(returned) == self._backing_address[name]:
                        v = self._out_views[name]  # Core ML wrote this predict's output into the backing
                    else:
                        if mode == "backed":
                            self.backing_misses += 1  # never trust a backing this predict did not return
                        dtype, shape, _ = self._out_meta[name]
                        if returned.dtype != dtype or list(returned.shape) != shape:
                            raise LayaAppleError(
                                f"Core ML returned {name} as {returned.dtype} {list(returned.shape)}, "
                                f"expected {np.dtype(dtype)} {shape}"
                            )
                        v = returned
                    result[name] = v.astype(np.float32) if v.dtype == np.float16 else np.array(v)
                c.output = None
            return result
        finally:
            self._busy.release()
