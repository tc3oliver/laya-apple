"""Core ML predict through PyObjC, with the GIL released for the Core ML call.

coremltools' `CompiledMLModel.predict` keeps the calling thread's GIL for the whole call
(research/gpu-ane-interference/, "GIL probe"). On a thread-placed ANE that blocks every other
thread of the caller's interpreter, including the GPU worker's dispatcher, whose reply then
waits in `take_gil` until `predict` returns (research/coreml-gil-completion-path/).

`NoGilModel` duck-types `CompiledMLModel.predict(feats) -> dict[str, np.ndarray]` and changes
only the binding. The binding was validated in research/coreml-gil-completion-path/
(scripts/nogil_predict.py) and research/coreml-nogil-product-mix/:
- **The same model.** It loads the same verified `model.mlmodelc` with
  `MLModel.modelWithContentsOfURL:configuration:error:` and the same compute units, which is
  what coremltools' `CompiledMLModel` does, with no other configuration set.
- **The GIL is released.** `predictionFromFeatures:error:` is sent through PyObjC, which
  releases the GIL around every Objective-C message it sends.
- **Bit-identical inputs.** Each input is copied into an `MLMultiArray` of the dtype and
  shape the model declares. The copy must be lossless: an input that would have to be
  rounded, or that has another shape, is refused rather than converted.
- **Bit-identical outputs.** Each output is copied back with coremltools' names, shapes and
  dtypes, honouring the array's strides. coremltools widens FP16 outputs to float32, and so
  does this; the widening is exact.
- **An autorelease pool per call.** Core ML returns its outputs (IOSurface-backed
  MLMultiArrays) autoreleased, and a Python thread has no autorelease pool of its own.
  Without a pool every call leaks them, until IOSurface allocation fails and the ANE slows
  down (#46's first run). Loading also runs inside a pool.

Which implementation runs is a predict-binding choice, never a device choice:
`select_predict` decides it once per backend, and the backend records it
(`ANEBackend.predict_impl` / `predict_reason`). Either implementation runs the same artifact
on the same compute units, and every load-time check, including the runtime placement probe,
runs through the implementation that will serve requests.

Needs pyobjc-framework-CoreML (part of the `[ane]` extra).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..errors import BackendUnavailableError, ComputeUnitMismatchError

ENV = "LAYA_APPLE_ANE_PREDICT"
NOGIL = "nogil"
COREMLTOOLS = "coremltools"
REQUESTS = ("auto", COREMLTOOLS)  # what a caller asks ANEBackend for
ENV_VALUES = ("", "auto", NOGIL, COREMLTOOLS)

# predict_reason values (plus the two below, which carry a detail after ": ")
DEFAULT = "default"  # nogil: thread placement and PyObjC importable
ENV_NOGIL = "env_nogil"  # nogil: LAYA_APPLE_ANE_PREDICT=nogil
ENV_COREMLTOOLS = "env_coremltools"  # coremltools: LAYA_APPLE_ANE_PREDICT=coremltools
NOT_THREAD_PLACED = "not_thread_placed"  # coremltools: inline or process placement
PYOBJC_UNAVAILABLE = "pyobjc_unavailable"  # coremltools: PyObjC Core ML cannot be imported
NOGIL_LOAD_FAILED = "nogil_load_failed"  # coremltools: the binding could not load the model

_UNITS = ("CPU_AND_NE", "CPU_ONLY")  # ANE_COMPUTE_UNITS, and the placement probe's reference


class NoGilBindingError(RuntimeError):
    """The PyObjC binding cannot load or run a model that Core ML itself may accept."""


def _modules():
    """(CoreML, Foundation, objc). One place to import them, so tests can substitute them."""
    import CoreML
    import Foundation
    import objc

    return CoreML, Foundation, objc


def pyobjc_import_error() -> str | None:
    """Why the binding cannot be used in this interpreter, or None if it can."""
    try:
        _, _, objc = _modules()
    except Exception as e:  # ImportError, or a broken install raising something else
        return f"{type(e).__name__}: {e}"
    if not hasattr(objc, "autorelease_pool"):
        return "objc.autorelease_pool is missing (PyObjC too old)"
    return None


def select_predict(requested: str) -> tuple[str, str]:
    """(implementation, reason) for an ANE backend.

    requested="coremltools": inline and process-placed backends; always coremltools.
    requested="auto": the thread-placed backend; nogil when PyObjC is importable, unless
    LAYA_APPLE_ANE_PREDICT=coremltools. LAYA_APPLE_ANE_PREDICT=nogil makes an unusable binding
    an error instead of a recorded fallback to coremltools.
    """
    if requested not in REQUESTS:
        raise ValueError(f"ANE predict request must be one of {REQUESTS}, got {requested!r}")
    env = os.environ.get(ENV, "").strip()
    if env not in ENV_VALUES:
        raise ValueError(f"{ENV} must be one of {[v for v in ENV_VALUES if v]}, got {env!r}")
    if requested == COREMLTOOLS:
        return COREMLTOOLS, NOT_THREAD_PLACED
    if env == COREMLTOOLS:
        return COREMLTOOLS, ENV_COREMLTOOLS
    why = pyobjc_import_error()
    if why is None:
        return NOGIL, ENV_NOGIL if env == NOGIL else DEFAULT
    if env == NOGIL:
        raise BackendUnavailableError(
            f"{ENV}=nogil, but PyObjC Core ML cannot be imported ({why}): install the [ane] extra"
        )
    return COREMLTOOLS, f"{PYOBJC_UNAVAILABLE}: {why}"


def forced() -> bool:
    """Whether LAYA_APPLE_ANE_PREDICT=nogil forbids falling back to coremltools."""
    return os.environ.get(ENV, "").strip() == NOGIL


def open_model(compiled_path: Path, compute_units: str) -> NoGilModel:
    """artifacts.load_verified's model opener for the nogil implementation."""
    return NoGilModel(compiled_path, compute_units)


def _dtypes(CoreML) -> dict:
    """MLMultiArrayDataType -> NumPy dtype, for the types coremltools converts."""
    return {
        CoreML.MLMultiArrayDataTypeFloat16: np.dtype(np.float16),
        CoreML.MLMultiArrayDataTypeFloat32: np.dtype(np.float32),
        CoreML.MLMultiArrayDataTypeDouble: np.dtype(np.float64),
        CoreML.MLMultiArrayDataTypeInt32: np.dtype(np.int32),
    }


class NoGilModel:
    """`predict(feats) -> dict[str, np.ndarray]`, as coremltools' CompiledMLModel, GIL released."""

    def __init__(self, compiled_path, compute_units: str = "CPU_AND_NE"):
        if compute_units not in _UNITS:
            raise ComputeUnitMismatchError(f"the nogil Core ML binding runs {_UNITS} only; {compute_units} requested")
        self.compiled_path, self.compute_units = Path(compiled_path), compute_units
        CoreML, Foundation, objc = _modules()
        self._CoreML, self._objc = CoreML, objc
        self._np = _dtypes(CoreML)
        units = {
            "CPU_AND_NE": CoreML.MLComputeUnitsCPUAndNeuralEngine,
            "CPU_ONLY": CoreML.MLComputeUnitsCPUOnly,
        }[compute_units]
        with objc.autorelease_pool():
            try:
                config = CoreML.MLModelConfiguration.alloc().init()
                config.setComputeUnits_(units)
                url = Foundation.NSURL.fileURLWithPath_(str(compiled_path))
                model, err = CoreML.MLModel.modelWithContentsOfURL_configuration_error_(url, config, None)
            except Exception as e:  # objc.error and friends: a binding failure, not an artifact one
                raise NoGilBindingError(f"PyObjC could not load {compiled_path}: {type(e).__name__}: {e}") from e
            if model is None:
                raise NoGilBindingError(f"Core ML could not load {compiled_path} through PyObjC: {err}")
            self._model = model
            self._inputs: dict[str, tuple] = {}  # name -> (MLMultiArrayDataType, NumPy dtype, shape)
            for name, desc in model.modelDescription().inputDescriptionsByName().items():
                c = desc.multiArrayConstraint()
                if c is None:
                    raise NoGilBindingError(f"input {name!r} of {compiled_path} is not a multi-array")
                code = c.dataType()
                if code not in self._np:
                    raise NoGilBindingError(f"input {name!r} of {compiled_path} has unsupported data type {code}")
                self._inputs[str(name)] = (code, self._np[code], tuple(int(s) for s in c.shape()))

    def predict(self, feats: dict) -> dict:
        # Everything Core ML hands back autoreleased is freed when this pool drains. Outputs
        # are copied out before that; nothing returned references Core ML's buffers.
        with self._objc.autorelease_pool():
            return self._predict(feats)

    def _predict(self, feats: dict) -> dict:
        CoreML = self._CoreML
        if set(feats) != set(self._inputs):
            raise ValueError(f"inputs {sorted(feats)} do not match the model's {sorted(self._inputs)}")
        values = {k: CoreML.MLFeatureValue.featureValueWithMultiArray_(self._array(k, v)) for k, v in feats.items()}
        provider, err = CoreML.MLDictionaryFeatureProvider.alloc().initWithDictionary_error_(values, None)
        if provider is None:
            raise NoGilBindingError(f"Core ML feature provider: {err}")
        out, err = self._model.predictionFromFeatures_error_(provider, None)  # PyObjC releases the GIL here
        if out is None:
            raise NoGilBindingError(f"Core ML predict failed: {err}")
        result = {}
        for name in out.featureNames():
            arr = out.featureValueForName_(name).multiArrayValue()
            if arr is None:
                raise NoGilBindingError(f"output {name!r} is not a multi-array")
            result[str(name)] = to_numpy(arr, self._np)
        return result

    def _array(self, name: str, value):
        code, dtype, shape = self._inputs[name]
        arr, err = self._CoreML.MLMultiArray.alloc().initWithShape_dataType_error_(list(shape), code, None)
        if arr is None:
            raise NoGilBindingError(f"MLMultiArray for {name!r}: {err}")
        fill(arr, dtype, shape, value, name)
        return arr


def fill(arr, dtype: np.dtype, shape: tuple, value, name: str = "input") -> None:
    """Copy `value` into an MLMultiArray of `dtype` / `shape`, losslessly or not at all.

    coremltools passes a NumPy input to Core ML in its own dtype, and Core ML converts it to
    the declared type. A lossless cast gives Core ML the same values either way. A lossy one
    (float32 into an FP16 input) could round differently from Core ML, so it is refused, as is
    any shape other than the declared one.
    """
    value = np.asarray(value)
    if value.shape != tuple(shape):
        raise ValueError(f"{name}: shape {value.shape} does not match the model's {tuple(shape)}")
    if not np.can_cast(value.dtype, dtype, "safe"):
        raise TypeError(f"{name}: {value.dtype} cannot be copied into a {dtype} input without rounding")
    view(arr, dtype)[...] = value


def to_numpy(arr, dtypes: dict) -> np.ndarray:
    """A copy of an MLMultiArray as coremltools returns it: its shape, FP16 widened to float32."""
    code = arr.dataType()
    if code not in dtypes:
        raise NoGilBindingError(f"output has unsupported data type {code}")
    v = view(arr, dtypes[code])
    return v.astype(np.float32) if v.dtype == np.float16 else np.array(v)


def view(arr, dtype) -> np.ndarray:
    """A NumPy view of an MLMultiArray's buffer, honouring its (possibly padded) strides.

    Valid only while `arr` is alive and inside the autorelease pool it came from."""
    dtype = np.dtype(dtype)
    shape = [int(s) for s in arr.shape()]
    strides = [int(s) for s in arr.strides()]
    if any(n == 0 for n in shape):
        return np.empty(shape, dtype)
    span = 1 + sum((n - 1) * s for n, s in zip(shape, strides))
    buf = arr.dataPointer().as_buffer(span * dtype.itemsize)
    flat = np.frombuffer(buf, dtype=dtype, count=span)
    return np.lib.stride_tricks.as_strided(flat, shape=shape, strides=[s * dtype.itemsize for s in strides])
