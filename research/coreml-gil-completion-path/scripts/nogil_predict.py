"""Core ML predict through PyObjC: the same compiled model, compute units and inputs as the
product's coremltools path, but the call releases the GIL while Core ML runs.

coremltools' `CompiledMLModel.predict` keeps the calling thread's GIL for the whole call
(research/gpu-ane-interference/README.md, "GIL probe"). PyObjC releases the GIL around every
Objective-C method call, so `predictionFromFeatures:error:` here runs without it. Everything
else stays the same: the `model.mlmodelc` the runtime verified and loaded, `CPU_AND_NE`, and
FP16 inputs copied into MLMultiArrays. The output is copied back to NumPy with the same names,
dtypes (FP16 outputs widened to float32) and shapes coremltools returns, so `ANEBackend.forward` runs unchanged on top of it.

Research only: requires pyobjc-framework-CoreML, which laya-apple does not depend on
(`uv run --with pyobjc-framework-CoreML ...`).
"""

from __future__ import annotations

import numpy as np

_NP = None  # MLMultiArrayDataType -> NumPy dtype, filled on first use


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


class NoGilModel:
    """Duck-types the `predict(feats) -> dict[str, np.ndarray]` of a coremltools model."""

    def __init__(self, compiled_path, compute_units: str = "CPU_AND_NE"):
        import CoreML
        import Foundation

        units = {
            "CPU_AND_NE": CoreML.MLComputeUnitsCPUAndNeuralEngine,
            "CPU_ONLY": CoreML.MLComputeUnitsCPUOnly,
            "ALL": CoreML.MLComputeUnitsAll,
        }[compute_units]
        config = CoreML.MLModelConfiguration.alloc().init()
        config.setComputeUnits_(units)
        url = Foundation.NSURL.fileURLWithPath_(str(compiled_path))
        model, err = CoreML.MLModel.modelWithContentsOfURL_configuration_error_(url, config, None)
        if model is None:
            raise RuntimeError(f"Core ML could not load {compiled_path}: {err}")
        self._model = model
        self._inputs = {}
        for name, desc in model.modelDescription().inputDescriptionsByName().items():
            c = desc.multiArrayConstraint()
            self._inputs[str(name)] = (c.dataType(), [int(s) for s in c.shape()])

    def _array(self, name, value):
        import CoreML

        dtype_code, shape = self._inputs[name]
        np_dtype = _types()[dtype_code]
        arr, err = CoreML.MLMultiArray.alloc().initWithShape_dataType_error_(shape, dtype_code, None)
        if arr is None:
            raise RuntimeError(f"MLMultiArray for {name}: {err}")
        dst = _view(arr, np_dtype)
        dst[...] = np.asarray(value, dtype=np_dtype).reshape(shape)
        return arr

    def predict(self, feats: dict) -> dict:
        import objc

        # Core ML's outputs (IOSurface-backed MLMultiArrays) are autoreleased, and a Python
        # thread has no autorelease pool of its own: without this pool every call leaks them,
        # until IOSurface allocation fails and the ANE slows down.
        with objc.autorelease_pool():
            return self._predict(feats)

    def _predict(self, feats: dict) -> dict:
        import CoreML

        values = {k: CoreML.MLFeatureValue.featureValueWithMultiArray_(self._array(k, v)) for k, v in feats.items()}
        provider, err = CoreML.MLDictionaryFeatureProvider.alloc().initWithDictionary_error_(values, None)
        if provider is None:
            raise RuntimeError(f"feature provider: {err}")
        out, err = self._model.predictionFromFeatures_error_(provider, None)  # GIL released here
        if out is None:
            raise RuntimeError(f"Core ML predict failed: {err}")
        result = {}
        for name in out.featureNames():
            arr = out.featureValueForName_(name).multiArrayValue()
            # a copy out of Core ML's buffer; coremltools hands FP16 outputs back as float32
            v = _view(arr, _types()[arr.dataType()])
            result[str(name)] = v.astype(np.float32) if v.dtype == np.float16 else np.array(v)
        return result


def _view(arr, np_dtype) -> np.ndarray:
    """A NumPy view of an MLMultiArray's buffer, honouring its (possibly padded) strides."""
    shape = [int(s) for s in arr.shape()]
    strides = [int(s) for s in arr.strides()]
    item = np.dtype(np_dtype).itemsize
    span = 1 + sum((n - 1) * s for n, s in zip(shape, strides))
    buf = arr.dataPointer().as_buffer(span * item)
    flat = np.frombuffer(buf, dtype=np_dtype, count=span)
    return np.lib.stride_tricks.as_strided(flat, shape=shape, strides=[s * item for s in strides])


def install(backend) -> None:
    """Swap every loaded bucket of one ANEBackend instance to the GIL-releasing path."""
    from laya_apple.artifacts import COMPILED, artifact_dir

    backend.models = {b: NoGilModel(artifact_dir(backend.spec, b) / COMPILED) for b in backend.models}
