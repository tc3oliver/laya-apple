"""The GIL-releasing Core ML predict binding (laya_apple/backends/coreml_nogil.py), mocked.

No model, no Core ML, no PyObjC: `coreml_nogil._modules` is replaced by fakes that behave like
the PyObjC objects the binding touches, and `coremltools` by a module that fails any use, so a
test also proves the coremltools path is never reached when the binding is selected. The real
binding is checked against coremltools on real artifacts by tests/parity/test_ane_nogil_parity.py,
tests/integration/test_ane_nogil.py and tests/stress/test_ane_nogil_soak.py.
"""

from __future__ import annotations

import contextlib
import sys
import time
import types
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from laya_apple import executor, model, routing, scheduling
from laya_apple.backends import coreml_ane, coreml_nogil
from laya_apple.backends.coreml_nogil import COREMLTOOLS, ENV, NOGIL, NoGilBindingError, NoGilModel, select_predict
from laya_apple.errors import ArtifactMissingError, BackendUnavailableError, ComputeUnitMismatchError
from laya_apple.registry import ANE_MAX_OPTIONS, resolve, routing_table
from laya_apple.trace import RequestTrace

MODEL = "laya-typed-decisions"

# MLMultiArrayDataType and MLComputeUnits values, as Core ML defines them.
FLOAT16, FLOAT32, DOUBLE, INT32 = 0x10010, 0x10020, 0x10040, 0x20020
CPU_ONLY, CPU_AND_NE = 0, 3
UNSPECIFIED, ENUMERATED, RANGE = 1, 2, 3  # MLMultiArrayShapeConstraintType
NP = {FLOAT16: np.float16, FLOAT32: np.float32, DOUBLE: np.float64, INT32: np.int32}


# ----------------------------------------------------------------------------- fake PyObjC


class FakeArray:
    """An MLMultiArray over a flat NumPy buffer, with element strides (possibly padded)."""

    def __init__(self, shape, code, strides=None):
        self._shape, self._code = [int(n) for n in shape], code
        if strides is None:
            strides, acc = [], 1
            for n in reversed(self._shape):
                strides.insert(0, acc)
                acc *= n
        self._strides = list(strides)
        span = 1 + sum((n - 1) * s for n, s in zip(self._shape, self._strides))
        self.buf = np.zeros(span, NP[code])
        self.short_by = 0  # bytes getBytesWithHandler: hands over fewer than the strides need

    def shape(self):
        return self._shape

    def strides(self):
        return self._strides

    def dataType(self):
        return self._code

    def dataPointer(self):
        buf = self.buf
        return SimpleNamespace(as_buffer=lambda n: memoryview(buf.view(np.uint8))[:n])

    def getBytesWithHandler_(self, handler):  # PyObjC hands the block a bytes copy and its size
        data = self.buf.tobytes()[: self.buf.nbytes - self.short_by]
        handler(data, len(data))

    def logical(self):
        item = self.buf.itemsize
        return np.lib.stride_tricks.as_strided(self.buf, self._shape, [s * item for s in self._strides])


class Pool:
    """objc.autorelease_pool(): counts pools and knows whether one is open."""

    def __init__(self):
        self.depth = self.opened = self.closed = 0

    @contextlib.contextmanager
    def __call__(self):
        self.depth += 1
        self.opened += 1
        try:
            yield
        finally:
            self.depth -= 1
            self.closed += 1


class FakeMLModel:
    """Declares inputs; its prediction is a pure function of them, with padded FP16 outputs."""

    def __init__(self, env, inputs, units):
        self.env, self.inputs, self.units = env, inputs, units
        self.seen: list[dict] = []  # the inputs as the model received them (their bytes)

    def modelDescription(self):
        descs = {
            name: SimpleNamespace(
                multiArrayConstraint=(
                    lambda c=code, s=shape: SimpleNamespace(
                        dataType=lambda: c,
                        shape=lambda: s,
                        shapeConstraint=lambda: SimpleNamespace(type=lambda: self.env.shape_constraint),
                    )
                )
            )
            for name, (code, shape) in self.inputs.items()
        }
        return SimpleNamespace(inputDescriptionsByName=lambda: descs)

    def predictionFromFeatures_error_(self, provider, error):
        self.env.calls.append(self.env.pool.depth)  # the pool depth while Core ML runs
        if self.env.fail_predict:
            return None, "predict failed (test)"
        x = {k: v.multiArrayValue().logical().copy() for k, v in provider.values.items()}
        self.seen.append(x)
        total = sum(float(v.astype(np.float64).sum()) for v in x.values())
        logits = FakeArray((1, 1, 1, 4), FLOAT16, strides=(64, 64, 16, 1))  # padded, as ANE outputs are
        logits.logical()[...] = np.array([total, -1.5, 65504, 2**-24], np.float16)
        pooled = FakeArray((1, 3), FLOAT32)
        pooled.logical()[...] = np.array([[1.25, total, -0.0]], np.float32)
        outs = {"logits": logits, "pooled": pooled}
        self.env.last_outputs = outs
        return SimpleNamespace(
            featureNames=lambda: list(outs),
            featureValueForName_=lambda n: SimpleNamespace(multiArrayValue=lambda: outs[n]),
        ), None


class FakePyObjC:
    """(CoreML, Foundation, objc) as the binding uses them."""

    def __init__(self, inputs=None):
        self.inputs = inputs or {"embeddings": (FLOAT16, [1, 4, 1, 8]), "type_vectors": (FLOAT16, [1, 4, 1, 1])}
        self.pool = Pool()
        self.calls: list[int] = []
        self.loaded: list[tuple] = []
        self.models: list[FakeMLModel] = []
        self.fail_load = self.fail_predict = False
        self.alloc_error: BaseException | None = None  # raised by MLMultiArray.alloc()
        self.shape_constraint = UNSPECIFIED
        env = self

        class MLModel:
            @staticmethod
            def modelWithContentsOfURL_configuration_error_(url, config, error):
                env.loaded.append((url, config.units, env.pool.depth))
                if env.fail_load:
                    return None, "cannot load (test)"
                m = FakeMLModel(env, env.inputs, config.units)
                env.models.append(m)
                return m, None

        class Config:
            units = None

            def setComputeUnits_(self, u):
                self.units = u

        self.CoreML = SimpleNamespace(
            MLMultiArrayDataTypeFloat16=FLOAT16,
            MLMultiArrayDataTypeFloat32=FLOAT32,
            MLMultiArrayDataTypeDouble=DOUBLE,
            MLMultiArrayDataTypeInt32=INT32,
            MLComputeUnitsCPUAndNeuralEngine=CPU_AND_NE,
            MLComputeUnitsCPUOnly=CPU_ONLY,
            MLMultiArrayShapeConstraintTypeUnspecified=UNSPECIFIED,
            MLMultiArrayShapeConstraintTypeEnumerated=ENUMERATED,
            MLMultiArrayShapeConstraintTypeRange=RANGE,
            MLModel=MLModel,
            MLModelConfiguration=SimpleNamespace(alloc=lambda: SimpleNamespace(init=Config)),
            MLMultiArray=SimpleNamespace(alloc=self._alloc_array),
            MLFeatureValue=SimpleNamespace(
                featureValueWithMultiArray_=lambda a: SimpleNamespace(multiArrayValue=lambda: a)
            ),
            MLDictionaryFeatureProvider=SimpleNamespace(
                alloc=lambda: SimpleNamespace(
                    initWithDictionary_error_=lambda values, err: (SimpleNamespace(values=values), None)
                )
            ),
        )
        self.Foundation = SimpleNamespace(NSURL=SimpleNamespace(fileURLWithPath_=lambda p: f"file://{p}"))
        self.objc = SimpleNamespace(autorelease_pool=self.pool)

    def _alloc_array(self):
        if self.alloc_error is not None:
            raise self.alloc_error
        return SimpleNamespace(initWithShape_dataType_error_=lambda shape, code, err: (FakeArray(shape, code), None))

    def modules(self):
        return self.CoreML, self.Foundation, self.objc


@pytest.fixture
def pyobjc(monkeypatch):
    fake = FakePyObjC()
    monkeypatch.setattr(coreml_nogil, "_modules", fake.modules)
    monkeypatch.delenv(ENV, raising=False)
    return fake


@pytest.fixture
def no_pyobjc(monkeypatch):
    def missing():
        raise ImportError("No module named 'CoreML'")

    monkeypatch.setattr(coreml_nogil, "_modules", missing)
    monkeypatch.delenv(ENV, raising=False)


@pytest.fixture
def no_coremltools(monkeypatch):
    """Any use of coremltools fails the test: the nogil path must never reach it."""

    class Poison(types.ModuleType):
        def __getattr__(self, name):
            raise AssertionError(f"coremltools.{name} used on the nogil path")

    monkeypatch.setitem(sys.modules, "coremltools", Poison("coremltools"))


def feats(seed=0):
    rng = np.random.default_rng(seed)
    return {
        "embeddings": rng.standard_normal((1, 4, 1, 8)).astype(np.float16),
        "type_vectors": rng.standard_normal((1, 4, 1, 1)).astype(np.float16),
    }


# ----------------------------------------------------------------------------- selection


def test_thread_placement_selects_nogil_when_pyobjc_imports(pyobjc):
    assert coreml_nogil.DEFAULT_AUTO == NOGIL  # until the product-mix experiment decides
    assert select_predict("auto") == (NOGIL, coreml_nogil.DEFAULT, False)


def test_default_auto_is_the_one_switch_to_opt_in(pyobjc, monkeypatch):
    monkeypatch.setattr(coreml_nogil, "DEFAULT_AUTO", COREMLTOOLS)
    assert select_predict("auto") == (COREMLTOOLS, coreml_nogil.DEFAULT, False)
    monkeypatch.setenv(ENV, "auto")
    assert select_predict("auto") == (COREMLTOOLS, coreml_nogil.DEFAULT, False)
    monkeypatch.setenv(ENV, "nogil")  # opting in still works
    assert select_predict("auto") == (NOGIL, coreml_nogil.ENV_NOGIL, True)


def test_default_auto_coremltools_does_not_need_pyobjc(no_pyobjc, monkeypatch):
    monkeypatch.setattr(coreml_nogil, "DEFAULT_AUTO", COREMLTOOLS)
    assert select_predict("auto") == (COREMLTOOLS, coreml_nogil.DEFAULT, False)


def test_env_forces_coremltools_even_when_pyobjc_imports(pyobjc, monkeypatch):
    monkeypatch.setenv(ENV, "coremltools")
    assert select_predict("auto") == (COREMLTOOLS, coreml_nogil.ENV_COREMLTOOLS, False)


def test_env_nogil_is_recorded_as_forced(pyobjc, monkeypatch):
    monkeypatch.setenv(ENV, "nogil")
    assert select_predict("auto") == (NOGIL, coreml_nogil.ENV_NOGIL, True)


def test_missing_pyobjc_falls_back_to_coremltools_with_the_reason(no_pyobjc):
    impl, reason, forced = select_predict("auto")
    assert impl == COREMLTOOLS and not forced
    assert reason.startswith(coreml_nogil.PYOBJC_UNAVAILABLE + ": ImportError") and "CoreML" in reason
    assert coreml_nogil.reason_code(reason) == coreml_nogil.PYOBJC_UNAVAILABLE


def test_missing_pyobjc_with_env_nogil_raises(no_pyobjc, monkeypatch):
    monkeypatch.setenv(ENV, "nogil")
    with pytest.raises(BackendUnavailableError, match="PyObjC Core ML cannot be imported"):
        select_predict("auto")


@pytest.mark.parametrize("env", [None, "nogil", "coremltools", "auto"])
def test_inline_and_process_placement_always_use_coremltools(pyobjc, monkeypatch, env):
    if env:
        monkeypatch.setenv(ENV, env)
    assert select_predict("coremltools") == (COREMLTOOLS, coreml_nogil.NOT_THREAD_PLACED, False)


@pytest.mark.parametrize("requested", ["auto", "coremltools"])
def test_unknown_env_value_is_refused_on_every_path(pyobjc, monkeypatch, requested):
    monkeypatch.setenv(ENV, "pyobjc")
    with pytest.raises(ValueError, match=ENV):
        select_predict(requested)


def test_unknown_request_is_refused(pyobjc):
    with pytest.raises(ValueError, match="predict request"):
        select_predict("nogil")


def test_pyobjc_without_autorelease_pool_is_unusable(pyobjc):
    del pyobjc.objc.autorelease_pool
    impl, reason, _ = select_predict("auto")
    assert impl == COREMLTOOLS and "autorelease_pool" in reason


# ----------------------------------------------------------------------------- NoGilModel


def test_loads_the_compiled_model_with_the_requested_compute_units_inside_a_pool(pyobjc):
    NoGilModel(Path("/a/model.mlmodelc"), "CPU_AND_NE")
    NoGilModel(Path("/a/model.mlmodelc"), "CPU_ONLY")
    assert pyobjc.loaded == [("file:///a/model.mlmodelc", CPU_AND_NE, 1), ("file:///a/model.mlmodelc", CPU_ONLY, 1)]
    assert pyobjc.pool.depth == 0


def test_compute_units_other_than_the_validated_ones_have_no_path(pyobjc):
    with pytest.raises(ComputeUnitMismatchError):
        NoGilModel(Path("/a"), "ALL")
    assert pyobjc.loaded == []


def test_a_load_core_ml_refuses_is_a_binding_error(pyobjc):
    pyobjc.fail_load = True
    with pytest.raises(NoGilBindingError, match="cannot load"):
        NoGilModel(Path("/a"))


def test_every_predict_runs_inside_its_own_autorelease_pool(pyobjc):
    m = NoGilModel(Path("/a"))
    opened = pyobjc.pool.opened
    for i in range(5):
        m.predict(feats(i))
    assert pyobjc.pool.opened - opened == 5 and pyobjc.pool.closed == pyobjc.pool.opened
    assert pyobjc.calls == [1] * 5  # Core ML ran inside exactly one pool each time
    assert pyobjc.pool.depth == 0


def test_the_pool_drains_when_predict_fails(pyobjc):
    m = NoGilModel(Path("/a"))
    pyobjc.fail_predict = True
    with pytest.raises(NoGilBindingError, match="predict failed"):
        m.predict(feats())
    assert pyobjc.pool.depth == 0 and pyobjc.pool.closed == pyobjc.pool.opened


def test_inputs_reach_core_ml_bit_identical(pyobjc):
    m = NoGilModel(Path("/a"))
    f = feats(3)
    f["embeddings"][0, 0, 0, :3] = np.array([np.nan, -0.0, 2**-24], np.float16)  # NaN, signed zero, subnormal
    m.predict(f)
    seen = pyobjc.models[-1].seen[-1]
    for k, v in f.items():
        assert seen[k].dtype == v.dtype and seen[k].shape == v.shape
        assert seen[k].tobytes() == v.tobytes()


def test_outputs_match_coremltools_names_dtypes_shapes_and_values(pyobjc):
    m = NoGilModel(Path("/a"))
    f = feats(4)
    out = m.predict(f)
    total = sum(float(v.astype(np.float64).sum()) for v in f.values())
    assert set(out) == {"logits", "pooled"}
    # coremltools widens FP16 outputs to float32; the widening is exact, strides are honoured
    assert out["logits"].dtype == np.float32 and out["logits"].shape == (1, 1, 1, 4)
    want = np.array([total, -1.5, 65504, 2**-24], np.float16).astype(np.float32).reshape(1, 1, 1, 4)
    assert out["logits"].tobytes() == want.tobytes()
    assert out["pooled"].dtype == np.float32 and out["pooled"].shape == (1, 3)
    assert out["pooled"].tobytes() == np.array([[1.25, total, -0.0]], np.float32).tobytes()


def test_outputs_are_copies_not_views_of_core_ml_buffers(pyobjc):
    m = NoGilModel(Path("/a"))
    out = m.predict(feats())
    before = {k: v.copy() for k, v in out.items()}
    for arr in pyobjc.last_outputs.values():  # Core ML reuses or frees its buffers after the pool drains
        arr.buf[...] = 7
    assert all(v.flags.writeable for v in out.values())
    assert all(before[k].tobytes() == out[k].tobytes() for k in out)


def test_lossy_or_misshaped_inputs_are_refused_not_converted(pyobjc):
    m = NoGilModel(Path("/a"))
    f = feats()
    with pytest.raises(TypeError, match="without rounding"):
        m.predict({**f, "embeddings": f["embeddings"].astype(np.float32)})
    with pytest.raises(ValueError, match="shape"):
        m.predict({**f, "embeddings": f["embeddings"].reshape(1, 4, 8, 1)})
    with pytest.raises(ValueError, match="do not match"):
        m.predict({"embeddings": f["embeddings"]})
    assert pyobjc.calls == [] and pyobjc.pool.depth == 0


def test_a_non_multiarray_input_is_a_binding_error(pyobjc, monkeypatch):
    def image_input(self):  # an input with no multi-array constraint (an image, say)
        return SimpleNamespace(
            inputDescriptionsByName=lambda: {"image": SimpleNamespace(multiArrayConstraint=lambda: None)}
        )

    monkeypatch.setattr(FakeMLModel, "modelDescription", image_input)
    with pytest.raises(NoGilBindingError, match="not a multi-array"):
        NoGilModel(Path("/a"))
    assert pyobjc.pool.depth == 0


@pytest.mark.parametrize("constraint", [ENUMERATED, RANGE])
def test_a_flexible_shape_input_is_a_binding_error(pyobjc, constraint):
    pyobjc.shape_constraint = constraint
    with pytest.raises(NoGilBindingError, match="flexible shape"):
        NoGilModel(Path("/a"))


def test_an_objc_exception_while_loading_is_a_binding_error(pyobjc, monkeypatch):
    def boom(url, config, error):
        raise RuntimeError("objc.error (test)")

    monkeypatch.setattr(pyobjc.CoreML.MLModel, "modelWithContentsOfURL_configuration_error_", boom)
    with pytest.raises(NoGilBindingError, match="objc.error"):
        NoGilModel(Path("/a"))
    assert pyobjc.pool.depth == 0


@pytest.mark.parametrize(
    "where",
    ["alloc", "feature_value", "provider", "predict", "read"],
)
def test_every_objc_call_in_predict_fails_as_a_binding_error(pyobjc, monkeypatch, where):
    m = NoGilModel(Path("/a"))
    err = RuntimeError(f"objc.error in {where} (test)")

    def boom(*a, **k):
        raise err

    if where == "alloc":
        pyobjc.alloc_error = err
    elif where == "feature_value":
        monkeypatch.setattr(pyobjc.CoreML.MLFeatureValue, "featureValueWithMultiArray_", boom)
    elif where == "provider":
        monkeypatch.setattr(pyobjc.CoreML, "MLDictionaryFeatureProvider", SimpleNamespace(alloc=boom))
    elif where == "predict":
        monkeypatch.setattr(m._model, "predictionFromFeatures_error_", boom, raising=False)
    else:
        monkeypatch.setattr(FakeArray, "getBytesWithHandler_", boom)
    with pytest.raises(NoGilBindingError, match=where) as e:
        m.predict(feats())
    assert e.value.__cause__ is err
    assert pyobjc.pool.depth == 0 and pyobjc.pool.closed == pyobjc.pool.opened


def test_caller_errors_stay_value_and_type_errors_before_any_objc_call(pyobjc):
    m = NoGilModel(Path("/a"))
    pyobjc.alloc_error = RuntimeError("must not be reached")
    f = feats()
    with pytest.raises(TypeError):
        m.predict({**f, "embeddings": f["embeddings"].astype(np.float32)})
    with pytest.raises(ValueError):
        m.predict({**f, "type_vectors": f["type_vectors"][..., :0]})
    assert pyobjc.pool.opened == 1  # only the load's pool: no predict pool was opened


# ----------------------------------------------------------------------------- conversion helpers


@pytest.mark.parametrize("code", [FLOAT16, FLOAT32, DOUBLE, INT32])
def test_write_and_to_numpy_honour_padded_strides(code):
    arr = FakeArray((2, 3), code, strides=(8, 2))
    value = np.array([[0, 2, 4], [8, 10, 12]], NP[code])
    coreml_nogil.write(arr, np.dtype(NP[code]), value)
    assert arr.buf[1] == 0 and arr.buf[3] == 0  # padding untouched
    assert arr.logical().tolist() == value.tolist()
    back = coreml_nogil.to_numpy(arr, {code: np.dtype(NP[code])})
    want = value.astype(np.float32) if code == FLOAT16 else value
    assert back.dtype == want.dtype and back.tobytes() == want.tobytes() and back.flags.writeable


def test_fp16_round_trip_is_exact_including_specials():
    arr = FakeArray((2, 3), FLOAT16, strides=(8, 2))
    value = np.array([[1, -2, 3.5], [np.inf, -0.0, 6e-8]], np.float16)
    coreml_nogil.write(arr, np.dtype(np.float16), coreml_nogil.check_input(value, np.dtype(np.float16), (2, 3)))
    back = coreml_nogil.to_numpy(arr, {FLOAT16: np.dtype(np.float16)})
    assert back.dtype == np.float32 and back.tobytes() == value.astype(np.float32).tobytes()


def test_check_input_accepts_lossless_numeric_widening_only():
    f32 = np.dtype(np.float32)
    assert coreml_nogil.check_input(np.array([1, 2, 3], np.float16), f32, (3,)).dtype == np.float16
    with pytest.raises(TypeError, match="without rounding"):
        coreml_nogil.check_input(np.array([1, 2, 3], np.float64), f32, (3,))
    with pytest.raises(TypeError, match="not a numeric"):
        coreml_nogil.check_input(np.array([True, False, True]), f32, (3,))
    with pytest.raises(ValueError, match="shape"):
        coreml_nogil.check_input(np.zeros((1, 3), np.float16), f32, (3,))


def test_to_numpy_refuses_an_unknown_data_type():
    arr = FakeArray((1,), INT32)
    with pytest.raises(NoGilBindingError, match="unsupported data type"):
        coreml_nogil.to_numpy(arr, {FLOAT16: np.dtype(np.float16)})


def test_to_numpy_fails_loudly_on_a_buffer_shorter_than_its_strides():
    arr = FakeArray((1, 1, 1, 4), FLOAT16, strides=(64, 64, 16, 1))
    arr.short_by = 2
    with pytest.raises(NoGilBindingError, match="needs 8"):
        coreml_nogil.to_numpy(arr, {FLOAT16: np.dtype(np.float16)})


# ----------------------------------------------------------------------------- ANEBackend selection


class CoremltoolsModel:
    """What the coremltools opener would return (the tests never import coremltools)."""

    def __init__(self, path, units):
        self.path, self.units, self.calls = path, units, 0

    def predict(self, feats):
        self.calls += 1
        return {}


class FakeHost:
    embedding = type_embedding = None

    def window(self, L):
        return None

    def tail(self, outputs, rows):
        return np.zeros((len(rows), ANE_MAX_OPTIONS), np.float32), np.zeros((len(rows), 3), np.float32)


@pytest.fixture
def backend_env(monkeypatch):
    """ANEBackend with verification, the probe and host weights replaced by recorders."""
    rec = SimpleNamespace(load=[], probe=[], fail_nogil_bucket=None, probe_error=None)
    monkeypatch.setattr(coreml_ane, "HostWeights", lambda checkpoint, local_attention: FakeHost())
    monkeypatch.setattr(coreml_ane, "ane_features", lambda rows, L, *a: {"L": np.array([L], np.float16)})

    def load_verified(spec, b, *, open_model=None):
        rec.load.append((b, open_model))
        path = Path(f"/artifacts/L{b}/model.mlmodelc")
        if open_model is None:
            m = CoremltoolsModel(path, "CPU_AND_NE")
        else:
            if rec.fail_nogil_bucket == b:
                raise NoGilBindingError(f"test: L{b} does not load through PyObjC")
            m = open_model(path, "CPU_AND_NE")
        return m, SimpleNamespace(artifact_sha256=f"{b:064d}")

    def probe(spec, b, m, path, feats, open_model=None):
        rec.probe.append((b, type(m).__name__, open_model))
        if rec.probe_error is not None:
            raise rec.probe_error
        return {"ane_ms": 1.0, "cpu_ms": 3.0, "ratio": 0.33}

    monkeypatch.setattr(coreml_ane, "load_verified", load_verified)
    monkeypatch.setattr(coreml_ane, "probe_placement", probe)
    return rec


def make_backend(predict, buckets=(64, 128), strict=True):
    return coreml_ane.ANEBackend(resolve(MODEL), Path("/ckpt"), 0, 128, buckets, strict=strict, predict=predict)


def test_thread_placed_backend_loads_probes_and_serves_through_nogil(pyobjc, backend_env, no_coremltools):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the default path warns about nothing
        b = make_backend("auto")
    assert (b.predict_impl, b.predict_reason) == (NOGIL, coreml_nogil.DEFAULT)
    assert all(isinstance(m, NoGilModel) for m in b.models.values())
    assert [o for _, o in backend_env.load] == [coreml_nogil.open_model] * 2
    # the placement probe measures the nogil model against a nogil CPU_ONLY instance
    assert backend_env.probe == [
        (64, "NoGilModel", coreml_nogil.open_model),
        (128, "NoGilModel", coreml_nogil.open_model),
    ]
    for m in b.models.values():  # the fake features of backend_env: one FP16 value
        m._inputs = {"L": (FLOAT16, np.dtype(np.float16), (1,))}
    b.forward([{"ids": [0] * 10, "markers": [1, 2], "qtype": 0}])
    assert pyobjc.calls == [1]  # one predict, through PyObjC, inside a pool; coremltools is poisoned
    assert pyobjc.models[0].seen[-1]["L"].tolist() == [64.0]  # the L64 model served the 10-token row


def test_env_override_serves_the_thread_placed_backend_through_coremltools(pyobjc, backend_env, monkeypatch):
    monkeypatch.setenv(ENV, "coremltools")
    b = make_backend("auto")
    assert (b.predict_impl, b.predict_reason) == (COREMLTOOLS, coreml_nogil.ENV_COREMLTOOLS)
    assert all(isinstance(m, CoremltoolsModel) for m in b.models.values())
    assert [o for _, o in backend_env.load] == [None, None]
    assert all(o is None for *_, o in backend_env.probe)
    assert pyobjc.loaded == []


def test_inline_and_process_backends_keep_coremltools(pyobjc, backend_env):
    b = coreml_ane.ANEBackend(resolve(MODEL), Path("/ckpt"), 0, 128, (64,), strict=True)
    assert (b.predict_impl, b.predict_reason) == (COREMLTOOLS, coreml_nogil.NOT_THREAD_PLACED)
    assert pyobjc.loaded == []


def test_missing_pyobjc_loads_through_coremltools_records_and_warns(no_pyobjc, backend_env):
    with pytest.warns(RuntimeWarning, match="pyobjc_unavailable: ImportError"):
        b = make_backend("auto")
    assert b.predict_impl == COREMLTOOLS and b.predict_reason.startswith(coreml_nogil.PYOBJC_UNAVAILABLE)
    assert all(isinstance(m, CoremltoolsModel) for m in b.models.values())


def test_a_binding_failure_reloads_every_bucket_through_coremltools_records_and_warns(pyobjc, backend_env):
    backend_env.fail_nogil_bucket = 128
    with pytest.warns(RuntimeWarning, match="nogil_load_failed: test: L128"):
        b = make_backend("auto")
    assert b.predict_impl == COREMLTOOLS
    assert b.predict_reason.startswith(coreml_nogil.NOGIL_LOAD_FAILED) and "L128" in b.predict_reason
    assert all(isinstance(m, CoremltoolsModel) for m in b.models.values())  # no mix of bindings
    assert sorted(b.models) == [64, 128] and sorted(b.probes) == [64, 128]
    assert [o for _, o in backend_env.load] == [coreml_nogil.open_model] * 2 + [None, None]
    # the same device, artifacts and compute units: only the binding changed
    assert b.device == "ane" and b.compute_units == "CPU_AND_NE"


def test_a_binding_failure_with_env_nogil_raises_instead(pyobjc, backend_env, monkeypatch):
    monkeypatch.setenv(ENV, "nogil")
    backend_env.fail_nogil_bucket = 64
    with pytest.raises(BackendUnavailableError, match="nogil"):
        make_backend("auto")


def test_a_failing_placement_probe_is_never_retried_through_coremltools(pyobjc, backend_env):
    backend_env.probe_error = ComputeUnitMismatchError("test: not running on the Neural Engine")
    with pytest.raises(ComputeUnitMismatchError):
        make_backend("auto", strict=True)
    assert all(o is coreml_nogil.open_model for _, o in backend_env.load)  # no second, coremltools attempt
    backend_env.load.clear()
    b = make_backend("auto", strict=False)  # auto: the bucket is dropped, as with coremltools
    assert b.models == {} and all(isinstance(e, ComputeUnitMismatchError) for e in b.load_errors.values())
    assert b.predict_impl == NOGIL


def test_a_missing_bucket_under_nogil_is_recorded_as_before(pyobjc, backend_env, monkeypatch):
    real = coreml_ane.load_verified

    def missing_128(spec, b, *, open_model=None):
        if b == 128:
            raise ArtifactMissingError("test: L128 missing")
        return real(spec, b, open_model=open_model)

    monkeypatch.setattr(coreml_ane, "load_verified", missing_128)
    b = make_backend("auto")
    assert sorted(b.models) == [64] and isinstance(b.load_errors[128], ArtifactMissingError)
    assert b.predict_impl == NOGIL


def test_a_binding_error_raised_by_the_probe_triggers_the_fallback(pyobjc, backend_env, monkeypatch):
    real = coreml_ane.probe_placement

    def probe(spec, b, m, path, feats, open_model=None):
        if open_model is not None:
            raise NoGilBindingError(f"test: L{b} predict failed through PyObjC")
        return real(spec, b, m, path, feats, open_model)

    monkeypatch.setattr(coreml_ane, "probe_placement", probe)
    with pytest.warns(RuntimeWarning, match="nogil_load_failed"):
        b = make_backend("auto")
    assert b.predict_impl == COREMLTOOLS and "predict failed through PyObjC" in b.predict_reason
    assert all(isinstance(m, CoremltoolsModel) for m in b.models.values()) and sorted(b.probes) == [64, 128]


# ----------------------------------------------------------------------------- placement probe


class SlowModel(CoremltoolsModel):
    def predict(self, feats):
        time.sleep(0.002)
        return super().predict(feats)


def test_probe_opens_its_cpu_reference_through_the_same_binding(no_coremltools):
    opened = []

    def opener(path, units):
        opened.append((path, units))
        return SlowModel(path, units)

    loaded = CoremltoolsModel(Path("/x"), "CPU_AND_NE")
    r = coreml_ane.probe_placement(resolve(MODEL), 64, loaded, Path("/x"), {}, opener)
    assert r["ratio"] < coreml_ane.PROBE_MAX_RATIO
    assert opened == [(Path("/x"), "CPU_ONLY")]
    assert loaded.calls == coreml_ane.PROBE_RUNS + 1  # the probe timed the model that serves requests


def test_an_mlmultiarray_failure_in_the_real_probe_falls_back_to_coremltools(pyobjc, monkeypatch, no_coremltools):
    """The real probe_placement, through the (fake) PyObjC binding: MLMultiArray.alloc raises."""
    monkeypatch.setattr(coreml_ane, "HostWeights", lambda checkpoint, local_attention: FakeHost())
    monkeypatch.setattr(coreml_ane, "ane_features", lambda rows, L, *a: feats())
    monkeypatch.setattr(coreml_ane, "_open_coremltools", lambda path, units: SlowModel(path, units))

    def load_verified(spec, b, *, open_model=None):
        path = Path(f"/artifacts/L{b}/model.mlmodelc")
        m = open_model(path, "CPU_AND_NE") if open_model else CoremltoolsModel(path, "CPU_AND_NE")
        return m, SimpleNamespace(artifact_sha256=f"{b:064d}")

    monkeypatch.setattr(coreml_ane, "load_verified", load_verified)
    pyobjc.alloc_error = RuntimeError("objc.error: MLMultiArray alloc (test)")
    with pytest.warns(RuntimeWarning, match="nogil_load_failed"):
        b = make_backend("auto")
    assert b.predict_impl == COREMLTOOLS
    assert coreml_nogil.reason_code(b.predict_reason) == coreml_nogil.NOGIL_LOAD_FAILED
    assert "MLMultiArray alloc" in b.predict_reason
    assert all(isinstance(m, CoremltoolsModel) for m in b.models.values())
    assert all(p["ratio"] < coreml_ane.PROBE_MAX_RATIO for p in b.probes.values())  # probed via coremltools


# ----------------------------------------------------------------------------- executor wiring


def test_thread_placed_ane_worker_requests_the_nogil_binding(monkeypatch):
    seen = {}

    def load(kind, args):
        seen[kind] = args
        return SimpleNamespace(name="coreml" if kind == "ane" else "mlx", device=kind)

    monkeypatch.setattr(executor, "load_backend", load)
    monkeypatch.setattr(executor, "warm", lambda *a: None)
    monkeypatch.setattr(executor, "backend_info", lambda kind, backend: {})
    for kind, args in (("ane", {"pad_id": 0}), ("gpu", {"pad_id": 0})):
        executor.DeviceWorker(kind, args, placement="thread").close()
    assert seen["ane"]["ane_predict"] == "auto"
    assert "ane_predict" not in seen["gpu"]
    executor.DeviceWorker("ane", {"pad_id": 0, "ane_predict": "coremltools"}, placement="thread").close()
    assert seen["ane"]["ane_predict"] == "coremltools"  # an explicit request wins


def test_process_placed_ane_worker_keeps_coremltools(monkeypatch):
    seen = {}

    class Conn:
        def send(self, msg):
            seen.setdefault("sent", []).append(msg)

        def recv(self):
            raise EOFError

    def load(kind, args):
        seen["args"] = args
        return SimpleNamespace()

    monkeypatch.setattr(executor, "load_backend", load)
    monkeypatch.setattr(executor, "warm", lambda *a: None)
    monkeypatch.setattr(executor, "backend_info", lambda kind, backend: {})
    executor._worker_main("ane", {"pad_id": 0}, Conn())
    assert "ane_predict" not in seen["args"]  # load_backend's default: coremltools


def test_load_backend_passes_the_request_to_ane_backend(monkeypatch):
    got = {}

    class Fake:
        def __init__(self, *a, **k):
            got.update(k)

    monkeypatch.setattr(coreml_ane, "ANEBackend", Fake)
    args = {"model": MODEL, "checkpoint": "/c", "pad_id": 0, "local_attention": 128, "buckets": [64], "strict": True}
    executor.load_backend("ane", args)
    assert got["predict"] == "coremltools"
    executor.load_backend("ane", dict(args, ane_predict="auto"))
    assert got["predict"] == "auto"


def test_backend_info_reports_the_binding_and_the_reason():
    shapes = coreml_ane.ANEShapes(resolve(MODEL), (64,), {64: "0" * 64}, {})
    shapes.predict_impl, shapes.predict_reason = NOGIL, coreml_nogil.DEFAULT
    info = executor.backend_info("ane", shapes)
    assert (info["ane_predict"], info["ane_predict_reason"]) == (NOGIL, coreml_nogil.DEFAULT)
    plain = coreml_ane.ANEShapes(resolve(MODEL), (64,), {64: "0" * 64}, {})
    assert (plain.predict_impl, plain.predict_reason) == (COREMLTOOLS, coreml_nogil.NOT_THREAD_PLACED)


# ----------------------------------------------------------------------------- RuntimeInfo / info()


def _laya(ane):
    laya = model.Laya.__new__(model.Laya)
    laya.spec, laya.device, laya.dtype, laya.execution = resolve(MODEL), "ane", "float16", "workers"
    laya.ane_placement = "thread"
    laya.mlx, laya.ane = None, ane
    laya._tie_buckets = ()
    laya.routing_profile = "shipped"
    laya.ane_state = routing.AneState(None, ane.buckets if ane else ())
    return laya


def test_runtime_info_and_info_record_the_binding_for_ane_requests_only():
    spec = resolve(MODEL)
    ane = coreml_ane.ANEShapes(spec, (64,), {64: "0" * 64}, {})
    detail = "nogil_load_failed: PyObjC Core ML predict failed: RuntimeError: objc.error (test)"
    ane.predict_impl, ane.predict_reason = COREMLTOOLS, detail
    laya = _laya(ane)
    prep = SimpleNamespace(
        sequence_length=10,
        question_count=1,
        input_tokens=10,
        truncated=False,
        items=[{"ids": [0] * 10, "markers": [1, 2]}],
    )
    r = laya._result(prep, routing.Decision("ane", routing.ANE_REQUESTED), ane, (64,), {}, 0, 1, 1)
    # RuntimeInfo carries the reason's code only; info() keeps the detail
    assert (r.runtime.ane_predict, r.runtime.ane_predict_reason) == (COREMLTOOLS, coreml_nogil.NOGIL_LOAD_FAILED)
    assert r.to_dict()["runtime"]["ane_predict"] == COREMLTOOLS
    gpu = model._GPUView(spec, "float16")
    r = laya._result(prep, routing.Decision("gpu", routing.GPU_REQUESTED), gpu, (), {}, 0, 1, 2)
    assert (r.runtime.ane_predict, r.runtime.ane_predict_reason) == (None, None)
    info = laya.info()
    assert (info["ane_predict"], info["ane_predict_reason"]) == (COREMLTOOLS, detail)


def test_finish_ane_carries_the_workers_binding_to_the_parent_view():
    laya = _laya(None)
    laya._workers = {
        "ane": SimpleNamespace(
            info={
                "offered": [64],
                "artifact_sha256": {"64": "0" * 64},
                "load_errors": {},
                "probes": {},
                "ane_predict": COREMLTOOLS,
                "ane_predict_reason": "pyobjc_unavailable: ImportError: test",
            }
        )
    }
    laya._finish_ane()
    assert laya.ane.predict_impl == COREMLTOOLS
    assert laya.ane.predict_reason == "pyobjc_unavailable: ImportError: test"


def test_request_trace_carries_the_binding():
    t = RequestTrace(1, 10, 1, "ane", "ane_requested", 1.0, None, None, *range(9), ane_predict=NOGIL)
    assert t.to_dict()["ane_predict"] == NOGIL
    assert RequestTrace(1, 10, 1, "gpu", "gpu_requested", 1.0, None, None, *range(9)).ane_predict is None


def test_submit_emits_a_trace_with_the_binding_that_served_the_request(monkeypatch):
    """Laya.submit's trace path, over a thread-placed fake ANE worker (no model)."""

    class FakeANE:
        name, device = "coreml", "ane"

        def forward(self, rows):
            return np.zeros((len(rows), ANE_MAX_OPTIONS), np.float32), np.zeros((len(rows), 3), np.float32)

    monkeypatch.setattr(executor, "load_backend", lambda kind, args: FakeANE())
    monkeypatch.setattr(executor, "warm", lambda *a: None)
    monkeypatch.setattr(executor, "backend_info", lambda kind, backend: {})
    monkeypatch.setattr(model, "format_answers", lambda prep, logits, act, cal: {})
    spec = resolve(MODEL)
    ane = coreml_ane.ANEShapes(spec, spec.ane_buckets, {b: "0" * 64 for b in spec.ane_buckets}, {})
    ane.predict_impl, ane.predict_reason = NOGIL, coreml_nogil.DEFAULT
    traces: list = []
    laya = _laya(ane)
    laya._closed, laya._ane_thread, laya._ane_dead_warned = False, None, False
    laya._trace, laya._trace_warned = traces.append, False
    laya._service = scheduling.ServiceModel(routing_table()["models"][spec.name]["service_ms"])
    laya._workers = {"ane": executor.DeviceWorker("ane", {"pad_id": 0}, placement="thread")}
    laya.calibration = None
    prep = SimpleNamespace(
        sequence_length=10,
        question_count=1,
        input_tokens=10,
        truncated=False,
        items=[{"ids": [0] * 10, "markers": [1, 2], "qtype": 0}],
    )
    laya.prepare = lambda context, questions: prep
    try:
        r = laya.submit("ctx", {"q": {}}).result(10)
    finally:
        laya._workers["ane"].close()
    assert (r.runtime.device, r.runtime.ane_predict, r.runtime.ane_predict_reason) == ("ane", NOGIL, "default")
    assert len(traces) == 1 and traces[0].target == "ane" and traces[0].ane_predict == NOGIL


# ----------------------------------------------------------------------------- laya-apple info


def test_cli_info_reports_the_thread_placement_binding(pyobjc, monkeypatch):
    from laya_apple import cli

    assert cli._ane_predict_info()["thread_placement"] == {
        "ane_predict": coreml_nogil.DEFAULT_AUTO,
        "reason": coreml_nogil.DEFAULT,
    }
    monkeypatch.setenv(ENV, "bogus")
    out = cli._ane_predict_info()
    assert out["env"] == "bogus" and out["thread_placement"]["ane_predict"] is None
    assert "ValueError" in out["thread_placement"]["reason"]
