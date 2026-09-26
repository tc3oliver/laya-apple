from __future__ import annotations

import copy
import os

import pytest

from laya_apple.hub import checkpoint_path
from laya_apple.parity import load_goldens
from laya_apple.prompt import Calibration, Tokenizer
from laya_apple.registry import ANE_COMPUTE_UNITS, ANE_GRAPH, ANE_MAX_OPTIONS, ANE_PRECISION, models

MODEL_NAMES = list(models().keys())
# LAYA_APPLE_TEST_MODELS=a,b limits the per-model tests to those checkpoints (the CI
# integration job caches two of the three). Unset: every supported checkpoint.
if os.environ.get("LAYA_APPLE_TEST_MODELS"):
    _wanted = [m.strip() for m in os.environ["LAYA_APPLE_TEST_MODELS"].split(",") if m.strip()]
    _unknown = sorted(set(_wanted) - set(MODEL_NAMES))
    if _unknown:
        raise ValueError(f"LAYA_APPLE_TEST_MODELS names unsupported checkpoints: {_unknown}")
    MODEL_NAMES = [m for m in MODEL_NAMES if m in _wanted]


def _hf_offline():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


@pytest.fixture(scope="session", params=MODEL_NAMES)
def model_name(request):
    return request.param


@pytest.fixture(scope="session")
def checkpoint(model_name):
    """Local checkpoint directory for `model_name`, offline (already in the HF cache)."""
    _hf_offline()
    spec = models()[model_name]
    return checkpoint_path(spec, local_files_only=True)


@pytest.fixture(scope="session")
def tokenizer(checkpoint):
    return Tokenizer(checkpoint / "tokenizer")


@pytest.fixture(scope="session")
def config(checkpoint):
    import json

    return json.loads((checkpoint / "rl_agent_config.json").read_text())


@pytest.fixture(scope="session")
def calibration(config):
    return Calibration(config)


@pytest.fixture(scope="session")
def golden(model_name):
    return load_goldens(model_name)


_LAYA_CACHE = {}


@pytest.fixture(scope="session")
def cached_laya(request):
    """Session-scoped `Laya` factory, cached per (model_name, device, dtype)."""
    _hf_offline()

    def _get(model_name, device="gpu", dtype="float16"):
        key = (model_name, device, dtype)
        if key not in _LAYA_CACHE:
            from laya_apple import Laya

            _LAYA_CACHE[key] = Laya.from_pretrained(model_name, device=device, dtype=dtype, local_files_only=True)
        return _LAYA_CACHE[key]

    return _get


@pytest.fixture
def laya_gpu(cached_laya, model_name):
    return cached_laya(model_name, device="gpu")


def make_manifest(spec, bucket, *, compute_units=ANE_COMPUTE_UNITS, **overrides):
    """A synthetic manifest dict that passes `verify_manifest` for spec/bucket, minus overrides."""
    data = {
        "format": "laya-apple-artifact",
        "format_version": 1,
        "status": "validated",
        "source": {
            "model": spec.name,
            "repo": spec.repo,
            "revision": spec.revision,
            "weights_sha256": spec.weights_sha256,
        },
        "artifact": {
            "graph": ANE_GRAPH,
            "length": bucket,
            "batch": 1,
            "precision": ANE_PRECISION,
            "max_options": ANE_MAX_OPTIONS,
        },
        "placement": {"compute_units": compute_units},
        "parity": {"passed": True},
        "integrity": {"artifact_sha256": "0" * 64},
        "platform": {"soc": "test", "macos": "1.0", "coremltools": "9.0"},
    }
    data = copy.deepcopy(data)
    for path, value in overrides.items():
        keys = path.split(".")
        d = data
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        if value is _DELETE:
            d.pop(keys[-1], None)
        else:
            d[keys[-1]] = value
    return data


class _Delete:
    def __repr__(self):
        return "<DELETE>"


_DELETE = _Delete()


@pytest.fixture
def manifest_factory():
    return make_manifest


@pytest.fixture
def delete_key():
    return _DELETE
