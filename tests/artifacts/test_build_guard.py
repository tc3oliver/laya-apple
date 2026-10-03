from __future__ import annotations

import json
import os
import sys
import tempfile
import types
from pathlib import Path

import pytest

import laya_apple.artifacts as A
import laya_apple.backends.coreml_ane as coreml_ane
import laya_apple.conversion.build as B
import laya_apple.parity.ane as parity_ane
from laya_apple.artifacts import COMPILED, artifact_dir, artifacts_root, load_verified
from laya_apple.conversion.build import build
from laya_apple.errors import UnsupportedShapeError
from laya_apple.registry import models

PROFILE = {"soc": "Apple M4 Max", "macos": "26.0.1", "macos_build": "25A1", "coremltools": "9.0"}
CLEAN_PLAN = {"compute_units": "CPU_AND_NE", "ops": {"ane": 1, "cpu": 0, "gpu": 0}, "transitions": 0}
PARITY = {"passed": True, "rows": 1, "prob_max_abs": 0.0, "action_prob_max_abs": 0.0, "hard_mismatches": 0}


def test_build_rejects_non_offered_bucket_before_doing_any_work(monkeypatch):
    spec = models()["laya"]
    bad_length = max(spec.ane_buckets) + 1
    assert bad_length not in spec.ane_buckets

    def _boom(*a, **k):
        raise AssertionError("build() should not touch the network/checkpoint for an unsupported bucket")

    monkeypatch.setattr("laya_apple.hub.checkpoint_path", _boom)
    with pytest.raises(UnsupportedShapeError):
        build(spec, bad_length)


@pytest.fixture
def stub_build(tmp_path, monkeypatch):
    """`build` without PyTorch, Core ML or a checkpoint: the conversion, the checkpoint
    loads and the parity gate are stand-ins, so the staging directory, the manifest and the
    move into place are the real ones."""
    ckpt = tmp_path / "ckpt"
    (ckpt / "encoder").mkdir(parents=True)
    (ckpt / "rl_agent_config.json").write_text("{}")
    (ckpt / "encoder/config.json").write_text(json.dumps({"local_attention": 4}))

    def save(path):
        (Path(path) / "Data").mkdir(parents=True)
        (Path(path) / "Data/model.mil").write_text("model")

    def compile_model(path):
        out = Path(tempfile.mkdtemp(dir=tmp_path)) / "model.mlmodelc"
        out.mkdir()
        (out / "weights.bin").write_bytes(b"weights")
        return str(out)

    ct = types.ModuleType("coremltools")
    ct.convert = lambda *a, **k: types.SimpleNamespace(save=save)
    ct.utils = types.SimpleNamespace(compile_model=compile_model)
    ct.TensorType = lambda **kw: kw
    ct.precision = types.SimpleNamespace(FLOAT16="FLOAT16")
    ct.target = types.SimpleNamespace(macOS15="macOS15")
    ct.ComputeUnit = types.SimpleNamespace(CPU_AND_NE="CPU_AND_NE")
    ct.models = types.SimpleNamespace(CompiledMLModel=lambda *a, **k: object())
    torch = types.ModuleType("torch")
    torch.set_num_threads = lambda n: None
    torch.randn = torch.zeros = lambda *shape: types.SimpleNamespace(shape=shape)
    torch.jit = types.SimpleNamespace(trace=lambda *a, **k: object())

    class InferenceMode:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    torch.inference_mode = InferenceMode
    body = types.SimpleNamespace(embedding_norm=types.SimpleNamespace(weight=types.SimpleNamespace(shape=(1, 8))))
    body.eval = lambda: body
    reference = types.ModuleType("laya_apple.conversion.torch_reference")
    reference.load_model = lambda *a, **k: object()
    bc1s = types.ModuleType("laya_apple.conversion.bc1s")
    bc1s.ConvBody = lambda source, length: body

    monkeypatch.setenv("LAYA_APPLE_CACHE", str(tmp_path / "cache"))
    for module in (ct, torch, reference, bc1s):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(B, "checkpoint_path", lambda spec, local_files_only=False: ckpt)
    monkeypatch.setattr(B, "verify_weights", lambda spec, path: "ok")
    monkeypatch.setattr(B, "Tokenizer", lambda path: object())
    monkeypatch.setattr(B, "_layout_check", lambda *a, **k: 0.0)
    monkeypatch.setattr(B, "compute_plan_summary", lambda *a, **k: dict(CLEAN_PLAN))
    monkeypatch.setattr(B, "platform_profile", lambda: dict(PROFILE))
    monkeypatch.setattr(A, "platform_profile", lambda: dict(PROFILE))
    monkeypatch.setattr(A, "compute_plan_summary", lambda *a, **k: dict(CLEAN_PLAN))
    monkeypatch.setattr(coreml_ane, "HostWeights", lambda *a, **k: object())
    monkeypatch.setattr(parity_ane, "ane_parity", lambda *a, **k: dict(PARITY))


@pytest.mark.parametrize("force", [False, True], ids=["new", "forced-rebuild"])
def test_build_keeps_the_build_marker_out_of_the_registered_artifact(stub_build, force):
    """The marker guards the staging directory while it is built; it is not in the
    registered artifact, and a rebuilt artifact that carried one leaves none behind."""
    spec = models()["laya"]
    final = artifact_dir(spec, 64)
    if force:
        (final / COMPILED).mkdir(parents=True)
        (final / "BUILDING.json").write_text(json.dumps({"pid": os.getpid()}))
    assert build(spec, 64, force=force) == final
    assert sorted(p.name for p in final.iterdir()) == sorted(["manifest.json", COMPILED])
    assert not list(final.rglob("BUILDING.json"))
    assert [p.name for p in final.parent.iterdir()] == [final.name]
    assert not any((artifacts_root() / ".staging").iterdir())
    load_verified(spec, 64, full=True)


def test_a_failed_move_into_place_restores_the_previous_artifact_on_a_forced_rebuild(stub_build, monkeypatch):
    spec = models()["laya"]
    final = artifact_dir(spec, 64)
    build(spec, 64)
    before = {str(f.relative_to(final)): f.read_bytes() for f in sorted(final.rglob("*")) if f.is_file()}
    real_rename = os.rename

    def rename(src, dst):
        if Path(src).name.startswith(f"{spec.name}-L64-"):  # the staged artifact moving into place
            raise OSError("simulated rename failure")
        return real_rename(src, dst)

    monkeypatch.setattr(os, "rename", rename)
    with pytest.raises(OSError, match="simulated rename failure"):
        build(spec, 64, force=True)
    monkeypatch.setattr(os, "rename", real_rename)
    assert {str(f.relative_to(final)): f.read_bytes() for f in sorted(final.rglob("*")) if f.is_file()} == before
    assert [p.name for p in final.parent.iterdir()] == [final.name]  # no leftover .old-* directory
    assert not any((artifacts_root() / ".staging").iterdir())
    load_verified(spec, 64, full=True)


def test_a_failed_restore_after_a_failed_forced_rebuild_says_where_the_previous_artifact_is(
    stub_build, monkeypatch, capsys
):
    spec = models()["laya"]
    final = artifact_dir(spec, 64)
    build(spec, 64)
    real_rename = os.rename

    def rename(src, dst):
        if Path(src).name.startswith(f"{spec.name}-L64-") or ".old-" in Path(src).name:
            raise OSError("simulated rename failure")  # the move into place, then the restore
        return real_rename(src, dst)

    monkeypatch.setattr(os, "rename", rename)
    with pytest.raises(OSError, match="simulated rename failure"):
        build(spec, 64, force=True)
    monkeypatch.setattr(os, "rename", real_rename)
    left = [p for p in final.parent.iterdir() if ".old-" in p.name]
    assert len(left) == 1 and not final.exists()
    assert f"could not restore the previous artifact; it was left at {left[0]}" in capsys.readouterr().err


def test_a_failed_removal_of_the_replaced_artifact_keeps_the_new_one_registered(stub_build, monkeypatch, capsys):
    spec = models()["laya"]
    final = artifact_dir(spec, 64)
    build(spec, 64)
    real_rmtree = B.shutil.rmtree

    def rmtree(path, *args, **kwargs):
        if ".old-" in Path(path).name:
            raise OSError("simulated removal failure")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(B.shutil, "rmtree", rmtree)
    assert build(spec, 64, force=True) == final
    monkeypatch.setattr(B.shutil, "rmtree", real_rmtree)
    assert "could not remove the replaced copy" in capsys.readouterr().err
    assert [p.name for p in final.parent.iterdir() if ".old-" in p.name]
    load_verified(spec, 64, full=True)


@pytest.mark.parametrize("force", [False, True], ids=["new", "forced-rebuild"])
def test_an_interrupt_right_after_the_move_into_place_leaves_the_new_artifact_registered(
    stub_build, monkeypatch, force
):
    spec = models()["laya"]
    final = artifact_dir(spec, 64)
    before = None
    if force:
        build(spec, 64)
        before = (final / "manifest.json").read_text()
    real_rename = os.rename

    def rename(src, dst):
        real_rename(src, dst)
        if Path(src).name.startswith(f"{spec.name}-L64-"):  # the staged artifact has just moved into place
            raise KeyboardInterrupt

    monkeypatch.setattr(os, "rename", rename)
    with pytest.raises(KeyboardInterrupt):
        build(spec, 64, force=force)
    monkeypatch.setattr(os, "rename", real_rename)
    assert sorted(p.name for p in final.iterdir()) == sorted(["manifest.json", COMPILED])
    assert not any((artifacts_root() / ".staging").iterdir())
    load_verified(spec, 64, full=True)
    left = [p for p in final.parent.iterdir() if p != final]
    if force:  # the previous artifact is kept beside it, intact
        assert len(left) == 1 and ".old-" in left[0].name
        assert (left[0] / "manifest.json").read_text() == before
    else:
        assert left == []


def test_an_interrupt_before_the_move_into_place_restores_the_previous_artifact(stub_build, monkeypatch):
    spec = models()["laya"]
    final = artifact_dir(spec, 64)
    build(spec, 64)
    before = (final / "manifest.json").read_text()
    real_rename = os.rename

    def rename(src, dst):
        if Path(src).name.startswith(f"{spec.name}-L64-"):
            raise KeyboardInterrupt
        return real_rename(src, dst)

    monkeypatch.setattr(os, "rename", rename)
    with pytest.raises(KeyboardInterrupt):
        build(spec, 64, force=True)
    monkeypatch.setattr(os, "rename", real_rename)
    assert (final / "manifest.json").read_text() == before
    assert [p.name for p in final.parent.iterdir()] == [final.name]
    assert not any((artifacts_root() / ".staging").iterdir())
