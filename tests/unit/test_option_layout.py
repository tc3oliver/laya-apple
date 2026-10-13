"""Checkpoints requesting a non-sequential option layout are rejected, never run as sequential."""

import json

import pytest

from laya_apple import model
from laya_apple.errors import LayaAppleError
from laya_apple.hub import check_option_layout


def _checkpoint(tmp_path, agent_cfg):
    (tmp_path / "encoder").mkdir()
    (tmp_path / "rl_agent_config.json").write_text(json.dumps(agent_cfg))
    (tmp_path / "encoder/config.json").write_text("{}")
    return tmp_path


@pytest.mark.parametrize("cfg", [{}, {"option_layout": "sequential"}])
def test_sequential_layout_is_accepted(cfg):
    check_option_layout(cfg)


@pytest.mark.parametrize("layout", ["parallel", "", "Sequential", None, 1])
def test_other_layouts_fail_closed(layout):
    with pytest.raises(LayaAppleError, match="option_layout"):
        check_option_layout({"option_layout": layout})


@pytest.mark.parametrize("device", ["auto", "gpu", "ane"])
def test_laya_rejects_parallel_checkpoint_before_any_backend(tmp_path, monkeypatch, device):
    ckpt = _checkpoint(tmp_path, {"option_layout": "parallel"})
    monkeypatch.setattr(model, "checkpoint_path", lambda *a, **k: ckpt)
    monkeypatch.setattr(model, "verify_weights", lambda *a, **k: "ok")
    with pytest.raises(LayaAppleError, match="option_layout='parallel'"):
        model.Laya.from_pretrained("laya", device=device)


def test_torch_reference_rejects_parallel_checkpoint(tmp_path):
    pytest.importorskip("torch")
    from laya_apple.conversion.torch_reference import load_model

    with pytest.raises(LayaAppleError, match="option_layout"):
        load_model(_checkpoint(tmp_path, {"option_layout": "parallel"}), 64)
