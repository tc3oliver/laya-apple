"""Switchyard on the real model: the oracle gate and a short end-to-end smoke run.

    LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 uv run pytest -q tests/integration/test_switchyard.py

Oracle gate (docs/switchyard.md): 12 platform patterns x 20 surface variants, argmax equals
the oracle for every one, on the GPU and on the ANE; every train prompt is <= 128 tokens.
"""

from __future__ import annotations

import json
import os

import pytest

from laya_apple.demos.switchyard import ane_state, result, world

pytestmark = pytest.mark.integration


def _laya(device):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from laya_apple import Laya

    return Laya.from_pretrained(world.MODEL, device=device, local_files_only=True)


def _oracle_gate(laya):
    misses = []
    for t in world.oracle_cases():
        context, questions = world.train_prompt(t)
        answer = laya.predict(context=context, questions=questions).answers[world.QUESTION_ID]["choice"]
        if answer != t.oracle:
            misses.append((world.PATTERNS[t.pattern], t.train_id, answer, t.oracle))
    return misses


def test_oracle_gate_gpu():
    laya = _laya("gpu")
    try:
        assert _oracle_gate(laya) == []
    finally:
        laya.close()


@pytest.mark.ane
def test_oracle_gate_ane():
    laya = _laya("ane")
    try:
        assert _oracle_gate(laya) == []
    finally:
        laya.close()


def test_every_train_prompt_fits_the_train_budget():
    laya = _laya("gpu")
    try:
        trains = [a.train for a in world.schedule(world.DEFAULT_SEED, world.DURATION_S) if a.train]
        trains += world.oracle_cases()
        lengths = [laya.prepare(*world.train_prompt(t)).sequence_length for t in trains]
    finally:
        laya.close()
    assert max(lengths) <= world.TRAIN_MAX_TOKENS


def _smoke(tmp_path, capsys):
    from laya_apple.cli import main

    out = tmp_path / "run"
    assert main(["--offline", "switchyard", "--duration", "3", "--no-open", "--out", str(out)]) == 0
    r, records = result.read(out)
    assert result.validate(r) == []
    per_round = r["workload"]["offered"]["requests"]
    assert len(records) == per_round * len(r["rounds"])  # every request traced, once
    for rnd in r["rounds"]:
        assert rnd["game"]["trains"] == r["workload"]["offered"]["trains"]
        assert rnd["game"]["misrouted"] == 0 and rnd["game"]["route_accuracy"] == 1.0
    assert r["comparison"]["decision_disagreements"] == 0
    assert (out / "replay.html").exists()
    return r, capsys.readouterr().out


def test_smoke_gpu_only(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(
        ane_state, "detect", lambda spec: ane_state.AneStatus(ane_state.SETUP_AVAILABLE, ane_state.NO_COREMLTOOLS)
    )
    r, text = _smoke(tmp_path, capsys)
    assert list(r["configs"]) == ["gpu_only"] and r["ane"]["setup_command"] == ane_state.SETUP_COMMAND
    assert ane_state.SETUP_COMMAND in text


@pytest.mark.ane
def test_smoke_hybrid(tmp_path, capsys):
    from laya_apple.registry import resolve

    if ane_state.detect(resolve(world.MODEL)).state != "ready":
        pytest.skip("the Neural Engine path is not ready on this machine")
    r, _ = _smoke(tmp_path, capsys)
    assert set(r["configs"]) == {"gpu_only", "hybrid"} and r["comparison"]["available"]
    assert r["ane"]["warmup_ane_requests"] > 0
    assert r["configs"]["hybrid"]["summary"]["secondary"]["devices"].get("ane", 0) > 0
    assert json.dumps(r).count(os.path.expanduser("~")) == 0
