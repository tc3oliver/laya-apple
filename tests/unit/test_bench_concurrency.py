"""Unit tests for scripts/bench_concurrency.py."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def bc():
    spec = importlib.util.spec_from_file_location("bench_concurrency", ROOT / "scripts" / "bench_concurrency.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_conditions_forces_c_locale_on_ps(monkeypatch, bc):
    """ps must run with LC_ALL=C so %cpu uses a period decimal."""
    seen_env = {}

    class FakeCompleted:
        stdout = " 29.5 /sbin/launchd\n  0.5 /usr/libexec/logd\n"

    def fake_run(_cmd, **kwargs):
        seen_env.update(kwargs.get("env") or {})
        return FakeCompleted()

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = bc.conditions()

    assert seen_env.get("LC_ALL") == "C"
    assert result["top_cpu"][0] == (29.5, "launchd")
