"""Unit tests for research/coreml-slow-state-trigger/scripts/background.py (research only): the window
groups, the per-process counts and ranges, and the CLI's --check, on synthetic runs."""

from __future__ import annotations

import gzip
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "research" / "coreml-slow-state-trigger" / "scripts" / "background.py"
spec = importlib.util.spec_from_file_location("sst_background_for_tests", SCRIPT)
background = importlib.util.module_from_spec(spec)
spec.loader.exec_module(background)

TOP = [[10.0, "Aerials"], [3.0, "Aerials"], [2.0, "helper"], [1.0, "x"], [0.5, "y"]]  # a name can repeat


def _window(cycle: int, condition: str, p99: float, load: float, top=TOP) -> dict:
    return {
        "cycle": cycle,
        "condition": condition,
        "streams": {"short": {"p99_ms": p99}},
        "conditions_before": {"loadavg": [load, load + 0.1, load + 0.2], "top_cpu": top},
    }


def _write(raw: Path, cell: str, rnd: int, windows: list[dict]) -> None:
    with gzip.open(raw / f"laya-{cell}-r{rnd}.json.gz", "wt") as fh:
        json.dump({"part_a": {"windows": windows}}, fh)


def _raw(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    spike = [[95.0, "spiker"], [10.0, "Aerials"], [2.0, "helper"], [1.0, "x"], [0.5, "y"]]
    _write(
        raw,
        "PB-R",
        1,
        [_window(0, "solo_short", 10.2, 1.0), _window(0, "hetero", 16.0, 2.0, spike), _window(1, "hetero", 12.0, 1.5)],
    )
    _write(raw, "A", 1, [_window(0, "hetero", 11.0, 1.2), _window(0, "gpu_only", 48.0, 1.3)])
    return raw


def test_groups_and_counts(tmp_path):
    ws = background.windows(_raw(tmp_path))
    assert sorted(w["group"] for w in ws) == sorted(
        [
            "not hetero, all cells",
            "not hetero, all cells",
            "hetero, PB-R, slow",
            "hetero, PB-R, normal",
            "hetero, other cells, normal",
        ]
    )
    md = background.tables(ws)
    assert "hetero, PB-R, slow (1)" in md and "hetero, PB-R, normal (1)" in md and "all (5)" in md
    # one entry per process and window: the highest of a repeated name
    aerials = next(line for line in md.splitlines() if line.startswith("| `Aerials` |"))
    assert aerials.endswith("| 5: 10.0 to 10.0 |")
    assert "| `spiker` | 1: 95.0 to 95.0 | - | - | - | 1: 95.0 to 95.0 |" in md
    assert "| PB-R | hetero, slow | 1 | 2.00 to 2.00 | 2.10 to 2.10 | 2.20 to 2.20 |" in md
    assert "| hetero, PB-R, slow | 1 | 95.0 | 95.0 | 95.0 | 1 | 1 | `spiker` (laya-PB-R-r1, cycle 0, hetero) |" in md


def test_load_rise_names_the_window_it_spans(tmp_path):
    md = background.tables(background.windows(_raw(tmp_path)))
    assert "| laya-PB-R-r1 | cycle 0, solo_short | 1.00 to 2.00 | +1.00 | cycle 0, hetero, slow |" in md


def test_check_mode(tmp_path):
    raw, out = _raw(tmp_path), tmp_path / "out"
    out.mkdir()
    run = [sys.executable, str(SCRIPT), "--raw", str(raw), "--out", str(out)]
    subprocess.run(run, check=True, capture_output=True)
    assert subprocess.run([*run, "--check"], capture_output=True).returncode == 0
    (out / "background.md").write_text("edited\n")
    stale = subprocess.run([*run, "--check"], capture_output=True)
    assert stale.returncode != 0 and b"stale" in stale.stderr
    empty = tmp_path / "empty"
    empty.mkdir()
    none = subprocess.run([sys.executable, str(SCRIPT), "--raw", str(empty), "--out", str(out)], capture_output=True)
    assert none.returncode != 0 and b"evidence.json" in none.stderr
