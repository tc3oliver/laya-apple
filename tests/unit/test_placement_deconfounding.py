"""Unit tests for research/coreml-placement-deconfounding/scripts/analyze.py (research only)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "research" / "coreml-placement-deconfounding" / "scripts" / "analyze.py"
spec = importlib.util.spec_from_file_location("deconfounding_analyze", PATH)
analyze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)


def fake(queues, received_late=0, errors=0):
    """One concurrent-cell window of 1 s with len(queues) GPU requests evenly spread."""
    n = len(queues)
    recs = []
    for i, q in enumerate(queues):
        arrival = 1000.0 + 1000.0 * i / n
        received = arrival + 1 + (5000 if i < received_late else 0)  # completion, independent of q here
        recs.append(
            {
                "cell": analyze.BOTH,
                "cycle": 0,
                "stream": "gpu_M",
                "t_ms": {"arrival": arrival, "received": received},
                "timing_ms": {"queue": q},
            }
        )
    run = {
        "windows": [
            {"cell": analyze.BOTH, "cycle": 0, "measure": [1.0, 2.0], "streams": {"gpu_M": {"errors": ["x"] * errors}}}
        ]
    }
    return run, recs


@pytest.mark.parametrize(
    ("queues", "late", "errors", "stable"),
    [
        ([10] * 100, 0, 0, True),
        ([10] * 75 + [29] * 25, 0, 0, True),  # +19 ms: within first + 20 ms
        ([10] * 75 + [31] * 25, 0, 0, False),  # +21 ms and more than 2x
        ([100] * 75 + [190] * 25, 0, 0, True),  # within 2x
        ([10] * 100, 3, 0, False),  # 97% completed within the window
        ([10] * 100, 0, 1, False),  # an error
    ],
)
def test_stability_rule(queues, late, errors, stable):
    run, recs = fake(queues, late, errors)
    got = analyze.stability(run, recs)["c0"]
    assert got["stable"] is stable
    assert got["offered_req_s"] == pytest.approx(len(queues))
