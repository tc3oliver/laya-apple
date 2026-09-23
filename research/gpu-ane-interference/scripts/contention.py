"""Prototype: a contention-aware completion estimate for scheduling.decide_queued (research only).

Calibrated from this study's device-plan measurements, never hand-set: for each device, the
service-time change its two measured shapes show while the OTHER device runs closed-loop
(the matrix cells) is fitted as

    service_concurrent(s) = s * (1 + a) + d        (s: that shape's solo service time)

through the two points, clamped to a, d >= 0. d captures a per-request additive cost (the GIL
wait of a thread-placed ANE: about the same number of ms for short and long GPU requests), a a
proportional one (the host slowdown of process placement).

`install` wraps laya_apple.scheduling.decide_queued in this process. The eligibility logic is
untouched: decide_queued still calls routing.decide first and only compares completion times
for requests it may place on either device; the wrapper only changes the numbers it compares:

    this request   gpu estimate -> gpu*(1+a_g)+d_g  when the ANE is busy (ane backlog > 0)
                   ane estimate -> ane*(1+a_a)+d_a  when the GPU is busy
    backlogs       gpu backlog  -> backlog*(1+a_g)+d_g  when both are busy (one running job's
                   additive cost; later queued jobs' is not counted), likewise for the ANE

`uninstall` restores the original. Nothing here is imported by laya_apple.
"""

from __future__ import annotations

import json
from pathlib import Path

import laya_apple.scheduling as scheduling

RESULTS = Path(__file__).resolve().parents[1] / "results.json"


def _fit(points):
    """points: [(solo_ms, concurrent_ms), (solo_ms, concurrent_ms)] -> (a, d), a, d >= 0."""
    (s1, c1), (s2, c2) = sorted(points)
    a = ((c2 - s2) - (c1 - s1)) / (s2 - s1)
    a = max(0.0, a)
    d = max(0.0, (c1 - s1) - a * s1)
    if a == 0.0:  # additive only: the mean added time
        d = max(0.0, ((c1 - s1) + (c2 - s2)) / 2)
    return a, d


def calibrate(model: str, placement: str, results: Path = RESULTS) -> dict:
    run = json.loads(Path(results).read_text())["runs"][f"device:{model}:{placement}"]
    cells = run["cells"]

    def mean_svc(cell, label):
        return cells[cell]["streams"][label]["service"]["mean"]

    gpu_pts, ane_pts = [], []
    for g in ("M", "L"):  # GPU shapes, against the ANE's longest bucket running closed-loop
        gpu_pts.append((mean_svc(f"solo:gpu_{g}", f"gpu_{g}"), mean_svc(f"matrix:gpu_{g}+ane_B", f"gpu_{g}")))
    for a in ("S", "B"):  # ANE shapes, against the GPU's long shape running closed-loop
        ane_pts.append((mean_svc(f"solo:ane_{a}", f"ane_{a}"), mean_svc(f"matrix:gpu_L+ane_{a}", f"ane_{a}")))
    a_g, d_g = _fit(gpu_pts)
    a_a, d_a = _fit(ane_pts)
    return {
        "model": model,
        "placement": placement,
        "source": f"{Path(results).name}: device:{model}:{placement} solo and matrix cells",
        "gpu": {"a": a_g, "d_ms": d_g, "points": gpu_pts},
        "ane": {"a": a_a, "d_ms": d_a, "points": ane_pts},
    }


class _Service:
    def __init__(self, base, p, ane_busy, gpu_busy):
        self.base, self.p, self.ane_busy, self.gpu_busy = base, p, ane_busy, gpu_busy

    def gpu_ms(self, length, questions=1):
        s = self.base.gpu_ms(length, questions)
        return s * (1 + self.p["gpu"]["a"]) + self.p["gpu"]["d_ms"] if self.ane_busy else s

    def ane_ms(self, bucket, questions=1):
        s = self.base.ane_ms(bucket, questions)
        return s * (1 + self.p["ane"]["a"]) + self.p["ane"]["d_ms"] if self.gpu_busy else s


_ORIGINAL = scheduling.decide_queued


def install(params: dict) -> None:
    def decide_queued(device, spec, question_count, sequence_length, ane, *, service, gpu_backlog_ms,
                      ane_backlog_ms, tie_buckets=()):
        ane_busy, gpu_busy = ane_backlog_ms > 0, gpu_backlog_ms > 0
        g, a = params["gpu"], params["ane"]
        if ane_busy and gpu_busy:
            gpu_backlog_ms = gpu_backlog_ms * (1 + g["a"]) + g["d_ms"]
            ane_backlog_ms = ane_backlog_ms * (1 + a["a"]) + a["d_ms"]
        return _ORIGINAL(
            device,
            spec,
            question_count,
            sequence_length,
            ane,
            service=_Service(service, params, ane_busy, gpu_busy),
            gpu_backlog_ms=gpu_backlog_ms,
            ane_backlog_ms=ane_backlog_ms,
            tie_buckets=tie_buckets,
        )

    scheduling.decide_queued = decide_queued


def uninstall() -> None:
    scheduling.decide_queued = _ORIGINAL


if __name__ == "__main__":
    import sys

    print(json.dumps(calibrate(sys.argv[1], sys.argv[2]), indent=1))
