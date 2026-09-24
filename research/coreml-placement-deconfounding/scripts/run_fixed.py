"""A / B / C at a fixed GPU offered load: the ANE's cost without the GPU-load confound.

    LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 uv run --with pyobjc-framework-CoreML==12.2.2 python \
        research/coreml-placement-deconfounding/scripts/run_fixed.py --gpu-rate 45 \
        --ane-predict pyobjc --ane-placement thread --model laya-typed-decisions --plan device \
        --cycles 3 --seconds 20 --out ...

This is research/coreml-gil-completion-path/scripts/run_cell.py (the gpu-ane-interference
harness, --ane-predict, the GPU reply stamps), with its device plan swapped for three cells:

  solo:ane_B          ANE L128 closed loop, alone
  fixed:gpu_M         GPU L128 open loop at --gpu-rate req/s (Poisson), alone
  fixed:gpu_M+ane_B   both at once

In #46 the GPU client was closed loop, so a configuration that unblocked the GPU also sent it
more work. Here the GPU is sent the same requests at the same scheduled times in every
configuration. The stream's RNG is seeded the same way in every window (seed 17), so the
Poisson arrival offsets and the request sequence repeat exactly. Each window's scheduled
offsets (ns after the window start) and request seeds are recorded under `fixed_load`, with a
SHA-256 over them, so A/B/C can be checked to be identical.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "coreml-gil-completion-path" / "scripts"))

import run_cell  # noqa: E402  (also puts the gpu-ane-interference scripts on sys.path)

interference = run_cell.interference


def main():
    argv = list(sys.argv[1:])
    i = argv.index("--gpu-rate")
    rate = float(argv[i + 1])
    del argv[i : i + 2]
    sys.argv = [sys.argv[0], *argv]
    Run, Stream = interference.Run, interference.DeviceStream
    arrivals: dict = {}  # window key -> [(offset_ns, seed)] as scheduled
    current = {}

    def device_cells(self):
        return [
            {"name": "solo:ane_B", "kind": "solo", "streams": [("ane", "B", "closed", 0)]},
            {"name": "fixed:gpu_M", "kind": "fixed", "streams": [("gpu", "M", "fixed", rate)]},
            {
                "name": "fixed:gpu_M+ane_B",
                "kind": "fixed",
                "streams": [("gpu", "M", "fixed", rate), ("ane", "B", "closed", 0)],
            },
        ]

    def make_streams(self, cell):
        # mode "fixed" = open loop at an absolute rate; same seed as the harness's first stream
        return [
            Stream(self, dev, shp, "poisson" if mode == "fixed" else mode, r if mode == "fixed" else 0.0, seed=17 + i)
            for i, (dev, shp, mode, r) in enumerate(cell["streams"])
        ]

    window = Run.window

    def run_window(self, cell, cycle, streams, aggressors):
        current["key"] = f"{cell['name']}|c{cycle}"
        return window(self, cell, cycle, streams, aggressors)

    run_open = Stream.run_open

    def stream_run_open(self, start_at, stop_at):
        self._start_at = start_at
        return run_open(self, start_at, stop_at)

    issue = Stream._issue

    def stream_issue(self, req, arrival):
        if self.mode != "closed":
            arrivals.setdefault(current["key"], []).append((arrival - self._start_at, req["seed"]))
        return issue(self, req, arrival)

    finish = Run.finish

    def run_finish(self):
        record = finish(self)
        per = {}
        for key, xs in arrivals.items():
            per[key] = {
                "offsets_ns": [x for x, _ in xs],
                "seeds": [s for _, s in xs],
                "sha256": hashlib.sha256(repr(xs).encode()).hexdigest(),
            }
        record["fixed_load"] = {"gpu_rate": rate, "arrivals": per}
        return record

    Run.device_cells, Run.make_streams, Run.window, Run.finish = device_cells, make_streams, run_window, run_finish
    Stream.run_open, Stream._issue = stream_run_open, stream_issue
    run_cell.main()


if __name__ == "__main__":
    main()
