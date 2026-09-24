"""#51's fixed-GPU-load run, with the whole process tree's CPU time sampled in every window.

    LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 uv run --with pyobjc-framework-CoreML==12.2.2 python \
        benchmarks/ane-equal-load-cpu/run_tree_cpu.py --gpu-rate 35 --ane-predict coremltools \
        --ane-placement process --model laya --plan device --cycles 3 --seconds 20 --out ...

This is research/coreml-placement-deconfounding/scripts/run_fixed.py unchanged (GPU L128 open
loop at a fixed Poisson rate with the same seeded arrival trace in every window; ANE L128 closed
loop), plus one thing: while each window runs, a thread reads user + system CPU time of this
process and of every child process every 50 ms. Each window's samples are stored under
`tree_cpu` as (t, {pid: cpu_s}) on the run's time axis, with the pid of each role:

  parent        this process: the harness clients, the dispatchers, a thread-placed ANE backend
  <inst>:<dev>  a process-placed worker of the "gpu" or "ane" Laya instance

The analysis interpolates the samples at the window's measurement bounds.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "research" / "coreml-placement-deconfounding" / "scripts"))

import run_fixed  # noqa: E402  (puts run_cell and the gpu-ane-interference harness on sys.path)

interference = run_fixed.interference
SELF = psutil.Process()
PERIOD_S = 0.05


def cpu_s(p: psutil.Process) -> float | None:
    try:
        t = p.cpu_times()
    except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
        return None  # AccessDenied: a transient setuid child such as ps, run between windows
    return t.user + t.system


def roles(run) -> dict:
    out = {"parent": os.getpid()}
    for inst, laya in run.lays.items():
        for dev, w in laya._workers.items():
            if w.placement == "process":
                out[f"{inst}:{dev}"] = w.pid
    return out


def main():
    Run = interference.Run
    window = Run.window

    def run_window(self, cell, cycle, streams, aggressors):
        samples, stop = [], threading.Event()
        procs = {}

        def sample():
            while True:
                try:
                    for c in SELF.children(recursive=True):
                        procs.setdefault(c.pid, c)
                except psutil.Error:
                    pass
                t = (interference.now() - self.t_ref) / 1e9
                row = {str(pid): cpu_s(p) for pid, p in [(os.getpid(), SELF), *procs.items()]}
                samples.append((t, row))
                if stop.is_set():
                    return
                stop.wait(PERIOD_S)

        th = threading.Thread(target=sample, daemon=True)
        th.start()
        try:
            out = window(self, cell, cycle, streams, aggressors)
        finally:
            stop.set()
            th.join()
        out["tree_cpu"] = {"roles": roles(self), "period_s": PERIOD_S, "samples": samples}
        return out

    Run.window = run_window  # run_fixed wraps this one, so the samples cover its window as run
    run_fixed.main()


if __name__ == "__main__":
    main()
