"""No-sudo energy sampler for Apple silicon: IOReport per-rail energy counters + SMC PSTR.

    python3 research/energy-sampler/scripts/sampler.py --out PATH.json [--interval 0.5] [--seconds N]

Standard library only (ctypes). Runs until SIGINT/SIGTERM (or --seconds), then writes PATH.json
atomically (temporary file + rename) and exits 0. Output keys:

  source               "ioreport"
  duration_s           t1 - t0 of the first and last IOReport sample
  energy_j             SoC energy: rails cpu + gpu + ane + dram (IOReport "Energy Model")
  mean_power_w         energy_j / duration_s
  rails_j, rails_mean_w   {cpu, gpu, ane, dram}
  system_energy_j, system_mean_power_w   SMC PSTR (total system power, zero-order hold over
                       its 1 Hz updates); null if the SMC key is unreadable
  t0, t1               {"uptime_ns", "unix_ns"} of the first / last sample
  samples              [[uptime_ns, unix_ns, {rail: cumulative J since t0}], ...]
  pstr                 [[uptime_ns, W], ...], one entry per observed change of PSTR
  meta                 SoC, macOS, interval, channel names, sampler CPU seconds

Nothing is idle-subtracted here. The analysis (analyze.py) interpolates the cumulative series
at window boundaries and subtracts idle windows recorded with the same sampler.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import SMC, IOReportEnergy  # noqa: E402

RAILS = {"cpu": "CPU Energy", "gpu": "GPU Energy", "ane": "ANE", "dram": "DRAM"}
SOC_RAILS = ("cpu", "gpu", "ane", "dram")
PSTR = "PSTR"


def uptime_ns() -> int:
    """CLOCK_UPTIME_RAW (mach_absolute_time): one system-wide clock, so the harness and the
    sampler process stamp on the same timeline. time.monotonic_ns() is not system-wide on
    every CPython build (the system python3 3.9 returns a per-process origin)."""
    return time.clock_gettime_ns(time.CLOCK_UPTIME_RAW)


def soc_name() -> str:
    try:
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return "unknown"


class Sampler:
    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self.io = IOReportEnergy()
        try:
            self.smc: SMC | None = SMC()
            self.smc.read(PSTR)
        except (OSError, KeyError, ValueError):
            self.smc = None
        self.samples: list = []
        self.pstr: list = []
        self._stop = threading.Event()
        self._only = tuple(RAILS.values())

    def stop(self) -> None:
        self._stop.set()

    def _poll_pstr(self, now_ns: int) -> None:
        if self.smc is None:
            return
        try:
            w = self.smc.read(PSTR)
        except (OSError, KeyError):
            return
        if not self.pstr or self.pstr[-1][1] != w:
            self.pstr.append([now_ns, w])

    def run(self) -> None:
        cpu0 = time.thread_time()
        try:
            self._run()
        finally:
            self.loop_cpu_s = time.thread_time() - cpu0

    def _run(self) -> None:
        cum = dict.fromkeys(RAILS, 0.0)
        prev = self.io.sample()
        t_mono, t_unix = uptime_ns(), time.time_ns()
        self.samples.append([t_mono, t_unix, dict(cum)])
        self._poll_pstr(t_mono)
        nxt = time.monotonic() + self.interval
        while True:
            stopping = self._stop.wait(max(0.0, nxt - time.monotonic()))
            nxt += self.interval
            cur = self.io.sample()
            t_mono, t_unix = uptime_ns(), time.time_ns()
            d = self.io.delta(prev, cur, self._only)
            self.io.release(prev)
            prev = cur
            for rail, chan in RAILS.items():
                cum[rail] += d.get(chan, 0.0)
            self.samples.append([t_mono, t_unix, dict(cum)])
            self._poll_pstr(t_mono)
            if stopping:
                break
        self.io.release(prev)

    def result(self) -> dict:
        s0, s1 = self.samples[0], self.samples[-1]
        dur = (s1[0] - s0[0]) / 1e9
        rails_j = dict(s1[2])
        energy = sum(rails_j[r] for r in SOC_RAILS)
        sys_e = sys_w = None
        if len(self.pstr) >= 1 and self.pstr[0][0] <= s0[0] + int(2e9) and dur > 0:
            from energy import integrate_power

            ts = [p[0] / 1e9 for p in self.pstr]
            ws = [p[1] for p in self.pstr]
            t0 = max(s0[0] / 1e9, ts[0])
            sys_e = integrate_power(ts, ws, t0, s1[0] / 1e9, hold=True)
            sys_w = sys_e / (s1[0] / 1e9 - t0) if s1[0] / 1e9 > t0 else None
        ru = resource.getrusage(resource.RUSAGE_SELF)
        return {
            "source": "ioreport",
            "duration_s": dur,
            "energy_j": energy,
            "mean_power_w": energy / dur if dur > 0 else None,
            "rails_j": rails_j,
            "rails_mean_w": {k: v / dur for k, v in rails_j.items()} if dur > 0 else None,
            "system_energy_j": sys_e,
            "system_mean_power_w": sys_w,
            "t0": {"uptime_ns": s0[0], "unix_ns": s0[1]},
            "t1": {"uptime_ns": s1[0], "unix_ns": s1[1]},
            "samples": self.samples,
            "pstr": self.pstr,
            "meta": {
                "soc": soc_name(),
                "macos": platform.mac_ver()[0],
                "python": platform.python_version(),
                "interval_s": self.interval,
                "ioreport_group": IOReportEnergy.GROUP,
                "ioreport_channels": RAILS,
                "smc_key": PSTR if self.smc is not None else None,
                "sampler_loop_cpu_s": getattr(self, "loop_cpu_s", None),
                "sampler_loop_cpu_frac": getattr(self, "loop_cpu_s", 0.0) / dur if dur > 0 else None,
                "process_cpu_s": ru.ru_utime + ru.ru_stime,
                "pid": os.getpid(),
            },
        }


def write_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--interval", type=float, default=0.5, help="IOReport sample interval, s (default 0.5)")
    ap.add_argument("--seconds", type=float, default=None, help="stop after this many seconds")
    a = ap.parse_args(argv)
    s = Sampler(a.interval)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: s.stop())
    if a.seconds is not None:
        threading.Timer(a.seconds, s.stop).start()
    th = threading.Thread(target=s.run, name="sampler")
    th.start()
    while th.is_alive():
        th.join(0.2)  # the main thread stays interruptible for the signal handlers
    write_atomic(a.out, s.result())
    return 0


if __name__ == "__main__":
    sys.exit(main())
