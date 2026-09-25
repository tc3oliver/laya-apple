"""J/decision at equal offered load: GPU-only, ANE-only and auto, idle baseline subtracted.

    LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 \\
      uv run python research/energy-sampler/scripts/harness.py \\
        --out research/energy-sampler/raw/campaign-r2-laya-typed-decisions.json.gz

Method (README.md, "Harness"):
- One process, three `Laya.from_pretrained(model, device=...)` instances (inline execution):
  device="gpu", device="ane", device="auto". All are loaded before the first window.
- The no-sudo sampler (sampler.py) runs as a separate process for the whole campaign and
  records cumulative IOReport rail energy every --interval s on CLOCK_UPTIME_RAW. This script
  stamps its windows on the same clock; analyze.py cuts the energy out of the sampler series.
- Windows run in ABBA order per shape: idle, gpu, ane, auto, idle, auto, ane, gpu, ..., idle.
  Every window is --warmup s (not measured) followed by --seconds s (measured). An idle
  window runs nothing; its measured span is --idle-seconds s.
- A loaded window offers requests at a fixed rate (open loop, deterministic spacing 1/rate).
  A request starts at its scheduled time, or at once if the previous one overran; lateness
  is recorded, not hidden. The same request sequence is offered to every config.
- Method version 2 (run 2, criteria-r2.json): idle windows measure --idle-seconds (60 s);
  every window records the CPU time used per process (ps, at its start and end) and its
  CPU-rail energy in 5 s bins over the measured span. A window whose CPU-rail bin mean
  exceeds the bin median by more than the r2 limit was hit by a burst of foreign CPU work;
  it is repeated at once (at most 2 times per window and 8 per run). Every attempt is
  kept; analyze.py uses the last attempt of each window.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from energy import abba_order, burst_excess, cpu_by_process, offered_arrivals, parse_ps  # noqa: E402
from sampler import uptime_ns  # noqa: E402
from sources import IOReportEnergy  # noqa: E402

METHOD_VERSION = 2
CRITERIA = "r2"
PROTOCOL = json.loads((HERE.parent / f"criteria-{CRITERIA}.json").read_text())["protocol"]

# Shapes: a fixed cycle of (length, n_questions, seed). Seeds pick question types from
# laya_apple.workload.QUESTION_POOL (0 choice, 1 score, 2 noul, 3 choice, ...).
SHAPES = {
    # short single-question decisions in the auto-ANE buckets
    "short": [(64, 1, 0), (128, 1, 1), (64, 1, 2), (128, 1, 3), (64, 1, 4), (128, 1, 5), (64, 1, 6), (128, 1, 2)],
    # mixed question shape: single-question requests plus one 4-question request per cycle
    # (auto sends the 4-question request to MLX, the rest to the ANE)
    "mixed": [(64, 1, 0), (128, 1, 2), (128, 4, 1), (96, 1, 4)],
}
DEFAULT_RATE = {"short": 30.0, "mixed": 20.0}  # requests/s; see README for the headroom


def machine_meta() -> dict:
    def sh(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return None

    import laya_apple

    return {
        "soc": sh("sysctl", "-n", "machdep.cpu.brand_string"),
        "model": sh("sysctl", "-n", "hw.model"),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "laya_apple": laya_apple.__version__,
        "git_commit": sh("git", "-C", str(HERE), "rev-parse", "HEAD"),
        "power_source": (sh("pmset", "-g", "batt") or "").split("\n")[0],
        # other GPU users (the local LLM server) must be stopped for a real run; recorded either way
        "omlx_running": bool(sh("pgrep", "-f", "omlx-server")),
        "method_version": METHOD_VERSION,
        "criteria": CRITERIA,
    }


def ps_snapshot() -> dict[int, tuple[str, float]]:
    """CPU time per process (~12 ms of CPU per call; taken outside the measured spans)."""
    try:
        out = subprocess.run(["ps", "-A", "-o", "pid=,time=,comm="], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    return parse_ps(out)


class CpuBins:
    """CPU-rail energy (IOReport "CPU Energy") every `bin_s` over a window's measured span,
    read by a background thread. ctypes releases the GIL during the IOReport call, and the
    same reads happen in idle and loaded windows, so their cost is the same in both."""

    def __init__(self, bin_s: float) -> None:
        self.bin_s = bin_s
        self.io = IOReportEnergy(("CPU Energy",))
        self._points: list = []
        self._th: threading.Thread | None = None

    def start(self, m0_ns: int, m1_ns: int) -> None:
        self._points = []
        n = max(1, round((m1_ns - m0_ns) / 1e9 / self.bin_s))
        at = [m0_ns + round(k * (m1_ns - m0_ns) / n) for k in range(n + 1)]
        self._th = threading.Thread(target=self._run, args=(at,), daemon=True)
        self._th.start()

    def _run(self, at: list[int]) -> None:
        (scale,) = self.io.scales
        for t in at:
            now = uptime_ns()
            if t > now:
                time.sleep((t - now) / 1e9)
            s = self.io.sample()
            try:
                (raw,) = self.io.read(s)
            finally:
                self.io.release(s)
            self._points.append([uptime_ns(), raw * scale])

    def result(self) -> list:
        if self._th is not None:
            self._th.join()
        e0 = self._points[0][1] if self._points else 0.0
        return [[t, e - e0] for t, e in self._points]


class SamplerProcess:
    """sampler.py as a child process; stop() returns its JSON."""

    def __init__(self, interval: float) -> None:
        fd, self.path = tempfile.mkstemp(prefix="energy-sampler-", suffix=".json")
        os.close(fd)
        os.unlink(self.path)
        self.proc = subprocess.Popen(
            [sys.executable, str(HERE / "sampler.py"), "--out", self.path, "--interval", str(interval)]
        )
        time.sleep(max(2.0, 3 * interval))  # a few samples before the first window
        if self.proc.poll() is not None:
            raise RuntimeError(f"sampler exited with {self.proc.returncode}")

    def stop(self) -> dict:
        time.sleep(1.0)  # one more sample after the last window
        self.proc.send_signal(signal.SIGINT)
        self.proc.wait(timeout=30)
        try:
            return json.loads(Path(self.path).read_text())
        finally:
            Path(self.path).unlink(missing_ok=True)


def load_models(model: str, configs: list[str]) -> dict:
    from laya_apple import Laya

    out = {}
    for c in configs:
        out[c] = Laya.from_pretrained(model, device=c, local_files_only=True)
    return out


def build_requests(laya, shape: str) -> list[tuple[str, dict]]:
    from laya_apple.workload import make_request

    return [make_request(laya.tokenizer, laya.config, L, n_questions=q, seed=s) for L, q, s in SHAPES[shape]]


def compact(result) -> dict:
    """Per-question decision for cross-config agreement (not a correctness gate)."""
    out = {}
    for name, a in result.answers.items():
        out[name] = a.get("choice", a.get("score", a.get("noul")))
    return out


def run_window(laya, reqs, rate: float | None, warmup: float, seconds: float, bins: CpuBins, own: dict) -> dict:
    """One window. rate None: idle. Times are CLOCK_UPTIME_RAW ns."""
    ps0 = ps_snapshot()
    t_start = uptime_ns()
    m0 = t_start + int(warmup * 1e9)
    m1 = m0 + int(seconds * 1e9)
    bins.start(m0, m1)
    records = []
    if rate is None:
        while uptime_ns() < m1:
            time.sleep(min(0.5, max(0.0, (m1 - uptime_ns()) / 1e9)))
    else:
        for k, off in enumerate(offered_arrivals(rate, warmup + seconds)):
            sched = t_start + int(off * 1e9)
            now = uptime_ns()
            if sched > now:
                time.sleep((sched - now) / 1e9)
            ctx, qs = reqs[k % len(reqs)]
            s = uptime_ns()
            err = None
            try:
                r = laya.predict(context=ctx, questions=qs)
                e = uptime_ns()
                rt = r.runtime
                rec_dev, reason, ans = rt.device, rt.routing_reason, compact(r)
            except Exception as ex:  # recorded; analyze.py marks the window invalid
                e = uptime_ns()
                rec_dev, reason, ans, err = None, None, None, f"{type(ex).__name__}: {ex}"
            records.append([k, k % len(reqs), len(qs), sched, s, e, rec_dev, reason, ans, err])
        while uptime_ns() < m1:  # the last arrival can finish before the window ends
            time.sleep(min(0.05, max(0.0, (m1 - uptime_ns()) / 1e9)))
    cpu_bins = bins.result()
    procs = cpu_by_process(ps0, ps_snapshot(), own)
    return {
        "start_ns": t_start,
        "measure_ns": [m0, m1],
        "end_ns": uptime_ns(),
        "requests": records,
        "cpu_bins": cpu_bins,
        "cpu_excess_w": burst_excess([(t / 1e9, e) for t, e in cpu_bins]),
        "cpu_by_process": procs,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default="laya-typed-decisions")
    ap.add_argument("--configs", nargs="+", default=["gpu", "ane", "auto"])
    ap.add_argument("--shapes", nargs="+", default=["short", "mixed"], choices=sorted(SHAPES))
    ap.add_argument("--rate", nargs="*", default=[], metavar="SHAPE=REQ_PER_S")
    ap.add_argument("--repeats", type=int, default=2, help="ABBA blocks per shape (default 2)")
    ap.add_argument("--warmup", type=float, default=PROTOCOL["warmup_s"])
    ap.add_argument(
        "--seconds", type=float, default=PROTOCOL["loaded_seconds"], help="measured span of a loaded window"
    )
    ap.add_argument(
        "--idle-seconds", type=float, default=PROTOCOL["idle_seconds"], help="measured span of an idle window"
    )
    ap.add_argument("--interval", type=float, default=PROTOCOL["sampler_interval_s"], help="sampler interval, s")
    a = ap.parse_args(argv)
    rates = dict(DEFAULT_RATE)
    for kv in a.rate:
        k, v = kv.split("=")
        rates[k] = float(v)

    meta = machine_meta()
    if meta["omlx_running"]:
        print("warning: omlx-server is running; this run is not a campaign run", file=sys.stderr)
    models = load_models(a.model, a.configs)
    reqs = {s: build_requests(models[a.configs[0]], s) for s in a.shapes}
    for s in a.shapes:  # untimed pass: compile/caches, and a request that cannot run fails here
        for c in a.configs:
            for ctx, qs in reqs[s]:
                models[c].predict(context=ctx, questions=qs)

    bins = CpuBins(PROTOCOL["disturbance_bin_s"])
    sampler = SamplerProcess(a.interval)
    own = {os.getpid(): "harness", sampler.proc.pid: "sampler"}
    limit_w = PROTOCOL["disturbance_cpu_excess_w"]
    per_window, left = PROTOCOL["disturbance_max_repeats_per_window"], PROTOCOL["disturbance_max_repeats_per_run"]
    windows = []
    try:
        for shape in a.shapes:
            for i, cfg in enumerate(abba_order(a.configs, a.repeats)):
                laya = None if cfg == "idle" else models[cfg]
                rate = None if cfg == "idle" else rates[shape]
                seconds = a.idle_seconds if cfg == "idle" else a.seconds
                for attempt in range(per_window + 1):
                    w = run_window(laya, reqs[shape], rate, a.warmup, seconds, bins, own)
                    ex = w["cpu_excess_w"]
                    disturbed = ex is not None and ex > limit_w
                    repeat = disturbed and attempt < per_window and left > 0
                    w.update(
                        shape=shape,
                        config=cfg,
                        index=i,
                        rate=rate,
                        attempt=attempt,
                        disturbed=disturbed,
                        superseded=repeat,
                    )
                    windows.append(w)
                    print(
                        f"{shape:6s} {i:2d} {cfg:5s} attempt={attempt} requests={len(w['requests'])} "
                        f"cpu_excess={'-' if ex is None else f'{ex:.3f}'} W{' REPEAT' if repeat else ''}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if not repeat:
                        break
                    left -= 1
    finally:
        samples = sampler.stop()
        for m in models.values():
            m.close()

    run = {
        "meta": meta,
        "args": {**vars(a), "out": str(a.out), "rates": {s: rates[s] for s in a.shapes}},
        "protocol": PROTOCOL,
        "shapes": {s: SHAPES[s] for s in a.shapes},
        "windows": windows,
        "sampler": samples,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.out, "wt") as f:
        json.dump(run, f)
    print(f"wrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
