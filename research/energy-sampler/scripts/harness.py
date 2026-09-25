"""J/decision at equal offered load: GPU-only, ANE-only and auto, idle baseline subtracted.

    LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 \\
      uv run python research/energy-sampler/scripts/harness.py --out research/energy-sampler/raw/campaign.json.gz

Method (README.md, "Harness"):
- One process, three `Laya.from_pretrained(model, device=...)` instances (inline execution):
  device="gpu", device="ane", device="auto". All are loaded before the first window.
- The no-sudo sampler (sampler.py) runs as a separate process for the whole campaign and
  records cumulative IOReport rail energy every --interval s on CLOCK_UPTIME_RAW. This script
  stamps its windows on the same clock; analyze.py cuts the energy out of the sampler series.
- Windows run in ABBA order per shape: idle, gpu, ane, auto, idle, auto, ane, gpu, ..., idle.
  Every window is --warmup s (not measured) followed by --seconds s (measured). An idle
  window runs nothing for the same time.
- A loaded window offers requests at a fixed rate (open loop, deterministic spacing 1/rate).
  A request starts at its scheduled time, or at once if the previous one overran; lateness
  is recorded, not hidden. The same request sequence is offered to every config.
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
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from energy import abba_order, offered_arrivals  # noqa: E402
from sampler import uptime_ns  # noqa: E402

METHOD_VERSION = 1

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
    }


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


def run_window(laya, reqs, rate: float | None, warmup: float, seconds: float) -> dict:
    """One window. rate None: idle. Times are CLOCK_UPTIME_RAW ns."""
    t_start = uptime_ns()
    m0 = t_start + int(warmup * 1e9)
    m1 = m0 + int(seconds * 1e9)
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
    return {"start_ns": t_start, "measure_ns": [m0, m1], "end_ns": uptime_ns(), "requests": records}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default="laya-typed-decisions")
    ap.add_argument("--configs", nargs="+", default=["gpu", "ane", "auto"])
    ap.add_argument("--shapes", nargs="+", default=["short", "mixed"], choices=sorted(SHAPES))
    ap.add_argument("--rate", nargs="*", default=[], metavar="SHAPE=REQ_PER_S")
    ap.add_argument("--repeats", type=int, default=2, help="ABBA blocks per shape (default 2)")
    ap.add_argument("--warmup", type=float, default=10.0)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--interval", type=float, default=0.5, help="sampler interval, s")
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

    sampler = SamplerProcess(a.interval)
    windows = []
    try:
        for shape in a.shapes:
            for i, cfg in enumerate(abba_order(a.configs, a.repeats)):
                laya = None if cfg == "idle" else models[cfg]
                rate = None if cfg == "idle" else rates[shape]
                w = run_window(laya, reqs[shape], rate, a.warmup, a.seconds)
                w.update(shape=shape, config=cfg, index=i, rate=rate)
                windows.append(w)
                print(f"{shape:6s} {i:2d} {cfg:5s} requests={len(w['requests'])}", file=sys.stderr, flush=True)
    finally:
        samples = sampler.stop()
        for m in models.values():
            m.close()

    run = {
        "meta": meta,
        "args": {**vars(a), "out": str(a.out), "rates": {s: rates[s] for s in a.shapes}},
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
