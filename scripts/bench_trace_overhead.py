"""What request tracing costs: closed-loop heterogeneous serving with trace off and on.

    LAYA_APPLE_CACHE=... uv run python scripts/bench_trace_overhead.py --condition none \
        --label branch --round 0 --output benchmarks/tracing/overhead.jsonl

One invocation loads one `Laya(device="auto", execution="workers")` for one condition and
appends one JSON line per measurement window to --output:

  none      trace=None (the default)
  noop      trace=lambda t: None
  recorder  trace=list.append (an in-memory recorder; nothing is serialised or written)

Workload: two short clients (L128, one question, which auto routes to the ANE) and one long client (L1024, which goes to MLX), each
closed-loop on its own thread, all through the one instance. After a warm-up, --windows
windows of --seconds each. Per window: completed requests per second, end-to-end latency
P50/P95/P99 measured by the client around `predict`, and CPU time (parent plus worker
processes, from psutil) per second of wall time.

The script runs against whichever laya_apple is importable, so the same file measures the
base branch from a checkout of it (it passes `trace` only when a condition sets one).
scripts/trace_overhead_report.py summarises the lines.
"""

from __future__ import annotations

import argparse
import json
import platform
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil

SHORT, LONG = 128, 1024
CLIENTS = (("short", SHORT), ("short", SHORT), ("long", LONG))


def cpu_seconds(proc: psutil.Process) -> float:
    total = 0.0
    for p in [proc, *proc.children(recursive=True)]:
        try:
            t = p.cpu_times()
        except psutil.NoSuchProcess:
            continue
        total += t.user + t.system
    return total


def client(laya, req, stop, out):
    lat, devices = [], Counter()
    while not stop.is_set():
        t0 = time.perf_counter()
        r = laya.predict(context=req[0], questions=req[1])
        lat.append((time.perf_counter() - t0) * 1e3)
        devices[r.runtime.device] += 1
    out["latency_ms"], out["devices"] = lat, devices


def window(laya, reqs, seconds):
    proc = psutil.Process()
    stop = threading.Event()
    outs = [{} for _ in CLIENTS]
    threads = [
        threading.Thread(target=client, args=(laya, reqs[name], stop, outs[i])) for i, (name, _) in enumerate(CLIENTS)
    ]
    c0, t0 = cpu_seconds(proc), time.perf_counter()
    for t in threads:
        t.start()
    time.sleep(seconds)
    stop.set()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    cpu = cpu_seconds(proc) - c0
    lat = np.concatenate([o["latency_ms"] for o in outs])
    per = {}
    for name in dict(CLIENTS):
        v = np.concatenate([o["latency_ms"] for (n, _), o in zip(CLIENTS, outs) if n == name])
        per[name] = {"n": int(v.size), "p50_ms": float(np.percentile(v, 50)), "p99_ms": float(np.percentile(v, 99))}
    devices = sum((o["devices"] for o in outs), Counter())
    return {
        "wall_s": wall,
        "requests": int(lat.size),
        "req_s": lat.size / wall,
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "p99_ms": float(np.percentile(lat, 99)),
        "mean_ms": float(lat.mean()),
        "cpu_s_per_s": cpu / wall,
        "streams": per,
        "devices": dict(devices),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["none", "noop", "recorder"], required=True)
    ap.add_argument("--label", required=True, help="which code is measured, e.g. main or branch")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--model", default="laya-typed-decisions")
    ap.add_argument("--warmup", type=float, default=3.0)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import laya_apple
    from laya_apple import Laya
    from laya_apple.workload import make_request

    kwargs = {}
    recorded: list = []
    if args.condition == "noop":
        kwargs["trace"] = lambda t: None
    elif args.condition == "recorder":
        kwargs["trace"] = recorded.append
    laya = Laya.from_pretrained(args.model, device="auto", execution="workers", local_files_only=True, **kwargs)
    try:
        tok, cfg = laya.tokenizer, laya.config
        reqs = {"short": make_request(tok, cfg, SHORT, 1, seed=1), "long": make_request(tok, cfg, LONG, 1, seed=2)}
        window(laya, reqs, args.warmup)
        lines = []
        for i in range(args.windows):
            recorded.clear()
            w = window(laya, reqs, args.seconds)
            if args.condition == "recorder":
                w["traces"] = len(recorded)
            lines.append(
                {
                    "label": args.label,
                    "condition": args.condition,
                    "round": args.round,
                    "window": i,
                    "model": args.model,
                    "laya_apple": laya_apple.__version__,
                    "ane_placement": laya.ane_placement,
                    "machine": platform.machine(),
                    "platform": platform.platform(),
                    "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    **w,
                }
            )
            print(
                args.label,
                args.condition,
                args.round,
                i,
                f"{w['req_s']:.1f} req/s",
                f"p50 {w['p50_ms']:.2f} p95 {w['p95_ms']:.2f} p99 {w['p99_ms']:.2f} ms",
                f"cpu {w['cpu_s_per_s']:.2f}",
                flush=True,
            )
    finally:
        laya.close()
    with Path(args.output).open("a") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    main()
