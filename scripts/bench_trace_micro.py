"""The runtime's own per-request cost in execution="workers", with trace off and on.

    uv run python scripts/bench_trace_micro.py --label branch --output benchmarks/tracing/micro.jsonl

bench_trace_overhead.py measures real serving, where a request spends milliseconds on a
device and a cost of microseconds disappears in the noise. This isolates that cost instead:
both devices are thread-placed DeviceWorkers around a backend whose forward returns at once,
tokenisation is skipped (a prepared request is passed in) and answer formatting is replaced
by a constant, so what remains is Laya.submit's routing, the queue snapshots, the
DeviceWorker queue and dispatcher hand-off, RuntimeInfo, and, when enabled, the trace.

One closed-loop client submits alternately a short (ANE) and a long (GPU) request and waits
for each. Per condition, --repeats blocks of --requests requests, conditions interleaved
block by block. Reports µs per request (wall time / requests) per block.

The same file measures the base branch from a checkout of it (it sets `trace` only where the
checkout has it).
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path


class Backend:
    def __init__(self, kind):
        self.name, self.device = ("mlx", "gpu") if kind == "gpu" else ("coreml", "ane")

    def forward(self, rows):
        return [len(r["ids"]) for r in rows], None


class Prep:
    def __init__(self, length):
        self.items = [{"ids": [0] * length, "markers": [1, 2], "qtype": 0}]
        self.sequence_length, self.question_count, self.input_tokens = length, 1, length


def make_laya(trace):
    from laya_apple import executor, model, routing, scheduling
    from laya_apple.backends.coreml_ane import ANEShapes
    from laya_apple.registry import resolve, routing_table

    executor.load_backend = lambda kind, args: Backend(kind)
    executor.warm = lambda kind, backend, pad_id: None
    executor.backend_info = lambda kind, backend: {"name": backend.name}
    model.format_answers = lambda prep, logits, act, cal: {"n": logits}
    spec = resolve("laya-typed-decisions")
    laya = model.Laya.__new__(model.Laya)
    laya.spec, laya.device, laya.dtype, laya.execution = spec, "auto", "float16", "workers"
    laya._closed, laya._ane_thread, laya._ane_dead_warned = False, None, False
    laya._trace, laya._trace_warned = trace, False
    laya._service = scheduling.ServiceModel(routing_table()["models"][spec.name]["service_ms"])
    laya._workers = {k: executor.DeviceWorker(k, {"pad_id": 0}, placement="thread") for k in ("gpu", "ane")}
    laya.mlx = model._GPUView(spec, "float16")
    laya.ane = ANEShapes(spec, spec.ane_buckets, {b: "0" * 64 for b in spec.ane_buckets}, {})
    laya.ane_state = routing.AneState(None, spec.auto_ane_buckets)
    laya._tie_buckets = ()
    laya.calibration = None
    laya.prepare = lambda context, questions: questions
    return laya


def block(laya, reqs, n):
    t0 = time.perf_counter()
    for i in range(n):
        laya.submit(questions=reqs[i & 1]).result()
    return (time.perf_counter() - t0) / n * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--requests", type=int, default=20000)
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import laya_apple

    has_trace = "trace" in laya_apple.Laya.from_pretrained.__code__.co_varnames
    recorded: list = []
    conds = {"none": None}
    if has_trace:
        conds.update(noop=lambda t: None, recorder=recorded.append)
    lays = {c: make_laya(cb) for c, cb in conds.items()}
    reqs = (Prep(128), Prep(1024))
    for laya in lays.values():
        block(laya, reqs, 2000)  # warm-up
    lines = []
    for rep in range(args.repeats):
        order = list(lays)[rep % len(lays) :] + list(lays)[: rep % len(lays)]
        for c in order:
            recorded.clear()
            us = block(lays[c], reqs, args.requests)
            lines.append(
                {
                    "label": args.label,
                    "condition": c,
                    "repeat": rep,
                    "requests": args.requests,
                    "us_per_request": us,
                    "laya_apple": laya_apple.__version__,
                    "python": platform.python_version(),
                    "machine": platform.machine(),
                    "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
            )
            print(args.label, c, rep, f"{us:.2f} us/request", flush=True)
    for laya in lays.values():
        laya.close()
    with Path(args.output).open("a") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    main()
