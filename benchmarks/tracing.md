# Request lifecycle tracing: overhead

What `Laya.from_pretrained(..., execution="workers", trace=...)` costs, measured against
`main` (4f95101) with the final tracing code (2ed4968). Four configurations:

- `main`, which has no tracing;
- this branch with `trace=None`, the default;
- this branch with a no-op callback (`lambda t: None`);
- this branch with an in-memory recorder (`list.append`).

Nothing in either measurement serialises a trace or writes one to disk.

**Setup:** Apple M4 Max, macOS 26.6.2 (25G83), Python 3.12.14, MLX 0.32.2, coremltools 9.0,
laya-apple 1.0.2, `laya-typed-decisions`, ANE thread placement (the measured default for
this model). On AC power. The oMLX service was stopped for the runs. A browser was open;
its load was the same for every configuration because they were interleaved.

## Serving: closed-loop heterogeneous load

`scripts/bench_trace_overhead.sh` (raw data: `tracing/overhead.jsonl`, summary:
`scripts/trace_overhead_report.py`).

**Workload:** one `Laya(device="auto", execution="workers")` instance with three
closed-loop client threads:
- two short clients: L128, one question, routed to the ANE;
- one long client: L1024, one question, routed to the GPU.

End-to-end latency is the client's own clock around `predict`. CPU is the user plus system
CPU time of the parent process and its worker processes, per second of wall time.

**Windows:**
- Each invocation loads its own instance, warms up for 3 s, then measures three 10 s
  windows.
- There are 5 rounds of the four configurations, with the order rotated each round. That
  gives 15 windows (about 17,700 requests) per configuration.

Each figure is the median over the 15 windows. The percentage is the change against `main`.

| Code | trace | Windows | Requests | Throughput (req/s) | e2e P50 (ms) | e2e P95 (ms) | e2e P99 (ms) | CPU (cores busy) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| main | none | 15 | 17691 | 117.25 | 20.05 | 101.54 | 102.10 | 0.16 |
| branch | none | 15 | 17656 | 117.16 (-0.08%) | 20.06 (+0.04%) | 101.52 (-0.02%) | 102.02 (-0.08%) | 0.16 (+1.58%) |
| branch | noop | 15 | 17675 | 117.48 (+0.19%) | 20.01 (-0.20%) | 101.43 (-0.11%) | 101.82 (-0.27%) | 0.15 (-2.01%) |
| branch | recorder | 15 | 17719 | 117.38 (+0.11%) | 20.02 (-0.13%) | 101.68 (+0.14%) | 102.03 (-0.08%) | 0.15 (-2.60%) |

**Noise band of `main`:** the spread of its own five per-round medians, relative to its
median.

| Metric | Spread |
|---|---|
| Throughput | -0.55% to +0.49% |
| P50 | -0.53% to +0.26% |
| P95 | -0.35% to +0.30% |
| P99 | -0.32% to +0.21% |
| CPU | -6.07% to +4.86% |

Every configuration's difference from `main` is inside that band and below 1%, on every
metric. At this load a request spends milliseconds on a device, so a per-request cost of a
few microseconds cannot show here. The next measurement isolates that cost.

## Host path: the runtime's own per-request cost

`scripts/bench_trace_micro.py` (raw data: `tracing/micro.jsonl`).

**Setup:**
- Both devices are thread-placed `DeviceWorker`s around a backend whose forward returns
  immediately.
- Tokenisation is skipped, and answer formatting is replaced by a constant.

What remains is the part of `Laya.submit` that tracing touches:
- routing, the queue readings, the FIFO queue and the dispatcher hand-off;
- `RuntimeInfo`, and the trace when it is enabled.

**Method:**
- One client submits a short and a long request alternately, waiting for each.
- Blocks of 20,000 requests are interleaved across configurations.
- `main` and the branch alternate process by process: 3 processes each, 10 blocks per
  configuration per process.

| Code | trace | Blocks | Requests | µs per request, median | min–max | vs main |
|---|---|---:|---:|---:|---:|---:|
| main | none | 30 | 600000 | 16.25 | 16.16–16.95 | |
| branch | none | 30 | 600000 | 17.22 | 16.91–19.02 | +0.96 µs (+5.9%) |
| branch | noop | 30 | 600000 | 19.26 | 19.11–19.58 | +3.01 µs (+18.5%) |
| branch | recorder | 30 | 600000 | 19.76 | 19.30–20.38 | +3.51 µs (+21.6%) |

**Reading it:**
- **`trace=None` costs about 1 µs per request of runtime overhead.** Against a
  10–100 ms request in the serving run above, that is 0.001–0.01%.
- **A first version cost +1.48 µs.** Two costs were tracing's own but were paid with
  tracing off:
  - two `QueueSnapshot` objects built per request for the router;
  - a clock read for `received_ns` on every job.

  Both now happen only when tracing is on. The serving figures above are from the final
  code.
- **Of the remaining +0.96 µs, an ablation puts about 0.3 µs on the request id and on the
  queue-depth and running counters.** The trace needs them. The rest is below what this
  measurement resolves: the same code varied by about 0.5 µs between sessions.
- **A trace costs about 2 µs with a callback enabled**, for building the `RequestTrace` and
  its two `QueueSnapshot`s and making the call. That cost lands on the device's dispatcher
  thread.

## What `trace=None` does not add

These were checked on the final code:

- **No trace objects.** A counting check over 5,000 requests built 0 `RequestTrace` and 0
  `QueueSnapshot` with `trace=None`, and 5,000 and 10,000 with a callback.
- **No lock.** `executor.py` and `model.py` have the same `with self._lock` sections as
  `main` (5 and 1). The router's one queue reading per worker replaces `backlog_ms()`'s
  one lock acquisition.
- **No IPC message.** The number of send and receive calls is the same as `main`.
  - The request message is still `(job_id, rows)`; `job_id` is now the request id instead
    of a per-worker counter.
  - The reply carries `(start_ns, end_ns)` in place of `device_ms`, in the same message.
- **No JSON and no file I/O** in `laya_apple/trace.py` or the request path.
