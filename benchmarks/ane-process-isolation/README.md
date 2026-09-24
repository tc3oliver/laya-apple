# ANE process isolation: production regression gate

Issue: [#52](https://github.com/tc3oliver/laya-apple/issues/52). This is the
performance-sensitive regression check before `ane_placement="auto"` moves laya and
laya-typed-decisions from a thread-placed ANE to a worker process. It measures the current
runtime on the canonical workload. It is not new research.

## Criteria (written and committed before the campaign ran)

For each of laya and laya-typed-decisions, B (process) is compared with A (thread, the
current production placement for both models). A model PASSes only if every criterion holds:

| # | criterion | definition |
|---|---|---|
| 1 | correctness | 0 answer mismatches in every window of every run, both placements |
| 2 | aggregate throughput | process ≥ 0.95 × thread |
| 3 | short-stream P99 | process ≤ 1.05 × thread |
| 4 | long (GPU) stream P99 | process ≤ 1.05 × thread |
| 5 | GPU completion isolation | process GPU return P50 ≤ 1 ms, **and** thread P50 / process P50 ≥ 5 |

The overall gate PASSes only if both models PASS. The thresholds are not changed after the
data is seen. A failure is reported as a failure, with the metric and its magnitude.

**Definitions:**
- **Windows.** Only the `hetero` windows count: both streams run through the one
  `device="auto"`, `execution="workers"` instance. Each placement has 2 runs × 3 cycles = 6
  windows.
- **Aggregate throughput.** For each stream, take the median of its req/s over the 6
  windows; add the two streams' medians. This is `bench_concurrency.py`'s own aggregate,
  over the pooled windows.
- **Stream P99.** The median over the 6 windows of that stream's per-window P99 latency.
  This is `bench_concurrency.py`'s own summary, over the pooled windows.
- **GPU return.** `received_ns − service_end_ns` of every GPU request whose reply arrives
  inside a `hetero` window, pooled over the 6 windows. P50 of that pool.

**Resource costs are not re-measured here.** This change is a policy and default change; it
does not touch process-placement execution. The evidence for those costs is:
- memory, start-up, IPC, shutdown, lifecycle and orphans:
  [`../ane-placement-gate/`](../ane-placement-gate/README.md) (#54, aggregate verdict
  FAIL as recorded);
- CPU at an equal offered load: [`../ane-equal-load-cpu/`](../ane-equal-load-cpu/README.md)
  (#56).

## Workload

This is the v1.0 closed-loop heterogeneous mix (`benchmarks/v1.0.md`, "Heterogeneous GPU +
ANE serving, closed loop"): `scripts/bench_concurrency.py --part a`, unchanged.
- **Shapes:** laya short L128 / long L512; laya-typed-decisions short L128 / long L1024. These
  are the v1.0 commands.
- **Windows:** 20 s each, 3 cycles per run, in alternating order (solo_short, solo_long,
  hetero, gpu_only). One closed-loop client per stream, the same request per stream every
  time.
- **Correctness:** answers are checked against inline references computed per device.
- **The only difference between A and B** is `--ane-placement thread` / `process`. The
  benchmark code, requests, warm-up and cache are the same for both.
- **Order per model:** thread, process, process, thread (ABBA), one run after another. The
  machine is idle on AC power, with oMLX stopped. The Core ML E5 cache is not cleared.
- **`run_mix.py`** wraps `bench_concurrency.py` without changing its workload. It records
  each closed-loop client's window bounds, and the return leg of every GPU request, from the
  `trace=` callback of the heterogeneous instance. Both placements pay the callback's cost.

laya-multilingual already uses process placement and is not part of the A/B. Its
correctness is covered by the integration and parity tests run for the change.

## Reproduce

```sh
# ~45 min, idle machine on AC power, other GPU/ANE services stopped
LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh benchmarks/ane-process-isolation/run.sh
uv run python benchmarks/ane-process-isolation/analyze.py      # --check to verify
```
