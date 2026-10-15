# Preregistration: open-loop S5, is the GPU LLM slowdown caused by W8?

Written and committed before any open-loop measurement. Nothing here changes after data is
seen. Last experiment of the S5 question: no follow-up experiment is opened from its result.

This is an independent record. It does not change the verdict of
[`../ane-w8-l64-gate/`](../ane-w8-l64-gate/README.md) (PR #174: Research Only / No Ship;
conditions 5 and 7 failed). Passing this experiment cannot move W8 toward production:
condition 7 (a real L64 request distribution) stays unproven, and no production code,
runtime default, golden, tolerance or release gate is touched.

## Question

In the closed-loop S3 runs of PR #174, the local MLX LLM generated about 4% fewer tokens/s
during arm B windows than during arm A windows (B/A 0.953 to 0.962 in 12 of 12 cells). Arm B
also completed about 24% more requests per second in those runs (closed-loop capacity under
LLM load: A 87.0 req/s, B 107.8 req/s, B/A 1.239, from `ane-w8-l64-gate/raw/phase3/s3/`).
Is the slowdown (a) inherent to running the W8 model on the ANE next to a GPU LLM, or (b) a
consequence of B serving more GPU-routed requests per second?

## What stays fixed

Same model (`laya-typed-decisions`, revision f9ab0b228f0f), L64 arms A (production FP16) and
B (`w8-pt`), same request mix (80% L64 single-question to ANE, 10% L64 3-question and 10% L256
single-question to GPU, `ane-w8-l64-gate/scripts/serve_mix.py`), same research-side injection
into an unmodified inline `Laya(device="auto")`, same LLM (Qwen3.5-0.8B-MLX-4bit, greedy,
128 new tokens, same prompt, `ane-w8-l64-gate/scripts/llm_load.py` unchanged), same LLM
interpreter (the oMLX release venv used as a read-only library interpreter, no service
touched), same macOS build (26A428), coremltools 9.0, mlx 0.32.2. The only changed variable is
the load model: closed-loop becomes open-loop. Artifacts are rebuilt in a new isolated
cache (the previous one was removed); their hashes and the unchanged parity result are recorded
and compared with `ane-w8-l64-gate/raw/phase2`. If a rebuilt B fails the unchanged parity or
placement gate, the experiment stops.

## Design

**Open-loop load.** A single dispatcher thread issues requests at the scheduled arrival times
on a monotonic clock into a fixed pool of 32 worker threads that call `laya.predict` on the
shared inline `Laya` (calls serialize on its lock, as before). The arrival schedule is a seeded
Poisson process (exponential inter-arrival times) with a seeded request sequence. Arms A and B
of one cycle replay the byte-identical schedule and request sequence; the schedule is a function
of (rate, cycle index) only, and is the same in every replicate. Latency is measured from the
scheduled arrival, not from issue time, so queueing and a late dispatcher are not hidden.
A request still outstanding when more than 256 are outstanding is dropped and counted.

**Rates, fixed now.** From the recorded closed-loop capacities (A 87.0 req/s under LLM load):
- `eq` = 52 req/s (about 0.60 of A's capacity): equal offered load for A and B. This is the
  primary cell. A runs at utilisation about 0.60 and B at about 0.48.
- `hi` = 65 req/s (about 0.60 of B's capacity, 1.25x `eq`): equal offered load, A at about 0.75
  and B at about 0.60. Used for the dose-response, not for the verdict.

**Windows.** 30 s of arrivals per window plus up to 5 s of drain, 2 s settle between windows.
8 cycles per cell. In each cycle the three cells run in this order: `eq` (A,B), `hi` (A,B),
`null` (A,A'); the A/B order alternates by cycle parity (AB, BA). `null` runs arm A twice at the
`eq` rate and measures the noise of the A/B alternation itself. Slow drift therefore reaches all
cells. Per replicate: 48 windows (about 30 minutes) plus an LLM-only block of 60 s at the start
and at the end. At least 3 replicates, each a fresh process, run sequentially.

**Screen first.** One process, `eq` only, 3 cycles (about 4 minutes), judged on validity only
(below). It is not a stop-for-no-effect screen: three replicates are required by the question.

## Metrics

Per window and arm, from raw per-request records:
- LLM tokens/s and LLM TTFT P95: computed exactly as in `ane-w8-l64-gate/scripts/llm_analyze.py`
  (generations lying fully inside the window, pooled per arm and cell).
- Offered load (scheduled arrivals / s), completed throughput, dropped requests, errors.
- Queue length: arrived minus started, sampled at each arrival; mean, P99 and max.
- Client lag: actual issue time minus scheduled time; P50, P99, max.
- ANE-routed requests: service time (start to end) P50/P99, and response time (scheduled
  arrival to end) P50/P99. GPU-routed requests the same way.
- Answer choice per request, compared A vs B on the identical request (mismatches).
- Loadavg and thermal state before and after each replicate. Energy and GPU/ANE utilisation:
  `unavailable`.

## Validity

A window is valid only if it has 0 drops, 0 errors, client-lag P99 <= 5 ms, and the queue
drains within 5 s after the last arrival. A cycle counts only if all its windows in that cell
are valid. A replicate is valid if every cell has at least 6 of its 8 cycles. An invalid replicate
is repeated once and the reason is recorded; a second invalid repeat ends the experiment as
invalid. Rates are never re-tuned after data, with one fixed exception: if more than 20% of
screen windows are invalid, the screen is repeated once at `eq` = 44 req/s and `hi` = 55 req/s
(about 0.50 of A's and of B's capacity respectively), and those two rates replace the ones above for the
whole experiment. This fallback is decided now and used at most once.

## Decision rules (original bounds, unchanged)

Primary comparison: B vs A at `eq`, per replicate, 3 valid replicates.

| Outcome | Rule |
|---|---|
| **No inherent W8 interference detected** | LLM tokens/s B/A >= 0.97 and LLM TTFT P95 B/A <= 1.05 in all 3 replicates |
| **W8 interference confirmed (at this load)** | LLM tokens/s B/A < 0.97 in at least 2 of 3 replicates, and the pooled mean B/A < 0.97 |
| **Inconclusive** | anything else; reported as inconclusive, not re-run |

Equal offered load makes the completed requests identical between A and B in a valid cycle, so
a slowdown at `eq` cannot come from B doing more work.

Reported without a verdict:
- `null` A'/A tokens/s and TTFT: the noise floor of the design. If it differs from 1 by more
  than 0.015 in any replicate, the result is flagged noise-limited; the outcome above is still
  decided by the fixed bounds.
- `hi` B/A on the identical schedule, and the dose-response of each arm to a 25% higher
  offered rate: A(`hi`)/A(`eq`) and B(`hi`)/B(`eq`) LLM tokens/s. This separates "interference
  at fixed request volume" (`eq` B/A) from "more total resource competition when more requests
  arrive" (the rate effect). Between-rate comparisons use different schedules and are descriptive.
- ANE service and response P50/P99 B/A at both rates, the original P99 bound (+5%) shown for
  reference only.

## Stop rules

- Rebuilt B fails parity, placement or probe: stop, record.
- A valid screen is required before the full replicates; an invalid screen is handled by the
  single fallback above.
- A replicate with an environment mismatch, thermal change or competing process is discarded and
  repeated once with the reason recorded.
- No run is described as a stable result before 3 valid replicates exist.
- When the replicates are analysed and written up, the S5 study ends. No further experiment is
  started from the outcome.

## Deliverables

`research/ane-w8-l64-s5-open-loop/`: this file, scripts, raw per-window and per-request data
(each file under 1 MiB), a `README.md` with the tables and the recorded outcome. A separate
pull request stacked on PR #174; no merge, no production PR.

## Amendment 1 (2026-10-15, before any measurement)

The validity rule "client-lag P99 <= 5 ms" cannot be met by arm A in this process, and this was
found by the harness smoke run before any screen or replicate. The smoke results are not data
and are not committed.

- Mechanism, checked in isolation: while a worker thread runs `laya.predict` on the L64 ANE path,
  a sleeping dispatcher thread in the same process is delayed by about 5 to 14 ms, even with
  `sys.setswitchinterval(0.0002)`. The Core ML predict holds the GIL. GPU-routed requests did not
  delay it. Smoke lag P99 was 7.4 to 7.6 ms in A windows and 4.7 ms in B windows, tracking one
  ANE predict (about 8 ms for A, 5 ms for B).
- Why this is not a departure from open-loop: `Laya` serializes calls on its lock. A request
  issued late only while a predict is running would have waited for that lock anyway, so its
  start time does not change. Latency is measured from the scheduled arrival, so it stays honest.
- Lowering the rates does not shorten a predict, so the rate fallback above does not address this.

Replacement for the lag condition in **Validity**: client-lag P99 <= 14 ms in every window, the
same bound for both arms (5 ms plus about one A-arm L64 predict of 8.2 ms, rounded up). Lag is
still recorded and reported per window, and the strict 5 ms figure is reported next to it for
information. All other validity conditions (0 drops, 0 errors, queue drained within 5 s, at
least 6 of 8 cycles per cell) are unchanged, as are the rates, cells, metrics, decision rules and
stop rules.

Harness-only changes that go with this: `sys.setswitchinterval(0.0002)`, and a chunked 0.2 ms
sleep followed by a spin instead of one long `time.sleep`, because a single long sleep overshoots
by 4 to 7 ms on this machine. The dispatcher stays a single thread in the process.

Two environment records made while rebuilding: the rebuilt A artifact has a different
`artifact_sha256` from the one in `ane-w8-l64-gate` (the rebuild is not byte-reproducible); its
parity, placement and probe results match. The isolated cache holds only the L64 artifacts, as in
the gate run.
