# Serve decisions beside a local LLM

`laya-apple serve` is meant to run on the same Mac as a local LLM server. This benchmark
measures what that costs each side while the LLM generates at saturation:
- does the LLM lose generation throughput while decisions run, and is the loss smaller when
  serve sends short decisions to the Neural Engine (`--device auto`) than when it runs every
  decision on the GPU (`--device gpu`)?
- do decisions stay fast and correct while the LLM holds the GPU?

`docs/serve.md` makes no performance claim about this until a campaign here has run
("Limits": *No performance claim yet*).

**Status: preregistered.** The criteria in [`criteria.json`](criteria.json) and below were
committed before any campaign data existed. Smoke runs of the harness are not results and are
not committed. Run 1 ([`m4-max/`](m4-max/)) was **invalid** under its own validity checks and
draws no result; run 2 is preregistered with two revised validity definitions and has not run
yet. See [Run 1](#run-1-m4-max-invalid) and [Run 2](#run-2-preregistered).

## Question and criteria

Throughput, latency and correctness are three separate results. There is no overall
PASS/FAIL. Each criterion is reported with its value, limit and verdict, and a failure is
reported as a failure, with its magnitude. The limits do not change after data is seen; a
different limit is a new preregistered campaign.

| Result | # | Criterion | Limit |
|---|---|---|---|
| Leaves the GPU free | G1 | LLM tok/s drop with serve `auto` running decisions, against the LLM alone | ≤ 5% |
| | G2 | drop with serve `gpu` − drop with serve `auto` | > the LLM-alone window spread |
| Decision latency beside the LLM | L1 | short-decision P99 under LLM load, serve `auto` | ≤ 50 ms |
| | L2 | the same P99, loaded / unloaded, serve `auto` | ≤ 1.5× |
| Correctness | C1 | every decision against the same serve process's unloaded reference: max probability error, max act-probability error, hard mismatches | ≤ 0.02, ≤ 0.02, 0; near-tie flips listed |
| | E1 | requests without an HTTP 200 answer (status, timeout, connection error) | 0 |

"Leaves the GPU free" is claimed only if G1 **and** G2 pass. Serve `gpu` has no limit of its
own: it is the comparison, reported with every metric.

**Definitions.**
- **LLM tok/s** of a window: completion tokens streamed inside the window's bounds, divided
  by the window's length. The server packs several tokens into one SSE chunk, so each chunk
  carries its share of the request's `usage.completion_tokens` in proportion to its share of
  the streamed text. A cell's value is the median over its windows.
- **Drop** of a config: 1 − median tok/s (config, `decisions_llm`) / median tok/s (`llm_alone`).
- **LLM-alone window spread:** (max − min) / median of tok/s over the `llm_alone` windows.
  This is the window-to-window noise of the thing being measured, taken from the same
  campaign.
- **Short-decision P99:** latency from the request's *scheduled* arrival to its response, so
  client-side lateness counts against the server. The per-window P99 of the `short_1q` class,
  then the median over the cell's windows (the repository's convention, as in
  `scripts/bench_concurrency.py`).
- **Reference:** each serve process, after warm-up and before its measured windows, answers
  every request variant once, sequentially, with the LLM idle. Every later answer of that
  process is compared with it under the FP16 gate in
  [`docs/correctness.md`](../../docs/correctness.md): hard mismatch = the decision differs and
  the reference top-1/top-2 margin ≥ 0.04; a flip inside that band is listed. A missing,
  extra or re-typed question is a hard mismatch. A decision answered on a different device
  from its reference is counted (`device_changed`) and compared all the same.

**Why these limits.**
- **G1, 5%.** With the default load, serve `auto` sends only the 20% multi-question class to
  the GPU: about 1.6 req/s of L128 three-question forwards, on the order of 15 ms each, so
  about 2–3% of GPU time. If the Neural Engine path works as intended, the LLM should lose
  about that much, plus HTTP and CPU overhead. 5% leaves room for that overhead and is below
  what a person reading the stream would notice: a 256-token reply at 36 tok/s takes 7.1 s,
  and 7.5 s at 5% less.
- **G2, beyond the noise.** "Leaves the GPU free" is a claim about the Neural Engine path,
  not about the decision load being small. If serve `gpu` costs the LLM no more than serve
  `auto` does, within the noise of the LLM-alone windows, the workload cannot tell the two
  apart, and the claim is not supported even if G1 passes.
- **L1, 50 ms.** In-process, laya short decisions on the Neural Engine have a P99 of 11.36 ms
  in the product mix ([`../ane-process-isolation/`](../ane-process-isolation/README.md),
  thread placement). Switchyard counts a decision as late above 100 ms
  ([`../switchyard/`](../switchyard/README.md)). 50 ms is half that deadline, and small next to
  an LLM turn that takes seconds, so a plugin that asks for a decision per tool call adds no
  wait a person notices.
- **L2, 1.5×.** An absolute limit alone would hide a tail that grows several-fold but starts
  small. 1.5× allows the LLM's CPU and memory-bandwidth pressure some cost, and fails a tail
  that the LLM's load dominates.
- **C1 / E1.** The repository's correctness gate, unchanged. A decision that changes under
  load is a regression whatever its speed.

### Validity checks

A failed check makes the affected results **invalid**, neither pass nor fail; the campaign is
rerun in a new directory. `analyze.py` reports each check.

| # | Check | Limit |
|---|---|---|
| V1 | open-loop client lag: P99 of (sent − scheduled), every decision window | ≤ 5 ms |
| V2 | LLM saturated: fraction of the window during which the LLM was streaming, and LLM errors, every LLM window | ≥ 0.90, 0 |
| V3 | LLM exclusive: the LLM server idle at window start, and its `total_requests` grew by exactly this harness's own requests across the window (oMLX `/api/status`) | every window |
| V4 | serve `auto` answered short decisions on the Neural Engine | ≥ 99% per window |
| V5 | references on the expected devices: `auto` short → ANE, `auto` mixed → GPU, `gpu` → GPU | all |

## Workload

- **Decision clients.** 8 plugin-like clients, each its own HTTP client with keep-alive,
  `POST /v1/systemone` with `model: "jev-latest"` as a Jev client sends it.
  - **Arrivals: open loop.** Each client has its own seeded Poisson timetable at 1 req/s,
    so 8 req/s offered in total. The same timetable (seed 11) replays in every window of every
    cell, so every cell gets identical offered load. Requests are sent at their scheduled
    times whether or not earlier ones have returned.
  - **Classes.** 80% `short_1q`: one question at exactly 96 tokens, rotating over 7 variants
    that cover `choice`, `score` and `noul`. This is the shape a Claude Code plugin sends, and
    in `--device auto` it routes to the Neural Engine. 20% `mixed_3q`: one call with three
    questions of mixed types (`choice`, `score`, `noul`) at exactly 128 tokens, 4 variants.
    Multi-question calls always run on the GPU.
  - **Text.** Generated by `laya_apple.workload.make_request` (synthetic support tickets, no
    third-party text).
- **Serve.** `laya-apple --offline serve --model laya --device {gpu,auto}` on a free loopback
  port, started by the harness for each block and stopped after it. `--model laya` pins the
  English checkpoint: every request gets the same checkpoint, so the configs differ only in
  the device. A block starts measuring only after `/health` reports the model's Neural Engine
  path `ready` (`auto`), warm-up passes and the references.
- **LLM load.** K = 2 closed-loop streaming `/v1/chat/completions` requests with a fixed
  prompt, `max_tokens` 256 and temperature 0, from a separate process so that stream parsing
  never delays the decision clients. The measured server runs one request at a time, so the
  second stream keeps the next request queued and the GPU never waits for the client. After
  the window no new request starts; in-flight requests finish (drain), so the server is never
  left generating for a client that went away. The model is the server's `default_model`,
  recorded in `campaign.json`.
- **Cells.**
  - `llm_alone`: LLM load, no serve process.
  - `gpu/decisions`, `auto/decisions`: decisions, LLM idle.
  - `gpu/decisions_llm`, `auto/decisions_llm`: decisions and LLM load together.
- **Order.** Per round, serve blocks in ABBA order (round 1: gpu, auto, auto, gpu; round 2:
  auto, gpu, gpu, auto), an `llm_alone` window before, between and after the blocks, and the
  window order inside a block alternating (`decisions` first, then `decisions_llm` first). The
  default is 2 rounds: 4 windows per serve cell, 9 `llm_alone` windows.
- **Windows and warm-up.** 60 s measured windows. LLM windows start 15 s after the LLM load
  starts; decision windows are preceded by 3 s of decisions at the same rate from a different
  seed; both are excluded. One unmeasured LLM request runs before the campaign. 5 s cool-down
  between windows.
- **Machine state.** Load average, busiest processes and `pmset -g therm` before and after
  every window, and the LLM server's `/api/status` before and after, are in the raw files.

**What this does not measure.**
- **A prefill-heavy LLM load.** The fixed prompt is about 90 tokens: shorter than the
  server's prefix-cache block, so it is prefilled on every request, but the load is dominated
  by decode (256 tokens per request). If the LLM server prefills on the Neural Engine, that
  short prefill competes with serve `auto` for it, and the effect is inside the window's
  tok/s. Long uncached prompts, where prefill dominates, are a separate campaign. The server
  TTFT reported per cell (oMLX `usage.time_to_first_token`, median) includes the wait behind
  the other stream, since the server runs one request at a time; it is reported, with no
  limit.
- **Other LLM servers or models,** other checkpoints (`--model auto` adds language routing and
  `laya-multilingual`), or other offered loads. Each is a new campaign with its own
  parameters recorded.

## Run

```sh
# ~50 min: other GPU/ANE consumers stopped (other benchmarks, browsers), on AC power. The LLM
# server stays up, serving its model, and must not be used by anything else meanwhile.
LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 OMLX_SETTINGS=<LLM server base path>/settings.json \
    sh benchmarks/serve/run.sh benchmarks/serve/<machine>
uv run python benchmarks/serve/analyze.py benchmarks/serve/<machine> --check
```

- `run.sh` refuses to start if another laya-apple process is running or the campaign
  directory already has data. It runs `scripts/bench_serve.py campaign` and then
  `analyze.py`, which writes `results.json` and `tables.md` next to `raw/`.
- The harness never starts, stops or reconfigures the LLM server. Before each window it waits
  while the server reports other active or waiting requests (up to 10 min, then stops).
- **The LLM server's API key** is read at run time, from `$LLM_API_KEY` or from `auth.api_key`
  in the JSON file named by `$OMLX_SETTINGS` / `--llm-settings`. It is sent only in the
  Authorization header and never written to `raw/` or printed. The settings path is not
  recorded either.
- **Runtime estimate.** `bench_serve.py` prints its own estimate for the plan first: about
  49 min for the defaults (17 steps; serve start-up is assumed at 45 s for `gpu` and 90 s for
  `auto`). `--rounds`, `--window`, `--clients`, `--rate`, `--llm-streams` and `--max-tokens`
  change the plan, and are recorded in `raw/campaign.json`.
- **Smoke test** of the harness, a few minutes, low load (2 clients, 10 s windows, one window
  per cell); its numbers are not a result and are never committed:

  ```sh
  LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 OMLX_SETTINGS=... \
      uv run python scripts/bench_serve.py campaign --smoke --out <scratch dir>
  uv run python benchmarks/serve/analyze.py <scratch dir>
  ```

## Raw data

`raw/campaign.json`: arguments, platform, plan, request shapes (lengths and questions), LLM
prompt and model, LLM server version. `raw/NNN-<kind>[-<config>].json`: one per window, with
every decision (scheduled, sent and done times, status, device, routing reason, comparison
with the reference), every LLM request (chunk times and sizes, usage), the LLM server status
and machine state before and after. `raw/refs-stepNN-<config>.json`: each serve process's
health at start-up and its reference answers. `raw/serve-stepNN-<config>.log`: the serve
process's log (warnings and errors only).

## Run 1 (`m4-max`): invalid

[`m4-max/`](m4-max/): M4 Max, LLM `Qwen3.8-27B-oQ4e-mtp` on oMLX, default parameters, analyzed
with [`criteria.json`](criteria.json) ([`m4-max/tables.md`](m4-max/tables.md),
[`m4-max/results.json`](m4-max/results.json)). Two of the five validity checks failed, so every
result of run 1 is **invalid**: neither pass nor fail. No result is drawn from it, and its
raw data, analysis and verdict stay as recorded. Its eight `raw/serve-stepNN-<config>.log`
files are not committed, because they carry the checkout's absolute path; each held the same
single warning (checkpoint temperatures outside [0.5, 5.0] replaced, as upstream does) and
nothing else. From run 2 the harness writes these logs with the checkout path replaced by
`<repo>`.

| # | Check | Limit | Run 1 | ok |
|---|---|---|---|---|
| V1 | open-loop client lag, P99, worst window | ≤ 5 ms | 3.13 ms | yes |
| V2 | LLM generating coverage, worst window; LLM errors | ≥ 0.90, 0 | 0.8955 (window 20, `gpu/decisions_llm`; the other 16 LLM windows 0.918–0.989); 0 | **no** |
| V3 | LLM exclusive | every window | every window | yes |
| V4 | `auto` short decisions on the Neural Engine, worst window | ≥ 99% | 95.2% (374 / 393, window 14) | **no** |
| V5 | references on the expected devices | all | all | yes |

Why the two checks failed. Neither is a harness fault or a product fault; both definitions
did not describe the system being measured.
- **V4.** Over the 8 `auto` windows, 3,144 short decisions: 3,008 on the Neural Engine with
  reason `validated_short_single_question_path`, and 136 (4.3%) on the GPU with reason
  `ane_backlog_shorter_on_gpu`. No other reason occurs. That is the product scheduler's
  designed spill ([`docs/no-silent-fallback.md`](../../docs/no-silent-fallback.md), row 2): a
  short request goes to the GPU when the Neural Engine backlog is the longer wait. The 99%
  limit assumed no spill at 8 req/s.
- **V2.** The LLM server runs one request at a time, and this 27B model takes about 6 s to
  first token (server TTFT, [`m4-max/tables.md`](m4-max/tables.md)). `generating_coverage`
  counts only the time tokens were streaming, so a request's prefill between the other
  stream's requests counts as idle although the server is busy with it.

## Run 2: preregistered

Criteria: [`criteria-r2.json`](criteria-r2.json). **Written after run 1's data was seen**, and
before any run-2 data. It changes **only** the V2 and V4 definitions below, for the reasons in
Run 1. Every result criterion and limit (G1, G2, L1, L2, C1, E1) is unchanged, as are V1, V3,
V5, the workload, the cells, the ABBA order, the windows and the warm-up. Run 1 is not
re-analyzed under these definitions.

| # | Check | Limit |
|---|---|---|
| V2 | LLM server busy: fraction of the window during which at least one of the harness's LLM requests is in flight at the server, each request from its start (`start_ns`) to its last chunk, so prefill counts; and LLM errors, every LLM window | ≥ 0.90, 0 |
| V4 | serve `auto` routed short decisions by the product policy's short path: on the Neural Engine, **or** on the GPU with reason `ane_backlog_shorter_on_gpu`. A short decision on the GPU for any other reason counts against it | ≥ 99% per window |

- Both are computed from fields every raw window file already records (per-request
  `start_ns` and chunk times; per-decision `device` and `reason`).
- **Reported, not gated:** generating coverage (run 1's V2 metric) per LLM window, and the
  spill share (short decisions on the GPU with `ane_backlog_shorter_on_gpu`) per `auto`
  window.
- **L1 and L2 include the spilled requests.** The short-decision P99 is over every `short_1q`
  answer of the `auto` windows, wherever it ran; it is the latency a client sees.
- The run directory is `m4-max-r2/`. `run.sh` copies the criteria file it runs under into
  the run directory (`criteria.json` there), and `analyze.py` uses that copy.
