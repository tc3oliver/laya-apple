# User guide

The README covers what laya-apple is and why. This page covers how to use it:
- the question schema;
- devices and routing;
- concurrent serving;
- building, managing and moving ANE artifacts;
- calibrating another machine;
- the failure policy, the CLI, offline use and runtime diagnostics.

## Question schema

`questions` is a dict of `{name: {"type": ..., "instructions": ..., "criteria": ...}}`,
matching upstream Laya's schema:

| `type` | `criteria` | Meaning |
|---|---|---|
| `choice` | non-empty dict of `{label: description}`, or a list of unique label strings | pick one label |
| `score` | non-empty list of level descriptions | pick a level, 0-indexed |
| `noul` | optional dict `{"false": ..., "true": ...}` | yes/no |

The rules follow upstream Laya 0.3.20:

- **`noul` criteria keys** are case-insensitive (`"True"` reads as `"true"`). Any key other
  than `true`/`false` raises `InvalidRequestError`.
- **`noul` labels.** An optional `"labels": {"false": "B", "true": "A"}` changes only the
  option text the model reads (by default `false:` and `true:`). The labels must be distinct,
  non-empty strings. `noul` is still P(true). `labels` on a `choice` or `score` question
  raises `InvalidRequestError`.
- **Non-string `instructions`** (a dict or list) are sent to the model as JSON, with
  non-ASCII text kept as it is.
- **A conversation `context`** (a list of turns) that exceeds the checkpoint's `max_len`
  keeps its newest turns: it is truncated from the start. Any other context keeps its start.
- **`questions={}`** returns empty `answers` with zero usage. Nothing is tokenized, routed or
  run, so `result.runtime` records `backend="none"`, `device="none"` and the routing reason
  `no_questions`, and workers mode emits no trace. This holds for `device="ane"` too: no
  device was used, so nothing fell back.

A list of plain strings is also accepted as a convenience; each string
becomes a `noul` question named after itself:

```python
model.predict(context="...", questions=["Does the customer request a refund?"])
```

Every answer has upstream's fields in upstream's order:

- `choice`: `type`, `choice`, `probabilities`, `confidence`, `answer_confidence`, `action`;
- `score`: `type`, `score`, `legend`, `probabilities`, `confidence`, `answer_confidence`,
  `action`;
- `noul`: `type`, `noul`, `confidence`, `answer_confidence`, `action`.

`answer_confidence` is max(p), the calibrated probability of the reported answer. It is the
quantity temperature scaling fits, so it is the one to gate on across question types. On
`choice` and `score`, `confidence` is the normalised entropy 1 − H(p)/log(k), a different
quantity. On `noul` the two are equal. All values are rounded to 4 decimals.

**Known limitation: option order.** For `choice` questions, the decision can depend on
the order of the `criteria` dict. This is a property of the upstream Laya model, not a
laya-apple defect: laya-apple's MLX FP32 backend reproduces upstream's decision for every
permutation exactly (see `research/option-order/README.md`). Keep option order fixed
across calls for a given question if you need repeatable decisions.

## Devices

`Laya.from_pretrained(model_id, device="auto" | "gpu" | "ane", ...)`.

- `device="gpu"`: MLX only. Every request runs on the GPU.
- `device="ane"`: Core ML / ANE only. A request that does not fit a validated
  artifact fails loudly (`UnsupportedShapeError`, `ArtifactMissingError`, etc.)
  — it never runs on MLX instead.
- `device="auto"` (default): MLX for everything, except a request routes to
  the ANE when **all** of the following hold:

  1. the request has exactly one question;
  2. the longest prompt row fits one of the model's auto-ANE buckets;
  3. a validated artifact for that bucket is present in the cache and passes
     its placement/parity checks at load;
  4. the platform profile matches a validated profile (currently only Apple
     M4 Max / macOS 26.6.2 / coremltools 9.0).

  Otherwise the request runs on MLX. Every result records exactly one
  routing reason in `result.runtime.routing_reason`:

  | Reason | Meaning |
  |---|---|
  | `gpu_requested` | `device="gpu"` was requested |
  | `ane_requested` | `device="ane"` was requested |
  | `validated_short_single_question_path` | auto chose the ANE |
  | `multiple_questions` | auto chose MLX: more than one question |
  | `sequence_exceeds_ane_auto_range` | auto chose MLX: the longest row is above the auto buckets |
  | `ane_artifact_unavailable` | auto chose MLX: no validated artifact for the bucket |
  | `ane_runtime_unavailable` | auto chose MLX: coremltools is not installed |
  | `platform_not_validated` | auto chose MLX: unknown hardware/OS profile |
  | `no_questions` | `questions={}`: nothing ran, on any `device` |

Auto-ANE buckets, generated from measured evidence
(`laya_apple/data/routing.json`):

| Model | Buckets for explicit `device="ane"` | Buckets used by `device="auto"` |
|---|---|---|
| `laya` | 64, 96, 128 | 64, 96, 128 |
| `laya-multilingual` | 64, 96, 128, 256 | 64, 96, 128 |
| `laya-typed-decisions` | 64, 96, 128 | 64, 96, 128 |

Longer validated lengths lose to MLX on latency and are not offered by
either explicit `device="ane"` or `auto`. Multi-question requests are never
auto-routed to the ANE: MLX batching wins at every measured length for 4 and
8 questions, and 2–3 questions were not measured, so they route to MLX
conservatively.

## Concurrent GPU + ANE execution

By default (`execution="inline"`) a `Laya` instance runs one request at a time in the
calling process. For serving many requests, use `execution="workers"`:

```python
from laya_apple import Laya

with Laya.from_pretrained("convaiinnovations/laya-typed-decisions", execution="workers") as laya:
    future = laya.submit(context="...", questions={...})        # concurrent.futures.Future
    result = laya.predict(context="...", questions={...})       # thread-safe, blocking
    # in asyncio code: result = await laya.apredict(context="...", questions={...})
```

- MLX (GPU) runs in its own worker process.
- Core ML (ANE) runs either on a dedicated thread in your process or in its own worker
  process. The choice is made per model from measurements (`ane_placement="auto"`), and
  you can override it with `"thread"` or `"process"`.
- Your process builds prompts, routes, and formats answers.
- The two devices serve requests at the same time. Each device runs its own queue in
  arrival order.
- `close()` (or leaving the `with` block) finishes queued work and stops the worker.

**Why this placement, and its limits.** Every alternative was measured (see
[`research/v0.2-concurrency/`](../research/v0.2-concurrency/)):
- Both backends in one interpreter cost the GPU 9–11% of its throughput.
- With both devices busy, a device whose requests arrive from another process runs its
  host-side work 4–6× slower.
- Core ML's Python `predict` holds the GIL for much of an ANE call, so running the ANE in
  your process costs more at high short-request rates.
- Results on the Phase -1 mix with the chosen placement (`benchmarks/v0.2.md`):

  | Model | Short-stream P99 vs solo | Long stream throughput vs solo | Aggregate vs GPU-only |
  |---|---:|---:|---:|
  | laya-typed-decisions | +8% | −11% | 4.6× |
  | laya | +5% | −12% | 2.9× |
  | laya-multilingual | +99% | −13% | 3.6× |

- Under open-loop load, short requests see 2.5–66× lower P99 than with GPU-only serving.
- Complete isolation (each stream within 10% of its solo P99) was not reached on this
  platform.

**Routing with queues.** On an idle machine, `auto` behaves exactly like `inline`. When a
device is busy, the router compares expected completion times: the device's backlog plus
the request's measured service time. Two additional reasons can then appear:

| Reason | Meaning |
|---|---|
| `ane_backlog_shorter_on_gpu` | a short single-question request went to MLX because the ANE queue was longer |
| `gpu_backlog_shorter_on_ane` | a request in the tie band (an explicit-only bucket, e.g. multilingual L256) went to the ANE because the GPU queue was longer |

Long requests and multi-question requests never go to the ANE under `auto`, however busy
the GPU is. Every `RuntimeInfo` records `execution`, `queue_wait_ms` and the backlog
estimates the router saw (`gpu_backlog_ms`, `ane_backlog_ms`).

If the ANE worker process dies:
- the request running on it raises `BackendUnavailableError`;
- later `auto` requests run on MLX with reason `ane_runtime_unavailable`, and a warning is
  issued once;
- explicit `device="ane"` requests keep raising.

### Adaptive ANE execution (in-process ANE, the default since 1.5)

With the ANE on a thread in your process (`laya`, `laya-typed-decisions`), coremltools'
`predict` holds the GIL for much of each ANE call, which delays the GPU worker's completions
in your process. Core ML's asynchronous prediction API with prebound buffers
(`backends/coreml_async.py`) releases it. On a contended Mac, though, that path can fall into
a slow state: its threads sit on the efficiency cores
([`research/coreml-async-transient/`](../research/coreml-async-transient/README.md)) and the
median short latency rises from about 10.0 to 12.3 ms. Adaptive execution (`laya_apple/handoff.py`) uses the fast path while it is healthy and
falls back to the 1.4 path when it is not.

**How it works:**
- Outside a GPU+ANE overlap, every ANE forward runs coremltools' `predict`, as in 1.4.
- **An episode starts** when an ANE forward arrives while the GPU worker has work queued or
  running, or finished it at most 1.0 s earlier.
  - Its first 64 ANE forwards run coremltools' `predict`, which steps over the transient at
    the start of an overlap.
  - Later forwards run the asynchronous path.
- **The breaker:** every completed asynchronous request's prepare time (tokenising and
  layout, `RequestTrace.prepare_ms`) is observed.
  - Three consecutive requests above 0.3 ms open the breaker.
  - From then on, every ANE forward of the episode runs coremltools' `predict`: no retry
    within the episode.
  - A request already running is never cancelled or re-sent.
- **The episode ends,** and the next overlap starts again with 64 synchronous forwards, when
  the GPU has been idle for more than 1.0 s or no ANE forward came for more than 1.0 s.
- **Only real requests move the state:** no background thread, no timer, no sleep, no CPU
  counters, no QoS call and no private API. The breaker needs no `trace=` callback.
- **"GPU active" means this instance's own GPU worker.** GPU work from another process, such
  as a local LLM, does not start an episode. Without laya-apple's own GPU work, the ANE runs
  the 1.4 path anyway.
- **Start-up cost:** it loads a second Core ML model per ANE bucket, plus its load checks.
  That adds to start-up time and memory; neither was measured.

**Measured** on one Apple M4 Max
([`research/coreml-adaptive-breaker/`](../research/coreml-adaptive-breaker/README.md)):
- **Healthy async:** in 154 production episodes of laya and laya-typed-decisions (product
  mix, bursts, repeated transitions and a 66-episode typed soak), every episode stayed on the
  fast path.
  - GPU return was 0.035–0.043 ms, against 4.3 ms (laya) and 8.6 ms (typed-decisions) on the
    1.4 path.
  - Throughput was 1.038–1.042 × the 1.4 path, and the median episode P99 was 0.18–0.53 ms
    lower.
- **The slow state:** none occurred naturally in those runs. The recovery experiment
  covered 12 episodes where the asynchronous path was already slow. In all 12 the breaker
  tripped within 40 ms of watching it, and latency was back to 1.4's within 164–414 ms.

**Control.** `ane_handoff=None` (the default) uses adaptive execution wherever it is eligible
and available:
- It is eligible with `execution="workers"`, `device="auto"` and a model whose measured ANE
  placement is `thread`, not overridden to `process`.
  - `laya-multilingual`, whose ANE runs in a worker process, never uses it, even with
    `ane_placement="thread"`.
- It is available when:
  - the ANE path starts;
  - `pyobjc-framework-CoreML` is importable (it is part of the `ane` extra);
  - Core ML has the asynchronous prediction API;
  - the asynchronous models load and pass their load checks.
- When an eligible instance cannot use it, the instance runs the 1.4 path. The reason is
  warned once per process and recorded in `info()["ane_handoff"]["disabled_reason"]`.

The other values:
- `ane_handoff=False` always runs the 1.4 path.
- `ane_handoff=True` requires adaptive execution:
  - an ineligible configuration raises `ValueError` before anything is downloaded;
  - `ane_startup="background"` raises `ValueError`, because a background start-up cannot
    raise to the caller;
  - an unavailable path raises `BackendUnavailableError` at start-up.

**Correctness.** Both paths run the same verified artifact on `CPU_AND_NE` in FP16; only the
Python call into Core ML differs.
- At load, each asynchronous bucket's outputs must be identical to coremltools' (names,
  shapes, dtypes and values), and it must pass the placement probe's ratio.
- The parity suite runs the asynchronous path against the goldens and requires every output
  to equal coremltools' exactly.
- Every asynchronous predict reads its outputs from the array Core ML returned for that
  predict.

**Failures.** The coremltools path is the production path, not a fallback device: a request
never changes device, compute units, artifact or precision because of adaptive execution.
- If an asynchronous predict fails, or does not complete within 5 s, that forward runs again
  on coremltools' `predict`, still on the Neural Engine. Adaptive execution is then disabled
  with one warning, and every later ANE forward runs the 1.4 path.
- If its own state is ever found inconsistent, it disables itself the same way.
- `info()["ane_handoff"]` reports:
  - the state, the guard, the episode count and the forwards per path;
  - the breaker's count, whether it is open and its trips;
  - whether it is disabled and why, and whether its state is consistent.

## The ANE path: building artifacts

`device="ane"` and the auto-ANE path need a Core ML artifact for the exact
(model, revision, bucket) tuple. Artifacts are never downloaded or
committed — build them locally:

```bash
laya-apple artifacts build laya-typed-decisions
```

This builds every offered bucket for the model (or pass `--length 64` to
build one). Each bucket goes through:

1. **layout** — the channel-first BC1S PyTorch body is checked against the
   upstream FP32 `DecisionModel` on an exact-length row;
2. **conversion** — traced and converted to Core ML with FP16 compute and
   FP16 I/O, fixed shape (`B=1 × L=<bucket>`), macOS 15 deployment target;
3. **compile** — to a `model.mlmodelc` (the intermediate `.mlpackage` is not
   kept, since loading it recompiles for the ANE on every process — 37.8s
   vs. 0.21s for a compiled artifact);
4. **placement** — the Core ML compute plan on `CPU_AND_NE` must show 100%
   of operations on the Neural Engine and 0 device transitions;
5. **parity** — every shipped golden row that fits the bucket, checked
   against the upstream PyTorch FP32 reference, with the unchanged Phase -1
   gate (see [Correctness](correctness.md));
6. **atomic registration** — the artifact is built in a temporary directory
   and renamed into place only after every step above passes. A partial or
   failed build never appears as usable; a failure leaves the manifest under
   `artifacts/rejected/` as evidence.

Building one bucket takes 1.5–3.5 minutes on the tested hardware, including the pre-warm
load at the registered path.

Artifacts are cached under `$LAYA_APPLE_CACHE` if set, else
`$XDG_CACHE_HOME/laya-apple`, else `~/.cache/laya-apple`. They live outside
the source tree and are never committed to Git or redistributed with the
package.

Re-check everything already registered:

```bash
laya-apple artifacts verify
```

This reloads every cached artifact and re-checks its manifest schema,
revision and weight hash, graph variant, bucket, compute units, parity status,
build platform profile, file hashes and Core ML placement.

At runtime the manifest, revision, profile and compute-unit checks run on every
load. The file hash and the placement check (about 1.1 s per bucket together)
run on first load and again whenever the artifact's files, the macOS build or
the coremltools version change; a passing result is stamped under
`<cache>/verified/`. `artifacts verify` ignores the stamp.

**Cold start.** Loading a compiled artifact triggers Core ML's on-device ANE
compile the first time a given artifact location is loaded (25–86 s per bucket
on the tested machine); the system caches the result. `artifacts build` pays
this cost once at the registered path, so later loads take 0.14–0.31 s per
bucket. If the system evicts its cache, the next load pays it again.

## Artifact lifecycle

- **Concurrent builds are safe.** One build per (model, revision, bucket) runs at a time
  across processes, under a file lock in `<cache>/artifacts/.locks/`. A second builder
  waits, then finds the artifact already registered.
- **Corruption is quarantined.** An artifact whose files no longer match their manifest
  hash, or whose manifest cannot be read, is moved to `<cache>/artifacts/quarantine/` when
  it is detected. The error names the rebuild command. Nothing corrupt stays where the
  runtime looks.
- **Cleanup is explicit.** `laya-apple artifacts prune` lists what it would delete and why:
  - other revisions or weights, unregistered models, buckets no longer offered;
  - builds from another platform profile, unvalidated or rejected builds;
  - quarantined entries, abandoned staging directories, orphaned verification stamps.
  Add `--yes` to delete. It only ever deletes inside the cache.
- **Warm after eviction.** `laya-apple artifacts warm MODEL` pays Core ML's on-device ANE
  compile ahead of the first request.

**Moving artifacts between machines.** Artifacts are never downloaded. You can build them
once and carry them to another machine yourself:

```bash
laya-apple artifacts export laya-typed-decisions --out exports/   # one .tar.gz per offered bucket
# on the other machine:
laya-apple artifacts import exports/laya-typed-decisions-L64.tar.gz
```

An import is registered only after the receiving machine has checked, itself:
- the manifest against the pinned checkpoint;
- the build platform profile;
- the file hash;
- the compute plan (100% ANE, 0 transitions);
- the full parity gate against the shipped goldens (the checkpoint must be downloaded).

The local results are recorded in the manifest under `imported`.

### Artifact provenance

`laya-apple artifacts list --capabilities` prints one JSON record per registered
artifact, whatever its manifest carries — a field the manifest does not have comes
back `null`, it never raises. Each record:

| Field | From the manifest |
|---|---|
| `model`, `repo`, `revision` | `source.model`, `source.repo`, `source.revision` |
| `source_weights_sha256` | `source.weights_sha256` |
| `conversion_revision.laya_apple_version` / `.git_revision` / `.code_sha256` | `conversion.laya_apple_version` / `.git_revision` / `.code_sha256` |
| `graph` | `artifact.graph` (the graph variant, e.g. `bc1s-masked`) |
| `bucket`, `batch` | `artifact.length`, `artifact.batch` (the fixed shape) |
| `compute_target.compute_units` / `.ops` / `.transitions` | `placement.compute_units` / `.ops` / `.transitions` |
| `precision` | `artifact.precision` |
| `platform` | `platform` (SoC, macOS, coremltools — the build profile) |
| `parity.passed` / `.tolerance` / `.prob_max_abs` / `.hard_mismatches` | the matching `parity.*` fields |
| `parity.near_tie_flips` | the count of `parity.near_tie_flips`, not the rows themselves |
| `artifact_sha256` | `integrity.artifact_sha256` |
| `offered_by_auto` | whether `bucket` is in the model's `auto_ane_buckets` (§ [Devices](#devices)) |
| `offered_explicit` | whether `bucket` is in the model's `ane_buckets` |

An unsupported request shape (too long, wrong graph, a dynamic length) is never padded
or reshaped to fit a bucket — it raises `UnsupportedShapeError` instead. Only buckets
that passed the parity gate **on this machine** (its SoC, macOS major version and
coremltools version — see [Calibrating another machine](#calibrating-another-machine))
are ever loaded; an artifact built elsewhere fails `verify_profile` rather than running
unverified.

**Cold start.** With `execution="workers"`, `ane_startup="background"` makes
`from_pretrained` return as soon as MLX is ready. The ANE finishes loading behind it,
including any on-device compile. Until it is ready, `auto` routes to MLX with reason
`ane_starting`. `laya.wait_for_ane()` blocks until it is ready, and
`info()["ane_ready"]` reports it. Measured start times are in
[`benchmarks/v0.3.md`](../benchmarks/v0.3.md).

## Calibrating another machine

The shipped routing table applies only to the profile it was measured on (SoC, macOS
major version, coremltools version). On any other profile, `auto` uses MLX only
(`platform_not_validated`). To enable the ANE on your machine:

```bash
laya-apple artifacts build laya-typed-decisions   # builds and parity-validates here
laya-apple calibrate laya-typed-decisions          # measures MLX and ANE latency here
```

`calibrate` applies the same rule that produced the shipped table to local measurements.
It writes `<cache>/profiles/<profile>.json`. Such a local profile is used only when no
shipped profile matches the machine. `Laya.info()["routing_profile"]` reports which table
is in effect: `shipped`, `local:<path>`, or `None` (MLX only).

## No silent fallback

- An explicit `device="ane"` request runs on the exact validated artifact
  with the exact validated compute units, or it raises. It never silently
  runs on MLX, CPU, or a different Core ML placement.
- Under `device="auto"`, a decision to use MLX instead of the ANE is made
  **before** the request runs, and the reason is recorded in
  `result.runtime.routing_reason`. It is never a reaction to Core ML
  changing placement mid-flight.
- The Core ML compute plan is checked against the declared placement (100%
  ANE, 0 transitions) at build time, at first load, and again after any change
  to the artifact files, the macOS build or coremltools. A mismatch raises
  `ComputeUnitMismatchError` instead of silently running on CPU.

### Failure policy

Every failure the runtime can detect is a specific exception from
`laya_apple.errors`:

| Condition | Error |
|---|---|
| model not registered | `UnsupportedModelError` |
| malformed questions | `InvalidRequestError` |
| ANE requested, request exceeds the largest offered bucket, or the model/shape pair is not validated | `UnsupportedShapeError` |
| ANE requested, no artifact in the cache | `ArtifactMissingError` |
| artifact built from another revision or other weights | `ArtifactRevisionError` |
| artifact files do not match their manifest hash | `ArtifactIntegrityError` |
| artifact has no passing parity record | `ArtifactParityError` |
| requested compute units differ from the validated ones, or the loaded compute plan is not 100% ANE / 0 transitions | `ComputeUnitMismatchError` |
| MLX or coremltools unavailable for the requested device | `BackendUnavailableError` |
| offline and not cached | `BackendUnavailableError` (with download instructions) |

## CLI reference

The `--offline` flag is **global** and must come before the subcommand:

```bash
laya-apple --offline predict laya-typed-decisions --context "..." --questions '{...}'
```

Commands:

```text
laya-apple predict MODEL --context TEXT --questions JSON [--device auto|gpu|ane] [--dtype float16|float32]
laya-apple info [MODEL]
laya-apple download MODEL...
laya-apple artifacts build MODEL [--length L ...] [--force] [--skip-existing]
laya-apple artifacts list [--capabilities]
laya-apple artifacts verify [MODEL] [--length L ...]
laya-apple artifacts warm [MODEL] [--length L ...]
laya-apple artifacts prune [--yes]
laya-apple artifacts export MODEL [--length L ...] [--out DIR]
laya-apple artifacts import ARCHIVE.tar.gz [--force]
laya-apple calibrate [MODEL ...] [--warmup N] [--iters N]
laya-apple parity MODEL [--device gpu|ane] [--dtype float16|float32]
laya-apple benchmark MODEL [--device auto|gpu|ane] [--lengths L ...] [--questions N] [--warmup N] [--iters N] [--output FILE]
laya-apple switchyard [--seed N] [--duration S] [--out DIR] [--no-open] [--setup-ane] [--replay DIR]
laya-apple serve [--model auto|MODEL] [--preload MODEL ...] [--host H] [--port P] [--device auto|gpu|ane] [--allow-remote] [--log-level L]
```

`serve` runs the local Jev-compatible decision server; see [`serve.md`](serve.md).

`--context` and `--questions` each accept inline text/JSON, `@file` to read
from a file, or `-` to read from stdin.

Example:

```bash
laya-apple --offline predict laya-typed-decisions \
  --context "The customer was charged twice." \
  --questions '{"refund": {"type": "noul", "instructions": "Does the customer request a refund?"}}'
```

```bash
laya-apple --offline info laya-typed-decisions
```

## Switchyard

The easiest way to see heterogeneous serving is to run it. `laya-apple switchyard` runs the
frozen `switchyard-v1` workload (bursty seeded arrivals, single-question routing decisions
as "trains", background long-context requests loading the GPU), then replays the recorded
run as a rail yard in your browser. All measurement happens headless, before anything
opens; the browser only replays the recorded requests.

```bash
uvx laya-apple switchyard
```

The first run downloads the pinned `laya-typed-decisions` checkpoint (about 800 MB) and runs
offline after that. A standard run (60 s timed window, plus a 5 s warmup, for one or two
rounds) takes about 2–3 minutes once the checkpoint is cached.

What you get depends on Neural Engine state:

- **No built ANE artifact:** `switchyard` measures a `gpu_only` round and prints a setup
  command:

  ```bash
  uvx --from "laya-apple[convert]" laya-apple switchyard --setup-ane
  ```

  This builds and parity-validates the ANE artifact, calibrates this machine if needed, then
  runs `switchyard` again. Missing ANE never fails the command — it runs `gpu_only` and shows
  why.
- **ANE ready:** `switchyard` measures `gpu_only` and `hybrid` (GPU + ANE) rounds on the
  identical seeded timetable, and the result card compares them: late trains, P99 decision
  latency, P99 queue wait.

Useful flags: `--seed`, `--duration`, `--out DIR` (default:
`./switchyard-results/<timestamp>-<soc>/`), `--no-open` (skip opening the browser),
`--replay DIR` (rebuild `replay.html` from a previous run's `result.json` and `trace.jsonl`
without re-measuring).

The workload, measurement boundaries, round order and result schema are frozen and described
in [`docs/switchyard.md`](switchyard.md). The official measured campaign is in
[`benchmarks/switchyard/README.md`](../benchmarks/switchyard/README.md).

## Offline use

Once checkpoints (and, if used, ANE artifacts) are cached, no network access
is needed. Three equivalent ways to force this:

- `Laya.from_pretrained(model_id, local_files_only=True)`
- `HF_HUB_OFFLINE=1` in the environment
- `laya-apple --offline <subcommand> ...` on the CLI

Running offline against an uncached checkpoint raises
`BackendUnavailableError` with download instructions instead of hanging or
silently going online.

## Runtime diagnostics

Every `Result.runtime` (a `RuntimeInfo`) records:

- `backend` (`mlx` / `coreml`) and `device` (`gpu` / `ane`);
- `model` and `model_revision`;
- `sequence_length` (longest prompt row, in tokens) and `question_count`;
- `routing_reason`;
- `artifact_revision` (the ANE artifact hash, or `mlx:<weights sha256 prefix>`);
- `compute_units` and `buckets` (Core ML only);
- `dtype`;
- `latency_ms`, from the call to the result (queueing included);
- `execution` (`inline` / `workers`); with workers, `queue_wait_ms` and the
  `gpu_backlog_ms` / `ane_backlog_ms` estimates the router used.

