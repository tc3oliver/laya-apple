# laya-apple

**English** | [繁體中文](README.zh-TW.md) | [简体中文](README.zh-CN.md)

[![CI](https://github.com/tc3oliver/laya-apple/actions/workflows/ci.yml/badge.svg)](https://github.com/tc3oliver/laya-apple/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/laya-apple)](https://pypi.org/project/laya-apple/)
![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

**Run [Laya](https://github.com/NandhaKishorM/laya) on your Mac's GPU and Neural Engine at
the same time, and use the Neural Engine only where it gives upstream's answers.**

Laya answers typed questions about a context (`choice`, `score`, `noul`) in one forward pass.
laya-apple runs upstream Laya on both engines and picks one per request.

New in 1.6.3: a first `artifacts fetch` compiles once instead of twice. One L128 import took
about 75 s instead of 146 s on an M4 Max with macOS 27.0 ([release notes](docs/releases/v1.6.3.md)).

## Why use it

- **Correct ANE results, not just fast ones.** On the tested Mac, a plain Core ML export ran
  on the Neural Engine (ANE) without a single error and still disagreed with upstream on up to
  85 decisions. laya-apple uses an ANE artifact only after it passes a parity gate on your
  machine ([Correctness](#correctness)).
- **Short decisions stop queueing behind long work.** Single-question requests go to the ANE
  when a validated artifact is present; long and multi-question ones stay on the GPU. With
  `execution="workers"` both engines serve at once. Measured on one Apple M4 Max; watch it
  with `uvx laya-apple switchyard`
  ([Switchyard](#see-it-yourself-switchyard), [GPU + ANE benchmarks](#gpu--ane-benchmarks)).
- **A local stand-in for the Jev API.** `laya-apple serve` answers existing Jev clients with
  upstream Laya on your Mac, without changes to the client
  ([Local Jev-compatible server](#local-jev-compatible-server)).
- **Every routing decision is explained.** Each result records its device and
  `routing_reason`. Ask for the ANE explicitly and you get the validated artifact or an
  exception, never a silent fallback ([How it works](#how-it-works)).

## Install

Apple silicon, Python 3.11–3.13:

```bash
pip install 'laya-apple[ane]'   # MLX GPU + the Neural Engine runtime
```

| Extra | Adds |
|---|---|
| (none) | The MLX GPU runtime only |
| `ane` | The Neural Engine runtime (coremltools 9.0, pyobjc-framework-CoreML) |
| `convert` | Building ANE artifacts on this Mac (torch 2.7.0); not needed for `artifacts fetch` |
| `serve` | `laya-apple serve`, the local Jev-compatible server |

Without the `ane` extra, or without a built ANE artifact, everything runs on the MLX GPU. To
run from source or develop laya-apple, see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Quickstart

```python
from laya_apple import Laya

model = Laya.from_pretrained("convaiinnovations/laya-typed-decisions", device="auto")

result = model.predict(
    context="The customer was charged twice for the same invoice and is frustrated.",
    questions={
        "urgency": {
            "type": "choice",
            "instructions": "How urgent is this?",
            "criteria": ["low", "medium", "high"],
        }
    },
)
rt = result.runtime
print(result.answers["urgency"]["choice"])
print(rt.device, rt.routing_reason, f"{rt.latency_ms:.1f} ms")
```

On the tested machine:

```text
high
ane validated_short_single_question_path 11.2 ms
```

- The first call downloads the pinned checkpoint. After that it works offline
  (`local_files_only=True`).
- Without an ANE artifact the same request runs on MLX, and `routing_reason` says why.
- On Apple M4 Max with macOS 26 or 27 and coremltools 9.0, fetch a prebuilt artifact; it is
  validated on your Mac: `laya-apple artifacts fetch laya-typed-decisions`.
- Anywhere else, build one (needs the `convert` extra):
  `laya-apple artifacts build laya-typed-decisions`.
- Probabilities, other question types and the full API: [`examples/`](examples/), the
  [user guide](docs/guide.md) and [`docs/api.md`](docs/api.md).

For concurrent GPU + ANE serving, use `execution="workers"` and submit requests from any thread:

```python
with Laya.from_pretrained("convaiinnovations/laya-typed-decisions", execution="workers") as model:
    futures = [model.submit(context=c, questions=q) for c, q in requests]   # thread-safe
```

### Fetch validated ANE artifacts, then route by language (1.6)

```bash
pip install -U 'laya-apple[ane]'
laya-apple artifacts fetch laya                  # prebuilt ANE artifacts, validated here
laya-apple artifacts fetch laya-multilingual
```

```python
from laya_apple import Laya

with Laya.from_pretrained("auto") as model:       # laya or laya-multilingual, per request
    result = model.predict(context=context, questions=questions)
    rt = result.runtime
    print(rt.model, rt.model_routing, rt.device, rt.routing_reason)
```

- **Language routing.** `Laya.from_pretrained("auto")` returns a `LayaRouter` that loads
  `laya` and `laya-multilingual` and picks one per request by the language of the context,
  with the same function `laya-apple serve --model auto` uses. Why is in
  `RuntimeInfo.model_routing` ([`docs/api.md`](docs/api.md)).
- **Prebuilt ANE artifacts, checked on your Mac.** `laya-apple artifacts fetch MODEL`
  downloads a prebuilt artifact from
  [`tc3oliver/laya-apple-artifacts`](https://huggingface.co/tc3oliver/laya-apple-artifacts)
  and registers it through the validating `artifacts import`: SHA-256 against the index,
  manifest and platform profile, compute plan, the full FP16 parity gate and the placement
  probe, all on the receiving machine. A download is trusted no more than a local build.
  Fetch reads that repository at a pinned, validated commit, never its mutable `main`. You
  skip the local build, and with it PyTorch and the `convert` extra. On a first install those
  checks run at the registered path, so the on-device compile is paid once, not twice: one
  laya-typed-decisions L128 import took about 75 s instead of 146 s (Apple M4 Max, macOS 27.0,
  [`research/import-compile-once/results.md`](research/import-compile-once/results.md)).
- **`predict` as before; `predict_shortlist` opt-in.** For `choice` questions with many
  labels, `predict_shortlist(..., embed_fn, k=20)` keeps the `k` labels most similar to the
  request, then runs one `predict`. `predict` itself is unchanged
  ([`examples/auto_fetch_shortlist.py`](examples/auto_fetch_shortlist.py)).

Limits:
- **Prebuilt artifacts exist for two platform profiles: Apple M4 Max with coremltools 9.0, on
  macOS 26 and (since 1.6.2) macOS 27.** Every other Mac gets `ArtifactMissingError` naming
  the build command, and builds locally as before (`laya-apple artifacts build MODEL`,
  `convert` extra).
- **The published artifacts were checked by a clean-cache download** on each profile: fetch,
  verify, parity and placement passed for all 10 model/bucket pairs
  ([macOS 26.6.2](benchmarks/prebuilt-artifacts-1.6.0.md),
  [macOS 27.0](benchmarks/prebuilt-artifacts-1.6.2-macos27.md)). Both checks ran on the build
  machine with an empty cache. Every receiving machine repeats the
  integrity, platform, parity and placement checks before it registers an artifact.
- **The first Core ML load still compiles on the device:** about 4.5 minutes on macOS 26. In
  one run, laya-typed-decisions (buckets 64/96/128) took 273.5 s cold at a new location against
  2.7 s warm ([`research/coreml-compile-cache/screen.md`](research/coreml-compile-cache/screen.md));
  about 210 s on macOS 27.0
  ([`research/coreml-compile-cache/results.md`](research/coreml-compile-cache/results.md)).
  Fetching does not remove it: the import pays it once, at the registered path.
  `ane_startup="background"` serves on MLX in the meantime.

Smaller 1.6 changes: an MLX fast path (a token-id cache on by default; `mx.compile`
and length-bucketed batching opt-in; see [Limitations](#limitations) for what was measured),
and W8 ANE research that ships nothing ([Limitations](#limitations)). Release notes:
[`docs/releases/v1.6.0.md`](docs/releases/v1.6.0.md).

## GPU + ANE benchmarks

Except for the community results, every number in this section comes from one Apple M4 Max
([Community benchmarks](#community-benchmarks)).

### See it yourself: Switchyard

```bash
uvx laya-apple switchyard
```

![Switchyard: the same recorded timetable replayed GPU-only, where trains queue at red signals during rush hour, and GPU + ANE, where they flow into their platforms; then the result card: 1,407 of 1,422 trains late GPU-only against 0 with GPU + ANE](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/media/switchyard-launch.gif)

**Every train is a real Laya decision** ("which platform is clear?"). A red signal means the
train is waiting for the model's answer; a train is late when it has no answer within 100 ms.
Rush hour adds bursts of load, with long background requests on the GPU.

| Same timetable, same model | MLX GPU only | MLX GPU + Neural Engine |
|---|---:|---:|
| Late trains | 1,407–1,408 / 1,422 | **0 / 1,422** |
| P99 decision latency | 3,107.7–3,170.6 ms | **54.5–54.7 ms** |
| P99 queue wait | 3,095.9–3,158.8 ms | **39.6–42.8 ms** |

- Three standard runs on one Apple M4 Max (macOS 26.6.2, `laya-typed-decisions`). In every
  run both rounds gave the same answer for every train.
- Most of the gap comes from short decisions no longer waiting in the GPU queue while the long
  requests keep running there.
- The benchmark runs headless; the animation is a replay, not the measurement. These numbers
  are not comparable with the v1.0 results below (different rate, boundary and workload).

First-run download, setup, method and raw data: [`docs/switchyard.md`](docs/switchyard.md)
and [`benchmarks/switchyard/README.md`](benchmarks/switchyard/README.md).

### Original heterogeneous serving benchmark (v1.0)

These results showed the value of GPU + ANE serving and were measured before 1.5 adaptive
execution.

| Model | Throughput vs GPU-only | Short P99, GPU-only | Short P99, GPU + ANE |
|---|---:|---:|---:|
| laya | **2.92×** | 1538.0 ms | **108.5 ms** |
| laya-multilingual | **4.34×** | 2052.3 ms | **29.6 ms** |
| laya-typed-decisions | **4.57×** | 1592.9 ms | **79.5 ms** |

One Apple M4 Max, a short and a long request stream through one `Laya(execution="workers")`.
Short P99 is measured from arrival under open-loop bursty load, so queueing counts. The gain
comes from running both engines at once, not from raw ANE speed. Method and raw data:
[`benchmarks/v1.0.md`](benchmarks/v1.0.md).

### Adaptive ANE execution (1.5)

With the ANE on a thread in the same process, the synchronous Core ML call holds the GIL for
much of each prediction, so a GPU request that had already finished could not hand its result
back until that call returned
([`research/coreml-gil-completion-path/`](research/coreml-gil-completion-path/README.md)).
In 1.5, laya and laya-typed-decisions run eligible ANE requests
through asynchronous Core ML execution, and on a sustained slowdown the runtime falls back to
the known-safe 1.4 synchronous path. With `execution="workers"` and `device="auto"` it is on by
default and needs no other code change.

![The GPU finishes but its result waits on the GIL held by the synchronous Core ML predict; with asynchronous Core ML it goes straight through, GPU result return 8.60 to 0.043 ms; then a recorded run where the asynchronous path turns slow, and a controlled recovery test where 1.5 detects it, falls back to the 1.4 path and recovers: 12 of 12 within 164–414 ms](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/media/release15-social.gif)

The first 10 s are a schematic. Sources for every number on screen:
[`docs/media/release15-social.md`](docs/media/release15-social.md).

| vs the 1.4 path (one Apple M4 Max) | laya | laya-typed-decisions |
|---|---:|---:|
| GPU return (P50) | 4.28–4.29 → **0.035–0.037 ms** | 8.60 → **0.043 ms** |
| Throughput | **1.042×** | **1.038×** |

*GPU return* is the time from the GPU worker finishing a request to its result reaching the
caller, not GPU compute time. On the 1.4 path, finished work waited 4.3–8.6 ms, almost all
of it for the GIL.

- **Validation.** 154 production validation episodes (76 laya, 78 typed-decisions, including
  bursts and soaks). Every episode stayed on the asynchronous path after the handoff, with no
  mismatches, routing failures, lost requests or crashes. No slow state occurred in these
  runs, so the fallback never triggered in them
  ([`val_tables.md`](research/coreml-adaptive-breaker/val_tables.md)).
- **Recovery, a separate controlled experiment.** In 12 of 12 episodes where the asynchronous
  path was already slow, the runtime detected it within 40 ms and was back to the 1.4 path's
  latency within 164–414 ms
  ([`phase1_tables.md`](research/coreml-adaptive-breaker/phase1_tables.md)).

The full story, from the GIL diagnosis to the adaptive breaker:
[`research/coreml-gil-completion-path/`](research/coreml-gil-completion-path/README.md),
[`research/coreml-adaptive-breaker/`](research/coreml-adaptive-breaker/README.md) and the
[1.5 release notes](docs/releases/v1.5.0.md).

### Community benchmarks

Every benchmark in this README ran on an M4 Max. The
[community matrix](docs/community-benchmarks.md) keeps results from other Macs separate, and
already has an M4 Pro, an M4 and an M2 Pro (MLX only). One command adds yours:

```bash
uv run python scripts/hardware_report.py --quick
```

The full steps, from `git clone` to the pull request, are in the
[guide](docs/community-benchmarks.md#add-your-mac).

## Local Jev-compatible server

`laya-apple serve` is a local stand-in for the Jev API. It serves the same API
(`POST /v1/systemone`) on loopback and answers with upstream Laya, running on your Mac. Point
an existing Jev client at it through its base-URL setting; the client's code does not change.

```bash
pip install 'laya-apple[serve,ane]'
laya-apple serve                                   # http://127.0.0.1:8642
export TYPESAFE_BASE_URL=http://127.0.0.1:8642     # then run your Jev client as usual
```

![A terminal: laya-apple serve starts locally; the released typesafe-sdk 0.7.1 for Python, unmodified, is pointed at it with TYPESAFE_BASE_URL and gets the answer fix_code; the same request with curl shows that laya-apple answered it with the laya checkpoint on the Apple Neural Engine](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/readme/serve-demo.svg)

- **Tested with 7 unmodified Jev clients** at their released versions, including the Python
  and JS SDKs and two Claude Code plugins
  ([`integrations/jev-plugins/README.md`](integrations/jev-plugins/README.md)).
- **The answers are Laya's, not Jev's.** 792 of 792 requests matched unmodified upstream
  `laya.serve` 0.3.20 within the FP16 parity gate
  ([`benchmarks/serve-compat/`](benchmarks/serve-compat/README.md)). No claim is made about
  accuracy relative to Jev.

API keys, models, security and client caveats: [`docs/serve.md`](docs/serve.md).

### Earlier serve benchmark

Measured on the 1.3 path with a 27B local LLM generating at saturation on the same Mac,
short-decision P99 was **47.2 ms** with heterogeneous `auto` serving against **122.3 ms**
GPU-only (one run on one M4 Max, `--model laya`). Run 3, on laya-apple 1.5.0 with its
defaults, measured **41.7 ms** (`auto`) against **79.5 ms** (`--device gpu`), and every
preregistered criterion passed. Adaptive execution was enabled, but its asynchronous path
engaged in 0 of 3,222 Neural Engine forwards, so run 3 measures 1.5 serve as shipped, not the
asynchronous path ([`benchmarks/serve/m4-max-r3/tables.md`](benchmarks/serve/m4-max-r3/tables.md)).
The runs are separate campaigns; compare them only descriptively.
LLM throughput cost, method and limits:
[`docs/serve.md`](docs/serve.md#beside-a-local-llm).

## How it works

A Mac has two engines that can run Laya, and each is faster for different requests.

![Requests of at most 128 tokens with one question and a validated artifact go to the Apple Neural Engine; longer, multi-question or unvalidated requests go to the MLX GPU; with execution="workers" both engines serve independent requests concurrently](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/readme/architecture.svg)

1. **Correct ANE execution.** A fast Core ML export is not necessarily a correct one.
   laya-apple uses an ANE artifact only after it passes a parity gate on the machine that uses
   it ([Correctness](#correctness)).
2. **Automatic routing.** The router picks a device before a request runs and records why in
   `routing_reason`. Under load it also compares the two queues' backlogs.
3. **Concurrent GPU + ANE serving.** With `execution="workers"`, the GPU runs in a worker
   process and the ANE on its own dispatcher, so short requests stop queueing behind long ones.
4. **Adaptive ANE execution (1.5).** After a conservative handoff at the start of each GPU + ANE
   overlap, eligible ANE requests use asynchronous Core ML, so finished GPU work is not held
   back by the synchronous path's long GIL hold. Per-request timing tells the runtime when
   that path slows down; a sustained slowdown sends the rest of the overlap back to the
   known-safe synchronous path. `ane_handoff=False` turns it off
   ([guide](docs/guide.md#adaptive-ane-execution-in-process-ane-the-default-since-15)).

| Request | Goes to | Why (measured on the tested Mac) |
|---|---|---|
| One question, ≤ 128 tokens, validated artifact present | **ANE** | Faster: laya-typed-decisions L128 takes 9.9 ms on the ANE against 12.2 ms on MLX (forward P50) |
| Longer context | **MLX GPU** | MLX is faster there: 19.2 ms at L256 and 71.0 ms at L1024 |
| Several questions | **MLX GPU** | MLX batches the questions; the ANE runs them one at a time |
| Unvalidated Mac, missing artifact, or no Core ML | **MLX GPU** | Recorded as `platform_not_validated`, `ane_artifact_unavailable` or `ane_runtime_unavailable` |

Full routing thresholds and calibration evidence: [`docs/support-matrix.md`](docs/support-matrix.md).

Completed requests can emit a `RequestTrace` (routing decision, queue, service and response
timing) through `trace=` ([`docs/api.md`](docs/api.md)). Adaptive execution watches the same
per-request timing; you do not need to pass `trace=` for it. Architecture:
[`docs/architecture.md`](docs/architecture.md).

### Supported models and platforms

| Model | max_len | MLX GPU | ANE buckets (explicit) | ANE buckets used by `auto` | Adaptive ANE execution |
|---|---:|---|---|---|---|
| [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) | 512 | FP16 / FP32, any length | 64, 96, 128 | 64, 96, 128 | Yes |
| [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual) | 1024 | FP16 / FP32, any length | 64, 96, 128, 256 | 64, 96, 128 | No (worker-process ANE) |
| [`convaiinnovations/laya-typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions) | 1024 | FP16 / FP32, any length | 64, 96, 128 | 64, 96, 128 | Yes |

Adaptive ANE execution applies with `execution="workers"` and `device="auto"`. Validated on one Apple M4 Max with macOS 26.6.2.
Shipped routing profiles cover the Apple M4 Max with coremltools 9.0 on macOS 26.6.2 and
macOS 27.0, so `auto` uses the ANE there once the artifacts are built or fetched. The macOS 27 profile
matches every 27.x, but only 27.0 was measured. On macOS 27 only build,
parity, routing and the prebuilt fetch check were measured; adaptive execution, `serve` and the release benchmarks were
not ([`benchmarks/routing-macos27/`](benchmarks/routing-macos27/README.md)). On other Macs,
`auto` stays on MLX until ANE artifacts are built and calibrated there (`laya-apple
calibrate`). Prebuilt artifacts (`laya-apple artifacts fetch`) exist only for Apple M4 Max with
coremltools 9.0, on macOS 26 or 27. Details: [`docs/compatibility.md`](docs/compatibility.md).

## Correctness

Parity against upstream Laya on PyTorch CPU FP32, over the shipped golden rows (v1.0).
Each cell gives hard mismatches, then the max probability error.

| Implementation on the tested Mac | laya | laya-multilingual | laya-typed-decisions |
|---|---|---|---|
| Ordinary Core ML export · `CPU_AND_NE` | ❌ 12, 0.56 | ❌ 85, 1.0 | ❌ 19, 0.42 |
| Ordinary Core ML export · `CPU_AND_GPU` | ✅ 0, 0.0066 | ✅ 0, 0.0059 | ✅ 0, 0.0028 |
| **laya-apple MLX FP16** | ✅ 0, 0.0037 | ✅ 0, 0.0045 | ✅ 0, 0.0017 |
| **laya-apple ANE FP16** | ✅ 0 (1 near-tie), 0.012 | ✅ 0, 0.013 | ✅ 0, 0.0077 |

- **The FP16 gate:** probability error ≤ 0.02 and 0 hard mismatches. Near-tie flips are
  listed, not hidden.
- **No silent fallback:** an explicit ANE request runs the validated artifact or raises.
- **The 1.5 asynchronous path** must match coremltools' output exactly on the goldens.

Definitions, every configuration tested and the fallback audit:
[`docs/correctness.md`](docs/correctness.md) and
[`docs/no-silent-fallback.md`](docs/no-silent-fallback.md).

## Limitations

- **One test machine.** Every benchmark, including the 1.5 validation and recovery runs, is
  from one Apple M4 Max: macOS 26.6.2, plus build, parity and routing on macOS 27.0. Routing
  thresholds are not assumed to hold on other Apple SoCs.
- **Adaptive execution hands off conservatively:** the first ANE requests of every GPU + ANE
  overlap run the 1.4 path. laya-multilingual, whose ANE runs in a worker process, does not
  use it.
- **The asynchronous path of 1.5 adaptive execution is not measured in `laya-apple serve`.**
  Serve uses adaptive execution by default. Its run-3 benchmark beside a local LLM ran with it
  enabled, but every one of 3,222 Neural Engine forwards ran on the 1.4 Core ML path
  ([`benchmarks/serve/m4-max-r3/tables.md`](benchmarks/serve/m4-max-r3/tables.md)).
- **Long and multi-question requests stay on the GPU,** which is faster for them. The ANE
  path is batch 1 only.
- **Isolation is partial.** Under concurrency, each stream's P99 is above its solo value.
- **Cold start** on a fresh artifact location costs 3–5 minutes of Core ML compile per
  model. `ane_startup="background"` serves on MLX in the meantime.
- **Prebuilt artifacts do not remove the cold start.** A fetched artifact still compiles on
  the device at first load: 273.5 s for laya-typed-decisions (buckets 64/96/128) in one run,
  against 2.7 s warm
  ([`research/coreml-compile-cache/screen.md`](research/coreml-compile-cache/screen.md)).
- **Core ML's compile cache grows without eviction:** 0.7–1.4 GB per bucket for every new
  artifact location (a fetch, an import, a moved cache). laya-apple never evicts it
  ([guide](docs/guide.md#artifact-lifecycle)). Core ML keeps it per process name, so another
  process may compile again.
- **Prebuilt artifacts exist for two platform profiles only** (Apple M4 Max with coremltools
  9.0, on macOS 26 and 27), and have been fetched only on the machine that built them, with an
  empty cache ([macOS 26.6.2](benchmarks/prebuilt-artifacts-1.6.0.md),
  [macOS 27.0](benchmarks/prebuilt-artifacts-1.6.2-macos27.md)). The macOS 27 profile matches
  every 27.x, but only 27.0 was measured.
  No independent check on a second machine of either profile has been done yet. We
  recommend one before publishing, but it is not required ([`docs/publishing.md`](docs/publishing.md));
  each receiving machine still validates every artifact before registering it.
- **No quantized artifact ships.** In the W8 research only the laya-typed-decisions L64
  `w8-pt` result is reproduced (0.650 of FP16 latency); laya and laya-multilingual failed the
  parity gate ([`research/ane-w8/README.md`](research/ane-w8/README.md)).
- **The MLX fast-path gains (about 1–4%) come from a screen,** not a full benchmark
  ([`benchmarks/mlx-fast-path-screen/README.md`](benchmarks/mlx-fast-path-screen/README.md)).
- **Switchyard does not counterbalance round order:** with the standard seed, GPU + ANE ran
  first in all three runs.

Benchmark-specific limitations are documented with each experiment, for example option
order in [`docs/correctness.md`](docs/correctness.md#option-order), serve and client caveats
in [`docs/serve.md`](docs/serve.md#limits), and what has not been measured in
[`docs/compatibility.md`](docs/compatibility.md#not-measured).

## Research, releases and reproduction

- **Reproduction.** Each headline number above traces to a report, raw data, a command and an
  environment in [`docs/reproducibility.md`](docs/reproducibility.md). The prebuilt-artifact
  checks are in [`benchmarks/prebuilt-artifacts-1.6.0.md`](benchmarks/prebuilt-artifacts-1.6.0.md)
  (macOS 26) and
  [`benchmarks/prebuilt-artifacts-1.6.2-macos27.md`](benchmarks/prebuilt-artifacts-1.6.2-macos27.md)
  (macOS 27).
  The 1.5 evidence is in
  [`research/coreml-adaptive-breaker/`](research/coreml-adaptive-breaker/README.md).
- **Research.** The research behind the 1.5 and 1.6 releases, failed routes included, is mapped in
  [`research/README.md`](research/README.md).
- **Releases.** Every change is in [`CHANGELOG.md`](CHANGELOG.md); release notes are in
  [`docs/releases/`](docs/releases/), most recently [1.6.3](docs/releases/v1.6.3.md),
  [1.6.2](docs/releases/v1.6.2.md) and [1.6.1](docs/releases/v1.6.1.md).

## Contributing

The most useful first contribution is a benchmark from a Mac other than an M4 Max
([how](docs/community-benchmarks.md)).

[`CONTRIBUTING.md`](CONTRIBUTING.md) covers setup, test tiers and parity checks; coding
agents follow [`AGENTS.md`](AGENTS.md). Open work is labelled `good first issue`,
`help wanted` and `research`.

More: [user guide](docs/guide.md) · [stable API](docs/api.md) ·
[architecture](docs/architecture.md) · [changes](CHANGELOG.md) · [security](SECURITY.md).

Apache-2.0; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Model weights are downloaded
from their pinned Hugging Face revisions and are not redistributed. This is an independent
project, not an official release of Convai Innovations, Apple or MLX.
