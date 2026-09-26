# laya-apple

**English** | [繁體中文](README.zh-TW.md) | [简体中文](README.zh-CN.md)

[![CI](https://github.com/tc3oliver/laya-apple/actions/workflows/ci.yml/badge.svg)](https://github.com/tc3oliver/laya-apple/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/laya-apple)](https://pypi.org/project/laya-apple/)
![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

**Correctness-validated heterogeneous [Laya](https://github.com/NandhaKishorM/laya) runtime
for Apple silicon:** the MLX GPU and the Apple Neural Engine, at the same time.

## Adaptive GPU + Neural Engine serving (1.5)

laya-apple sends short, single-question decisions to the Apple Neural Engine and keeps long
or multi-question ones on the MLX GPU, both engines serving at once. Since 1.5, laya and
laya-typed-decisions send Neural Engine requests through Core ML's asynchronous API, avoiding
the synchronous path's long GIL hold, so they no longer hold up the GPU's results. If that path slows down, a
breaker sends the rest of that GPU + ANE overlap back to the 1.4 path. It is on by default and
needs no code change.

| Against the 1.4 path, one Apple M4 Max | laya | laya-typed-decisions |
|---|---:|---:|
| GPU return (P50) | 4.28–4.29 → **0.035–0.037 ms** | 8.60 → **0.043 ms** |
| Throughput | **1.042×** | **1.038×** |
| Median episode P99 | **0.18–0.19 ms lower** | **0.53 ms lower** |

- **154 production validation episodes** (76 laya, 78 typed-decisions, including bursts and
  soaks): all remained on the asynchronous path after handoff, with 0 mismatches, routing
  failures, lost requests or crashes ([`val_tables.md`](research/coreml-adaptive-breaker/val_tables.md)).
- **Recovery was measured separately,** because no slow state occurred in validation. In 12 of
  12 episodes where the asynchronous path was already slow, the breaker tripped within 40 ms
  and latency was back to the 1.4 path's within 164–414 ms
  ([`phase1_tables.md`](research/coreml-adaptive-breaker/phase1_tables.md)).

[Try Switchyard](#see-it-yourself-switchyard) ·
[Install](#install) · [Local Jev-compatible server](#local-jev-compatible-server)

## Install

Apple silicon, Python 3.11–3.13:

```bash
pip install 'laya-apple[ane]'   # MLX GPU + the Neural Engine runtime
laya-apple artifacts build laya-typed-decisions   # optional: build + parity-validate ANE artifacts here (~5 min)
```

| Extra | Adds |
|---|---|
| (none) | The MLX GPU runtime only |
| `ane` | The Neural Engine runtime (coremltools 9.0, pyobjc-framework-CoreML) |
| `convert` | Building ANE artifacts on this Mac (torch 2.7.0) |
| `serve` | `laya-apple serve`, the local Jev-compatible server |

Without the `ane` extra, or without a built artifact, everything runs on the MLX GPU. To run from
source or develop laya-apple, see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Quickstart (30 seconds)

```python
from laya_apple import Laya

model = Laya.from_pretrained(
    "convaiinnovations/laya-typed-decisions",
    device="auto",
)

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
print(result.answers["urgency"]["choice"], result.answers["urgency"]["probabilities"])

rt = result.runtime
print(rt.backend, rt.device, rt.routing_reason, f"{rt.latency_ms:.1f} ms")
```

On the tested machine:

```text
high {'low': 0.1713, 'medium': 0.3358, 'high': 0.4929}
coreml ane validated_short_single_question_path 11.2 ms
```

- The first call downloads the pinned checkpoint. After that it works offline
  (`local_files_only=True`).
- Without ANE artifacts, the same request runs on MLX and `routing_reason` says why.
- More: [`examples/`](examples/) (`basic.py`, `auto_routing.py`,
  `heterogeneous_serving.py`) and the [user guide](docs/guide.md).

For concurrent GPU + ANE serving, use `execution="workers"` and submit requests from any thread:

```python
with Laya.from_pretrained("convaiinnovations/laya-typed-decisions", execution="workers") as model:
    futures = [model.submit(context=c, questions=q) for c, q in requests]   # thread-safe
```

## How it works

Laya answers typed questions about a context (`choice`, `score`, `noul`) in one forward pass.
A Mac has two engines that can run it, with different strengths.

![Requests of at most 128 tokens with one question and a validated artifact go to the Apple Neural Engine; longer, multi-question or unvalidated requests go to the MLX GPU; with execution="workers" both engines serve independent requests concurrently](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/readme/architecture.svg)

1. **Correct ANE execution.** A fast Core ML export is not necessarily correct. On the tested
   Mac, the ordinary Core ML export ran on the Neural Engine without any error and disagreed
   with upstream PyTorch on up to 85 decisions. laya-apple ships an ANE artifact only after it
   passes a parity gate on the machine that uses it ([Correctness](#correctness)).
2. **Automatic routing.** The router decides before a request runs and records why in
   `routing_reason`. Under load it also compares the two queues' backlogs.
3. **Concurrent GPU + ANE serving.** With `execution="workers"`, the GPU runs in a worker
   process and the ANE on its own dispatcher. Short requests stop queueing behind long ones.
4. **Adaptive ANE execution (1.5).** In every GPU + ANE overlap, the first 64 Neural Engine
   requests run the 1.4 path; later ones use Core ML's asynchronous API. Three consecutive requests
   with prepare time above 0.3 ms send the rest of the overlap back to the 1.4 path
   ([guide](docs/guide.md#adaptive-ane-execution-in-process-ane-the-default-since-15)).
   `ane_handoff=False` turns it off.

| Request | Goes to | Why (measured on the tested Mac) |
|---|---|---|
| One question, ≤ 128 tokens, validated artifact present | **ANE** | Faster: laya-typed-decisions L128 takes 9.9 ms on the ANE against 12.2 ms on MLX (forward P50) |
| Longer context | **MLX GPU** | MLX is faster there: 19.2 ms at L256 and 71.0 ms at L1024 |
| Several questions | **MLX GPU** | MLX batches the questions; the ANE runs them one at a time |
| Unvalidated Mac, missing artifact, or no Core ML | **MLX GPU** | Recorded as `platform_not_validated`, `ane_artifact_unavailable` or `ane_runtime_unavailable` |

**The production threshold is more conservative than the measured crossover.**
laya-multilingual is slightly faster on the ANE at exactly 256 tokens (8.3 ms against
8.6 ms), but that bucket stays explicit-only, because it does not beat MLX at the previous
bucket, 128 tokens.

To see why a request was slow, pass a callback (workers only). It receives one
`RequestTrace` per completed request: the queue snapshot the router decided on, the device
and reason it chose, and monotonic timestamps from submit through queue, service and
response ([`docs/api.md`](docs/api.md)). With the default `trace=None` nothing is recorded.

```python
def on_trace(trace):   # runs on the device's dispatcher thread: keep it cheap
    print(trace.request_id, trace.target, trace.routing_reason, trace.queue_ms, trace.e2e_ms)

model = Laya.from_pretrained("convaiinnovations/laya-typed-decisions", execution="workers", trace=on_trace)
```

Thresholds: [`docs/support-matrix.md`](docs/support-matrix.md). Architecture:
[`docs/architecture.md`](docs/architecture.md).

## Local Jev-compatible server

`laya-apple serve` is a local alternative to the Jev API. It serves a Jev-compatible API
(`POST /v1/systemone`) on loopback and answers with upstream Laya, running on your Mac.
Point an existing Jev client at it through its base-URL setting; its code does not change.

```bash
pip install 'laya-apple[serve,ane]'
laya-apple serve                                   # http://127.0.0.1:8642
export TYPESAFE_BASE_URL=http://127.0.0.1:8642     # then run your Jev client as usual
```

A Jev client needs some API key to start. Any placeholder works, such as
`TYPESAFE_API_KEY=local-placeholder`, unless you set `LAYA_API_KEY` on the server.

![A terminal: laya-apple serve starts locally; the released typesafe-sdk 0.7.1 for Python, unmodified, is pointed at it with TYPESAFE_BASE_URL and gets the answer fix_code; the same request with curl shows that laya-apple answered it with the laya checkpoint on the Apple Neural Engine](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/readme/serve-demo.svg)

- **Tested with 7 unmodified Jev clients** at their released versions, including the Python
  and JS SDKs and two Claude Code plugins. Each was pointed here only through its base-URL
  setting and completed its requests ([`integrations/jev-plugins/README.md`](integrations/jev-plugins/README.md)).
- **The answers are Laya's, not Jev's.** What is measured is fidelity to upstream Laya:
  792 of 792 requests matched unmodified upstream `laya.serve` 0.3.20 within the FP16
  parity gate, 365 of them answered on the Neural Engine
  ([`benchmarks/serve-compat/`](benchmarks/serve-compat/README.md)). Laya is a different,
  smaller model, so there is no Jev-level accuracy claim, and a client whose thresholds
  were tuned on Jev may take its fallback path more often.
- laya-apple is not affiliated with TypeSafe or Jev. The figure above is a recorded run
  ([`scripts/capture_serve_demo.py`](scripts/capture_serve_demo.py)).

Models, API, security and limits: [`docs/serve.md`](docs/serve.md).

### Beside a busy local LLM

**This benchmark predates 1.5:** it was measured on the 1.3 execution path. Serve now uses
adaptive ANE execution by default, which has not been measured in serve.

![laya-apple serve beside a busy local LLM, one run on an Apple M4 Max with Qwen3.8-27B-oQ4e-mtp: short-decision P99 47.2 ms with serve auto against 122.3 ms with --device gpu while the LLM generates (43.2 and 55.9 ms with the LLM idle); LLM throughput 42.2 tok/s alone, 40.0 beside serve --device gpu and 40.5 beside serve auto](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/readme/serve-llm-load.svg)

One run of `laya-apple serve --model laya` on one Apple M4 Max, with a 27B local LLM
(`Qwen3.8-27B-oQ4e-mtp` on oMLX) generating at saturation and 8 decision requests per second
offered beside it, 80% of them short single-question decisions
([`benchmarks/serve/`](benchmarks/serve/README.md), run 2). The default `--model auto` was not
measured.

- **Short-decision P99 with the LLM busy: 47.2 ms with `auto`, against 122.3 ms with
  `--device gpu`.** With the LLM idle it is 43.2 ms with `auto` and 55.9 ms with
  `--device gpu`.
- **The LLM's throughput drops 4.0% beside serve `auto` and 5.3% beside `--device gpu`,**
  from 42.2 tok/s alone. The gap between the two, 1.31 points, is just above the LLM's own
  window-to-window spread in the same run, 1.28 points.
- **Correct under load:** 0 hard mismatches and 0 errors over 7,728 decisions, each checked
  against the same server's answer with the LLM idle.
- About 4% of `auto`'s short decisions went to the GPU by design, when the Neural Engine's
  backlog was the longer wait; the P99 above includes them.
- Run 1 of this benchmark failed its own validity checks, so no result is drawn from it.

## See it yourself: Switchyard

```bash
uvx laya-apple switchyard
```

![Switchyard: the same recorded timetable replayed GPU-only, where trains queue at red signals during rush hour, and GPU + ANE, where they flow into their platforms; then the result card: 1,407 of 1,422 trains late GPU-only against 0 with GPU + ANE](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/media/switchyard-launch.gif)

**Every train is a real Laya decision** ("which platform is clear?"). A red signal means the
train is waiting for the model's answer; a train is late when it has no answer within 100 ms.
Rush hour adds bursty load, with long background requests on the GPU.

| Same timetable, same model | MLX GPU only | MLX GPU + Neural Engine |
|---|---:|---:|
| Late trains | 1,407–1,408 / 1,422 | **0 / 1,422** |
| P99 decision latency | 3,107.7–3,170.6 ms | **54.5–54.7 ms** |
| P99 queue wait | 3,095.9–3,158.8 ms | **39.6–42.8 ms** |

- Three standard runs on one Apple M4 Max (macOS 26.6.2, `laya-typed-decisions`); other
  Macs will differ. In every run both rounds gave the same answer for every train.
- Most of the gap is short decisions no longer waiting in the GPU queue while the long
  requests keep running there.
- The benchmark runs headless first (about 2–3 minutes after an 800 MB first download). The
  browser then replays the recorded request traces; the animation is not part of the
  measurement. Without a built Neural Engine artifact it runs GPU-only and prints the setup
  command, `uvx --from "laya-apple[convert]" laya-apple switchyard --setup-ane`.
- Round order is fixed by the seed, so `hybrid` ran first in all three runs. These numbers
  are not comparable with the v1.0 table below: the rate, the measurement boundary and the
  workload differ.

Workload, method and raw data: [`docs/switchyard.md`](docs/switchyard.md) and
[`benchmarks/switchyard/README.md`](benchmarks/switchyard/README.md).

## Correctness

Parity against upstream Laya on PyTorch CPU FP32, over the shipped golden rows (v1.0).
Each cell gives hard mismatches, then the max probability error.

| Implementation on the tested Mac | laya | laya-multilingual | laya-typed-decisions |
|---|---|---|---|
| Ordinary Core ML export · `CPU_AND_NE` | ❌ 12, 0.56 | ❌ 85, 1.0 | ❌ 19, 0.42 |
| Ordinary Core ML export · `CPU_AND_GPU` | ✅ 0, 0.0066 | ✅ 0, 0.0059 | ✅ 0, 0.0028 |
| **laya-apple MLX FP16** | ✅ 0, 0.0037 | ✅ 0, 0.0045 | ✅ 0, 0.0017 |
| **laya-apple ANE FP16** | ✅ 0 (1 near-tie), 0.012 | ✅ 0, 0.013 | ✅ 0, 0.0077 |

- **The FP16 gate:** probability error ≤ 0.02 and 0 hard mismatches.
- **Near-tie flips** (upstream's top-two margin < 0.04) are listed, not hidden.
- **Explicit ANE requests never fall back.** They run the validated artifact or raise, and
  every loaded artifact is also timed against `CPU_ONLY` to catch a silent CPU placement.
- **The 1.5 asynchronous path** must equal coremltools' output exactly on the goldens
  (`tests/parity/test_ane_async_parity.py`).
- Definitions, every configuration tested, and the fallback audit are in
  [`docs/correctness.md`](docs/correctness.md) and
  [`docs/no-silent-fallback.md`](docs/no-silent-fallback.md).

## Heterogeneous serving benchmark (v1.0)

![The same model playing Lane Runner on the MLX GPU and on the Apple Neural Engine to the same score, then the same burst of requests served GPU-only and GPU + ANE: GPU-only short requests wait in the GPU queue for up to 1.5 s, while under GPU + ANE the router sends them to the ANE and they run as they arrive](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/media/heterogeneous-serving.gif)

![Mixed-workload throughput against GPU-only serving: laya 41.9 to 122.5 req/s (2.92×), laya-multilingual 55.7 to 241.8 req/s (4.34×), laya-typed-decisions 24.0 to 109.6 req/s (4.57×)](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/readme/hero-throughput.svg)

**Short-request P99 under open-loop bursty arrivals**, measured from arrival with queueing
included (same arrival sequence for both):

| Model | GPU-only | GPU + ANE |
|---|---:|---:|
| laya (46.2 req/s offered) | 1538.0 ms | 108.5 ms |
| laya-multilingual (83.8 req/s) | 2052.3 ms | 29.6 ms |
| laya-typed-decisions (35.8 req/s) | 1592.9 ms | 79.5 ms |

- One Apple M4 Max with macOS 26.6.2: one short and one long request stream through one
  `Laya(execution="workers")` instance. It predates 1.5's adaptive execution.
- The gain comes from running both engines at once, not from raw ANE latency: a single short
  request is only somewhat faster on the ANE (for example 9.9 against 12.2 ms).
- The animation first replays a recorded
  [Lane Runner](https://github.com/tc3oliver/laya-playground-apple) game, the same model on
  each device, then one burst of the laya-typed-decisions bursty workload. Every number's
  source is in [`docs/media/README.md`](docs/media/README.md).

Method and raw data: [`benchmarks/v1.0.md`](benchmarks/v1.0.md).

## Supported models and platforms

| Model | max_len | MLX GPU | ANE buckets (explicit) | ANE buckets used by `auto` | Adaptive ANE execution |
|---|---:|---|---|---|---|
| [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) | 512 | FP16 / FP32, any length | 64, 96, 128 | 64, 96, 128 | Yes |
| [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual) | 1024 | FP16 / FP32, any length | 64, 96, 128, 256 | 64, 96, 128 | No (worker-process ANE) |
| [`convaiinnovations/laya-typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions) | 1024 | FP16 / FP32, any length | 64, 96, 128 | 64, 96, 128 | Yes |

**Tested:**
- Apple M4 Max, macOS 26.6.2, MLX 0.32.2, coremltools 9.0;
- Python 3.11–3.13.

**Other Apple silicon:**
- MLX is expected to work.
- `auto` stays on MLX until artifacts are built and calibrated on that machine
  (`laya-apple calibrate`).

See [`docs/compatibility.md`](docs/compatibility.md).

## Community benchmarks

Every benchmark above covers only an M4 Max. The [community matrix](docs/community-benchmarks.md)
collects results from other Macs as separate runs, not mixed into the numbers above. It has
external results for an M4 Pro, an M4 and an M2 Pro (MLX only). If you have another Mac, one command adds it.

- M1 / M2 → [#1](https://github.com/tc3oliver/laya-apple/issues/1)
- M3 Max → [#2](https://github.com/tc3oliver/laya-apple/issues/2)
- M5 or any other Mac → [add your Mac](docs/community-benchmarks.md#add-your-mac)

```bash
uv run python scripts/hardware_report.py --quick
```

The linked issues and guide have the full steps, from `git clone` to opening the pull request; it takes about 10 minutes.

## Reproduction

Each headline number above traces to a report, raw data, a command and an environment in
[`docs/reproducibility.md`](docs/reproducibility.md). The full v1.0 suite, which compares
PyTorch CPU/MPS, the ordinary Core ML export, MLX and laya-apple, is in
[`benchmarks/v1.0.md`](benchmarks/v1.0.md). The 1.5 detector replay, recovery experiment and
release validation are in
[`research/coreml-adaptive-breaker/`](research/coreml-adaptive-breaker/README.md).

## Limitations

- **One test machine.** Every benchmark is from one Apple M4 Max on macOS 26.6.2. Routing
  thresholds are not assumed to hold on other Apple SoCs.
- **Long contexts stay on MLX,** which is faster there. The ANE path is batch 1 only.
- **Isolation is partial.** Under concurrency, each stream's P99 is above its solo value.
- **Adaptive ANE execution was measured on one M4 Max only.**
  - No slow state occurred in its 154 validation episodes, so recovery after a trip was
    measured in a separate 12-episode experiment on the same machine.
  - The first 64 Neural Engine requests of every GPU + ANE overlap run the 1.4 path.
  - laya-multilingual, whose Neural Engine runs in a worker process, does not use it.
  - `laya-apple serve` uses it by default, but it was not measured in serve or beside a local
    LLM. The serve numbers above were measured on the 1.3 path.
- **Cold start** on a fresh artifact location costs 3–5 minutes of Core ML compile per
  model. `ane_startup="background"` serves on MLX in the meantime.
- **`choice` decisions can depend on option order.** This comes from upstream Laya, and
  laya-apple reproduces it exactly ([`research/option-order/`](research/option-order/)).
- **Not measured yet:** energy use, quantized artifacts and cross-SoC validation.
- **`laya-apple serve` answers with Laya, not Jev.** Only fidelity to upstream Laya is
  measured, not Jev-level accuracy. Clients with thresholds tuned on Jev may take their
  fallback path more often; two of the seven tested did
  ([`integrations/jev-plugins/README.md`](integrations/jev-plugins/README.md)).
- **`laya-apple serve` performance is one run in one setting:** one M4 Max, one LLM
  (`Qwen3.8-27B-oQ4e-mtp` on oMLX, decode-heavy, short prompts), `--model laya`, 8 decision
  requests per second offered ([`benchmarks/serve/`](benchmarks/serve/README.md)). Not
  measured: other LLM servers and models, prefill-heavy LLM loads, other request rates,
  serve's maximum decision throughput (run 2 used a fixed 8 req/s offered load), and the
  other checkpoints (`laya-typed-decisions`, `--model laya-multilingual`) and
  `--model auto`.
- **Serve `auto` still costs the LLM throughput:** 4.0% in that run, only 1.31 points less
  than `--device gpu`, against a 1.28-point window-to-window spread of the LLM alone.
- **Multi-question decisions always run on the GPU,** and their P99 grows with the LLM
  busy: 117.1 ms against 84.1 ms idle with `auto`.
- **Client compatibility was tested once,** on 2026-09-25, at the client versions listed and
  on the tested M4 Max. A later client release may change what it sends or accepts.
- **`serve --model auto` routes between English and multilingual only.** Upstream's opt-in
  typed-decisions workflow detection (`LAYA_AUTO_TASK`) and caller language hints are not
  implemented ([`docs/serve.md`](docs/serve.md#limits)).
- **Switchyard ran on one M4 Max with a fixed round order** (`design.counterbalance` is
  `"none"`, so `hybrid` always ran first) ([`docs/switchyard.md`](docs/switchyard.md)).

## Contributing

The most useful first contribution is a benchmark from a Mac other than an M4 Max: run
the command above and open a PR with `hardware-results/`
([how](docs/community-benchmarks.md)).

- [`CONTRIBUTING.md`](CONTRIBUTING.md) covers setup, test tiers (which tests a change
  actually needs), parity checks and backend changes.
- Open work is labelled `good first issue`, `help wanted` and `research`.
- Coding agents: [`AGENTS.md`](AGENTS.md) has the repository rules.

## More

- User guide: [`docs/guide.md`](docs/guide.md).
- Local Jev-compatible server: [`docs/serve.md`](docs/serve.md).
- Stable API: [`docs/api.md`](docs/api.md).
- Architecture: [`docs/architecture.md`](docs/architecture.md).
- Changes: [`CHANGELOG.md`](CHANGELOG.md).
- Security: [`SECURITY.md`](SECURITY.md).

Apache-2.0; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Model weights are downloaded
from their pinned Hugging Face revisions and are not redistributed. This is an independent
project, not an official release of Convai Innovations, Apple or MLX.
