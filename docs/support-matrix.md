# Support matrix

The authoritative reference for platform scope, model support, compute-unit status and
routing derivation. [`compatibility.md`](compatibility.md) restates its conclusions for
someone deciding whether to run `laya-apple` on a given machine.

## Platform scope

| Scope | Status |
|---|---|
| Apple M4 Max, macOS 26.6.2, MLX 0.32.2, coremltools 9.0 | **Tested (release validation).** The shipped routing thresholds, the release benchmarks and the full validation of all three models come from this profile |
| Apple M4 Max 64 GB, macOS 27.0 (26A428), MLX 0.32.2, coremltools 9.0 | **Shipped routing profile (since 1.6.1), no release benchmarks.** The release machine after its upgrade from 26.6.2, so only macOS changed. Build and parity validation: [`research/macos27-validation/`](../research/macos27-validation/README.md) ([#17](https://github.com/tc3oliver/laya-apple/issues/17)). All three models, every explicit ANE bucket (10 pairs): built, 100% Neural Engine with 0 transitions, FP16 parity passed with 0 hard mismatches, placement probe and `artifacts verify` passed. Same op counts and parity error as on 26.6.2. Routing: `laya-apple calibrate`, two passes, gave the same auto ANE buckets as 26.6.2 for every model (64, 96, 128), shipped as `laya_apple/data/profiles/Apple_M4_Max-macos27-coremltools9.0.json` ([`benchmarks/routing-macos27/`](../benchmarks/routing-macos27/README.md)). `auto` uses the ANE here once the artifacts are built locally (no macOS 27 prebuilt artifacts). The release benchmarks, `serve` and adaptive-execution results are 26.6.2 only |
| Apple M4 Pro 48 GB, macOS 27.0, MLX 0.32.2, coremltools 9.0 | **Community measurement**, one `--quick` run of `laya-typed-decisions` only: [`hardware-results/apple-m4-pro-macos27/`](../hardware-results/apple-m4-pro-macos27/summary.md) ([#32](https://github.com/tc3oliver/laya-apple/pull/32)). MLX and ANE parity passed with 0 hard mismatches, a locally calibrated profile made `auto` use the ANE, and the heterogeneous check passed. Not a shipped routing profile and not release-validated |
| Apple M4 32 GB, macOS 26.2, MLX 0.32.2, coremltools 9.0 | **Community measurement**, one `--quick` run of `laya-typed-decisions` only: [`hardware-results/apple-m4-macos26/`](../hardware-results/apple-m4-macos26/summary.md) ([#41](https://github.com/tc3oliver/laya-apple/pull/41)). MLX and ANE parity passed with 0 hard mismatches. `laya-apple calibrate` was not run, so `auto` stayed on MLX (`platform_not_validated`) and the heterogeneous check did not run. Not a shipped routing profile and not release-validated |
| Apple M2 Pro 16 GB, macOS 26.6.2, MLX 0.32.2, no coremltools | **Community measurement**, one MLX-only `--quick` run of `laya-typed-decisions`: [`hardware-results/apple-m2-pro-macos26/`](../hardware-results/apple-m2-pro-macos26/summary.md) ([#47](https://github.com/tc3oliver/laya-apple/pull/47)). MLX FP16 parity passed. coremltools was not installed, so no ANE artifacts were built, `auto` stayed on MLX (`ane_runtime_unavailable`) and the heterogeneous check did not run. The first measured result on an SoC outside the M4 family. Not a shipped routing profile and not release-validated |
| Other Apple M-series SoCs, macOS 15–26 | **Expected** to run the MLX backend correctly (hypothesis; measured only on the M2 Pro row above). ANE placement, correctness and routing thresholds are **unknown** |
| macOS 27.x | **Two measurements, both macOS 27.0 with coremltools 9.0:** the M4 Max build, parity and routing validation (all three models; a shipped routing profile for M4 Max, macOS 27, coremltools 9.0) and the M4 Pro community run (`laya-typed-decisions`, not a shipped profile), above. Every other SoC and macOS 27 profile is **unknown**. Prior third-party work (laya-coreml, M3 Max, macOS 27.2) reported an enumerated-shape package running on the GPU. On macOS 27.0, the M4 Max run recorded 100% Neural Engine placement with 0 transitions for every fixed-shape BC1S artifact. The M4 Pro run's placement is inferred: its artifacts registered, which requires the same compute-plan check. Neither run tested enumerated shapes or macOS 27.2 |
| iOS / iPadOS | Out of scope |

Two profiles are validated for `device="auto"`: M4 Max / macOS 26 / coremltools 9.0
(`routing.json`) and M4 Max / macOS 27 / coremltools 9.0 (a shipped calibrated profile).
On an unvalidated hardware/OS profile, `device="auto"` uses MLX only
(`platform_not_validated`). `device="ane"` there requires building and
parity-validating artifacts on that machine yourself.

## Prebuilt ANE artifacts (since 1.6)

`laya-apple artifacts fetch` selects only archives built on the running machine's platform
profile (SoC, macOS major, coremltools), and each one passes the full import validation and
the placement probe on the receiving machine ([guide](guide.md#artifact-lifecycle)).

| Platform profile | Models and buckets | Status |
|---|---|---|
| Apple M4 Max, macOS 26, coremltools 9.0 | `laya` 64/96/128, `laya-multilingual` 64/96/128/256, `laya-typed-decisions` 64/96/128 | **Published** in [`tc3oliver/laya-apple-artifacts`](https://huggingface.co/tc3oliver/laya-apple-artifacts). Downloaded into an empty cache on the build machine: fetch, verify, parity and placement passed for all 10 model/bucket pairs ([`benchmarks/prebuilt-artifacts-1.6.0.md`](../benchmarks/prebuilt-artifacts-1.6.0.md)). No independent check on a second machine of the same profile yet (recommended, not required: [`publishing.md`](publishing.md)); every receiving machine re-validates before registering |
| Any other profile | — | **None published.** `artifacts fetch` raises `ArtifactMissingError` naming the build command; build locally with `laya-apple artifacts build` |

A fetched artifact still pays Core ML's on-device compile at first load: 273.5 s cold against
2.7 s warm for laya-typed-decisions (buckets 64/96/128) in one run
([`research/coreml-compile-cache/screen.md`](../research/coreml-compile-cache/screen.md)).

## Models

| Model | Repo | Pinned revision | `model.safetensors` SHA-256 (prefix) | Encoder | max_len | MLX dtypes | Explicit ANE buckets (`device="ane"`) | Auto ANE buckets (`device="auto"`) |
|---|---|---|---|---|---:|---|---|---|
| `laya` | `convaiinnovations/laya` | `c5d78730f349…` | `891102d372688fc2…` | ModernBERT-large, 28 layers (10 global, 18 local) | 512 | float16, float32 | 64, 96, 128 | 64, 96, 128 |
| `laya-multilingual` | `convaiinnovations/laya-multilingual` | `052592a15d19…` | `9d628fd971b70038…` | mmBERT-base, 22 layers | 1024 | float16, float32 | 64, 96, 128, 256 | 64, 96, 128 |
| `laya-typed-decisions` | `convaiinnovations/laya-typed-decisions` | `f9ab0b228f0f…` | `4fa56de72383a9d3…` | ModernBERT-large, 28 layers | 1024 | float16, float32 | 64, 96, 128 | 64, 96, 128 |

The runtime verifies `model.safetensors` against the pinned hash before use;
a mismatch raises `ArtifactRevisionError` rather than a warning. A model's
longer validated ANE lengths (parity passed up to `max_len`) are not offered
by either explicit `device="ane"` or `auto` — they lose to MLX on latency
(see "How the auto-ANE buckets were derived" below).

## Core ML configuration status

| Model | Backend / graph | Shape | Compute units | Profile | Status |
|---|---|---|---|---|---|
| all three | MLX FP16 / FP32 | any L ≤ max_len, any batch | Metal GPU | M4 Max / 26.6.2 | **validated** |
| all three | Core ML BC1S (`bc1s-masked`) | fixed B=1, L ∈ {64, 96, 128, 256, 512, 1024}\* | `CPU_AND_NE` | M4 Max / 26.6.2 | **validated** |
| all three | Core ML BC1S | fixed B=1 | `CPU_ONLY` | M4 Max / 26.6.2 | **invalid** — FP16 conv/matmul precision on Core ML's CPU path |
| all three | Core ML BC1S | fixed B=1 | `CPU_AND_GPU` | M4 Max / 26.6.2 | validated for correctness, not used (slower than MLX) |
| all three | Core ML BC1S | fixed B=1 | `ALL` | M4 Max / 26.6.2 | not used — placement is not a stable device (invariant I-6) |
| all three | Core ML ordinary (BxLxC/SDPA) graph | fixed | `CPU_AND_NE`, `ALL` | M4 Max / 26.6.2 | **invalid** — numerically wrong on the ANE (up to 85/187 hard decision mismatches) |
| all three | Core ML ordinary graph | fixed | `CPU_AND_GPU` | M4 Max / 26.6.2 | validated, not in the product |
| all three | Core ML ordinary graph | enumerated/flexible shape | any | M4 Max / 26.6.2 | **invalid for acceleration** — runs 100% on CPU |
| `laya-typed-decisions` | Core ML BC1S windowed attention | fixed B=1, L128–1024 | `CPU_AND_NE` | M4 Max / 26.6.2 | validated, research only (not shipped) |
| all three | Core ML BC1S | fixed B=4 / B=8 | `CPU_AND_NE` | M4 Max / 26.6.2 | **unknown** — FP32 layout check only, parity not run |
| all three | Core ML BC1S (`bc1s-masked`) | fixed B=1, the explicit ANE buckets only | `CPU_AND_NE` | M4 Max / 27.0 | **validated**: build, placement and parity passed ([`research/macos27-validation/`](../research/macos27-validation/README.md)); routing calibrated and shipped ([`benchmarks/routing-macos27/`](../benchmarks/routing-macos27/README.md)) |
| any | any Core ML | any | any | any other profile | **unknown** |

\* Parity was measured at these lengths (`laya` up to 512). Which of them the
product *offers* is the separate routing decision above.

## `laya-apple serve`

The local Jev-compatible server (since 1.3.0, `[serve]` extra) uses the same checkpoints,
backends and routing as the Python API; this table covers only what the server adds.

| Scope | Status |
|---|---|
| Wire format of upstream `laya.serve` 0.3.20 | **Tested** on the release profile: 792 of 792 requests matched unmodified upstream, 365 of them answered on the ANE ([`benchmarks/serve-compat/`](../benchmarks/serve-compat/README.md)). The status codes that deliberately differ are listed in [`serve.md`](serve.md) |
| `--model auto` language routing | **Tested** against upstream's router on the golden cases (the `auto-*` files in the same directory). English vs multilingual only; `LAYA_AUTO_TASK` and caller language hints are **not implemented** |
| Jev clients | **Tested**, wire compatibility only: 7 clients at their released versions, unmodified ([`integrations/jev-plugins/`](../integrations/jev-plugins/README.md)). Other versions are **unknown** |
| Decision latency and the effect on a local LLM on the same GPU, at a fixed offered load | **Tested twice**, in one setting: on the release profile beside `Qwen3.8-27B-oQ4e-mtp` on oMLX, `--model laya`, 8 req/s offered. Run 2 (laya-apple 1.3.0, published in 1.4.0): short-decision P99 with the LLM busy 47.2 ms (`auto`) against 122.3 ms (`--device gpu`); LLM tok/s −4.0% against −5.3%. Run 3 (1.5.0 defaults): 41.7 ms against 79.5 ms; −2.2% against −4.7%. Both runs: 0 hard mismatches, 0 errors ([`serve.md`](serve.md#beside-a-local-llm), [`benchmarks/serve/`](../benchmarks/serve/README.md)) |
| Adaptive ANE execution (1.5, the default for laya and laya-typed-decisions under `workers` + `auto`) | **Tested once**, on the release profile with the library's closed-loop harness: 154 production episodes, 0 correctness failures ([`research/coreml-adaptive-breaker/`](../research/coreml-adaptive-breaker/README.md)). In `serve` beside a local LLM (run 3 above), it was enabled but the asynchronous path **never engaged**: 3,222 of 3,222 Neural Engine forwards ran on the 1.4 path, 0 breaker trips. The asynchronous path is **not measured** in `serve`, beside a local LLM, or on any other Mac |
| Maximum decision throughput | **Not measured** (runs 2 and 3 used a fixed 8 req/s offered load) |
| The same beside other LLM servers and models, under prefill-heavy LLM loads, at other request rates; the other checkpoints (`laya-typed-decisions`, `--model laya-multilingual`) and `--model auto` | **Not measured** |
| Bind address | Loopback by default. A non-loopback bind needs `--allow-remote` and `LAYA_API_KEY` |

## Compute-unit terminology

Configurations are always named explicitly and never merged in benchmark
tables or capability records:

| Name | Meaning | Phase -1 behaviour for Laya |
|---|---|---|
| **MLX / Metal GPU** | MLX framework, not Core ML | correct everywhere; the default backend |
| **Core ML CPU+ANE** (`CPU_AND_NE`) | Core ML may use CPU and ANE | BC1S graph: 100% ANE, correct. Ordinary graph: partitioned ANE↔CPU (6 transitions), wrong; at L1024 a compile failure silently drops to CPU |
| **Core ML CPU+GPU** (`CPU_AND_GPU`) | Core ML may use CPU and GPU | both graphs correct. Ordinary graph ≈ MLX speed; BC1S graph slower (99.26 vs 71.06 ms at L1024) |
| **Core ML ALL** | Core ML chooses among CPU, GPU and ANE | placement changes with length. Ordinary graph: ANE at L ≤ 128 (typed, laya) and L ≤ 96 (multilingual), where it is wrong; GPU above. BC1S graph: ANE at L ≤ 256, 96–100% GPU at L512–L1024 |
| **Core ML CPU only** (`CPU_ONLY`) | CPU only | BC1S graph wrong (FP16 precision); ordinary graph correct for typed-decisions, marginal/failing for others |

`CPU_AND_NE` means "CPU and ANE are *allowed*," not "runs on the ANE." Every
ANE artifact's compute plan is checked at load: 100% of operations on the
Neural Engine and 0 device transitions, or it is rejected
(`ComputeUnitMismatchError`).

## How the auto-ANE buckets were derived

`laya_apple/data/routing.json` is generated, not hand-written, by
[`scripts/derive_routing.py`](../scripts/derive_routing.py) from the
committed Phase -1 evidence (`research/phase-0-feasibility/raw/bench/latency.jsonl`
and `research/phase-0-feasibility/raw/parity/<model>/ane_units-cpu_ne.json`).

Rule, per model, walking the ANE buckets in increasing order: a bucket `b_i`
joins auto routing if and only if the BC1S graph passed parity at `b_i` on
`CPU_AND_NE`, **and** the ANE forward P50 at `b_i` is below the MLX forward
P50 at the previous bucket `b_{i-1}` (for `b_1`, compared against MLX at
`b_1` itself, since no shorter MLX length was measured). The walk stops at
the first bucket that fails this test. Multi-question requests never
auto-route to the ANE: MLX batching wins at every measured length for 4 and
8 questions, and 2–3 questions were not measured.

The rule is more conservative than the measured crossover. laya-multilingual is slightly
faster on the ANE at exactly 256 tokens (8.3 ms against 8.6 ms on MLX), but that bucket
stays explicit-only, because the ANE at 256 does not beat MLX at the previous bucket,
128 tokens (6.4 ms) ([`benchmarks/v1.0.md`](../benchmarks/v1.0.md#auto-routing)).

Regenerate and check the committed file with:

```bash
uv run python scripts/derive_routing.py            # writes laya_apple/data/routing.json
uv run python scripts/derive_routing.py --check    # fails if the committed file is stale
```

### Shipped calibrated profiles

A profile validated after `routing.json` was derived ships as the unedited output of
`laya-apple calibrate` on that profile, in `laya_apple/data/profiles/<profile-key>.json`, with
its raw measurements committed under `benchmarks/`. It uses the same rule
(`laya_apple/derivation.py`) and replaces the auto buckets and service times for its profile
only, the way a local calibrated profile does. It ships only if two calibration passes agree
on every model's auto buckets. A unit test checks that each file equals its committed
evidence and re-derives its buckets.

| Profile | File | Evidence | Auto ANE buckets |
|---|---|---|---|
| Apple M4 Max, macOS 27, coremltools 9.0 | `Apple_M4_Max-macos27-coremltools9.0.json` | [`benchmarks/routing-macos27/`](../benchmarks/routing-macos27/README.md) | 64, 96, 128 for all three models (as on 26.6.2) |

It was measured with `calibrate`, not the Phase -1 harness behind `routing.json`; the
differences are listed in that record.

Profiles match on the macOS major version, so this one covers every macOS 27.x release with
coremltools 9.0 on an M4 Max, although only 27.0 (26A428) was measured. Prior third-party
work (laya-coreml, M3 Max, macOS 27.2) reported an enumerated-shape Core ML package running
on the GPU ([`typed-decisions-ane.md`](../research/phase-0-feasibility/typed-decisions-ane.md)).
laya-apple ships only fixed-shape artifacts, and every artifact must still pass the compute-plan
check (100% Neural Engine, 0 transitions) and the placement probe on the machine that loads
it, but 27.2 itself is unmeasured here.
