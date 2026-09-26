# Support matrix

The authoritative reference for platform scope, model support, compute-unit status and
routing derivation. [`compatibility.md`](compatibility.md) restates its conclusions for
someone deciding whether to run `laya-apple` on a given machine.

## Platform scope

| Scope | Status |
|---|---|
| Apple M4 Max, macOS 26.6.2, MLX 0.32.2, coremltools 9.0 | **Tested (release validation).** The shipped routing thresholds, the release benchmarks and the full validation of all three models come from this profile |
| Apple M4 Pro 48 GB, macOS 27.0, MLX 0.32.2, coremltools 9.0 | **Community measurement**, one `--quick` run of `laya-typed-decisions` only: [`hardware-results/apple-m4-pro-macos27/`](../hardware-results/apple-m4-pro-macos27/summary.md) ([#32](https://github.com/tc3oliver/laya-apple/pull/32)). MLX and ANE parity passed with 0 hard mismatches, a locally calibrated profile made `auto` use the ANE, and the heterogeneous check passed. Not a shipped routing profile and not release-validated |
| Apple M4 32 GB, macOS 26.2, MLX 0.32.2, coremltools 9.0 | **Community measurement**, one `--quick` run of `laya-typed-decisions` only: [`hardware-results/apple-m4-macos26/`](../hardware-results/apple-m4-macos26/summary.md) ([#41](https://github.com/tc3oliver/laya-apple/pull/41)). MLX and ANE parity passed with 0 hard mismatches. `laya-apple calibrate` was not run, so `auto` stayed on MLX (`platform_not_validated`) and the heterogeneous check did not run. Not a shipped routing profile and not release-validated |
| Apple M2 Pro 16 GB, macOS 26.6.2, MLX 0.32.2, no coremltools | **Community measurement**, one MLX-only `--quick` run of `laya-typed-decisions`: [`hardware-results/apple-m2-pro-macos26/`](../hardware-results/apple-m2-pro-macos26/summary.md) ([#47](https://github.com/tc3oliver/laya-apple/pull/47)). MLX FP16 parity passed. coremltools was not installed, so no ANE artifacts were built, `auto` stayed on MLX (`ane_runtime_unavailable`) and the heterogeneous check did not run. The first measured result on an SoC outside the M4 family. Not a shipped routing profile and not release-validated |
| Other Apple M-series SoCs, macOS 15–26 | **Expected** to run the MLX backend correctly (hypothesis; measured only on the M2 Pro row above). ANE placement, correctness and routing thresholds are **unknown** |
| macOS 27.x | **One community measurement** (the M4 Pro row above: macOS 27.0, coremltools 9.0, `laya-typed-decisions`). Every other SoC, model and macOS 27 profile is **unknown**. Prior third-party work observed different Core ML placement behaviour there (enumerated shapes on the GPU) |
| iOS / iPadOS | Out of scope |

On an unvalidated hardware/OS profile, `device="auto"` uses MLX only
(`platform_not_validated`). `device="ane"` there requires building and
parity-validating artifacts on that machine yourself.

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
| Decision latency and the effect on a local LLM on the same GPU, at a fixed offered load | **Tested once** (published in 1.4.0; measured on 1.3.0), in one setting: one run on the release profile beside `Qwen3.8-27B-oQ4e-mtp` on oMLX, `--model laya`, 8 req/s offered. Short-decision P99 with the LLM busy 47.2 ms (`auto`) against 122.3 ms (`--device gpu`); LLM tok/s −4.0% against −5.3%; 0 hard mismatches, 0 errors ([`serve.md`](serve.md#beside-a-local-llm), [`benchmarks/serve/`](../benchmarks/serve/README.md)) |
| Adaptive ANE execution (1.5, the default for laya and laya-typed-decisions under `workers` + `auto`) | **Tested once**, on the release profile with the library's closed-loop harness: 154 production episodes, 0 correctness failures ([`research/coreml-adaptive-breaker/`](../research/coreml-adaptive-breaker/README.md)). **Not measured** in `serve`, beside a local LLM, or on any other Mac |
| Maximum decision throughput | **Not measured** (run 2 used a fixed 8 req/s offered load) |
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
