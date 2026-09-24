# Hardware report: Apple M2 Pro, macOS 26.6.2

Matrix row (docs/community-benchmarks.md):

| Mac | MLX | ANE | Auto uses ANE | Heterogeneous |
|---|---|---|---|---|
| Apple M2 Pro (16 GB, macOS 26.6.2) | ✓ | untested | no | untested |

## Environment

| | |
|---|---|
| SoC | Apple M2 Pro (Mac14,12) |
| Memory | 16 GB |
| macOS | 26.6.2 (25G83) |
| Python | 3.12.8 |
| mlx | 0.32.2 |
| coremltools | None |
| numpy | 2.1.3 |
| laya-apple | 1.0.2 @ 0ae219ea05b8 |
| Shipped routing profile matches | no |

Configuration: quick, warmup 3, iters 20, latency measured in-process (one process for every configuration). Wall time 31 s.

## laya-typed-decisions

Revision `f9ab0b228f0fc0f14d873dbc99038f135c2da1b2`, weights sha256 `4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e`.

- MLX FP16 parity: passed
- ANE parity: unavailable: BackendUnavailableError: device='ane' needs coremltools: install the [ane] extra (uv sync --extra ane)
  - to build and validate here: `uv run laya-apple artifacts build laya-typed-decisions`
- auto routing (profile: None):
  - L64 q1: gpu (ane_runtime_unavailable)
  - L96 q1: gpu (ane_runtime_unavailable)
  - L128 q1: gpu (ane_runtime_unavailable)
  - L256 q1: gpu (ane_runtime_unavailable)
  - L512 q1: gpu (ane_runtime_unavailable)
  - L1024 q1: gpu (ane_runtime_unavailable)
  - L64 q4: gpu (ane_runtime_unavailable)
  - L128 q4: gpu (ane_runtime_unavailable)
- heterogeneous: skipped: no validated ANE artifacts on this machine

Warm latency, in-process:

| L | q | device | forward P50/P95/P99 ms | predict P50/P95/P99 ms |
|---:|---:|---|---|---|
| 64 | 1 | gpu | 16.05 / 16.63 / 16.96 | 16.75 / 17.74 / 18.01 |
| 128 | 1 | gpu | 28.07 / 31.39 / 31.53 | 29.26 / 31.95 / 32.75 |
| 512 | 1 | gpu | 96.30 / 97.02 / 97.26 | 96.96 / 97.48 / 97.53 |
| 128 | 4 | gpu | 90.47 / 90.87 / 90.99 | 91.16 / 91.76 / 91.86 |
