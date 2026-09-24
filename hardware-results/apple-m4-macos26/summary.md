# Hardware report: Apple M4, macOS 26.2

Matrix row (docs/community-benchmarks.md):

| Mac | MLX | ANE | Auto uses ANE | Heterogeneous |
|---|---|---|---|---|
| Apple M4 (32 GB, macOS 26.2) | ✓ | ✓ | no | untested |

## Environment

| | |
|---|---|
| SoC | Apple M4 (Mac16,10) |
| Memory | 32 GB |
| macOS | 26.2 (25C56) |
| Python | 3.12.13 |
| mlx | 0.32.2 |
| coremltools | 9.0 |
| numpy | 2.1.3 |
| laya-apple | 1.0.2 @ e0d11a7249bf |
| Shipped routing profile matches | no |

Configuration: quick, warmup 3, iters 20, latency measured in-process (one process for every configuration). Wall time 166 s.

## laya-typed-decisions

Revision `f9ab0b228f0fc0f14d873dbc99038f135c2da1b2`, weights sha256 `4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e`.

- MLX FP16 parity: passed
- ANE parity: passed
- auto routing (profile: None):
  - L64 q1: gpu (platform_not_validated)
  - L96 q1: gpu (platform_not_validated)
  - L128 q1: gpu (platform_not_validated)
  - L256 q1: gpu (platform_not_validated)
  - L512 q1: gpu (platform_not_validated)
  - L1024 q1: gpu (platform_not_validated)
  - L64 q4: gpu (platform_not_validated)
  - L128 q4: gpu (platform_not_validated)
- heterogeneous: skipped: device='auto' does not route to the ANE here (platform_not_validated); `uv run laya-apple calibrate laya-typed-decisions` writes a local profile that enables it

Warm latency, in-process:

| L | q | device | forward P50/P95/P99 ms | predict P50/P95/P99 ms |
|---:|---:|---|---|---|
| 64 | 1 | gpu | 20.75 / 21.30 / 21.31 | 21.24 / 22.77 / 22.78 |
| 128 | 1 | gpu | 37.05 / 39.93 / 40.58 | 37.35 / 38.96 / 40.61 |
| 512 | 1 | gpu | 133.95 / 137.64 / 138.07 | 137.97 / 139.10 / 140.05 |
| 128 | 4 | gpu | 130.54 / 132.00 / 132.27 | 132.56 / 133.90 / 134.38 |
| 64 | 1 | ane | 13.42 / 13.57 / 13.58 | 13.84 / 14.16 / 14.24 |
| 96 | 1 | ane | 14.17 / 14.39 / 14.54 | 14.20 / 14.43 / 14.45 |
| 128 | 1 | ane | 14.54 / 15.02 / 15.04 | 14.75 / 15.18 / 15.23 |
