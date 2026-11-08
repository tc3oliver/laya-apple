# Hardware report: Apple M4, macOS 26.2

Matrix row (docs/community-benchmarks.md):

| Mac | MLX | ANE | Auto uses ANE | Heterogeneous |
|---|---|---|---|---|
| Apple M4 (16 GB, macOS 26.2) | ✓ | ✓ | yes | ✓ |

## Environment

| | |
|---|---|
| SoC | Apple M4 (Mac16,10) |
| Memory | 16 GB |
| macOS | 26.2 (25C56) |
| Python | 3.12.16 |
| mlx | 0.32.2 |
| coremltools | 9.0 |
| numpy | 2.1.3 |
| laya-apple | 1.7.0 @ 557784767646 |
| Shipped routing profile matches | no |

Configuration: quick, warmup 3, iters 20, latency measured in-process (one process for every configuration). Wall time 106 s.

## laya-typed-decisions

Revision `f9ab0b228f0fc0f14d873dbc99038f135c2da1b2`, weights sha256 `4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e`.

- MLX FP16 parity: passed
- ANE parity: passed
- auto routing (profile: local:<cache>/profiles/Apple_M4-macos26-coremltools9.0.json):
  - L64 q1: ane (validated_short_single_question_path)
  - L96 q1: ane (validated_short_single_question_path)
  - L128 q1: ane (validated_short_single_question_path)
  - L256 q1: gpu (sequence_exceeds_ane_auto_range)
  - L512 q1: gpu (sequence_exceeds_ane_auto_range)
  - L1024 q1: gpu (sequence_exceeds_ane_auto_range)
  - L64 q4: gpu (multiple_questions)
  - L128 q4: gpu (multiple_questions)
- heterogeneous: GPU-only 6.7 req/s, GPU+ANE 70.7 req/s (10.52x), 0 mismatches

Warm latency, in-process:

| L | q | device | forward P50/P95/P99 ms | predict P50/P95/P99 ms |
|---:|---:|---|---|---|
| 64 | 1 | gpu | 18.01 / 18.35 / 18.42 | 18.22 / 18.48 / 18.51 |
| 128 | 1 | gpu | 32.31 / 32.65 / 32.72 | 32.56 / 32.89 / 33.10 |
| 512 | 1 | gpu | 122.78 / 124.12 / 125.11 | 123.15 / 124.06 / 124.08 |
| 128 | 4 | gpu | 115.73 / 116.91 / 117.61 | 116.36 / 119.80 / 121.18 |
| 64 | 1 | ane | 13.23 / 14.20 / 14.31 | 12.87 / 14.46 / 14.60 |
| 96 | 1 | ane | 13.22 / 14.41 / 14.89 | 13.58 / 13.96 / 14.35 |
| 128 | 1 | ane | 14.39 / 15.63 / 15.87 | 14.07 / 14.48 / 15.30 |
