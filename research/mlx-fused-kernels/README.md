# MLX fused kernels: candidates (research, no data yet)

Status: **candidate list only.** No code and no measurements. Any experiment on these is
preregistered in a GitHub issue first, with criteria fixed before a run: the unchanged FP16
parity gate (`laya_apple/parity/__init__.py`), decisions identical to the eager MLX
forward on every golden row, and a latency comparison with `scripts/compare_bench.py`
against the same-session control. A kernel that fails any of them is a documented no-ship.

Scope: `laya_apple/models/modernbert_mlx.py`, the one MLX graph that serves all three models,
laya-multilingual included. `mx.compile` (elementwise fusion), the token-id cache and
length-bucketed batching are separate pull requests and are not repeated here.

## Step 0: find where the time goes

Before any kernel is written, profile the eager forward per op class at L64, L512 and
L1024 (q1 and q4), using a Metal capture (`mx.metal.start_capture`) or per-layer timing
with synchronisation. Record matmul, attention, norm, mask and head shares. Rank the
candidates below by that measured share, not by this list's order.

## Candidates

| # | Candidate | Why it might pay | Exactness expectation |
|---|---|---|---|
| 1 | **Banded (sliding-window) attention.** Two of every three encoder layers attend only within ±64 tokens (`local_attention` 128), but `mx.fast.scaled_dot_product_attention` computes the full L×L scores and masks them. A kernel that visits only the band | At L1024, the sliding layers do ~8× the needed attention work. Likely the largest win at long L; nothing at L ≤ 128 | Masked scores are exact zeros today, so it is exact up to reduction order. Related to the windowed ANE graph research (#15) |
| 2 | **Mask construction once per shape.** `attention_masks` rebuilds a B×1×L×L boolean sliding mask on every forward (16 MB at B16, L1024) | Removes L² memory traffic per call. Cacheable per (L, window) because the positional band is input-independent | Bitwise identical booleans. Arguably a plain optimisation, not a kernel: a candidate for a small PR of its own |
| 3 | **Residual add + LayerNorm.** Each sublayer writes `x + f(x)` and then reads it back for the next pre-norm (`mx.fast.layer_norm`) | Saves one hidden-state read and write per sublayer (~2 per layer). Bandwidth-bound at every length | Same arithmetic order is achievable, so bitwise identity is plausible |
| 4 | **RoPE on q and k in one kernel, or inside attention.** Two `mx.fast.rope` launches per layer, after a strided split of `Wqkv` | Fewer launches and one less q/k round trip. Matters most at short L, where launches dominate | The same rotation formula is exact if the angle computation matches `mx.fast.rope` |
| 5 | **GeGLU epilogue.** `split` → `gelu(value) * gate` after `Wi` | `mx.compile` already fuses the elementwise part. Only a matmul-epilogue fusion (a custom GEMM) would add more, which is likely not worth owning | GELU approximation must match `nn.gelu` (erf). Risky for FP16 rounding |
| 6 | **Decision head tail.** Marker gather, scorer, masked softmax, entropy, top-2 sort and the action head: ~20 tiny kernels on a few hundred values | Fixed per-request overhead, visible only at short L. One `mx.fast.metal_kernel` could do all of it | Reductions over ≤ 64 values: exact order is easy to match |
| 7 | **Varlen (packed, unpadded) attention.** Pack rows of different lengths into one sequence with block-diagonal masks | Removes padding entirely for multi-question requests | Needs per-token RoPE positions (`mx.fast.rope` takes one offset per batch) plus a block-diagonal mask or a varlen kernel. Highest complexity. Only after #1–#3 |

Out of scope: weight quantisation or lower precision on the GPU path. That changes numerics
and needs its own gate discussion, like the W8 ANE work.
