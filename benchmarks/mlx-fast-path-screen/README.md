# MLX fast path: screen (#128)

**This is a screen only, not a full `release_bench`.** It was run for
[#128](https://github.com/tc3oliver/laya-apple/pull/128) (the MLX fast path) and is copied
here from that pull request's description so that the numbers are in a committed file. No
speedup is claimed beyond this table.

The screen can only stop a candidate; it never declares a win. It stops a candidate on any
regression above 3% in forward or predict P50, or when no primary configuration gains beyond
noise. All three candidates survived, with gains of about 1–4%. The full
`scripts/release_bench.py` + `scripts/compare_bench.py` run was not done.

## Method

- **Machine:** Apple M4 Max, MLX 0.32.2 (as recorded in #128).
- **Workload:** laya-typed-decisions, `--device gpu`,
  `laya-apple benchmark --warmup 10 --iters 100`.
- **Order:** one fresh process per arm, A-B-B-A.
- **Arms:** A = `origin/main` (7cb5c87); B = each candidate branch with its flag.
- **Metrics:** P50 in ms. Δ is B vs A; noise is the larger spread between the two A runs or
  the two B runs.

## Results

| Candidate | L | q | metric | A1 | A2 | B1 | B2 | Δ | noise |
|---|---|---|---|---|---|---|---|---|---|
| compile | 64 | 1 | forward | 9.164 | 9.374 | 8.999 | 9.044 | −2.7% | 2.3% |
| compile | 128 | 1 | forward | 12.245 | 12.204 | 11.751 | 11.924 | −3.2% | 1.5% |
| compile | 512 | 1 | forward | 35.217 | 35.107 | 34.718 | 34.805 | −1.1% | 0.3% |
| token cache | 64 | 1 | predict | 9.479 | 9.414 | 9.286 | 9.417 | −1.0% | 1.4% |
| token cache | 128 | 1 | predict | 12.223 | 12.375 | 12.285 | 12.160 | −0.6% | 1.2% |
| token cache | 512 | 1 | predict | 35.584 | 35.576 | 35.241 | 35.262 | −0.9% | 0.1% |
| length buckets | 128 | 32 | forward | 231.851 | 231.997 | 222.861 | 222.823 | −3.9% | 0.1% |

The largest regression seen in any configuration was +0.7% (token-cache L128 forward;
length-buckets L128 q1 predict).

The token-cache pass is marginal: only L512 gains beyond noise. `laya-apple benchmark`
repeats one identical request, so its `predict` result is a best case. The length-bucket
1-question configurations run the same code as `main` and stayed within noise.

## What was decided from it

- Only the token-id cache is on by default. The compiled forward
  (`LAYA_APPLE_MLX_COMPILE=1`) and length-bucketed batching
  (`LAYA_APPLE_MLX_LENGTH_BUCKETS=1`) stay opt-in.
- The gains are about 1–4% and are not a release headline.
- Changing the default for compile or length buckets would need a full `release_bench`
  comparison first.

The raw per-run output is not committed; the table above is the record.

## Correctness (from #128, not part of the screen)

Recorded in #128 on the same machine; the integration tests named here print these values
when run.

- **Compiled vs eager:** bitwise identical on every golden row (max |Δlogit| 0, max |Δaction
  logit| 0) for all 3 models, in FP16 and FP32
  (`tests/integration/test_mlx_compile_checkpoints.py`).
- **Bucketed vs in-order batching,** on golden cases padded with filler rows to more than 16
  rows (`tests/integration/test_mlx_length_buckets_checkpoints.py`): the parity gate passes
  and decisions are identical everywhere; outputs are bitwise identical except
  laya-multilingual FP32 (max |Δlogit| 9.5e-07).
- **Token-id cache:** warm-cache token ids equal the goldens for all 3 tokenizers
  (`tests/integration/test_token_cache_goldens.py`).
- **Both flags on:** `LAYA_APPLE_MLX_COMPILE=1 LAYA_APPLE_MLX_LENGTH_BUCKETS=1 laya-apple
  parity <model> --device gpu` passes for all 3 models, with 0 hard mismatches and 0 token
  mismatches.
