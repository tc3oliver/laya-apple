# Addendum 2: replication of the typed L64 `w8-pt` latency result

`criteria.md` and `addendum-1.md` are unchanged. This addendum was written after the first run's
results (`raw/laya-typed-decisions/L64-w8-pt/latency.json`, 0.649× FP16 predict P50,
95% CI [0.647, 0.650]) and before any replication data. That first result is one paired run in
one session; the README says it needs a separately preregistered replication. This is it. It
adds measurements only and changes no criterion, limit or verdict rule of the original cell.

**What is measured.** The same cell (`laya-typed-decisions`, L64, `w8-pt`), the same
artifacts, the same protocol:
- the existing W8 artifact from the build run, not rebuilt. Its size must equal the build record
  (`compiled_bytes` 372,335,130; `weight.bin` 369,653,888). The first run recorded no W8 hash, so
  its tree SHA-256 is recorded now. The FP16 baseline is loaded through `load_verified`, which
  checks the manifest hash; it must be `1273fcd3…e720d0`, as in the first run;
- `scripts/latency.py` unchanged in method: warm-up 20 per arm, 10 cycles × one 100-forward window
  per arm, A B in even cycles and B A in odd cycles, paired predict-P50 ratio, geometric mean,
  95% Student t interval on the log ratios (df = 9), plus the runtime placement probe;
- the only script change is an output-directory flag (`--raw`) that refuses to overwrite, so the
  first run's raw file is never touched.

**Runs.** 3 independent runs, each a fresh process, in an exclusive slot (oMLX stopped, no other
benchmark, build, test or install). Raw data goes to `raw/replication-l64/run-<n>/`.

**Screen and stop.** Run 1 is the screen. If its 95% CI includes 1.0, the replication stops
there and the result is **not reproduced**. A screen can only stop, never pass. Runs 2 and 3
follow only if run 1's upper bound is below 1.0.

**Reproduced** means all three hold:
1. all 3 runs completed;
2. every run's paired log-ratio 95% CI upper bound is < 1.0;
3. the pooled point estimate, the geometric mean of all 30 pair ratios, is within ±0.1 of 0.649
   (i.e. in [0.549, 0.749]).

Anything else is **not reproduced**. A run interrupted by a documented external event is re-run
once in full and its data kept under `raw/replication-l64/aborted/`. No re-run because a result
is unwelcome, and no extra runs or cycles.

**Also recorded, not part of the reproduced rule** (reported side by side, never merged):
- per run: FP16 and W8 predict P50 (median window P50) and P99 (all 1,000 samples per arm),
  forward-ratio, load average before and after, the 1.05 verdict of `criteria.md`;
- run-to-run spread of the per-run ratios;
- placement probe per run (W8 ratio to `CPU_ONLY` ≤ 0.8);
- after timing, with the slot released: the unchanged placement gate (100% ANE, 0 transitions)
  and the unchanged FP16 parity gate (probability error ≤ 0.02, 0 hard mismatches, near-tie flips
  listed) on the same W8 artifact (`scripts/replicate_gate.py`);
- size: `weight.bin` and `model.mlmodelc` bytes, W8 against the shipped FP16 artifact.

**Analysis.** `scripts/replicate_analyze.py` writes `raw/replication-l64/replication.json`.
It uses `analyze.latency_stats` unchanged.

Commands (environment as in the README):

```bash
uv run --extra ane python research/ane-w8/scripts/latency.py --config w8-pt \
    --model laya-typed-decisions --length 64 --raw research/ane-w8/raw/replication-l64/run-<n>
uv run python research/ane-w8/scripts/replicate_analyze.py
uv run --extra ane python research/ane-w8/scripts/replicate_gate.py --config w8-pt \
    --model laya-typed-decisions --length 64 --raw research/ane-w8/raw/replication-l64
```

Nothing is shipped, registered or routed to by this replication, whatever the result.
