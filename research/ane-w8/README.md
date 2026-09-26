# W8-palettized ANE artifacts

**Status: preregistered, not run.** The criteria are in [`criteria.md`](criteria.md),
committed before any data. The preregistration is a comment on
[#11](https://github.com/tc3oliver/laya-apple/issues/11).

This is research only. The `laya_apple` package never imports it, and no artifact built here
is registered, offered or routed to.

**Question.** Do 8-bit palettized weights (coremltools k-means, per-tensor LUT) keep every
shipped BC1S ANE artifact inside the unchanged artifact gate? The cells are the shipped buckets:
64/96/128 for `laya` and `laya-typed-decisions`, and 64/96/128/256 for `laya-multilingual`.
The gate is the 100% ANE compute plan, the placement probe and the FP16 parity gate. What
happens to size and to short-L latency against the shipped FP16 artifact?

## How to run

Environment for every step:

```bash
export LAYA_APPLE_CACHE=/Volumes/Data/cache/laya-apple       # production cache: baselines, read only
export HF_HOME=/Volumes/Data/cache/huggingface HF_HUB_OFFLINE=1
# research artifacts go to /Volumes/Data/cache/laya-apple-research/w8 (override with --root)
```

1. **Build + placement + parity**, primary config. CPU-heavy, then the ANE for parity; not
   timing-sensitive:
   ```bash
   uv run --extra ane --extra convert python research/ane-w8/scripts/build_w8.py --config w8-pt
   ```
2. **Secondary config**, only for cells whose `w8-pt` parity failed. The script skips the others:
   ```bash
   uv run --extra ane --extra convert python research/ane-w8/scripts/build_w8.py --config w8-gc32 --only-failed-primary
   ```
3. **Latency + placement probe.** TIMING-SENSITIVE, exclusive slot, oMLX stopped:
   ```bash
   uv run --extra ane python research/ane-w8/scripts/latency.py --config w8-pt
   uv run --extra ane python research/ane-w8/scripts/latency.py --config w8-gc32   # if step 2 built any
   ```
4. **Tables**, no hardware:
   ```bash
   uv run python research/ane-w8/scripts/analyze.py
   ```

`--model` and `--length` restrict any step to a subset of cells.

## Files

- `criteria.md`: the preregistration.
- `scripts/build_w8.py`: the production conversion, plus `palettize_weights`, compile, the
  placement gate and parity.
- `scripts/latency.py`: paired A/B windows against the shipped FP16 artifact, and the runtime
  placement probe.
- `scripts/analyze.py`: per-cell verdicts, per-row parity, `results.json` and `tables.md`.
- `scripts/common.py`: the environment guard, sizes, the plan diagnostic and the parity
  recording pass.
- `scripts/paired.py`: the paired gate statistics. They are copied unchanged from
  `research/coreml-prebind-predict/scripts/gate.py`.
- `raw/`: everything measured, written by the scripts.
