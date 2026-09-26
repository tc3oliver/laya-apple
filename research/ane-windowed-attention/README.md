# Windowed (local) attention BC1S graph

**Status: preregistered, not run.** The criteria are in [`criteria.md`](criteria.md),
committed before any data. The preregistration is a comment on
[#15](https://github.com/tc3oliver/laya-apple/issues/15), linked from
[#14](https://github.com/tc3oliver/laya-apple/issues/14).

This is research only. The `laya_apple` package never imports it, and no artifact built here
is registered, offered or routed to.

**Question.** Phase −1's exact block-local rewrite (`ConvAttention._windowed`,
`research/phase-0-feasibility/ane-long-context.md` H2) is carried onto the production BC1S
graph for the 18 local layers of `laya` and `laya-typed-decisions`. Does it pass the unchanged
artifact gate: the 100% ANE compute plan with 0 transitions, the placement probe and the FP16
parity gate? Is it at least 5% faster than the masked graph in a paired comparison?

- At the shipped buckets 64/96/128 (#15), against the shipped artifacts.
- At 256/512 (#14), against a masked build from the same script, with MLX FP16 as a
  descriptive reference.

## How to run

Environment for every step:

```bash
export LAYA_APPLE_CACHE=/Volumes/Data/cache/laya-apple       # production cache: shipped baselines, read only
export HF_HOME=/Volumes/Data/cache/huggingface HF_HUB_OFFLINE=1
# research artifacts go to /Volumes/Data/cache/laya-apple-research/windowed (override with --root)
```

1. **Build + placement + parity.** CPU-heavy, then the ANE for parity; not timing-sensitive:
   ```bash
   uv run --extra ane --extra convert python research/ane-windowed-attention/scripts/build_windowed.py --variant windowed
   uv run --extra ane --extra convert python research/ane-windowed-attention/scripts/build_windowed.py --variant masked   # 256/512 only
   ```
2. **Latency + placement probe.** TIMING-SENSITIVE, exclusive slot, oMLX stopped:
   ```bash
   uv run --extra ane python research/ane-windowed-attention/scripts/latency.py
   ```
3. **Tables**, no hardware:
   ```bash
   uv run python research/ane-windowed-attention/scripts/analyze.py
   ```

`--model` and `--length` restrict any step to a subset of cells.

## Files

- `criteria.md`: the preregistration.
- `scripts/windowed_body.py`: the rewrite, applied on top of the production `ConvBody`.
- `scripts/build_windowed.py`: the production conversion, the FP32 layout and exactness checks,
  compile, the placement gate, first-load time and parity.
- `scripts/latency.py`: paired A/B windows, the MLX windows at 256/512, and the runtime
  placement probe.
- `scripts/analyze.py`: per-cell verdicts, per-row parity, `results.json` and `tables.md`.
- `scripts/common.py`, `scripts/paired.py`: the same helpers as `research/ane-w8/` (#118), with this
  track's constants. The paired statistics are copied unchanged from
  `research/coreml-prebind-predict/scripts/gate.py`.
- `raw/`: everything measured, written by the scripts.
