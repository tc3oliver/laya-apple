# Addendum 1: scope cut to a staged L512 screen

`criteria.md` is unchanged. This addendum was written before any windowed or masked artifact of
this experiment was built or measured. It follows a maintainer policy of short screens with early
stops. It removes and orders cells. It changes no criterion, limit, protocol or verdict rule.

## 1. The shipped buckets are dropped (#15 cells L64, L96, L128)

`criteria.md` already records, before any data, that the rewrite computes 100% of the dense
score entries at L ≤ 128: every 64-query block's key span covers the whole sequence. It
therefore predicts a latency FAIL there. These cells are not built or timed. For #15 they are
reported as "not run: no score work can be removed at L ≤ 128 (criteria.md, 'What the rewrite
can save'; addendum 1)". That is a structural no-change, not a measured one.

## 2. Staged run for the long cells (#14)

**Stage 1, the screen: `laya-typed-decisions` L512 only.**
1. `build_windowed.py --variant windowed --model laya-typed-decisions --length 512`
2. `build_windowed.py --variant masked --model laya-typed-decisions --length 512`
3. In the exclusive slot, `latency.py --model laya-typed-decisions --length 512`, with the
   unchanged protocol: 10 ABBA cycles of 100 forwards per arm, then the MLX window.

**Stage 2 runs only if the stage-1 latency verdict is PASS** (upper 95% bound of windowed/masked
predict P50 ≤ 0.95, the preregistered limit, meaning at least a 5% gain):
- `laya-typed-decisions` L256;
- `laya` L256 and L512;

each with the same build, parity and latency steps.

**If stage 1 is FAIL or INCONCLUSIVE:**
- the experiment stops;
- the stage-2 cells are reported as "not run (stage-1 screen did not pass; addendum 1)";
- no other variant or length is tried.

Every stage-1 result is reported as preregistered, whatever it is. That covers placement, parity,
probe, latency, first-load time, exactness and the MLX comparison.
