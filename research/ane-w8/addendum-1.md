# Addendum 1: latency scope cut

`criteria.md` is unchanged. This addendum was written after the build and parity results of
both configurations (`raw/*/L*-w8-*/build.json`, commit edd38e5) and before any latency or
probe data. It follows a maintainer policy of short screens with early stops. It removes
measurements and changes no criterion, limit or verdict rule.

**Change.** Latency and the placement probe (gate results 3 and 4) run only on the cells whose
build passed both placement and parity:

| model | L | config | command |
|---|---:|---|---|
| `laya-typed-decisions` | 64 | `w8-pt` | `latency.py --config w8-pt --model laya-typed-decisions --length 64` |
| `laya-typed-decisions` | 96 | `w8-gc32` | `latency.py --config w8-gc32 --model laya-typed-decisions --length 96` |
| `laya-typed-decisions` | 128 | `w8-gc32` | `latency.py --config w8-gc32 --model laya-typed-decisions --length 128` |

**Consequences.**
- Every other cell and configuration failed parity, which already makes it a documented no-ship
  under `criteria.md`. Timing it cannot change that verdict.
- Its probe and latency results are reported as "not timed (parity FAIL; addendum 1)", never as
  PASS or FAIL.
- The latency protocol, the 1.05 limit, the paired statistic and the wording rule are unchanged
  for the three cells above.
