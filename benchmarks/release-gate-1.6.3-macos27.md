# Release gate — laya-apple 1.6.3

- generated: 2026-10-03T14:25:26+0800
- git revision: `836b5284281e9a828e0a1d8c50cef74ab3fdf059`
- python: 3.12.14 (main, Sep  1 2026, 14:09:38) [Clang 22.1.3 ]
- platform: {"soc": "Apple M4 Max", "macos": "27.0", "macos_build": "26A428", "coremltools": "9.0"}
- quick mode: False
- soak seconds: 0
- **result: PASS**

| step | status | duration (s) | required |
|---|---|---|---|
| ruff check | pass | 0.48 | True |
| ruff format --check | pass | 0.04 | True |
| derive_routing --check | pass | 0.07 | True |
| make_goldens --check | pass | 0.05 | True |
| derive_placement --check | pass | 0.03 | True |
| pytest | pass | 1285.42 | True |
| clean install matrix | pass | 117.7 | True |
| artifacts verify | pass | 307.09 | True |
| manifest schema | pass | 0.0 | True |
| no silent fallback tests | pass | 0.01 | True |
| docs versions | pass | 0.0 | True |
| soak | skipped | 0.0 | False |

## Clean-wheel fetch check (after the gate)

The 1.6.3 wheel built from this commit, installed into a fresh Python 3.12 venv outside the source tree, fetched `laya-typed-decisions` (L64, L96, L128) from the pinned prebuilt revision into an empty cache: 254.6 s wall for the three buckets, every bucket's parity gate at its registered path (25.2 / 26.6 / 26.9 s), placement probe ratios 0.507 / 0.365 / 0.365, `artifacts verify` OK for all three, and no `BUILDING.json` or `PENDING.json` left in the cache. Same machine as the gate: Apple M4 Max, macOS 27.0 (26A428), coremltools 9.0, local LLM service stopped. One run; not a benchmark.
