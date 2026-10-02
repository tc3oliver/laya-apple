# Release gate — laya-apple 1.6.0

- generated: 2026-10-03T00:25:24+0800
- git revision: `f41fb41c5e9052a041aba97f4b0c2d8b5d5bc784`
- python: 3.12.14 (main, Sep  1 2026, 14:09:38) [Clang 22.1.3 ]
- platform: {"soc": "Apple M4 Max", "macos": "27.0", "macos_build": "26A428", "coremltools": "9.0"}
- quick mode: False
- soak seconds: 0
- **result: PASS**

| step | status | duration (s) | required |
|---|---|---|---|
| ruff check | pass | 0.06 | True |
| ruff format --check | pass | 0.02 | True |
| derive_routing --check | pass | 0.07 | True |
| make_goldens --check | pass | 0.05 | True |
| derive_placement --check | pass | 0.04 | True |
| pytest | pass | 1097.1 | True |
| clean install matrix | pass | 25.08 | True |
| artifacts verify | pass | 15.21 | True |
| manifest schema | pass | 0.0 | True |
| no silent fallback tests | pass | 0.01 | True |
| docs versions | pass | 0.0 | True |
| soak | skipped | 0.0 | False |
