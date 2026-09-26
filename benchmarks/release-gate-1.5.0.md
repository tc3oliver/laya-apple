# Release gate — laya-apple 1.5.0

- generated: 2026-09-26T19:41:09+0800
- git revision: `50707a70ddf48dc0a6336dc96f11977f9da0f7e1` (dirty)
- python: 3.12.14 (main, Sep  1 2026, 14:09:38) [Clang 22.1.3 ]
- platform: {"soc": "Apple M4 Max", "macos": "26.6.2", "macos_build": "25G83", "coremltools": "9.0"}
- quick mode: False
- soak seconds: 0
- **result: PASS**

| step | status | duration (s) | required |
|---|---|---|---|
| ruff check | pass | 0.03 | True |
| ruff format --check | pass | 0.02 | True |
| derive_routing --check | pass | 0.07 | True |
| make_goldens --check | pass | 0.05 | True |
| derive_placement --check | pass | 0.04 | True |
| pytest | pass | 405.01 | True |
| clean install matrix | pass | 53.04 | True |
| artifacts verify | pass | 14.15 | True |
| manifest schema | pass | 0.0 | True |
| no silent fallback tests | pass | 0.01 | True |
| docs versions | pass | 0.0 | True |
| soak | skipped | 0.0 | False |
