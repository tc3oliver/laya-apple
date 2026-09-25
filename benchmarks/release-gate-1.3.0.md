# Release gate — laya-apple 1.3.0

- generated: 2026-09-25T16:25:05+0800
- git revision: `48a16e96b258f9bef52690697d898c9853f8ff81`
- python: 3.12.14 (main, Sep  1 2026, 14:09:38) [Clang 22.1.3 ]
- platform: {"soc": "Apple M4 Max", "macos": "26.6.2", "macos_build": "25G83", "coremltools": "9.0"}
- quick mode: False
- soak seconds: 0
- **result: PASS**

| step | status | duration (s) | required |
|---|---|---|---|
| ruff check | pass | 0.06 | True |
| ruff format --check | pass | 0.02 | True |
| derive_routing --check | pass | 0.08 | True |
| make_goldens --check | pass | 0.06 | True |
| derive_placement --check | pass | 0.04 | True |
| pytest | pass | 347.87 | True |
| clean install matrix | pass | 35.52 | True |
| artifacts verify | pass | 16.71 | True |
| manifest schema | pass | 0.0 | True |
| no silent fallback tests | pass | 0.01 | True |
| docs versions | pass | 0.0 | True |
| soak | skipped | 0.0 | False |
