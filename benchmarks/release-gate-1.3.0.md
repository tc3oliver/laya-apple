# Release gate — laya-apple 1.3.0

- generated: 2026-09-25T16:50:01+0800
- git revision: `49767b21780d15734a78631c22766a3845e93fe5` (dirty)
- python: 3.12.14 (main, Sep  1 2026, 14:09:38) [Clang 22.1.3 ]
- platform: {"soc": "Apple M4 Max", "macos": "26.6.2", "macos_build": "25G83", "coremltools": "9.0"}
- quick mode: False
- soak seconds: 600
- **result: PASS**

| step | status | duration (s) | required |
|---|---|---|---|
| ruff check | pass | 0.02 | True |
| ruff format --check | pass | 0.02 | True |
| derive_routing --check | pass | 0.07 | True |
| make_goldens --check | pass | 0.05 | True |
| derive_placement --check | pass | 0.03 | True |
| pytest | pass | 286.65 | True |
| clean install matrix | pass | 35.89 | True |
| artifacts verify | pass | 15.06 | True |
| manifest schema | pass | 0.0 | True |
| no silent fallback tests | pass | 0.01 | True |
| docs versions | pass | 0.0 | True |
| soak | pass | 630.66 | True |
