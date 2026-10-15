# Release gate — laya-apple 1.7.0

- generated: 2026-10-15T19:26:02+0800
- git revision: `c9090e8a7ae353a3415ab7a71fb698eeee51d779`
- python: 3.12.14 (main, Sep  1 2026, 14:09:38) [Clang 22.1.3 ]
- platform: {"soc": "Apple M4 Max", "macos": "27.0", "macos_build": "26A428", "coremltools": "9.0"}
- quick mode: False
- soak seconds: 0
- **result: PASS**

| step | status | duration (s) | required |
|---|---|---|---|
| ruff check | pass | 0.03 | True |
| ruff format --check | pass | 0.03 | True |
| derive_routing --check | pass | 0.07 | True |
| make_goldens --check | pass | 0.05 | True |
| derive_placement --check | pass | 0.03 | True |
| pytest | pass | 1013.89 | True |
| clean install matrix | pass | 31.54 | True |
| artifacts verify | pass | 14.99 | True |
| manifest schema | pass | 0.0 | True |
| no silent fallback tests | pass | 0.01 | True |
| docs versions | pass | 0.0 | True |
| soak | skipped | 0.0 | False |
