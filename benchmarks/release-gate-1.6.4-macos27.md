# Release gate — laya-apple 1.6.4

- generated: 2026-10-13T20:15:27+0800
- git revision: `6a8a4803d0528b88d546cb1d79f19f1afe215f96`
- python: 3.12.14 (main, Sep  1 2026, 14:09:38) [Clang 22.1.3 ]
- platform: {"soc": "Apple M4 Max", "macos": "27.0", "macos_build": "26A428", "coremltools": "9.0"}
- quick mode: False
- soak seconds: 0
- **result: PASS**

| step | status | duration (s) | required |
|---|---|---|---|
| ruff check | pass | 0.31 | True |
| ruff format --check | pass | 0.05 | True |
| derive_routing --check | pass | 0.07 | True |
| make_goldens --check | pass | 0.05 | True |
| derive_placement --check | pass | 0.04 | True |
| pytest | pass | 1037.41 | True |
| clean install matrix | pass | 118.19 | True |
| artifacts verify | pass | 311.08 | True |
| manifest schema | pass | 0.0 | True |
| no silent fallback tests | pass | 0.01 | True |
| docs versions | pass | 0.0 | True |
| soak | skipped | 0.0 | False |
