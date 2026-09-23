# AGENTS.md

These are the rules for coding agents that work on laya-apple. This file is the single
source of truth for agents; `CLAUDE.md` only imports it. Human contributors should read
[`CONTRIBUTING.md`](CONTRIBUTING.md), which has the full setup, the test tiers and the
release policy. These rules summarise it and do not replace it.

## Scope

laya-apple is an Apple-native inference runtime for Laya models. It covers:
- model loading and the MLX GPU and Core ML / ANE backends;
- routing and heterogeneous serving;
- correctness validation, artifact provenance and benchmarking.

Application-level or agent-level logic does not belong in this repository.

## Git workflow

- **`main` is stable and always releasable.** Never commit or push to it directly.
- **Check the branch before changing anything:** `git status --short --branch`. If it
  shows `main`, create a branch first. Use one of these prefixes:
  - `feat/`, `fix/`, `docs/`;
  - `bench/`, `research/`, `release/`.
- **Every change goes through a pull request.** Merge only after CI passes. Prefer
  squash merge.
- **Protect published history:**
  - never force-push `main`;
  - never rewrite `main`'s published history;
  - never move, delete or recreate a published tag.
- **Never commit generated or local-only files:**
  - generated Core ML models (`*.mlpackage`, `*.mlmodelc`);
  - model weights;
  - credentials;
  - machine-specific absolute paths.

## Validation

Run what the change needs. The table in `CONTRIBUTING.md` ("Which tests a PR needs")
gives the exact commands.

| Change | Minimum validation |
|---|---|
| Any change | `uv run ruff check .`, `uv run ruff format --check laya_apple scripts tests`, and the fast suite: `uv run pytest -q -m "not integration and not parity and not ane and not stress"` |
| Docs only | Fast suite and CI. For packaging or README changes, also run `uv build` and `uvx twine check --strict dist/*` |
| Runtime, routing, scheduler, backend, model or ANE code | The relevant `integration`, `parity` and `ane` tests on Apple silicon, plus the `scripts/derive_*.py --check`, `scripts/make_goldens.py --check` and `scripts/generate_readme_svgs.py --check` data checks |
| Performance-sensitive change | Re-run the affected benchmark and compare it with the recorded baseline in `benchmarks/` (`scripts/compare_bench.py`). Report the comparison in the PR |

- **Never loosen a gate to make CI or a benchmark pass.** This covers:
  - the parity tolerances in `laya_apple/parity/__init__.py`;
  - routing thresholds;
  - release-gate criteria.

  A tolerance or threshold change needs its own PR with measurements behind it.
- **Never edit historical raw benchmark data** (`benchmarks/*/`, `research/*/`) to change
  a result. A new measurement goes in a new file or directory, with its methodology.
- **Every performance or correctness claim in `README.md` or `docs/` must trace to
  recorded measurement data.** Name the file it comes from.

## Correctness and benchmark principles

- **Correctness comes before speed.**
  - A faster path that changes decisions against upstream Laya is a regression.
  - The FP16 gate is: probability error ≤ 0.02, 0 hard mismatches, and near-tie flips
    listed, not hidden ([`docs/correctness.md`](docs/correctness.md)).
- **Explicit ANE requests never fall back silently.** `device="ane"` either runs a
  validated artifact on the Neural Engine or raises
  ([`docs/no-silent-fallback.md`](docs/no-silent-fallback.md)).
- **Unvalidated platforms and artifacts are handled conservatively.** `auto` stays on MLX
  unless the artifact passed parity on the machine that uses it.
- **Benchmark methodology changes are recorded explicitly.** This includes changes to
  workloads, windows, arrival patterns, warm-up and hardware state. Numbers measured
  under different methodologies are never compared without saying so.
- **Keep throughput, latency, correctness and isolation as separate results.** Never
  merge them into one PASS/FAIL that hides a failing dimension.
- **Published limitations stay published.** Do not remove or soften a limitation to make
  a release look better. Remove one only when new measurements show it no longer applies.

## Releases

- **Not every PR needs a release.** A release is its own PR, or a PR whose purpose is the
  version change.
- **Keep versions in sync.** The version in `pyproject.toml`, `laya_apple.__version__`
  and the `CHANGELOG.md` heading must agree.
- **Tag only after the release PR is merged and `main` is healthy** (CI green). Tag the
  merge commit on `main` with `vX.Y.Z`.
- **Never move, delete or rewrite a published tag.** A bad release is fixed by a new
  patch version.
- **Publish to PyPI only through `.github/workflows/release.yml`,** which uses Trusted
  Publishing (GitHub OIDC). Never add a PyPI API token or password anywhere.
- **After publishing, verify a clean install:**
  - `pip install laya-apple==X.Y.Z` in a fresh environment;
  - import it and check the version;
  - check that the GitHub release, the tag and the PyPI version agree.

  Details: [`docs/publishing.md`](docs/publishing.md).
