# Contributing

## Dev setup

To run laya-apple from source (Apple silicon, Python 3.11–3.13):

```bash
git clone https://github.com/tc3oliver/laya-apple && cd laya-apple
uv sync --extra ane --extra convert          # MLX + Neural Engine + artifact builds
uv run laya-apple artifacts build laya-typed-decisions   # optional: build + parity-validate ANE artifacts here (~5 min)
```

To develop, pick one of these (`uv sync` removes extras it is not given):

```bash
uv sync --extra dev                          # tests, ruff, psutil
uv sync --extra dev --extra ane              # + Neural Engine backend (coremltools==9.0)
uv sync --extra dev --extra ane --extra convert     # + building ANE artifacts (torch==2.7.0)
uv sync --extra dev --extra reference        # + regenerating PyTorch FP32 goldens
```

`ane`, `convert` and `reference` pin `numpy<2.2` and specific `torch`/`transformers`/`laya`
versions; do not relax those pins in a PR unless the PR is specifically about updating them.

## Running tests

Tests are marked with `integration`, `parity`, `ane` and `stress` (see
`[tool.pytest.ini_options]` in `pyproject.toml`). The fast suite is everything else and
needs no downloaded checkpoints or built artifacts:

```bash
uv run pytest -q -m "not integration and not parity and not ane and not stress"
```

`integration`, `parity` and `ane` tests need a checkpoint already in the Hugging Face
cache (run once online, or `laya-apple download <model>`, first):

```bash
HF_HUB_OFFLINE=1 uv run pytest -q -m "integration or parity or ane"
```

`LAYA_APPLE_CACHE` points laya-apple's own cache (build artifacts, verification stamps)
at a directory of your choice; it defaults to `~/.cache/laya-apple`. Set `HF_HUB_OFFLINE=1`
once checkpoints are cached to keep tests from touching the network.

Stress tests are opt-in and long-running:

```bash
LAYA_APPLE_STRESS=1 uv run pytest -q -m stress                    # ~60s
LAYA_APPLE_STRESS=1 LAYA_APPLE_STRESS_SECONDS=600 uv run pytest -q -m stress   # release soak
```

### Which tests a PR needs

Run the fast suite for every PR. Run the other tiers only when they apply to what you
changed — most PRs need only the fast suite plus lint.

| Tier | Command | Hardware / setup | Run it when your PR touches |
|---|---|---|---|
| Fast (unit) | `uv run pytest -q -m "not integration and not parity and not ane and not stress"` | None — no checkpoints, no ANE, runs in CI | Everything. Always run this before opening a PR. |
| Checkpoint / integration | `HF_HUB_OFFLINE=1 uv run pytest -q -m integration` | A downloaded checkpoint (`laya-apple download <model>`) | Model loading, prompts, the schema, or anything that calls `predict()` end to end |
| Parity | `HF_HUB_OFFLINE=1 uv run pytest -q -m parity` | A downloaded checkpoint; `reference` extra if you're regenerating goldens | Parity tolerances, calibration math, or the MLX/PyTorch numerics |
| ANE validation | `HF_HUB_OFFLINE=1 uv run pytest -q -m ane` | Apple Silicon with the ANE, `ane` extra, built artifacts (`laya-apple artifacts build <model>`) | The Core ML backend, artifact conversion, the placement probe, or the compute-plan check |
| Stress / soak | `LAYA_APPLE_STRESS=1 uv run pytest -q -m stress` | Apple Silicon; minutes to ~10 min for the release soak | Worker lifecycle, queueing, or anything claiming improved reliability under load |
| Full benchmark suite | see "How to benchmark" below | Apple Silicon, significant wall-clock time | A new benchmark report or a routing-threshold change — **not** routine PRs |

By what a PR touches:

- **Docs-only** (`docs/`, `README.md`, this file): fast suite only, for anything that
  might have broken a doctest or a linked example; often nothing to run beyond lint.
- **Scripts** (`scripts/`): fast suite; add integration if the script exercises
  `predict()` or artifact loading.
- **Backend code** (`laya_apple/backends/`, `laya_apple/model.py`): fast suite +
  integration; add ANE validation if you touched the Core ML backend, and parity if the
  change could affect numerics.
- **Routing** (`laya_apple/routing.py`, `laya_apple/scheduling.py`,
  `laya_apple/derivation.py`, `laya_apple/profiles.py`): fast suite + integration; a
  threshold change needs new measurements (see "Artifact release policy" below) and the
  relevant `scripts/derive_*.py --check`.
- **Artifacts** (`laya_apple/artifacts.py`, `laya_apple/conversion/`,
  `laya_apple/lifecycle.py`): fast suite + ANE validation; a new or changed artifact
  configuration needs the full gate in "Artifact release policy" below.

Do not run the full benchmark suite or a release soak for a change unrelated to
performance or reliability — it is expensive and its purpose is to validate a release,
not every PR.

## Lint and format

```bash
uv run ruff check .
uv run ruff format laya_apple scripts tests
```

`research/` is excluded from ruff entirely. `laya_apple/models/modernbert_mlx.py` and
`laya_apple/conversion/torch_reference.py` are vendored upstream code kept byte-identical
to their source revision (see `NOTICE`) and are excluded from `ruff format`; leave their
formatting alone.

## Release gate

`scripts/release_gate.py` runs the full pre-release check: the install matrix, the slow
pytest markers, and (optionally) a soak test. See `--help` for the current flags,
including `--quick` (skip the install matrix and slow markers) and `--soak SECONDS`. Run
it before proposing a release, not as a substitute for the fast suite during normal
development.

Publishing to PyPI is automated by `.github/workflows/release.yml` with Trusted
Publishing; how to build locally and cut a release is in
[`docs/publishing.md`](docs/publishing.md).

## How to run parity checks

To check one model against the shipped PyTorch FP32 goldens on a specific device:

```bash
laya-apple parity laya-typed-decisions --device gpu   # MLX, FP16 by default
laya-apple parity laya-typed-decisions --device ane    # Core ML, needs built ANE artifacts
laya-apple parity laya-typed-decisions --device gpu --dtype float32
```

This is the same gate used in `CONTRIBUTING.md`'s "Artifact release policy" and reports
hard mismatches, near-tie flips and max probability error against the tolerances in
`laya_apple/parity/__init__.py`. Run it for any change that could affect numerics
(model code, calibration, dtype handling, the ANE graph).

## How to benchmark

- **Quick, cross-model sanity check:** `scripts/hardware_report.py --quick` — the fast
  pass used for a community hardware-results bundle (see "How to add a hardware result"
  below).
- **Single model, single device:** `laya-apple benchmark <model> [--device gpu|ane]` —
  useful while iterating on one backend without running the full suite.
- **Full release-quality benchmark:** the methodology, exact commands and raw-data
  layout are in [`docs/reproducibility.md`](docs/reproducibility.md). This is what
  produces a report like `benchmarks/v1.0.md`; it takes significant wall-clock time and
  is for release reports, not routine PRs (see "Which tests a PR needs" above).

## How to add a hardware result

laya-apple's own numbers all come from one machine (`docs/support-matrix.md`).
Community results on other Apple Silicon extend that coverage. See
[`docs/community-benchmarks.md`](docs/community-benchmarks.md) for the bundle format,
the `hardware-results/<soc>-macos<major>/` layout `scripts/hardware_report.py`
produces, and how to submit one as a PR. Several starter issues for specific hardware
are seeded under the `benchmark` and `hardware` labels — see "Where issues are" below.

## How to modify a backend

Backends live in `laya_apple/backends/` (`base.py` defines the interface, `mlx.py` is
the GPU backend, `coreml_ane.py` is the Neural Engine backend), with the ANE artifact
build and graph rewrite in `laya_apple/conversion/`. Whatever you change, these must
stay true:

- **Parity.** The backend's output still passes the parity gate at the shipped
  tolerances — run `laya-apple parity <model> --device <device>` (above) before and
  after your change to see the delta. See "Non-negotiable rules" below: tolerances are
  never loosened to make a change pass.
- **No silent fallback.** If your change adds or touches a device, bucket, precision or
  `except`-around-backend-selection decision, add a row to
  [`docs/no-silent-fallback.md`](docs/no-silent-fallback.md) describing the new path and
  add a test that pins the behavior (see the existing rows and their tests for the
  pattern).
- **The placement probe.** Any change to how or when the ANE model loads should keep the
  runtime placement probe intact (`docs/no-silent-fallback.md`, "Runtime placement
  probe"); do not bypass or weaken `PROBE_MAX_RATIO`.

A new or changed ANE artifact configuration additionally needs the full gate in
"Artifact release policy" below before it can ship.

## Where issues are

Open issues are labeled by area (`coreml`, `mlx`, `ane`, `scheduler`, `correctness`,
`documentation`, `benchmark`, `hardware`, `research`, `performance`) and by how
self-contained they are (`good first issue`, `help wanted`). `.github/labels.yml` is the
canonical label set; `scripts/github_seed.sh` applies it to the repository.

## Non-negotiable rules

These are enforced by tests and reviewed as blocking, not stylistic:

- **Parity tolerances are never loosened.** The shipped tolerances
  (`laya_apple/parity/__init__.py`) are FP32 `1e-4`, FP16 `0.02` max absolute probability
  difference, and zero hard mismatches. A PR that needs a looser tolerance to pass is a
  regression, not a passing test; fix the regression instead.
- **No silent fallback.** A request never runs on a different device, bucket, precision,
  or the network (when offline was requested) without that being visible in the result or
  the exception. See [`docs/no-silent-fallback.md`](docs/no-silent-fallback.md). If your
  change adds or touches a fallback-relevant code path (a device or bucket decision, an
  `except` clause around backend selection, a routing decision), add a row to that page
  and a test that pins the behavior.
- **Generated Core ML artifacts and model weights are never committed.** `.mlpackage`,
  `.mlmodelc`, `.safetensors` and similar are gitignored; build or download them locally.
- **Benchmarks record raw data and are reproducible.** A new or changed benchmark follows
  [`docs/benchmarks.md`](docs/benchmarks.md): it writes raw per-request data, not only
  summary statistics, and states exactly how to reproduce it.
- **Public API changes follow `docs/api.md`.** The public API, the CLI, the stable
  `info()` keys, the routing-reason strings, and the artifact manifest format are covered
  by Semantic Versioning as of 1.0. Read the deprecation policy in
  [`docs/api.md`](docs/api.md) before changing or removing any of them.

## Artifact release policy

An ANE artifact configuration can ship only if it passes the same correctness gate as
the shipped ones. That applies to a new graph variant, a new bucket, a new compute-unit
setting, and any quantized or otherwise optimized export.

**The gate**, all on the tested profile:
- a 100% ANE compute plan with 0 device transitions;
- the runtime placement probe;
- the full parity gate against the upstream PyTorch FP32 goldens, with FP16
  probability error ≤ 0.02 and 0 hard mismatches.

**Other rules:**
- **Failed variants stay research.** A variant that fails, such as a 4-, 6- or 8-bit
  quantization, or a graph that is faster but numerically off, is recorded under
  `research/` with its raw data. It is never registered, offered or routed to.
- **Fixed shapes only.** The ANE runs fixed-shape buckets that each passed parity. A
  request that fits no validated bucket raises `UnsupportedShapeError`. Dynamic or
  enumerated shapes are not offered unless they are re-validated on the current runtime.
  On the tested profile they ran entirely on the CPU.
- **Routing changes need evidence.** A change to routing thresholds or auto buckets
  needs new measurements, derived with `laya_apple/derivation.py` and recorded in
  `CHANGELOG.md`.

## Commits and pull requests

- `main` is protected: every change, including a maintainer's, lands through a pull
  request from a branch named `feat/…`, `fix/…`, `docs/…`, `bench/…`, `research/…` or
  `release/…`. CI (`test`) must pass, and pull requests are squash-merged. Force pushes
  to `main` and rewriting published tags are not allowed.
- Write commit subjects in the imperative mood ("Add X", not "Added X" or "Adds X"),
  under about 70 characters.
- Explain *why* a change was made in the body, not just what changed; the diff already
  shows what changed.
- Keep a pull request to one logical change. Update `CHANGELOG.md` under `[Unreleased]`
  for any user-visible change.
- Before opening a PR: run the fast test suite, `ruff check .`, and `ruff format --check`.
  If your change touches parity, routing, placement or the artifact manifest, also run
  the relevant `scripts/derive_*.py --check` or `scripts/make_goldens.py --check`.
- Coding agents follow [`AGENTS.md`](AGENTS.md), which condenses these rules; `CLAUDE.md`
  imports it for Claude Code.

### Pull request titles

Pull requests are squash-merged, so the title becomes the commit subject on `main`. Use
`<type>(<scope>): <summary>`:

- **type**: `feat`, `fix`, `perf`, `docs`, `test`, `refactor`, `build`, `ci`, `chore` or
  `bench`;
- **scope** (optional): a short lowercase area such as `routing`, `ane`, `mlx`,
  `hardware` or `community`;
- **summary**: imperative, lowercase after the colon, no trailing period.

```text
bench(hardware): add M3 Max benchmark result
fix(ane): reject artifacts that fail placement validation
perf(scheduler): reduce short-request queueing
docs(community): clarify hardware benchmark submissions
```

A title in a different format does not block a pull request; the maintainer can adjust it
when merging. Titles of already-merged pull requests are not changed.
