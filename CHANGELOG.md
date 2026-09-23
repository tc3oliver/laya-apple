# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- **README serving animation.** `docs/media/heterogeneous-serving.gif` (960×540, 20 s,
  2.2 MB) is the first figure in the README, after the one-line description and before the
  throughput figure. It replays one burst of the laya-typed-decisions bursty workload, GPU-only against GPU + ANE, from a
  per-request trace of the benchmark's arrival sequence. The P99 values it shows are the
  published v1.0 numbers.
- **Serving video and its data.** `docs/media/` holds the shareable benchmark media: the
  animation, a 1080p MP4, a poster frame, the per-request trace and a README that gives the
  source of every number on screen. `docs/readme/` keeps only README-specific static
  graphics.

### Changed

- **README throughput figure** no longer carries a release version, so it does not have to
  change with every release. The text next to it names the v1.0 benchmark report, its
  methodology and raw data; `benchmarks/v1.0.md` keeps its version.
- **`docs/launch-checklist.md`** records the final launch state. Every item is done, and each
  row gives its evidence: the public repository, `main` protection, the releases through
  1.0.2, PyPI, the seeded issues and the social preview.
- **Community benchmark onboarding.** The README has a short "Help benchmark Apple
  Silicon" section that points M1/M2, M3 Max and M4 Pro owners to issues #1, #2 and #3.
  `docs/community-benchmarks.md` lists those issues as contribution slots, marks untested
  SoCs (M5 family included) as **Wanted** instead of `?`, and its "Add your Mac" steps
  start from a base `uv sync` and `--quick`, with a pre-PR check of the bundle. The three
  issue drafts in `.github/issue-drafts/` are rewritten as copy-and-paste walkthroughs.
  No runtime, routing, parity or benchmark code changed.

## [1.0.2] - 2026-09-23

Packaging and documentation only. The library code is unchanged since 1.0.1; this
release exists so the PyPI project page shows the current README, since 1.0.1 was
built from a README that still said the package was not on PyPI.

### Added

- **PyPI release workflow** (`.github/workflows/release.yml`), using Trusted Publishing
  with GitHub OIDC; no API token is stored. It builds on an Apple silicon runner and
  checks that the tag is on `main` and matches `pyproject.toml`, `__version__` and the
  wheel metadata. Distribution contents are checked against an allowlist, with no local
  paths and `LICENSE`/`NOTICE` in both files. It imports the built wheel in a clean
  environment before the reviewer-gated `pypi` environment publishes it. How to use it:
  [`docs/publishing.md`](docs/publishing.md).
- **README figures** generated from data by `scripts/generate_readme_svgs.py`: the v1.0
  mixed-workload throughput gain, and the request-level routing between the ANE and the
  MLX GPU. CI fails if a committed figure drifts from `benchmarks/v1.0/` or
  `routing.json`.

### Changed

- **README:** `pip install laya-apple` is the primary install, with the `ane` and
  `convert` extras optional. Running from source moved to `CONTRIBUTING.md`. The figures
  use absolute image URLs so they also render on PyPI.
- **`benchmarks/v1.0.md`** explains why the raw part-A gate records `passed: false`.
  The flag also bounds each stream's P99 against its solo value, and the long streams
  exceed that bound. The throughput ratio and the mismatch count, which are also part
  of the flag, hold.
- **CI** pins its actions to commit SHAs.

### Notes

- **History was rewritten on 2026-09-23 before publication:**
  - a private planning document was removed;
  - local absolute paths were scrubbed from committed raw data and logs.

  The tags were recreated on the rewritten commits. v1.0.0 and v1.0.1 have exactly
  the same content as before. At v0.1.0 to v0.3.0, the embedded sha256 of the raw
  Phase -1 files in `routing.json` and the goldens no longer match: those files lost
  their path strings. As a result `derive_routing.py --check` and `make_goldens.py
  --check` report them stale at those tags. No measured value changed.

## [1.0.1] - 2026-09-23

Documentation, examples and tooling only. The library code is unchanged since 1.0.0.

### Added

- `examples/basic.py` (one prediction with `device="auto"`) and `examples/auto_routing.py`
  (short, long and multi-question requests through one `auto` instance, with the routing
  reason for each).
- `scripts/hardware_report.py`: one command that records this machine's environment and
  runs MLX and ANE parity, latency, routing and a short heterogeneous mix, writing a
  bundle under `hardware-results/` for the community matrix.
- `docs/community-benchmarks.md`: the Apple Silicon results matrix and how to add a Mac
  to it. `hardware-results/README.md` describes the bundle format, and
  `hardware-results/apple-m4-max-macos26/` is the first entry.
- New documents:
  - `docs/architecture.md`;
  - `docs/correctness.md` (fast ≠ correct: what was tested, on which platform);
  - `docs/reproducibility.md` (every headline number traced to its report, raw data,
    command and environment);
  - `docs/releases/v1.0.0.md`;
  - `docs/launch-checklist.md`.
- Contributor surface:
  - `.github/labels.yml` and seeded issue drafts in `.github/issue-drafts/`
    (`good first issue`, `help wanted`, `research`);
  - `scripts/github_seed.sh` creates them on GitHub. It is dry-run by default.

### Changed

- `examples/heterogeneous_routing.py` is renamed `examples/heterogeneous_serving.py`;
  its behaviour is unchanged.
- The README leads with the v1.0 mixed-workload result and the three differentiators;
  deeper material moved to `docs/`.
- `CONTRIBUTING.md` maps each kind of change to the tests it needs, and covers parity
  checks, benchmarking, hardware results and backend changes.
- Package description and keywords.

## [1.0.0] - 2026-09-23

First stable release. There are no breaking changes since 0.3.0; one CLI flag is added.
This release declares what is stable and re-runs the full comparative benchmark suite.

### Stable

The following now follow Semantic Versioning, with the deprecation policy in
[`docs/api.md`](docs/api.md):
- the public Python API: `Laya`, `Result`, `RuntimeInfo`, and the exception hierarchy;
- the stable `info()` keys and the routing-reason strings;
- the CLI and its exit codes;
- the artifact manifest (`format_version: 1`, `laya_apple/data/manifest.schema.json`);
- the export archive and the local-profile format.

### Added

- `docs/compatibility.md`: what is tested, expected and unknown for each dimension, and
  what happens on an untested machine.
- `docs/benchmarks.md`: how to reproduce every benchmark report.
- `benchmarks/v1.0.md`: v0.1 single-request latency and the v0.2 concurrency exit-gate
  mix, re-run on the release code and compared figure by figure with the recorded data
  (`scripts/compare_bench.py`).
- `scripts/release_gate.py --soak SECONDS`: the v1.0 gate includes the 600 s sustained
  load.
- `laya-apple artifacts list --capabilities`: one provenance record per registered
  artifact. It covers:
  - model and revision, the source weight hash, and the conversion revision;
  - graph, bucket, compute target, precision and platform;
  - the parity result and the artifact checksum;
  - whether `auto` offers the bucket.
- `examples/heterogeneous_routing.py` (renamed `heterogeneous_serving.py` after 1.0.0): a
  mixed batch served concurrently, showing backend, device and routing reason for each
  request.
- **Option-order robustness check** (`scripts/option_order.py`, `research/option-order/`,
  `tests/integration/test_option_order.py`):
  - `choice` decisions depend on option order in upstream Laya itself;
  - laya-apple MLX FP32 reproduces upstream permutation by permutation, and FP16/ANE
    differ only inside the near-tie band;
  - this is documented as a known limitation of the model.
- **The v1.0 comparative benchmark suite:**
  - `scripts/bench_v1.sh` measures upstream PyTorch CPU/MPS and the ordinary Core ML
    graph, with their parity, and open-loop serving;
  - `scripts/v1_report.py` renders every table in the README and `benchmarks/v1.0.md`
    from the raw data.
- `CONTRIBUTING.md` gains an artifact release policy: a quantized or other optimized
  variant ships only if it passes the unchanged correctness gate, and failed variants stay
  research.

### Changed

- The README positions laya-apple as a correctness-validated heterogeneous runtime. Every
  headline figure comes from the v1.0 re-run. The usage reference moved to
  `docs/guide.md`.
- The Phase -1 parity script can write a re-run elsewhere (`PARITY_OUT`), so the recorded
  evidence is never overwritten.
- Research scripts default to a portable cache location.

## [0.3.0] - 2026-09-23

### Added

- **Artifact lifecycle** (`laya_apple/lifecycle.py`):
  - Builds and imports of one (model, revision, bucket) are serialised across processes
    by a file lock.
  - An artifact that fails its hash check, or whose manifest cannot be read, is
    quarantined when detected, and the error names the rebuild command.
  - `laya-apple artifacts prune` lists stale, foreign-profile, unvalidated, rejected,
    quarantined and abandoned entries with a reason. It deletes them only with `--yes`,
    and only inside the cache.
  - `laya-apple artifacts warm` pre-pays Core ML's on-device ANE compile.
- **Artifact distribution without trust:** `laya-apple artifacts export` / `import`.
  - An imported artifact is registered only after the receiving machine re-checks the
    manifest, platform profile, file hash, compute plan, and the full parity gate.
  - Local results are recorded under `imported`. There are no automatic downloads.
- **Local capability profiles:** `laya-apple calibrate`.
  - On a platform profile with no shipped routing table, it measures MLX and ANE latency
    and applies the shipped derivation rule (`laya_apple/derivation.py`).
  - It writes `<cache>/profiles/<profile>.json`.
  - `Laya.info()["routing_profile"]` reports which table is in effect.
- **Background ANE start-up:** `ane_startup="background"` (workers, `auto`).
  - `from_pretrained` returns once MLX is ready.
  - Requests go to MLX with reason `ane_starting` until the ANE is loaded.
  - `wait_for_ane()` blocks until then, and `info()["ane_ready"]` reports it.
- **Release tooling and stress tests:**
  - `scripts/release_gate.py`: lint, format, data-drift checks, the full test suite,
    a clean-install matrix (Python 3.11–3.13 × base/ane with the README quickstart), and
    artifact verification. It writes a JSON + Markdown report.
  - `tests/stress/` (opt-in, `LAYA_APPLE_STRESS=1`): sustained mixed load through
    workers, repeated load/close, inline memory growth, inline thread safety.
  - `scripts/bench_coldstart.py` measures cold start.

- **Artifact manifest schema:**
  - `laya_apple/data/manifest.schema.json` is the published `format_version: 1` schema.
  - It is enforced on every load, and a failure names the offending field.
- **Runtime placement probe:**
  - Every loaded ANE bucket is timed against a `CPU_ONLY` instance of the same artifact.
  - A ratio above 0.8 means the model is not running on the Neural Engine. Explicit `ane`
    raises `ComputeUnitMismatchError`; `auto` drops the bucket with a warning.
  - Measured ratios are 0.32–0.50 (`benchmarks/v1.0/probe.json`), and results appear in
    `info()["ane_probes"]`.
- `docs/no-silent-fallback.md`: an audit of every path that could run a request somewhere
  other than recorded, each pinned by a test (`tests/integration/test_no_silent_fallback.py`).

### Fixed

- `dtype` was ignored for `device="ane"`. Now `ane` accepts only `float16`, and an unknown
  `dtype` is refused for every device.
- A local capability profile could widen the auto buckets beyond the validated ones. A
  profile whose buckets are not a prefix of the offered ones, or which is unreadable, is
  now ignored with a warning.
- Under `execution="workers"`:
  - A dead ANE worker is detected before the next request is routed, not after that
    request fails on it.
  - A worker reported ready before its queue existed.
  - `close()` during background ANE start-up leaked the loading backend and emitted a
    spurious warning.
- ANE worker processes no longer hide Core ML / coremltools / NumPy warnings.

### Changed

- The build's parity step uses the torch-free `laya_apple.parity.ane.ane_parity`, which
  imports also use.
- The routing derivation rule moved from `scripts/derive_routing.py` into the package
  (`laya_apple.derivation`), so `calibrate` and the shipped table share one
  implementation.
- `artifacts export` defaults to every offered bucket and the current directory.

## [0.2.0] - 2026-09-23

### Added

- **`execution="workers"`: heterogeneous GPU + ANE serving.**
  - MLX runs in a worker process. Core ML runs either on a dispatcher thread in the
    caller's process or in a worker process, chosen per model from measurements
    (`ane_placement="auto"`, `laya_apple/data/placement.json`,
    `scripts/derive_placement.py`).
  - Each device has its own FIFO queue.
  - `Laya.submit()` returns a `concurrent.futures.Future`. `Laya.apredict()` is
    awaitable. `predict()` is thread-safe. `Laya.close()` and the context-manager form
    stop the workers.
  - Workers start as `python -m laya_apple.executor` over an authenticated AF_UNIX
    connection, so the caller's `__main__` is never re-imported.
  - Workers load in parallel and warm every compiled shape before serving.
- **Queue-aware, isolation-first routing** (`laya_apple/scheduling.py`).
  - Expected completion is the device backlog plus a measured service time (Phase -1
    forward P50s, now in `routing.json` as `service_ms`).
  - It is identical to the v0.1 rule on an idle machine, which is tested exhaustively.
  - New reasons: `ane_backlog_shorter_on_gpu` and `gpu_backlog_shorter_on_ane` (the
    tie-band bucket, loaded in workers mode).
  - Long and multi-question requests never go to the ANE.
- **`RuntimeInfo` fields:** `execution`, `queue_wait_ms`, `device_ms`, and the
  `gpu_backlog_ms` / `ane_backlog_ms` values the router saw.
- **Failure handling:**
  - A dead GPU worker fails its requests loudly. Nothing is moved to the other device.
  - A dead ANE worker process is reported as `ane_runtime_unavailable` under `auto`, with
    one warning.
  - `close()` fails any still-queued request instead of leaving it pending.
- **Benchmarks and research:**
  - `scripts/bench_concurrency.py` covers the closed-loop exit-gate mix plus open-loop
    Poisson and bursty mixed workloads. Every answer is checked against the inline result
    for the same device.
  - `benchmarks/v0.2.md` is the release report.
  - `research/v0.2-concurrency/` holds the step-1 gate and the request-driven findings.

### Changed

- Vendored model code is excluded from `ruff format` so it stays byte-identical to its
  upstream revision.

### Known limitations

- **Isolation is partial for request-driven serving** on the tested platform:
  - The step-1 criterion "each stream's P99 within 10% of solo" was not met by any of the
    tested designs; the measured values are in `benchmarks/v0.2.md`.
  - With both devices busy, a device fed over IPC runs its host-side work several times
    slower.
  - Core ML's Python `predict` holds the GIL for much of an ANE call.

## [0.1.0] - 2026-09-23

### Added

- **Runtime:** `Laya.from_pretrained()` and `Laya.predict()` with a common
  request/result schema and `RuntimeInfo` diagnostics.
- All three pinned checkpoints (`laya`, `laya-multilingual`,
  `laya-typed-decisions`) at fixed revisions, with weight-hash verification
  on load.
- MLX backend (default), FP16 and FP32.
- Core ML / ANE backend for the offered fixed-shape buckets, `bc1s-masked`
  graph, `CPU_AND_NE` compute units, `B=1`; multi-question requests can run
  on the ANE sequentially only when explicitly requested (`device="ane"`),
  and never auto-route there.
- Model and capability registry, including recorded invalid configurations.
- `device="gpu" | "ane" | "auto"` routing, with deterministic, per-model
  auto-routing thresholds generated from measured Phase -1 evidence
  (`laya_apple/data/routing.json`, `scripts/derive_routing.py`).
- Strict validation for every unsupported request or artifact state, each
  raising a specific `laya_apple.errors` exception; no silent fallback
  between devices or Core ML compute-unit placements.
- Artifact pipeline: build → compile → placement check → parity gate →
  atomic registration, with a full-provenance manifest per artifact.
- Parity infrastructure: shipped PyTorch FP32 goldens, `laya-apple parity`,
  and a `tests/parity` suite.
- CLI (`laya-apple`): `predict`, `info`, `download`, `artifacts
  build|list|verify`, `parity`, `benchmark`, with a global `--offline` flag.
- Offline inference via `local_files_only=True`, `HF_HUB_OFFLINE=1`, or
  `--offline`; no telemetry.
- Documentation: README, this changelog, `docs/support-matrix.md`,
  `LICENSE`/`NOTICE`.

### Release gate, Apple M4 Max / macOS 26.6.2

1. **Clean install:** base and `[ane]` in fresh venvs on Python 3.11, 3.12 and 3.13.
2. **README quickstart:** runs unchanged, offline, in all six environments.
3. **MLX:** passes parity for all three checkpoints in FP16 (max probability error ≤ 0.0045)
   and FP32 (≤ 1e-4), 0 hard mismatches.
4. **ANE:** all 10 offered buckets build from the release commit and pass placement (100%
   ANE, 0 transitions) and parity (≤ 0.0128, 0 hard mismatches). Each loads in 0.14–0.31 s
   from the cache.
5. **Unsupported paths:** every row of the failure table raises its own error, with tests.
6. **Auto routing:** matches the derived rule in every benchmarked configuration, and is
   tested for determinism.
7. **CLI:** every command works.
8. **Offline:** `HF_HUB_OFFLINE=1` and `local_files_only=True` work end to end.
9. **Provenance:** recorded in every manifest, including a clean release-commit revision.
10. **Benchmark:** `benchmarks/v0.1.md`, 100 of 100 configurations ok.
11. **Documentation:** checked against code and data by an independent review pass.
12. **Tests:** the full suite passes, 257 tests including integration and parity.

### Not included in v0.1

- Concurrency or request queues (planned for v0.2).
- Core ML on the GPU (`CPU_AND_GPU`) as a product path — validated for
  correctness in Phase -1 but not exposed.
- Batched (`B>1`) ANE artifacts — parity at `B>1` is unmeasured.
- Long-context ANE optimization, windowed attention, quantization, and
  energy/power measurement — all tracked as non-blocking research.
- Validation on any hardware/OS profile other than Apple M4 Max / macOS
  26.6.2 / coremltools 9.0.
