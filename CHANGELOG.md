# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [1.4.0] - 2026-09-25

No runtime change. This release publishes the measurements and research recorded since
1.3.0: `laya-apple serve` measured beside a busy local LLM, the FAIL of GIL-released Core ML
predict on the product mix and the prebound-predict result (no production change), new
research tooling, and Chinese READMEs. The package code, routing thresholds, parity
tolerances and the published v1.0 measurements are unchanged.

### Added

- **`laya-apple serve` measured beside a local LLM** (run 2,
  [`benchmarks/serve/m4-max-r2/`](benchmarks/serve/m4-max-r2/tables.md)). One run on an
  Apple M4 Max, `Qwen3.8-27B-oQ4e-mtp` on oMLX generating at saturation, 8 decision requests
  per second offered beside it (80% short), `serve --model laya`; the default
  `--model auto` was not measured:
  - short-decision P99 with the LLM busy: 47.2 ms with `--device auto`, 122.3 ms with
    `--device gpu`; with the LLM idle, 43.2 ms and 55.9 ms;
  - LLM tok/s: −4.0% beside serve `auto`, −5.3% beside `--device gpu`, from 42.2 tok/s alone.
    The 1.31-point gap is just above the LLM-alone window spread of 1.28 points;
  - 0 hard mismatches and 0 errors over 7,728 decisions. About 4% of `auto` short decisions
    went to the GPU by the designed backlog spill, and the P99 includes them;
  - every preregistered criterion passed (G1, G2, L1, L2, C1, E1), each reported as its own
    result. Run 1 ([`benchmarks/serve/m4-max/`](benchmarks/serve/m4-max/tables.md)) was
    invalid under its own validity checks, and no result is drawn from it. Run 2 used
    revised V2 and V4 definitions, preregistered before its data
    ([`benchmarks/serve/README.md`](benchmarks/serve/README.md)).
- **Chinese READMEs:** [`README.zh-TW.md`](README.zh-TW.md) (Traditional Chinese) and
  [`README.zh-CN.md`](README.zh-CN.md) (Simplified Chinese), translated from `README.md`,
  with a language switcher at the top of all three. The English README stays authoritative.
- **Research tooling** (not part of the package; no results claimed from it):
  - a same-machine comparison harness for Apple-silicon Laya runtimes
    ([`benchmarks/compare-v1.4/`](benchmarks/compare-v1.4/README.md)): method, pins and
    tooling; no campaign has run yet;
  - the no-sudo energy sampler and its J/decision harness
    ([`research/energy-sampler/`](research/energy-sampler/README.md)). Energy sampler run 1
    recorded: it is invalid under its own criteria, and run 2 is preregistered. Energy use
    stays unmeasured.

### Changed

- **README leads with `serve` beside a busy local LLM** ("Short decisions stay fast while
  your local LLM is busy"), measured with `--model laya`. The new first section has the
  three-line quickstart and a chart of short-decision P99 and LLM tok/s, `--device gpu`
  against `auto`, with the LLM idle and busy, labelled with the machine, the LLM model and
  that it is one run. The chart, `docs/readme/serve-llm-load.svg`, is drawn by
  `scripts/generate_readme_svgs.py` from `benchmarks/serve/m4-max-r2/results.json` (and
  covered by its `--check`); it refuses a run that is not valid under its own checks. The
  1.3 serve section follows, unchanged.
- **The `serve` "no performance claim" limitation is replaced** by what run 2 measured and
  what it does not cover (other LLM servers and models, prefill-heavy loads, other request
  rates, serve's maximum decision throughput, the other checkpoints and `--model auto`),
  plus the costs it did measure: `auto` still costs
  the LLM 4.0% tok/s, and multi-question decisions, which always run on the GPU, reach a P99
  of 117.1 ms with the LLM busy. `docs/serve.md` gains a "Beside a local LLM" section;
  `docs/support-matrix.md`, `docs/compatibility.md` and `docs/reproducibility.md` are
  updated to match.
- **GIL-released Core ML predict on the product mix: FAIL, production unchanged**
  ([`research/coreml-nogil-product-mix/`](research/coreml-nogil-product-mix/README.md)).
  Neither candidate (C: GIL-released predict with the GPU in its worker process; D: the same
  with the GPU on a thread) passed the preregistered production gate on any of the three
  models. Correctness and GPU completion isolation passed everywhere; short (ANE) P99
  regressed in every cell (+11.8% to +83.0%) and aggregate throughput failed in four of six
  (down to −15.6%). Production stays thread-placed coremltools for `laya` and
  `laya-typed-decisions`, and process placement for `laya-multilingual`.
  - The study's stop rule named a Swift-worker study as the next step. A post-hoc diagnosis
    of the recorded data found dozens of PyObjC/GIL handoffs per ANE forward on the request
    path, so a smaller, request-path study (prebound predict, #80) was preregistered first;
    the Swift worker is deferred.
- **Prebound Core ML predict: no production change**
  ([`research/coreml-prebind-predict/`](research/coreml-prebind-predict/README.md), #80).
  PB prebinds the inputs' and output's backings: 4 PyObjC/GIL crossings per ANE forward
  against C's 108. Preregistered paired verdict against production (ratio, 95% interval):
  PASS on `laya` (aggregate 1.060 [1.057, 1.062], short P99 0.957 [0.953, 0.962]), PASS on
  `laya-typed-decisions` (aggregate 1.049, short P99 0.963 [0.952, 0.974]), INCONCLUSIVE on
  `laya-multilingual` (short P99 0.851 [0.524, 1.384]). As preregistered for this outcome,
  production is unchanged and a larger preregistered replication decides. PB's short P99 was
  about 3–5% lower than C's in the same campaign.
  - Unplanned observation: under this experiment's hetero-only protocol, C (#77's binding)
    also passes on `laya` and `laya-typed-decisions`, where #77 recorded it as FAIL. #77's
    regression therefore depends on the protocol; its verdict stands, and the protocol
    dependence is an open question.

## [1.3.0] - 2026-09-25

Adds `laya-apple serve`: a local, loopback-only decision server with a Jev-compatible API,
so an existing Jev client can use Laya on the Mac by changing only its base URL. The answers
are upstream Laya's, not Jev's: 792 of 792 requests matched unmodified upstream `laya.serve`
0.3.20 within the FP16 parity gate. Parity now tracks upstream Laya 0.3.20. Routing
thresholds and the published v1.0 measurements are unchanged.

### Added

- **`laya-apple serve`, a local Jev-compatible decision server** (`pip install
  'laya-apple[serve]'`). It exposes `POST /v1/systemone` with upstream `laya.serve`'s wire
  format (v0.3.20): request fields, limits, status codes, and the `routing` block.
  - **Jev clients work unchanged:** point a client's base URL at `http://127.0.0.1:8642`.
  - **Default `--model auto`:** the checkpoint is chosen per request by the state's language,
    as upstream's Router does. Upstream's language detection is vendored unchanged
    (`laya_apple/lang.py`). Every response also carries a `laya_apple` block saying which
    checkpoint and device answered, and why.
  - **Where we differ from upstream:**
    - it binds loopback only by default;
    - there is no global inference lock, so the GPU and the Neural Engine serve
      concurrently;
    - the Neural Engine warms in the background;
    - the default port is 8642, not 8000.
  - **Fidelity:** against unmodified upstream `laya.serve` 0.3.20 serving the same pinned
    weights, 792 of 792 requests matched in wire shape, language routing and values
    within the FP16 parity gate; 365 of them were answered on the Neural Engine
    ([`benchmarks/serve-compat/`](benchmarks/serve-compat/README.md)).
  - **Clients:** seven Jev clients, at their released versions and unmodified, completed
    their requests against it
    ([`integrations/jev-plugins/README.md`](integrations/jev-plugins/README.md)). This is
    wire compatibility, not Jev-level decisions.
  - Docs: [`docs/serve.md`](docs/serve.md).
- **`RuntimeInfo.truncated`.** It is true when a state was cut to fit the checkpoint's
  maximum length. Upstream truncates too, without reporting it.

### Fixed

- **Upstream Laya 0.3.20 parity.** Prompt building, question validation and the answer
  format now follow upstream Laya 0.3.20 (NandhaKishorM/laya@23a1752), not 0.3.5. Token ids
  are unchanged for every request that 0.3.5 and 0.3.20 already treated alike. The rest
  differs as follows:
  - A conversation `context` (a list of turns) that exceeds `max_len` keeps its newest
    turns. It is now truncated from the start, not the end.
  - Non-string `instructions` keep non-ASCII text, where they were `\uXXXX`-escaped before.
  - `noul` accepts optional `labels` (`{"false": ..., "true": ...}`), which replace the
    `false:` / `true:` option prefixes. `labels` on `choice` or `score`, or invalid labels,
    raise `InvalidRequestError`.
  - `noul` criteria keys are case-insensitive. Keys other than `true`/`false` raise
    `InvalidRequestError`; before, they were replaced by the default texts without a
    word.
  - Every answer gains `answer_confidence`, max(p), and the answer fields follow upstream's
    order.
  - `questions={}` returns empty answers with zero usage instead of raising. Nothing is
    tokenized, routed or run: `runtime` records `backend="none"`, `device="none"` and
    `routing_reason="no_questions"`, and workers mode emits no trace.
  - A checkpoint temperature that is not a number now loads with a warning and falls back
    to 1.0, where it raised before. Out-of-range values in `temperature` are now warned
    about as well as those in `temperature_by_options`.
- **Goldens recorded from upstream 0.3.20.** The `[reference]` extra pins `laya==0.3.20`.
  `scripts/make_reference.py` records the references into
  `research/upstream-reference/raw/laya-0.3.20/`. Every Phase -1 case is identical to its
  0.3.5 record, which `scripts/make_goldens.py` enforces. Five new cases per model (13
  rows) cover the changes above, and each golden now stores upstream's answers. The v1.0
  parity tables predate these cases ([`docs/correctness.md`](docs/correctness.md#golden-set)).

### Changed

- **README leads with `laya-apple serve`.** The new first section has the three-line
  quickstart, the client and fidelity results, and a terminal figure,
  `docs/readme/serve-demo.svg`. The figure is rendered by `scripts/generate_readme_svgs.py`
  (and covered by its `--check`) from `docs/readme/serve-demo.json`, which
  `scripts/capture_serve_demo.py` records from a real run: the released `typesafe-sdk` 0.7.1,
  unmodified, against a local server. The Switchyard section follows it, unchanged. The
  README's limitations gain the `serve` limits.
- **Switchyard launch media.** `docs/media/switchyard-launch.gif` and `.mp4` are re-cut for
  readability. Both are cropped to the rail yard, with the whole-round result in large type.
  The result card is simplified to trains late and P99 decision latency. The MP4 is now 20 s
  and opens on the result; the GIF drops the command, which the README shows above it. Same
  run (run-003), same numbers; sources in
  [`docs/media/switchyard-launch.md`](docs/media/switchyard-launch.md). The README states the
  result in text under the GIF.

## [1.2.0] - 2026-09-24

Adds `laya-apple switchyard`: a one-command, headless open-loop benchmark of GPU-only
against GPU + ANE serving, replayed in the browser as a rail-yard game from the recorded
request traces. The runtime, routing and published v1.0 measurements are unchanged.

### Added

- **`laya-apple switchyard`.** A headless benchmark of the frozen `switchyard-v1` workload
  (bursty seeded arrivals, single-question routing decisions as "trains", background
  long-context requests loading the GPU), replayed as a rail-junction game. The CLI runs the
  timed `gpu_only` and (when the Neural Engine is ready) `hybrid` rounds, writes `result.json`
  and `trace.jsonl`, builds a self-contained `replay.html` and opens it (`--no-open` to skip).
  `--setup-ane` builds and parity-validates the ANE artifact, calibrates if needed, then runs.
  Missing ANE never fails the command — it runs `gpu_only` and prints the setup command. See
  [`docs/switchyard.md`](docs/switchyard.md) for the workload, measurement boundaries and
  result schema.
  - Official campaign (`benchmarks/switchyard/v1-m4-max/`, 3 runs, Apple M4 Max, macOS
    26.6.2, `laya-typed-decisions`, standard workload: seed 11, 60 s, 40 req/s nominal, 1,422
    trains, 100 ms deadline): `gpu_only` 1,407–1,408/1,422 late (P99 decision
    3,107.7–3,170.6 ms, P99 queue 3,095.9–3,158.8 ms); `hybrid` (GPU + ANE) 0 late (P99
    decision 54.5–54.7 ms, P99 queue 39.6–42.8 ms); 0 decision disagreements, 0 misrouted in
    all 3 runs. Round order is fixed by seed parity (`hybrid` first in this campaign),
    `design.counterbalance` is `"none"`, and these numbers are not comparable with the v1.0
    open-loop table (different rate, measurement boundary and workload) — see
    [`docs/switchyard.md`](docs/switchyard.md). Report:
    [`benchmarks/switchyard/README.md`](benchmarks/switchyard/README.md).
- **Switchyard launch media.** `docs/media/switchyard-launch.gif` (README) and
  `docs/media/switchyard-launch.mp4`, a capture of the recorded replay of campaign run-003;
  sources in [`docs/media/switchyard-launch.md`](docs/media/switchyard-launch.md). The README
  now opens with Switchyard.

### Fixed

- **`laya-apple switchyard` opens the replay on macOS.** `webbrowser.open()` reports success
  for a `file://` URL without opening anything, so the replay never appeared. It is now
  opened with `/usr/bin/open`, and the path is printed if no browser opens.

## [1.1.0] - 2026-09-24

Adds public request lifecycle tracing for `execution="workers"` (`RequestTrace`,
`QueueSnapshot`, `RuntimeInfo.request_id`). It is additive: `trace=None`, the default,
changes no routing decision, answer or measured serving throughput. Also in this release:
a disk preflight for the benchmark scripts; community hardware results for the M2 Pro,
M4 and M4 Pro; the README serving animation and shareable benchmark media; and a fix for
the hardware report on comma-decimal locales.

### Added

- **Request lifecycle tracing** (#6, runtime part). `Laya.from_pretrained(...,
  execution="workers", trace=callback)` calls the callback once per completed request with
  an immutable `laya_apple.RequestTrace`. It holds:
  - the per-device queue state the router decided on (`QueueSnapshot`: backlog, queued jobs,
    running);
  - the target and routing reason;
  - the service estimate charged to the queue;
  - `time.monotonic_ns()` timestamps: submit, prepared, routed, queue enter, dispatch,
    service start and end (taken in the process that runs the backend and returned on the
    existing reply), received, and response.

  Durations are derived properties.
  - `RuntimeInfo` gains `request_id`, unique in the process across instances and also the
    job id on the worker protocol.
  - `queue_wait_ms`, `device_ms` and `latency_ms` are now computed from the same timestamps.
  - `trace=None`, the default, builds no trace object and adds no lock, message or I/O. Its
    serving throughput and P50/P95/P99 are within 0.1% of `main` and inside `main`'s own
    run-to-run spread. The host-path cost is +0.96 µs per request (`benchmarks/tracing.md`).
  - Routing decisions and answers are unchanged.
- **Benchmark disk preflight** (`scripts/bench_preflight.py`). `release_bench.py` and
  `bench_v1.sh` now stop before loading any model if free disk on the output filesystem is
  below 200 GiB. They print each Core ML E5 cache under `~/Library/Caches` with its size,
  and warn without stopping if the caches exceed 50 GiB. Nothing is deleted automatically;
  `docs/benchmarks.md` explains the cache and how to clean it up by hand. No runtime,
  routing or parity code changed.
- **Community hardware result: Apple M2 Pro** (16 GB, macOS 26.6.2), in
  `hardware-results/apple-m2-pro-macos26/` (#47, thanks @bozin1990), the first result outside the M4 family.
  An MLX-only run: MLX FP16 parity passed for `laya-typed-decisions`; coremltools was not
  installed, so `auto` stayed on MLX (`ane_runtime_unavailable`) and the ANE and
  heterogeneous checks did not run. One change updates the M2 Pro row of
  `docs/community-benchmarks.md` (and notes `ane_runtime_unavailable` as an expected
  `Auto uses ANE: no` reason), a community-measurement row in `docs/support-matrix.md`,
  the community results and the SoC cell in `docs/compatibility.md`, the README's
  community status, and the draft of issue #1, which stays open for the other M1/M2 chips
  and now asks for `Refs #1` instead of `Closes #1`.
  No runtime, routing, parity or benchmark code changed.
- **Community hardware result: Apple M4** (32 GB, macOS 26.2), in
  `hardware-results/apple-m4-macos26/` (#41, thanks @ShaoAnFang). MLX and ANE parity passed for
  `laya-typed-decisions`; `laya-apple calibrate` was not run, so `auto` stayed on MLX
  (`platform_not_validated`) and the heterogeneous check did not run. One change updates
  the M4 row of `docs/community-benchmarks.md`, a community-measurement row in
  `docs/support-matrix.md`, the community results and the SoC and macOS cells in
  `docs/compatibility.md`, and the README's community status. The draft of issue #17 no
  longer says nobody has run the pipeline on another macOS: it cites #32 (macOS 27.0) and
  #41 (macOS 26.2) and narrows the open question to another coremltools release. No
  runtime, routing, parity or benchmark code changed.
- **First community hardware result: Apple M4 Pro** (48 GB, macOS 27.0), in
  `hardware-results/apple-m4-pro-macos27/` (#32, thanks @AirRunner). `docs/community-benchmarks.md` fills its
  matrix row from that bundle and marks `Auto uses ANE` as coming from a local calibrated
  profile, not the shipped routing table. The M4 Pro contribution slot is removed now that
  issue #3 is closed. No runtime, routing, parity or benchmark code changed.
- **README serving animation.** `docs/media/heterogeneous-serving.gif` (800×450, 5 fps, 30 s,
  1.7 MB) is the first figure in the README, after the one-line description and before the
  throughput figure. It opens with a recorded Lane Runner game from
  [laya-playground-apple](https://github.com/tc3oliver/laya-playground-apple), the same
  model on the MLX GPU and the Apple Neural Engine, recorded separately and labelled as a
  single-request comparison. It then turns to GPU + ANE serving: one burst of the
  laya-typed-decisions bursty workload, GPU-only against GPU + ANE, from a per-request trace
  of the benchmark's arrival sequence, ending on the published v1.0 short-request P99.
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
- **Cross-SoC validation status.** `docs/support-matrix.md` and `docs/compatibility.md` no
  longer say every measurement in the repository comes from the M4 Max. They separate the
  release-validation profile (M4 Max, macOS 26.6.2), which still backs the shipped routing
  and release benchmarks, from community measurements, starting with the M4 Pro / macOS
  27.0 `--quick` result from #32, scoped to one machine and `laya-typed-decisions`.
  `docs/community-benchmarks.md` no longer offers the closed issue #3, and the stale
  `.github/issue-drafts/add-m4-pro-benchmark-result.md` is removed.
- **README community benchmark status.** The "Help benchmark Apple Silicon" section no
  longer calls other Macs untested: it says the headline benchmark is M4 Max only and that
  the community matrix has its first external result, an M4 Pro, kept separate from the
  headline numbers. The pointer to the closed M4 Pro issue #3 is removed.
- **Community benchmark onboarding.** The README has a short "Help benchmark Apple
  Silicon" section that points M1/M2, M3 Max and M4 Pro owners to issues #1, #2 and #3.
  `docs/community-benchmarks.md` lists those issues as contribution slots, marks untested
  SoCs (M5 family included) as **Wanted** instead of `?`, and its "Add your Mac" steps
  start from a base `uv sync` and `--quick`, with a pre-PR check of the bundle. The three
  issue drafts in `.github/issue-drafts/` are rewritten as copy-and-paste walkthroughs.
  No runtime, routing, parity or benchmark code changed.

### Fixed

- **Hardware report on comma-decimal locales** (#38, thanks @AirRunner). `ps` printed
  `%cpu` as e.g. `29,2` under locales such as French, which crashed the heterogeneous step
  of `scripts/hardware_report.py --quick`. `scripts/bench_concurrency.py` now runs `ps`
  with `LC_ALL=C`.

### Notes

- The ANE placement for `execution="workers"` is unchanged: laya and laya-typed-decisions
  run the ANE on a thread, laya-multilingual in a worker process. Moving the first two to
  a process failed its production regression gate (`benchmarks/ane-process-isolation/`).

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
