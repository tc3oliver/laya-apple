# Compatibility statement (v1.0)

This page states, for each dimension of the runtime, what is **tested** (the
release-validation profile below), what is **expected** (a hypothesis, not measured), and
what is **unknown**. Where it disagrees with
[`support-matrix.md`](support-matrix.md), that page wins; this page restates its
conclusions for someone deciding whether to run `laya-apple` on a given machine.

## Tested profile

The shipped routing thresholds, the release benchmarks and the parity records behind the
release were all measured on one profile:

| | |
|---|---|
| SoC | Apple M4 Max |
| macOS | 26.6.2 (25G83) |
| MLX | 0.32.2 |
| coremltools | 9.0 |
| NumPy | 2.1.3 (the `[ane]` extra pins `numpy>=1.26,<2.2`) |
| Python | 3.12.14, but 3.11–3.13 are all in the install matrix ([`release_gate.py`](../scripts/release_gate.py) builds and smoke-tests base and `ane` extras on 3.11, 3.12, 3.13) |

Community hardware results are recorded separately, in
[`community-benchmarks.md`](community-benchmarks.md) and `hardware-results/`. They do not
change the shipped routing table or the status in this page's **Tested** column. The first
one is an Apple M4 Pro (48 GB, macOS 27.0, coremltools 9.0): a `--quick` run of
`laya-typed-decisions` with a locally calibrated profile,
[`hardware-results/apple-m4-pro-macos27/`](../hardware-results/apple-m4-pro-macos27/summary.md) ([#32](https://github.com/tc3oliver/laya-apple/pull/32)).

## Per-dimension status

| Dimension | Tested | Expected | Unknown |
|---|---|---|---|
| SoC | Apple M4 Max | Other Apple M-series SoCs run MLX correctly (hypothesis: MLX itself is validated across Apple Silicon upstream) | ANE placement, correctness and routing thresholds on any other SoC, except one community data point: on an M4 Pro (macOS 27.0, coremltools 9.0), `laya-typed-decisions` passed MLX and ANE parity with 0 hard mismatches, a locally calibrated profile routed short requests to the ANE, and the heterogeneous check passed ([community matrix](community-benchmarks.md#matrix)). That covers one machine and one model, not every M4 Pro and not the shipped routing |
| macOS | 26.6.2 | macOS 15–26 run MLX correctly | ANE behaviour on any macOS other than 26.6.2; macOS 27.x specifically — prior third-party work saw different Core ML placement there (enumerated shapes moved to the GPU). The M4 Pro community result above is the only macOS 27 evidence; other SoCs, models and macOS 27 profiles are unmeasured |
| Python | 3.12.14 (benchmarks); 3.11–3.13 (install matrix) | — | Any Python outside 3.11–3.13 (unsupported, not merely untested) |
| MLX | 0.32.2 | Other MLX versions within the package's declared constraint are expected to work for the GPU backend | Numerical or performance drift on a materially different MLX version |
| coremltools | 9.0 (pinned exactly by the `ane`/`convert` extras) | — | Any other coremltools version — the placement and parity gates have not been run against one, and coremltools 9.0 constrains NumPy to `<2.2` for a reason: it breaks on NumPy ≥ 2.5 |
| NumPy | 2.1.3, and `>=1.26,<2.2` when the `ane` extra is installed | — | Behaviour outside that pinned range with the ANE backend |
| `laya` × MLX | validated | — | — |
| `laya` × Core ML/ANE | validated (buckets 64/96/128, `CPU_AND_NE`) | — | Batched (`B>1`), other compute units, other profiles |
| `laya-multilingual` × MLX | validated | — | — |
| `laya-multilingual` × Core ML/ANE | validated (buckets 64/96/128/256, `CPU_AND_NE`) | — | Batched (`B>1`), other compute units, other profiles |
| `laya-typed-decisions` × MLX | validated | — | — |
| `laya-typed-decisions` × Core ML/ANE | validated (buckets 64/96/128, `CPU_AND_NE`) | — | Batched (`B>1`), other compute units, other profiles |
| Compute units | `CPU_AND_NE` (ANE, shipped), Metal GPU (MLX, shipped) | `CPU_AND_GPU` is correct but unused (slower than MLX) | — |
| Compute units | — | — | `CPU_ONLY` is invalid (FP16 precision on Core ML's CPU path); `ALL` is never used because placement is not stable across runs |
| Execution: `inline` | validated (single request at a time, thread-safe) | — | — |
| Execution: `workers` | validated: GPU always in its own worker process; ANE placement (`thread`/`process`) chosen per model from measurement on the tested profile | The same GPU-process / ANE-thread-or-process split should hold on other Apple Silicon, since the mechanism (GIL contention, IPC cost) is not M4-Max-specific | Whether the per-model `thread` vs `process` choice in `laya_apple/data/placement.json` is the right one on a different SoC — it was derived from measurements on this machine only |
| `ane_placement="thread"` | validated for `laya`, `laya-typed-decisions` | — | — |
| `ane_placement="process"` | validated for `laya-multilingual` | — | — |
| Offline operation | validated (`local_files_only=True`, `HF_HUB_OFFLINE=1`, `--offline`) — no network access once checkpoints/artifacts are cached | — | — |

## What happens on an untested profile

`laya-apple` never guesses. On a platform profile (SoC, macOS major version,
coremltools version) that does not match a shipped, validated profile:

- `device="auto"` uses MLX only. Every result records
  `routing_reason == "platform_not_validated"`.
- `device="ane"` still works, but only after you validate it yourself on that machine:

  ```bash
  laya-apple artifacts build laya-typed-decisions   # builds locally; each bucket must
                                                     # pass its own placement and parity
                                                     # gate on this machine before it is
                                                     # registered
  laya-apple calibrate laya-typed-decisions          # measures MLX and ANE latency here
  ```

  `calibrate` applies the same rule that produced the shipped routing table
  (`docs/support-matrix.md`, "How the auto-ANE buckets were derived") to local
  measurements, and writes `<cache>/profiles/<profile>.json`. A local profile is used
  only when no shipped profile matches the running machine — it never overrides a
  validated shipped profile. `Laya.info()["routing_profile"]` reports which table is in
  effect: `"shipped"`, `"local:<path>"`, or `None` (MLX only, no profile matched).
  A local profile is bound to the platform profile, the pinned model revision and the
  artifact hashes it measured. After an upgrade that re-pins a model, or after rebuilding
  an artifact, it is ignored with a warning until `calibrate` is run again.

- Artifacts are tied to the profile that built them. An artifact built on one SoC,
  macOS major version, or coremltools version is refused when loaded or imported on a
  different one (`ArtifactRevisionError`) — it is never silently accepted and re-checked
  loosely.
- `artifacts export` / `artifacts import` do not trust the exporting machine's checks.
  The importing machine re-validates everything itself: the manifest against the pinned
  checkpoint, the build platform profile, the file hash, the compute plan (100% ANE, 0
  device transitions), and the full parity gate against the shipped goldens (which
  requires the checkpoint to be downloaded or already cached there).

This is the same "no silent fallback" policy the runtime applies everywhere else (see
[`no-silent-fallback.md`](no-silent-fallback.md)): an unvalidated configuration is never
quietly used as if it were validated.

## Where to look for more detail

- Per-model, per-compute-unit validation status, exact pinned revisions and weight
  hashes: [`support-matrix.md`](support-matrix.md).
- How the auto-ANE bucket list and routing thresholds were derived from evidence:
  `support-matrix.md`, "How the auto-ANE buckets were derived".
- Platform scope, including per-(SoC, macOS major, coremltools) capability profiles:
  `support-matrix.md`, "Platform scope", and `laya-apple calibrate` above.
- The stable public API and what compatibility promises apply to it (routing reasons,
  CLI, file formats): [`api.md`](api.md).
