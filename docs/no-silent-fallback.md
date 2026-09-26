# No silent fallback: audit

v1.0 requires no known silent fallback path.

**What a silent fallback is.** Any way a request could run somewhere other than where the
caller or the router said, with nothing in the result or the exception showing it. That
covers:
- another device;
- other compute units;
- another artifact or bucket;
- another precision;
- the network, when offline was requested.

**What this page records:**
1. every path found where that could happen;
2. how the runtime handles each one;
3. the test that pins it.

The audit read every `except` clause and every device, bucket, compute-unit and dtype
decision in `laya_apple/`, as of the 1.0.0 release.

## Rules

1. **Explicit devices.** `device="gpu"` runs on MLX or raises. `device="ane"` runs the
   exact validated Core ML artifact on `CPU_AND_NE` in FP16, or raises. There is no
   other outcome.
2. **Routing decides before a request runs.** Under `device="auto"`, the device is chosen
   before the request runs, and the reason is recorded in `RuntimeInfo.routing_reason`.
   Routing never reacts to a failure by re-running the request elsewhere.
3. **Failures on a device stay on that device.** A failure on one device fails the
   request. Later `auto` requests may avoid that device, but only with a recorded reason
   and a warning.
4. **Unverified artifacts are never loaded.** An artifact that fails any check is not
   loaded:
   - manifest schema, revision, weights hash, graph, shape and precision;
   - compute units, platform profile, file hash and compute plan;
   - parity status;
   - the runtime placement probe (below).

## Paths and how they are handled

| # | Path | Handling | Pinned by |
|---|---|---|---|
| 1 | `auto` → MLX for multi-question, long, missing-artifact, unavailable-runtime, unvalidated-platform or ANE-starting requests | Decided before the request runs; the reason is recorded | `tests/routing/test_decide.py` (every reason reachable), `tests/integration/test_routing_e2e.py` |
| 2 | `auto` with busy queues → the other device | Decided before submission; recorded as `ane_backlog_shorter_on_gpu` / `gpu_backlog_shorter_on_ane`. Long and multi-question requests never go to the ANE | `tests/routing/test_scheduling.py` |
| 3 | Explicit `ane`, request longer than the largest bucket | `UnsupportedShapeError`, raised before any work is queued | `test_routing_e2e.py`, `test_workers.py::test_explicit_ane_workers_refuses_long_rows_before_queueing` |
| 4 | Explicit `ane`, the fitting bucket's artifact is missing while a larger one exists | Raises that bucket's `ArtifactMissingError`; never pads up to a larger bucket | `test_no_silent_fallback.py::test_explicit_ane_never_pads_up_to_a_larger_bucket` |
| 5 | Explicit `ane`, an artifact fails verification | Raises at load | `test_no_silent_fallback.py::test_explicit_ane_refuses_a_corrupt_bucket_at_load`, `tests/artifacts/*` |
| 6 | `auto`, an artifact fails verification | Bucket dropped; `RuntimeWarning` naming the error; its requests go to MLX with `ane_artifact_unavailable` | `test_no_silent_fallback.py::test_auto_with_a_rejected_bucket_warns_and_records_the_reason` |
| 7 | Corrupt artifact files or an unreadable manifest | `ArtifactIntegrityError`; the artifact is quarantined so nothing corrupt stays in place | `tests/artifacts/test_lifecycle.py` |
| 8 | Verification stamp skips the hash and compute-plan checks | The stamp is keyed on every file's size, mtime and inode, plus the macOS build and coremltools version; any change re-runs both checks | `test_no_silent_fallback.py::test_changed_artifact_files_invalidate_the_verification_stamp` |
| 9 | Core ML places operations off the ANE at compile time | The compute plan must be 100% ANE with 0 transitions, checked at build, import, first load and after any change, else `ComputeUnitMismatchError` | `tests/artifacts/test_placement.py` |
| 10 | Core ML runs the loaded model somewhere other than the ANE, which the static compute plan cannot see (audit V1) | Runtime placement probe (below) | `test_no_silent_fallback.py::test_placement_probe_*` |
| 11 | `dtype` other than FP16 requested for the ANE (audit V2, fixed in 1.0) | `ValueError`; an unknown dtype is refused for every device. Under `auto`, `dtype` applies to MLX, and `RuntimeInfo.dtype` records what ran | `test_no_silent_fallback.py::test_ane_refuses_*`, `::test_unknown_dtype_*` |
| 12 | Compute units other than `CPU_AND_NE` | Hard-wired; `ALL` and `CPU_ONLY` have no code path; a manifest or argument that says otherwise raises `ComputeUnitMismatchError` | `tests/artifacts/test_manifest.py` |
| 13 | Dead GPU worker | The request fails with `BackendUnavailableError` and so does every later one; nothing is moved to the ANE | `test_workers.py::test_dead_gpu_worker_fails_loudly_and_never_falls_back` |
| 14 | Dead ANE worker, `auto` | The in-flight request fails; later requests go to MLX with `ane_runtime_unavailable` and one `RuntimeWarning` | `test_no_silent_fallback.py::test_dead_ane_worker_under_auto_is_recorded_and_warned_once` |
| 15 | Dead ANE worker, explicit `ane` | Every request raises `BackendUnavailableError` | `test_no_silent_fallback.py::test_dead_ane_worker_under_explicit_ane_keeps_raising` |
| 16 | Background ANE start-up fails | `RuntimeWarning`; `wait_for_ane()` is False; `auto` records `ane_runtime_unavailable` | `test_no_silent_fallback.py::test_background_startup_failure_is_warned_and_recorded` |
| 17 | Router picks a device whose backend is not loaded (internal inconsistency) | `LayaAppleError`; the other backend is never called | `test_no_silent_fallback.py::test_route_to_a_missing_backend_raises_instead_of_using_the_other` |
| 18 | A local capability profile lists buckets that are not offered, or not a prefix of the offered buckets (audit V3, fixed in 1.0) | Ignored with a `RuntimeWarning`; `auto` uses MLX only (`platform_not_validated`) | `test_no_silent_fallback.py::test_local_profile_with_unoffered_buckets_is_ignored` |
| 19 | An unreadable or unknown-format local profile | Ignored with a `RuntimeWarning` | `test_no_silent_fallback.py::test_unreadable_local_profile_is_warned` |
| 20 | Offline requested and the checkpoint is not cached | `BackendUnavailableError` with download instructions; never goes online | `test_no_silent_fallback.py::test_offline_with_an_uncached_checkpoint_raises_and_never_downloads` |
| 21 | Worker processes' warnings | Only laya-apple's own load warnings, which the parent re-issues, are filtered; Core ML, coremltools and NumPy warnings stay visible | — |

**Paths that catch an exception and carry on.** Every `except` clause in the package
falls into one of these kinds:
- **re-raises** a specific error;
- **fails closed**: an unreadable stamp means full verification, and an unknown platform
  field means a profile mismatch;
- **records and warns**: rows 6, 14, 16, 18 and 19;
- **resource clean-up**: `close()` and `__del__`, which cannot change the device of any
  request.

## Runtime placement probe

`CPU_AND_NE` allows Core ML to use the CPU. The compute plan check (row 9) inspects a
separate static compile of the model. If the system's on-device ANE compile failed or was
evicted and not redone, the loaded model could run on the CPU while the plan still says
ANE. Before 1.0 nothing observed this at run time.

The probe works like this:
- **When:** every ANE bucket is probed each time it is loaded, inline or in a worker, for
  explicit `ane` and for `auto`.
- **Measurement:** the fastest of 5 predictions on that bucket's shape, first with the
  loaded model, then with a `CPU_ONLY` instance of the same artifact, back to back.
  Machine load slows both alike, so the ratio does not depend on how busy the machine is
  or on any stored reference time. It therefore also runs on uncalibrated machines.
- **Evidence:** on the tested profile the ratio was 0.32–0.50 for every shipped artifact.
  A model running on the CPU measures about 1.0. Raw data is in
  [`benchmarks/v1.0/probe.json`](../benchmarks/v1.0/probe.json), from
  `scripts/bench_probe.py`.
- **Threshold:** `PROBE_MAX_RATIO = 0.8`.
- **Cost:** 0.10–0.19 s to load the CPU instance, plus 12 predictions per bucket (5 timed and 1 warm-up on each instance).
- **Outcome for explicit `ane`:** a failing bucket raises `ComputeUnitMismatchError`.
- **Outcome for `auto`:** a failing bucket is dropped with a warning, as in row 6.
- **Visibility:** `info()["ane_probes"]` reports every bucket's measurement.

**Limitation.** The probe runs at load time. A placement change after a model has loaded
and passed is not observed. No such change was ever seen on the tested profile: repeated
runs of the same artifact were bitwise identical and stable in latency over the 600 s
soak.
