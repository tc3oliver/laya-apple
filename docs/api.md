# Public API and stability policy

laya-apple follows [Semantic Versioning](https://semver.org/) from 1.0.0. This page
defines what "the public API" means. Anything not listed here is internal and may change
in any release.

## Stable (covered by SemVer)

### Python

Import these from the top-level `laya_apple` package:

| Name | Contract |
|---|---|
| `Laya.from_pretrained(model_id, device="auto", *, dtype="float16", local_files_only=False, batch_size=16, execution="inline", ane_placement="auto", ane_startup="wait", trace=None, ane_handoff=None)` | Loads a pinned checkpoint. Invalid arguments raise `ValueError`. Every other failure raises a `LayaAppleError` subclass |
| `Laya.predict(context=None, questions=None, *, state=None) -> Result` | Blocking and thread-safe |
| `Laya.submit(...) -> concurrent.futures.Future[Result]` | Same arguments as `predict` |
| `await Laya.apredict(...) -> Result` | Same arguments as `predict` |
| `Laya.close()`, `with Laya.from_pretrained(...) as laya:` | Idempotent. Queued work fails with `BackendUnavailableError` |
| `Laya.wait_for_ane(timeout=None) -> bool` | Only meaningful with `ane_startup="background"` |
| `Laya.info() -> dict` | The keys below are stable. New keys may be added |
| `Result` | Fields `answers`, `usage`, `runtime`, `model`, `extra`, and `to_dict()` |
| `RuntimeInfo` | Every field listed in `laya_apple/result.py` at 1.0.0. New optional fields may be added |
| `RequestTrace`, `QueueSnapshot` | The fields and duration properties in `laya_apple/trace.py`. `trace=` (workers only) calls the callback once per completed request, before its Future resolves; a callback exception is warned once and never fails the request. New fields may be added |
| The exception classes in `laya_apple.errors`, re-exported at top level | Their hierarchy: every one derives from `LayaAppleError`, and the artifact errors from `ArtifactError` |
| `laya_apple.__version__` | |

**Stable `info()` keys:**
- `model`, `repo`, `revision`;
- `device`, `execution`, `ane_placement`, `dtype`;
- `mlx`, `ane_buckets`, `ane_load_errors`;
- `routing_profile`, `ane_ready`, `auto_ane`;
- `ane_handoff` (since 1.5), present only on instances eligible for adaptive execution: its state, and whether it is enabled and why not. When adaptive execution could not start, the dict has only `enabled`, `disabled` and `disabled_reason`.

**Argument values:**

| Argument | Values |
|---|---|
| `device` | `auto`, `gpu`, `ane` |
| `dtype` | `float16`, `float32` (MLX only; `device="ane"` accepts only `float16`) |
| `execution` | `inline`, `workers` |
| `ane_placement` | `auto`, `thread`, `process` |
| `ane_startup` | `wait`, `background` |
| `ane_handoff` | `None` (default: adaptive ANE execution where eligible and available), `False` (always the 1.4 path), `True` (required; [`guide.md`](guide.md#adaptive-ane-execution-in-process-ane-the-default-since-15)) |

### Routing reasons

The strings in `RuntimeInfo.routing_reason` are stable:
- `gpu_requested`, `ane_requested`, `validated_short_single_question_path`;
- `multiple_questions`, `sequence_exceeds_ane_auto_range`;
- `ane_artifact_unavailable`, `ane_runtime_unavailable`, `platform_not_validated`;
- `ane_backlog_shorter_on_gpu`, `gpu_backlog_shorter_on_ane`, `ane_starting`.

New reasons may be added in a minor release. Code that switches on the reason must accept
unknown values.

**Which device serves a request is not part of the API.** Routing thresholds, auto
buckets and service-time estimates are measured data. They can change in any release,
with the evidence recorded in `CHANGELOG.md`. The guarantee is that every decision is
recorded in `RuntimeInfo`, never that a given request lands on a given device.

### Command line

The `laya-apple` subcommands, and their arguments as listed in the CLI reference in
[`docs/guide.md`](guide.md), are stable:
- `predict`, `info`, `download`;
- `artifacts build|list [--capabilities]|verify|warm|prune|export|import`;
- `parity`, `calibrate`, `benchmark`;
- `serve` (since 1.3.0, needs the `[serve]` extra).

**Exit codes:**
- `0` means success;
- `1` means a check failed (`artifacts verify`, `parity`);
- `2` means a laya-apple error or an invalid argument value.

### Files and environment

| Item | Contract |
|---|---|
| `LAYA_APPLE_CACHE`, `XDG_CACHE_HOME`, `HF_HUB_OFFLINE` | Their meaning as documented in [`docs/guide.md`](guide.md) |
| Artifact manifest, `format: "laya-apple-artifact"`, `format_version: 1` | The shipped JSON Schema is [`laya_apple/data/manifest.schema.json`](../laya_apple/data/manifest.schema.json). Fields may be added without a version change, and readers ignore fields they do not know. Removing a field or changing its meaning requires `format_version: 2`. A release that reads v2 keeps reading v1 for at least one major version |
| Export archive (`artifacts export`) | A `.tar.gz` holding `manifest.json` and `model.mlmodelc/` |
| Local capability profile, `format: "laya-apple-profile"`, `format_version: 1` | `<cache>/profiles/<profile>.json` |

### HTTP API (`laya-apple serve`, since 1.3.0)

Stable, as documented in [`docs/serve.md`](serve.md):
- the endpoints `POST /v1/systemone`, `GET /v1/models`, `GET /health`, `GET /healthz`;
- the request fields and the status codes;
- the top-level response keys `model`, `answers`, `usage`, `routing`, `laya_apple`, and the
  keys of the `laya_apple` block;
- the `--model` values, the default port 8642, and the loopback-only default bind;
- the environment variables `LAYA_APPLE_SERVE_MODEL`, `LAYA_APPLE_SERVE_HOST`,
  `LAYA_APPLE_SERVE_PORT` and `LAYA_API_KEY`.

The wire format tracks upstream `laya.serve`. When upstream changes it, laya-apple follows
in a minor release, with the change recorded in `CHANGELOG.md`. New response keys may be
added at any time. The text of `routing.reason` and of error `detail` values is not stable.

The cache *layout* (the directories under `<cache>/artifacts/`) is not public. Use the
CLI or the manifest, not paths.

## Internal (no compatibility promise)

These are not covered by SemVer:
- `Laya.prepare`, `Laya.route`, `Laya.backlogs`, `Laya.queue_snapshots`, and the `mlx` / `ane`
  attributes;
- every submodule other than `laya_apple.errors`: `laya_apple.routing`, `scheduling`,
  `executor`, `artifacts`, `lifecycle`, `profiles`, `derivation`, `backends`,
  `conversion`, `parity`, `schema`, `workload`, `benchmark`, `serve`, `lang` (use the
  CLI and the HTTP API);
- the bundled data files, except the manifest schema;
- everything under `scripts/`.

## Deprecation policy

- A stable name or behaviour is deprecated before it is removed. Deprecated functionality
  keeps working for at least one minor release and is removed only in the next major
  release.
- Using a deprecated name or argument emits a `DeprecationWarning` naming the
  replacement and the version that removes it.
- Every deprecation and removal is listed under *Deprecated* or *Removed* in
  `CHANGELOG.md`.
- **Exception:** a correctness or safety defect may be fixed in any release, even when the
  fix changes behaviour. An example is a path that could silently run on an unvalidated
  configuration. The CHANGELOG says so explicitly.
- A pinned model revision changes only in a minor or major release. The old revision's
  artifacts are then reported by `artifacts prune`.
