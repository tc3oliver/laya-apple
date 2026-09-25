# `laya-apple serve`: a local Jev-compatible decision backend

`laya-apple serve` runs Laya on your Mac behind the same HTTP API as TypeSafe's hosted Jev
decision API (`POST /v1/systemone`). A client written for Jev, such as an SDK, a CLI or a
Claude Code plugin, keeps working when you point its base URL at this server. You don't
change its code.

```bash
pip install 'laya-apple[serve,ane]'
laya-apple serve                                   # http://127.0.0.1:8642
export TYPESAFE_BASE_URL=http://127.0.0.1:8642     # then run your Jev client as usual
```

laya-apple is not affiliated with TypeSafe or Jev. The API is compatible; the model is
not the same.
- **What is guaranteed:** the server's answers are upstream Laya's answers, within the parity
  gate in [`correctness.md`](correctness.md).
- **What is not:** Jev-level accuracy. Laya is a different, smaller model. The upstream Laya
  README compares the two on its benchmarks.

## Which model answers

Jev clients send a Jev model id such as `jev-latest`, which names no Laya checkpoint.
`--model` decides what those requests get:

| `--model` | Requests that name no checkpoint go to | Use it when |
|---|---|---|
| `auto` (default) | `laya` for English text, `laya-multilingual` otherwise, chosen per request from the state's script and language | You don't know, or your agents work in several languages |
| `laya` | always the English checkpoint (ModernBERT-large, 512 tokens) | All your traffic is English |
| `laya-multilingual` | always the multilingual checkpoint (mmBERT-base, 1,024 tokens, 100+ languages) | All your traffic is non-English |
| `laya-typed-decisions` | always the typed-decisions checkpoint (ModernBERT-large, 1,024 tokens) | Your decisions match the four workflows it was fine-tuned on |

- **Where `auto` comes from.** It is upstream Laya's own router (`laya/router.py`, v0.3.20),
  and the language detection is vendored unchanged ([`laya_apple/lang.py`](../laya_apple/lang.py)).
- **Why English goes to `laya`.** Upstream measured the English checkpoint collapsing to
  near chance on non-Latin scripts.
- **Why `laya-typed-decisions` is not the default.** Upstream recommends against making it a
  silent default: it is fine-tuned on four synthetic workflows.
- **Checking a response.** Every response names the checkpoint and the reason in `routing`,
  for example `"reason": "non-Latin script (han, 100% of letters); ..."`.

A request whose `model` names a checkpoint always gets that checkpoint. Accepted names:
- ours: `laya`, `laya-multilingual`, `laya-typed-decisions`;
- upstream's names and aliases: `english`, `multilingual`, `typed-decisions`, `typed`, `ml`, …;
- the standalone Hugging Face ids.

Any other value, including `convaiinnovations/laya`, goes to the server default, as upstream
does.

Set the default once with `--model` or `LAYA_APPLE_SERVE_MODEL`. `/v1/models` lists the
choices.

## API

The wire format follows upstream `laya.serve` (Laya v0.3.20).

### `POST /v1/systemone`

The request is `{"model"?: string, "state": any JSON, "questions": {id: question}}`. The
question schema is Laya's `choice` / `score` / `noul` ([`guide.md`](guide.md)).

A response looks like this:

```json
{
  "model": "laya-rl-agent",
  "answers": {"next": {"type": "choice", "choice": "run_tests", "confidence": 0.20,
                       "probabilities": {"run_tests": 0.62, "ask_user": 0.09, "commit": 0.29},
                       "action": {"act_probability": 1.0}}},
  "usage": {"input_tokens": 35, "output_tokens": 0},
  "routing": {"model": "english", "repo": "convaiinnovations/laya", "reason": "English Latin text",
              "detection": {"script": "latin", "language": "en", "...": "..."}, "workflow": null},
  "laya_apple": {"model": "laya", "device": "ane", "backend": "coreml",
                 "routing_reason": "validated_short_single_question_path", "request_id": 1,
                 "truncated": false, "sequence_length": 35, "question_count": 1,
                 "latency_ms": 38.6, "queue_wait_ms": 0.02, "device_ms": 38.1}
}
```

- **Fields as upstream:** `model`, `answers`, `usage` and `routing` are upstream's. Jev clients
  read `answers` and `usage` and ignore the rest.
- **The `laya_apple` block** is our addition:
  - which checkpoint and which device (GPU or Neural Engine) answered, and why;
  - `truncated: true` when the state was cut to fit the model's maximum length. Upstream
    cuts the same way, silently.
- **Timing headers:** `Server-Timing: inference;dur=<ms>` and `X-Inference-Time-Ms`, as
  upstream.

Status codes are the same as upstream:

| Status | When |
|---|---|
| 200 | Answered. An empty `questions: {}` answers `{}` with zero usage |
| 400 | The body is not JSON, is not an object, lacks `questions`, or `questions` is not an object |
| 401 | `LAYA_API_KEY` is set and the `Authorization: Bearer` value does not match |
| 413 | Body over 2 MiB, more than 64 questions, or a state over 50,000 characters |
| 422 | A question fails validation, for example a missing `instructions` (upstream requires it too) |
| 500 | `{"detail": "inference failed"}`. The cause is in the server log, never in the response |

### Other endpoints

- **`GET /v1/models`** returns Jev's shape: `{"models": [{"name", "description",
  "release_date", "loaded", "default"}]}`.
- **`GET /health`** and **`GET /healthz`** need no auth. They return the version, the default,
  the loaded checkpoints and each one's Neural Engine state: `warming`, `ready` or
  `unavailable` (with the reason).

## Running it

```text
laya-apple serve [--model auto|laya|laya-multilingual|laya-typed-decisions]
                 [--preload MODEL ...] [--host 127.0.0.1] [--port 8642]
                 [--device auto|gpu|ane] [--allow-remote] [--log-level info]
```

| Setting | Flag | Environment | Default |
|---|---|---|---|
| Default model | `--model` | `LAYA_APPLE_SERVE_MODEL` | `auto` |
| Bind address | `--host` | `LAYA_APPLE_SERVE_HOST` | `127.0.0.1` |
| Port | `--port` | `LAYA_APPLE_SERVE_PORT` | `8642` |
| Bearer key | none | `LAYA_API_KEY` | unset: any key, or none, is accepted |
| Offline | `laya-apple --offline serve` | `HF_HUB_OFFLINE=1` | online |

**Startup.** It loads the default's checkpoints, which are `laya` and `laya-multilingual` for
`auto`, plus any `--preload`. Then it answers on the MLX GPU while the Neural Engine
artifacts load in the background. A checkpoint that a request names but that is not loaded
loads on first use.

**Concurrency.** There is no global lock. Each checkpoint is one
`Laya(execution="workers")`, so requests run concurrently: short single-question decisions
go to the Neural Engine, long or multi-question ones to the GPU. Upstream serializes every
request through one worker.

**Port.** The default is not 8000, because upstream `laya.serve` and local LLM servers such
as oMLX already use it.

**Security.**
- The server binds loopback only. `--host` with any non-loopback address is refused unless
  you also pass `--allow-remote`, and then it prints a warning.
- Set `LAYA_API_KEY` before exposing it. The key is compared in constant time and never
  logged.
- Without `LAYA_API_KEY`, any bearer value is accepted. Jev clients require some key to
  start, so give them a placeholder.

## Client compatibility

Seven Jev clients were tested at their released versions, unmodified, pointed here only
through their base-URL setting. All seven completed their requests
([`integrations/jev-plugins/README.md`](../integrations/jev-plugins/README.md) has the
versions, commands and details):

| Client | Version | Base-URL setting | Result |
|---|---|---|---|
| typesafe-sdk (Python) | 0.7.1 | `TYPESAFE_BASE_URL` | Pass |
| @typesafe-ai/sdk (JS) | 0.6.0 | `TYPESAFE_BASE_URL` | Pass |
| jev-axi | 0.7.2 | `TYPESAFE_BASE_URL` | Pass |
| jev-belay (Claude Code plugin) | v0.2.0 | `JEV_BASE_URL` | Pass |
| Canny (Claude Code plugin) | v0.3.0 | `CANNY_JEV_URL` (full `/v1/systemone` URL) | Pass |
| typesafe-mcp | v0.4.5 | `TYPESAFE_BASE_URL` | Pass |
| switchboard | 0.1.0 | `TYPESAFE_BASE_URL` | Pass |

"Pass" means wire compatibility: the client decodes the response without an error. It does
not mean the client makes the same decisions as with Jev. Canny and switchboard act only
on confidences above thresholds tuned for Jev, and Laya's answers did not cross them in
these runs, so both took their own fallback paths.

## Limits

- **No Jev accuracy claim.** Only runtime fidelity to upstream Laya is guaranteed and tested.
- **Client thresholds tuned on Jev** may rarely trigger on Laya's confidences (see
  "Client compatibility").
- **`auto` routes between English and multilingual only.** Upstream's opt-in typed-decisions
  workflow detection (`LAYA_AUTO_TASK`) and caller language hints are not implemented. Name
  the checkpoint or use `--model` instead.
- **No performance claim yet.** No claim is made about how the server affects a local LLM on
  the same GPU; that is measured separately.
- **Truncation.** Long states are cut to the checkpoint's maximum length: 512 tokens for
  `laya`, 1,024 for the others. `laya_apple.truncated` reports it.
