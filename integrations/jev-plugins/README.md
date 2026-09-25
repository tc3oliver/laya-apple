# Jev clients against `laya-apple serve`

Each client below was installed at its released version in a throwaway directory. Its
source was not modified. It was pointed at a local `laya-apple serve` through its own
base-URL setting, with a placeholder key. SDK retries were off, so a failure could not be
retried away.

- **Tested:** 2026-09-25, on an Apple M4 Max.
- **Server:** the `feat/serve` branch (released as 1.3.0), `--model auto`, all three
  checkpoints loaded, Neural Engine ready.
- **Pass criterion:** the client completes its normal request and decodes the response
  without an error.

This is wire compatibility, not decision quality; see "Calibration" below.

| Client | Version tested | How it was pointed here | Requests | Result |
|---|---|---|---|---|
| [typesafe-sdk](https://github.com/typesafe-ai/typesafe-sdk-python) (Python) | 0.7.1 (PyPI) | `TYPESAFE_BASE_URL=http://127.0.0.1:8642` | `POST /v1/systemone` (choice, score, noul), `GET /v1/models` | Pass. Strict pydantic parsing |
| [@typesafe-ai/sdk](https://github.com/typesafe-ai/typesafe-sdk-js) (JS) | 0.6.0 (npm) | `TYPESAFE_BASE_URL` | `POST /v1/systemone` (choice, score, noul), `GET /v1/models` | Pass |
| [jev-axi](https://github.com/shiftynick/jev-axi) | 0.7.2 (npm) | `TYPESAFE_BASE_URL` | `pick`, `check`, `rate`, `models` | Pass |
| [jev-belay](https://github.com/valentynkit/jev-belay) (Claude Code plugin) | v0.2.0 (git tag) | `JEV_BASE_URL` | Stop hook: 3 noul + 1 choice | Pass, with a placeholder key and with no key |
| [Canny](https://github.com/qkal/Canny) (Claude Code plugin) | v0.3.0 (git tag) | `CANNY_JEV_URL=http://127.0.0.1:8642/v1/systemone` (the full URL) | `claims_done` noul | Pass: 30–66 ms against its 3,000 ms timeout |
| [typesafe-mcp](https://github.com/itsmostafa/typesafe-mcp) (`evaluate`) | v0.4.5 (release binary, SHA-256 checked) | `TYPESAFE_BASE_URL` | 1 request, then 8 concurrent | Pass |
| [switchboard](https://github.com/ruban-24/switchboard) | 0.1.0 (npm `@ruban24/switchboard`) | `TYPESAFE_BASE_URL` | 6 routing decisions: choice questions, array `instructions` | Pass: 101–144 ms against its 3,000 ms deadline |

## What clients see differently from Jev

- **`model`.** Every response names `laya-rl-agent`, as upstream Laya does.
  - jev-axi caches `jev-latest → laya-rl-agent` as an alias.
  - switchboard reports `resolvedModel: null`, a diagnostic field only.
- **`GET /v1/models`** lists `auto` and the three checkpoints, with an empty `release_date`.
- **A question without `instructions`** gets `422 question is missing instructions`.
  - Jev's schema makes the field optional, and the Python SDK's `Noul()` omits it when
    unset. Upstream Laya requires it, so we do too.
  - Both SDKs raise a clear 422 error. Give every question `instructions`.
- **Extra keys:** the `routing` and `laya_apple` blocks and each answer's `action` are
  ignored by the strict Python SDK and passed through by the others.

## Calibration: the protocol works, the thresholds may not

Laya is not Jev. Its `confidence` is upstream Laya's calibrated value, not Jev's. Clients
with fixed thresholds tuned on Jev may therefore rarely act on Laya's answers. Two cases
from these runs:

- **Canny acts on `claims_done` only at ≥ 0.9 or ≤ 0.1.**
  - Laya answered 0.84 for a done-claim message and 0.30 for a clearly not-done one.
  - Neither crossed a threshold, so Canny fell back to its own edit ledger both times.
- **switchboard needs confidence ≥ 0.7 on its 4–5-option routing questions.**
  - Laya's `confidence` was 0.002–0.09, so both tasks took switchboard's "uncertain"
    fallback route.
  - The detected task types themselves looked right: `edit` at 0.96, `architecture` at 0.70.

If a client lets you set its thresholds, set them from Laya's outputs on your own traffic.
Otherwise expect the client's fallback path more often than with Jev.

## Reproducing

Every client reads its base URL from the environment variable in the table. Start the
server with `laya-apple serve`, export the variable and a placeholder key such as
`TYPESAFE_API_KEY=local-placeholder`, then run the client's normal command.

Point plugins that write state (jev-belay, Canny, switchboard) at a scratch `HOME` or at
their own home variable, so the test leaves nothing in your real configuration.
