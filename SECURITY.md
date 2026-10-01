# Security Policy

## Supported versions

Security fixes go to the latest minor release line, as a new patch release. Older lines
are not patched: upgrade to the latest release to get a fix.

| Version | Supported |
|---|---|
| 1.6.x | Yes |
| < 1.6 | No |

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Report it privately
through [GitHub's private vulnerability reporting](https://github.com/tc3oliver/laya-apple/security/advisories/new)
on this repository. If that is not available, use GitHub's security advisories for
`tc3oliver/laya-apple` directly.

Include a description of the issue, the affected version, and steps to reproduce if you
have them. You should get an initial response within a few days.

## Scope

`laya-apple` runs models locally on Apple silicon. The Python API and every CLI command
except `laya-apple serve` open no network listener. Areas relevant to a security report:

### `laya-apple serve` (HTTP server)

`laya-apple serve` (the `[serve]` extra, since 1.3) is a plain-HTTP decision server with a
Jev-compatible API. Its full behaviour is in [`docs/serve.md`](docs/serve.md); the code is
`laya_apple/serve.py`.

- **Loopback by default.** It binds `127.0.0.1:8642`. A bind address is loopback when it is
  `localhost` or a loopback IP address (all of `127.0.0.0/8`, and `::1`).
- **Host validation (DNS rebinding).** On a loopback bind, every request, on every endpoint,
  must carry a `Host` header that passes the same loopback test, with or without a port
  (`127.0.0.1:8642`, `localhost:8642`, `[::1]:8642`). Any other `Host`, such as the
  hostname of a web page that rebinds its DNS to `127.0.0.1`, or a malformed bracketed
  value, gets **421**.
- **JSON only.** `POST /v1/systemone` requires `Content-Type: application/json` (otherwise
  **415**). A browser cannot send that cross-site without a CORS preflight, and the server
  grants none.
- **Request limits.** A body over 2 MiB gets **413**, whether `Content-Length` declares it or
  the streamed body exceeds it; so do more than 64 questions or a state over 50,000
  characters. A body that is not valid JSON, including JSON nested deep enough to raise
  `RecursionError`, gets **400**. An unexpected failure answers
  `{"detail": "inference failed"}` with **500**; the cause goes to the server log, never to
  the response.
- **Bearer authentication.** With `LAYA_API_KEY` set, `POST /v1/systemone` and
  `GET /v1/models` require `Authorization: Bearer <key>` (otherwise **401**). The value is
  compared in constant time and never logged. Without `LAYA_API_KEY`, any bearer value, or
  none, is accepted, so on a loopback bind any local process can query the server.
- **Unauthenticated endpoints.** `GET /health` and `GET /healthz` never require the key;
  they return the version, the default model, the loaded checkpoints, the device and each
  checkpoint's Neural Engine state. FastAPI's generated `GET /docs`, `GET /redoc` and
  `GET /openapi.json` are served without the key too. On a loopback bind, all of them are
  still subject to the `Host` check.
- **Remote binding.** A non-loopback `--host` (or `LAYA_APPLE_SERVE_HOST`) is refused unless
  `--allow-remote` is passed **and** `LAYA_API_KEY` is set; the server then prints a warning.
  A remote bind does not check the `Host` header, so the bearer key is the only access
  control, and the unauthenticated endpoints above are reachable from the network. The
  server speaks plain HTTP: the key and the request content travel unencrypted unless you
  put a TLS-terminating proxy in front of it.

### Artifact import and fetch

- **Import.** Importing an exported artifact archive (`laya-apple artifacts import ARCHIVE`
  / `laya_apple.lifecycle.import_artifact`) rejects an archive that holds anything other
  than the manifest and the compiled model, any symlink or hard link member, or more than
  4 GiB of files. It then extracts it into a staging directory with Python's `tarfile`
  `data` extraction filter, which rejects absolute paths, `..` traversal and links that
  would land outside the destination, and needs Python >= 3.11.4. Before the artifact is
  registered, it is re-validated on this machine: the manifest against the pinned
  checkpoint revision and weight hash, the build platform profile, every file against the
  manifest, the compute plan (100% Neural Engine) and the full parity gate. An archive
  that fails any check is rejected and its staging directory removed; it is never
  registered.
- **Fetch.** `laya-apple artifacts fetch` downloads a prebuilt archive from a Hugging Face
  repository, checks its SHA-256 against the repository index, and registers it only
  through the same validating import. Nothing is trusted because it was downloaded.

### Worker processes

The Neural Engine can run in a separate worker process (`laya_apple.executor`). Workers
connect back to the parent over a Unix domain socket (`AF_UNIX`) authenticated with a
per-run key; there is no network listener.

### Network access

Apart from `laya-apple serve`, `laya-apple` makes no network calls of its own beyond
downloading pinned Hugging Face checkpoint revisions and, with `laya-apple artifacts fetch`,
prebuilt artifacts. Each downloaded weights file is verified against a pinned SHA-256
before use. Setting `local_files_only=True` (`laya-apple --offline`) keeps checkpoint
loading off the network; a checkpoint not already cached then fails explicitly instead of
falling back to the network. `laya-apple artifacts fetch` exists to download, and
`--offline` limits only its checkpoint lookup, not the archive download; set
`HF_HUB_OFFLINE=1` to block every Hugging Face download, including that one.

If you find a way for the server, an artifact import or fetch, a worker connection, or the
offline path to behave differently from the above, that is exactly what this policy wants
reported.
