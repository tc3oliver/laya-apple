# Examples

Each example is a single file and runs directly. Without the `[ane]` extra they still
run: requests go to the GPU and `routing_reason` says why.

- `basic.py` loads `laya-typed-decisions` with `device="auto"`, asks one question and
  prints the answer with its backend, device, routing reason and latency.

  ```bash
  uv run --extra ane python examples/basic.py
  ```

- `auto_routing.py` sends a short single-question request, a long one and a
  multi-question one through one `device="auto"` instance and prints a table of
  where each ran and why.

  ```bash
  uv run --extra ane python examples/auto_routing.py
  ```

- `heterogeneous_serving.py` submits a mix of short single-question, long and
  multi-question requests concurrently to one `Laya(execution="workers")` instance,
  so the GPU and the ANE serve at the same time, and prints which device each request
  ran on and why.

  ```bash
  uv run --extra ane python examples/heterogeneous_serving.py
  ```

- `auto_fetch_shortlist.py` (since 1.6) loads `Laya.from_pretrained("auto")`, which picks
  `laya` or `laya-multilingual` per request by language, sends an English and a German
  request through `predict` and one 16-label `choice` question through `predict_shortlist`,
  and prints the checkpoint, device and routing reasons for each. It never downloads ANE
  artifacts itself; its docstring gives the `laya-apple artifacts fetch` commands, which
  work on Apple M4 Max, macOS 26, coremltools 9.0 only. Without artifacts it runs on the GPU.

  ```bash
  uv run --extra ane laya-apple artifacts fetch laya                # optional, see above
  uv run --extra ane laya-apple artifacts fetch laya-multilingual   # optional, see above
  uv run --extra ane python examples/auto_fetch_shortlist.py
  ```

- `serve_client.py` sends one Jev-style decision request to a running `laya-apple serve`
  over HTTP (standard library only) and prints the answer, the checkpoint the server
  routed to, and the device that answered.

  ```bash
  uv run --extra serve --extra ane laya-apple serve      # in one terminal
  uv run python examples/serve_client.py                 # in another
  ```
