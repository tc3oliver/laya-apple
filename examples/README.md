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

- `serve_client.py` sends one Jev-style decision request to a running `laya-apple serve`
  over HTTP (standard library only) and prints the answer, the checkpoint the server
  routed to, and the device that answered.

  ```bash
  uv run --extra serve --extra ane laya-apple serve      # in one terminal
  uv run python examples/serve_client.py                 # in another
  ```
