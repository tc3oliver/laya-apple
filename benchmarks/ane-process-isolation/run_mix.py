"""The v1.0 closed-loop heterogeneous mix, with the GPU completion path recorded.

    LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 uv run python benchmarks/ane-process-isolation/run_mix.py \
        --model laya --short 128 --long 512 --ane-placement process --part a --output ...

This runs scripts/bench_concurrency.py unchanged (Part A: solo_short, solo_long, hetero,
gpu_only; 3 cycles of 20 s; one closed-loop client per stream; answers checked against inline
references). It adds two records to the output file, and nothing to the workload:

  windows_t   for every closed-loop client: its stream, start and end (time.monotonic_ns)
  gpu_return  for every GPU request of the device="auto" workers instance: the reply's arrival
              (received_ns) and its return leg, received_ns - service_end_ns, in µs

The return leg is the time from the GPU worker's forward returning to the parent's dispatcher
resolving the job. A thread-placed Core ML predict holding the GIL delays it (#46, #54).
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import bench_concurrency  # noqa: E402

from laya_apple import Laya  # noqa: E402


def main():
    returns, windows, lock = [], [], threading.Lock()

    def record(t):
        if t.target == "gpu":
            with lock:
                returns.append((t.received_ns, (t.received_ns - t.service_end_ns) // 1000))

    from_pretrained = Laya.from_pretrained.__func__

    def patched(cls, model_id, *a, **kw):
        # only the heterogeneous instance (device="auto", execution="workers") is traced
        if kw.get("device") == "auto" and kw.get("execution") == "workers":
            kw["trace"] = record
        return from_pretrained(cls, model_id, *a, **kw)

    Laya.from_pretrained = classmethod(patched)

    closed_loop = bench_concurrency.closed_loop

    def traced_loop(laya, req, start_at, end_at, out, reference):
        closed_loop(laya, req, start_at, end_at, out, reference)
        stream = "short" if req["length"] == args_short() else "long"
        with lock:
            windows.append(
                {
                    "stream": stream,
                    "instance": "auto" if laya.device == "auto" else laya.device,
                    "start_ns": int(start_at * 1e9),
                    "end_ns": int(end_at * 1e9),
                }
            )

    def args_short():
        i = sys.argv.index("--short")
        return int(sys.argv[i + 1])

    bench_concurrency.closed_loop = traced_loop
    bench_concurrency.main()
    out = Path(sys.argv[sys.argv.index("--output") + 1])
    rec = json.loads(out.read_text())
    rec["windows_t"] = windows
    rec["gpu_return"] = {"received_ns": [r for r, _ in returns], "return_us": [u for _, u in returns]}
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print("gpu returns recorded:", len(returns))


if __name__ == "__main__":
    main()
