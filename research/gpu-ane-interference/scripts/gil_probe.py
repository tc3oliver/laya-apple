"""Does a backend call hold the GIL, and for how long does it delay another Python thread?

    LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 uv run python \
        research/gpu-ane-interference/scripts/gil_probe.py --model laya-typed-decisions \
        --out research/gpu-ane-interference/raw/gil-probe-laya-typed-decisions.json

A "load" thread runs one of the loads below in a loop for `--seconds`. Meanwhile two probes
run in turn on other threads of the same interpreter:

  counter   a pure-Python loop counting iterations: its rate relative to the idle rate is
            the share of wall time this thread could hold the GIL
  wake      a thread that sleeps 1 ms and records how late it wakes and re-acquires the
            GIL (the delay a dispatcher thread sees on its IPC reply)

Loads (each in this process, on its own thread):
  idle        nothing (baseline)
  sleep       time.sleep(0.010) in a loop: releases the GIL (negative control)
  numpy       a 1024x1024 float32 matmul loop: NumPy releases the GIL in BLAS (control)
  spin        a pure-Python loop: holds the GIL, released every switch interval (control)
  ane_predict Core ML predict only, ANE bucket B (features built once)
  ane_forward the full ANEBackend.forward (features + predict + action head), bucket B
  mlx_forward MLXBackend.forward at the model's maximum length (in-process MLX)

Nothing here changes laya-apple; it loads the product backends and times them.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import threading
import time
from pathlib import Path

import numpy as np

perf = time.perf_counter


def counter_probe(seconds: float) -> float:
    n, end = 0, perf() + seconds
    while perf() < end:
        for _ in range(1000):
            n += 1
    return n / seconds


def wake_probe(seconds: float, period: float = 0.001) -> list:
    late, end = [], perf() + seconds
    while perf() < end:
        t = perf()
        time.sleep(period)
        late.append((perf() - t - period) * 1e3)
    return late


def stats(a) -> dict:
    a = np.asarray(a)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="laya-typed-decisions")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import coremltools
    import mlx.core as mx

    import laya_apple
    from laya_apple.artifacts import platform_profile
    from laya_apple.backends.coreml_ane import ANEBackend, ane_features
    from laya_apple.backends.mlx import MLXBackend
    from laya_apple.hub import checkpoint_path
    from laya_apple.prompt import Tokenizer, prepare
    from laya_apple.registry import resolve
    from laya_apple.workload import make_request

    spec = resolve(args.model)
    ckpt = checkpoint_path(spec, local_files_only=True)
    cfg = json.loads((ckpt / "rl_agent_config.json").read_text())
    enc = json.loads((ckpt / "encoder/config.json").read_text())
    tok = Tokenizer(ckpt / "tokenizer")
    B = max(spec.auto_ane_buckets)
    L = int(cfg.get("max_len", 512))
    ane = ANEBackend(spec, ckpt, tok.pad_token_id, int(enc["local_attention"]), [B], strict=True)
    gpu = MLXBackend(spec, ckpt, tok.pad_token_id)
    rows_b = prepare(tok, cfg, *make_request(tok, cfg, B, n_questions=1, seed=1)).items
    rows_l = prepare(tok, cfg, *make_request(tok, cfg, L, n_questions=1, seed=1)).items
    feats = ane_features(rows_b, B, 1, ane.host.embedding, ane.host.type_embedding, ane.host.window(B), ane.pad_id)
    model = ane.models[B]
    m1 = np.ones((1024, 1024), np.float32)
    for _ in range(3):  # warm
        ane.forward(rows_b)
        gpu.forward(rows_l)

    def spin_once():
        x = 0
        for i in range(20000):
            x ^= i

    loads = {
        "idle": None,
        "sleep": lambda: time.sleep(0.010),
        "numpy": lambda: m1 @ m1,
        "spin": spin_once,
        "ane_predict": lambda: model.predict(feats),
        "ane_forward": lambda: ane.forward(rows_b),
        "mlx_forward": lambda: gpu.forward(rows_l),
    }
    results = []
    for rep in range(args.repeats):
        for name, fn in loads.items():
            stop = threading.Event()
            calls = []

            def run(fn=fn, calls=calls, stop=stop):
                while not stop.is_set():
                    t = perf()
                    fn()
                    calls.append((perf() - t) * 1e3)

            th = None
            if fn is not None:
                th = threading.Thread(target=run, daemon=True)
                th.start()
                time.sleep(0.5)
            rate = counter_probe(args.seconds)
            late = wake_probe(args.seconds)
            stop.set()
            if th is not None:
                th.join()
            results.append(
                {
                    "repeat": rep,
                    "load": name,
                    "counter_rate": rate,
                    "wake_late_ms": stats(late),
                    "load_call_ms": stats(calls) if calls else None,
                }
            )
            print(rep, name, f"counter {rate / 1e6:.2f} M/s", {k: round(v, 3) for k, v in stats(late).items()}, flush=True)
    idle = np.median([r["counter_rate"] for r in results if r["load"] == "idle"])
    for r in results:
        r["counter_share_of_idle"] = r["counter_rate"] / idle
    out = {
        "experiment": "gil-probe",
        "args": vars(args),
        "environment": {
            "laya_apple": laya_apple.__version__,
            "platform": platform_profile(),
            "python": platform.python_version(),
            "mlx": mx.__version__,
            "coremltools": coremltools.__version__,
            "switch_interval_s": sys.getswitchinterval(),
        },
        "ane_bucket": B,
        "mlx_length": L,
        "results": results,
    }
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
