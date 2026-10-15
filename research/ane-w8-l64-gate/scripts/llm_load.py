"""Continuous local MLX LLM generation on the GPU (S3 / S5 load). Run with an interpreter that has mlx_lm.

    llm_load.py --model <local dir> --out <jsonl> [--max-tokens 64] [--secs 0 (until the stop file appears)]

One JSON line per generation: wall start/end (time.time()), TTFT, tokens, decode tokens/s. Greedy
decoding, a fixed synthetic prompt. Stops when <out>.stop exists or after --secs.
"""

import argparse
import json
import time
from pathlib import Path

from mlx_lm import load, stream_generate

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-tokens", type=int, default=64)
ap.add_argument("--secs", type=float, default=0)
a = ap.parse_args()
model, tok = load(a.model)
prompt = tok.apply_chat_template(
    [{"role": "user", "content": " ".join(f"Item {i} costs {i * 7 % 100} units." for i in range(60)) + " Summarise."}],
    add_generation_prompt=True,
    tokenize=False,
)
stop, t_end = Path(a.out + ".stop"), (time.time() + a.secs if a.secs else None)
with open(a.out, "a") as f:
    print("ready", flush=True)
    while not stop.exists() and (t_end is None or time.time() < t_end):
        t0 = time.time()
        ttft, n, tps, ptoks = None, 0, None, None
        for r in stream_generate(model, tok, prompt, max_tokens=a.max_tokens):
            if ttft is None:
                ttft = time.time() - t0
            n, tps, ptoks = r.generation_tokens, r.generation_tps, r.prompt_tokens
        f.write(json.dumps({"t0": t0, "t1": time.time(), "ttft_s": ttft, "tokens": n, "decode_tps": tps, "prompt_tokens": ptoks}) + "\n")
        f.flush()
