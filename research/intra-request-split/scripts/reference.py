"""Write raw/reference-<model>.json: the FP32 reference answers for every fixture request.

    HF_HUB_OFFLINE=1 uv run --extra ane --extra convert python research/intra-request-split/scripts/reference.py --model laya

The reference is the PyTorch CPU FP32 port that `laya-apple artifacts build` checks every ANE
artifact against (laya_apple/conversion/torch_reference.py, the vendored upstream model), run one
row at a time with no padding, on the original checkpoint weights. Its answers go through the
product's own `format_answers`, so every arm's answers are compared with it by the FP16 gate
(scripts/bench_serve.compare_answers: probability error, hard mismatches, near-tie flips).

CPU only (no GPU, no ANE), but it keeps several cores busy for a few minutes: never run it beside
a timing-sensitive step.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    a = p.parse_args(argv)

    import os

    import torch

    from laya_apple.conversion.torch_reference import load_model
    from laya_apple.hub import checkpoint_path, verify_weights
    from laya_apple.prompt import Calibration, Tokenizer, format_answers, prepare
    from laya_apple.registry import resolve

    out_path = common.RAW / f"reference-{a.model}.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; references are written once")
    fx = common.load_fixtures(a.model)
    spec = resolve(a.model)
    ckpt = checkpoint_path(spec, local_files_only=True)
    verify_weights(spec, ckpt)
    cfg = json.loads((ckpt / "rl_agent_config.json").read_text())
    tok = Tokenizer(ckpt / "tokenizer")
    calib = Calibration(cfg)
    torch.set_num_threads(max(1, (os.cpu_count() or 8) // 2))
    model = load_model(ckpt, spec.max_len, attention_implementation="explicit")
    t_start = time.perf_counter()
    out = {
        "model": a.model,
        "revision": spec.revision,
        "reference": "laya_apple.conversion.torch_reference (PyTorch CPU FP32, one row at a time, no padding)",
        "fixtures_sha256": common.sha256_file(common.fixtures_path(a.model)),
        "environment": common.environment(),
        "workloads": {},
    }
    for name, w in fx["workloads"].items():
        reqs = []
        for r in w["requests"]:
            prep = prepare(tok, cfg, r["state"], r["questions"])
            if [len(it["ids"]) for it in prep.items] != r["lengths"]:
                raise SystemExit(f"{name} seed {r['seed']}: tokenisation differs from the fixture")
            width = max(2, max(len(it["markers"]) for it in prep.items))
            logits = np.full((len(prep.items), width), -1e4, np.float32)
            acts = []
            with torch.inference_mode():
                for i, it in enumerate(prep.items):
                    m, k = len(it["ids"]), len(it["markers"])
                    ids = torch.tensor([it["ids"]], dtype=torch.long)
                    att = torch.ones((1, m), dtype=torch.long)
                    mpos = torch.tensor([it["markers"]], dtype=torch.long)
                    mmask = torch.ones((1, k), dtype=torch.bool)
                    lg, ac = model(ids, att, mpos, mmask, torch.tensor([it["qtype"]]))
                    logits[i, :k] = lg.numpy()[0, :k]
                    acts.append(ac.numpy()[0])
            act = np.stack(acts)
            reqs.append(
                {
                    "seed": r["seed"],
                    "answers": format_answers(prep, logits, act, calib),
                    "logits": logits.tolist(),
                    "action_logits": act.tolist(),
                }
            )
            print(f"{a.model} {name} seed {r['seed']}: {len(prep.items)} rows", flush=True)
        out["workloads"][name] = reqs
    out["seconds"] = round(time.perf_counter() - t_start, 1)
    common.write_json(out_path, out)
    print(f"wrote {out_path} in {out['seconds']} s")


if __name__ == "__main__":
    main()
