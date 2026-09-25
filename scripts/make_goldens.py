"""Derive the shipped parity goldens from the upstream PyTorch FP32 references.

    uv run python scripts/make_goldens.py            # writes laya_apple/parity/goldens/*.json
    uv run python scripts/make_goldens.py --check    # fails if the shipped goldens are stale

The references themselves are produced by unmodified upstream Laya (0.3.20, PyTorch CPU
FP32) with scripts/make_reference.py (the `[reference]` extra); regenerate them there and
then rerun this script. They cover every Phase -1 fixture, whose items and logits must be
identical to the Phase -1 record made with upstream 0.3.5, plus cases for the upstream
changes after 0.3.5. The shipped goldens keep what the parity gate and the answer-format
tests read: prompt items, unrounded decision logits, action logits and upstream's answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "research/upstream-reference/raw/laya-0.3.20"
OUT = ROOT / "laya_apple/parity/goldens"
MODELS = ("laya", "laya-multilingual", "laya-typed-decisions")
KEEP = ("name", "state", "questions", "items", "logits", "action_logits", "length")


def derive(model: str) -> str:
    src = SRC / f"{model}.json"
    raw = json.loads(src.read_text())
    if not all(c.get("tokenizer_match", True) for c in raw["cases"]):
        raise SystemExit(f"{model}: reference cases where the Rust tokenizer differs from upstream")
    if not all(c.get("phase0_identical", True) for c in raw["cases"]):
        raise SystemExit(f"{model}: reference cases that differ from the Phase -1 record")
    pk = raw["environment"]["packages"]
    out = {
        "model": raw["model"],
        "reference": raw["reference"],
        "source_weights_sha256": raw["source_weights_sha256"],
        "generated_from": {
            "file": str(src.relative_to(ROOT)),
            "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        },
        "reference_packages": {k: pk[k] for k in ("torch", "transformers", "laya", "tokenizers")},
        "cases": [{**{k: c[k] for k in KEEP if k in c}, "answers": c["result"]["answers"]} for c in raw["cases"]],
    }
    return json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    stale = []
    for model in MODELS:
        text, path = derive(model), OUT / f"{model}.json"
        if args.check:
            if json.loads(path.read_text()) != json.loads(text):
                stale.append(model)
        else:
            path.write_text(text)
            print(f"{path.relative_to(ROOT)}: {len(json.loads(text)['cases'])} cases")
    if stale:
        sys.exit(f"stale goldens: {stale}")
    if args.check:
        print("goldens are up to date")


if __name__ == "__main__":
    main()
