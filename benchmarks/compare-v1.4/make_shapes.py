"""Write shapes.json: the request shapes timed by the comparison (text written for this project).

    uv run python benchmarks/compare-v1.4/make_shapes.py [--check]

Every runtime answers exactly these requests. Token counts are recorded per model with
laya_apple's tokenizer (it reproduces upstream Laya's prompt exactly; laya_apple/parity), so
the README can say which shapes fit which fixed-shape artifact:

- short:    1 choice question, <= 96 prompt tokens on every model (fits a 96-token ANE bundle)
- long:     1 noul question, a state near the English checkpoint's 512-token limit
- mixed3:   choice + score + noul in one call (the mixed-type shape of laya-coreml issue #5)
- uniform3: three noul questions in one call (same state; the control for mixed3)
- batch16:  16 short questions of all three types in one call (throughput)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "shapes.json"
MODELS = ("laya", "laya-multilingual", "laya-typed-decisions")

SHORT_STATE = (
    "Order 7731 arrived yesterday with a cracked screen. The customer attached two photos of the damage, "
    "says the box was dented on one corner, and asks for a replacement before the weekend."
)
LONG_PARAGRAPH = (
    "The warehouse scanner logged parcel 7731 at the loading dock at 06:12, then again at the sorting "
    "belt at 06:40, where the weight reading was two hundred grams under the manifest. The courier note "
    "says the outer box was dented on one corner. The customer later reported that the screen was "
    "cracked on arrival and attached two photos of the damage. "
)
LONG_STATE = LONG_PARAGRAPH * 5

CHOICE = {
    "type": "choice",
    "instructions": "What should support do next?",
    "criteria": ["send replacement", "issue refund", "ask for more details"],
}
SCORE = {
    "type": "score",
    "instructions": "How severe is the damage?",
    "criteria": ["cosmetic", "partly usable", "unusable"],
}
NOUL = {"type": "noul", "instructions": "Did the customer provide evidence of the damage?"}
NOUL_B = {"type": "noul", "instructions": "Is the customer asking for a refund?"}
NOUL_C = {"type": "noul", "instructions": "Does the message mention an order number?"}


def batch16():
    qs = {}
    for i in range(16):
        base = (CHOICE, SCORE, NOUL, NOUL_B)[i % 4]
        q = dict(base)
        q["instructions"] = f"{base['instructions']} (check {i + 1})"
        qs[f"q{i:02d}"] = q
    return qs


def shapes():
    return {
        "short": {"state": SHORT_STATE, "questions": {"next_step": CHOICE}},
        "long": {"state": LONG_STATE, "questions": {"evidence": NOUL}},
        "mixed3": {"state": SHORT_STATE, "questions": {"next_step": CHOICE, "severity": SCORE, "evidence": NOUL}},
        "uniform3": {"state": SHORT_STATE, "questions": {"evidence": NOUL, "refund": NOUL_B, "order": NOUL_C}},
        "batch16": {"state": SHORT_STATE, "questions": batch16()},
    }


def token_lengths(sh):
    sys.path.insert(0, str(ROOT))
    from laya_apple.hub import checkpoint_path
    from laya_apple.prompt import Tokenizer, prepare
    from laya_apple.registry import resolve

    out = {}
    for m in MODELS:
        spec = resolve(m)
        ck = checkpoint_path(spec, local_files_only=True)
        cfg = json.loads((ck / "rl_agent_config.json").read_text())
        tok = Tokenizer(ck / "tokenizer")
        out[m] = {
            k: [len(it["ids"]) for it in prepare(tok, cfg, v["state"], v["questions"]).items] for k, v in sh.items()
        }
    return out


def build():
    sh = shapes()
    lens = token_lengths(sh)
    for m in MODELS:
        assert max(lens[m]["short"]) <= 96, (m, lens[m]["short"])
        assert 350 <= max(lens[m]["long"]) <= 512, (m, lens[m]["long"])
    return {
        "note": "Request shapes for benchmarks/compare-v1.4 (make_shapes.py). Text written for this project.",
        "prompt_tokens": lens,
        "shapes": sh,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    text = json.dumps(build(), ensure_ascii=False, indent=1) + "\n"
    if a.check:
        if not OUT.exists() or OUT.read_text() != text:
            print(f"{OUT} is out of date; run make_shapes.py")
            return 1
        print(f"{OUT} up to date")
        return 0
    OUT.write_text(text)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
