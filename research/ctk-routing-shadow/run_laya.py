"""Laya arm: one single-choice Typed Decision question per request. Read-only shadow run.

Usage: run_laya.py SPLIT [--variant names|described]
Writes results/laya_<variant>_<split>.jsonl, or results/laya_BLOCKED.txt when Laya cannot load.
"""

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
LABELS = ["lint-fix", "small-bugfix", "refactor", "docs", "needs-human", "needs-more-info"]
DESCRIBED = [
    "lint-fix: fix a linter, formatter or style warning",
    "small-bugfix: fix a specific, localized bug or failing test",
    "refactor: restructure or rename code without changing behaviour",
    "docs: write or correct documentation or comments",
    "needs-human: risky, security, production-data, permission or design decision",
    "needs-more-info: the request is too vague to act on",
]
INSTR = "Classify this software work request."


def main():
    split = sys.argv[1]
    variant = sys.argv[sys.argv.index("--variant") + 1] if "--variant" in sys.argv else "described"
    crit = DESCRIBED if variant == "described" else LABELS
    out = HERE / "results" / f"laya_{variant}_{split}.jsonl"
    try:
        from laya_apple import Laya

        model = Laya.from_pretrained("convaiinnovations/laya-typed-decisions", device="gpu", local_files_only=True)
    except Exception as e:  # BLOCKED, not a crash: the harness must still finish the other arms
        (HERE / "results" / "laya_BLOCKED.txt").write_text(f"{type(e).__name__}: {e}\n")
        print("BLOCKED", e)
        return
    q = {"route": {"type": "choice", "instructions": INSTR, "criteria": crit}}
    rows = [json.loads(line) for line in (HERE / "data" / f"{split}.jsonl").open()]
    model.predict(context="warmup", questions=q)
    with out.open("w") as f:
        for r in rows:
            t0 = time.perf_counter()
            res = model.predict(context=r["text"], questions=q)
            ms = (time.perf_counter() - t0) * 1000
            a = res.answers["route"]
            pred = a["choice"]
            pred = next((x for x in LABELS if pred.startswith(x)), pred)
            probs = {next((x for x in LABELS if k.startswith(x)), k): v for k, v in a["probabilities"].items()}
            f.write(json.dumps({"id": r["id"], "pred": pred, "conf": a["answer_confidence"], "probs": probs,
                                "latency_ms": ms, "device": res.runtime.device}) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
