"""The Switchyard oracle gate, for every train-prompt wording tried (docs/switchyard.md).

    LAYA_APPLE_CACHE=... uv run python scripts/switchyard_oracle.py --device gpu \
        --output benchmarks/switchyard/oracle/gpu.json
    LAYA_APPLE_CACHE=... uv run python scripts/switchyard_oracle.py --device ane \
        --wordings 1 14 --output benchmarks/switchyard/oracle/ane.json

Each wording is a variation of the original contract wording (WORDINGS below; #14 is the
frozen `world.WORDING`). The gate set is 12 patterns x 20 surface variants
(`world.oracle_cases()`, seed 0). Wordings that pass it also run on the held-out set,
12 patterns x 50 variants from seed 1. The model is loaded once, inline, on --device.

Per case the output keeps the pattern, line, train id, oracle, answer and margin (the
oracle's probability minus the best other one; negative when the answer is wrong), so every
accuracy and minimum margin in docs/switchyard.md can be recomputed from the file.
"""

from __future__ import annotations

import argparse
import json
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path

from laya_apple.demos.switchyard import world

ORIGINAL = {
    "header": "Line {line} inbound. Train {train} is approaching the junction.",
    "sentences": {
        world.OCCUPIED: "Platform {platform} is occupied by train {other}.",
        world.CLOSED: "Platform {platform} is closed for maintenance.",
        world.CLEAR: "Platform {platform} is clear.",
    },
    "question": "Which platform should train {train} be routed to?",
    "criteria": list(world.PLATFORMS),
}
Q_CLEAR_FOR = "Which platform is clear for train {train}?"
Q_ROUTE_CLEAR = "Train {train} must be routed to the clear platform. Which platform is clear?"
CLEAR_OPEN = {world.CLEAR: "Platform {platform} is clear and open."}
BLOCKED = {world.OCCUPIED: "Platform {platform} is blocked by train {other}."}

# id -> (description, changes to ORIGINAL). Order and ids as in docs/switchyard.md.
WORDINGS = {
    1: ("original contract wording", {}),
    2: ("question: clear for train", {"question": Q_CLEAR_FOR}),
    3: ("question: which platform is clear", {"question": "Which platform is clear?"}),
    4: ("question: must be routed to the clear platform", {"question": Q_ROUTE_CLEAR}),
    5: ("question: free for train to enter", {"question": "Which platform is free for train {train} to enter?"}),
    6: ("clear: clear and open", {"sentences": CLEAR_OPEN}),
    7: (
        "clear: clear and available for train arrivals",
        {"sentences": {world.CLEAR: "Platform {platform} is clear and available for train arrivals."}},
    ),
    8: ("closed: closed", {"sentences": {world.CLOSED: "Platform {platform} is closed."}}),
    9: (
        "occupied: blocked by train, closed: closed",
        {"sentences": BLOCKED | {world.CLOSED: "Platform {platform} is closed."}},
    ),
    10: (
        "criteria with descriptions",
        {"criteria": {"A": "platform A", "B": "platform B", "C": "platform C"}},
    ),
    11: ("#2 + #6", {"question": Q_CLEAR_FOR, "sentences": CLEAR_OPEN}),
    12: ("#2 + #6 + occupied: blocked by train", {"question": Q_CLEAR_FOR, "sentences": CLEAR_OPEN | BLOCKED}),
    13: (
        "question: clear and open for train, + #6",
        {"question": "Which platform is clear and open for train {train}?", "sentences": CLEAR_OPEN},
    ),
    14: ("#4 + #6 (frozen)", {"question": Q_ROUTE_CLEAR, "sentences": CLEAR_OPEN}),
}
SETS = {"gate": (20, 0), "held_out": (50, 1)}  # (variants per pattern, seed)
CASE_FIELDS = ["pattern", "line", "train_id", "oracle", "answer", "margin", "tokens"]


def wording(changes: dict) -> dict:
    w = json.loads(json.dumps(ORIGINAL))
    for key, value in changes.items():
        if key == "sentences":
            w["sentences"].update(value)
        else:
            w[key] = value
    return w


def evaluate(laya, w: dict, cases) -> dict:
    rows, correct, margins = [], 0, []
    for t in cases:
        context, questions = world.train_prompt(t, w)
        tokens = laya.prepare(context, questions).sequence_length
        answer = laya.predict(context=context, questions=questions).answers[world.QUESTION_ID]
        p = answer["probabilities"]
        margin = round(p[t.oracle] - max(v for k, v in p.items() if k != t.oracle), 4)
        correct += answer["choice"] == t.oracle
        if answer["choice"] == t.oracle:
            margins.append(margin)
        rows.append([t.pattern, t.line, t.train_id, t.oracle, answer["choice"], margin, tokens])
    return {
        "cases": len(rows),
        "correct": correct,
        "min_margin_correct": min(margins) if margins else None,
        "tokens": [min(r[-1] for r in rows), max(r[-1] for r in rows)],
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["gpu", "ane"], required=True)
    ap.add_argument("--wordings", type=int, nargs="+", default=list(WORDINGS))
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import laya_apple
    from laya_apple import Laya
    from laya_apple.artifacts import platform_profile

    warnings.filterwarnings("ignore", message=".*checkpoint temperatures.*")
    laya = Laya.from_pretrained(world.MODEL, device=args.device, local_files_only=True)
    runs = []
    try:
        for wid in args.wordings:
            description, changes = WORDINGS[wid]
            w = wording(changes)
            run = {"wording_id": wid, "description": description, "wording": w, "sets": {}}
            for name, (variants, seed) in SETS.items():
                if name == "held_out" and run["sets"]["gate"]["correct"] < run["sets"]["gate"]["cases"]:
                    continue  # only wordings that pass the gate run on the held-out set
                run["sets"][name] = evaluate(laya, w, world.oracle_cases(variants, seed))
                s = run["sets"][name]
                print(f"#{wid} {name}: {s['correct']}/{s['cases']} min margin {s['min_margin_correct']}", flush=True)
            runs.append(run)
    finally:
        laya.close()
    record = {
        "experiment": "switchyard oracle gate, train-prompt wordings",
        "laya_apple": laya_apple.__version__,
        "time": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "execution": "inline",
        "platform": platform_profile() | {"python": platform.python_version()},
        "model": {"name": laya.spec.name, "revision": laya.spec.revision},
        "frozen_prompt_template_sha256": world.prompt_template_sha256(),
        "sets": {k: {"variants_per_pattern": v, "seed": s} for k, (v, s) in SETS.items()},
        "case_fields": CASE_FIELDS,
        "runs": runs,
    }
    assert wording(WORDINGS[14][1]) == world.WORDING, "wording #14 must be the frozen world.WORDING"
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, separators=(",", ":")) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
