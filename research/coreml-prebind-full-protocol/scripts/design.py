"""The run order and the looks of criteria.md's "fast fail, slow pass" addendum (pure, unit-tested).

    uv run python research/coreml-prebind-full-protocol/scripts/design.py runs
    uv run python research/coreml-prebind-full-protocol/scripts/design.py next

Blocks, per model (a configuration's round r is its r-th run of the campaign):
  block 1  P C PB PB C P   rounds 1-2 of P, C and PB (6 matched hetero pairs per candidate)
  block 2  PB P P PB       rounds 3-4 of P and PB
  block 3  P PB PB P       rounds 5-6 of P and PB
Every block is ABBA-balanced for P and PB (and for C in block 1); a round r of PB and of P
always sit in the same block. There is no C after block 1 and no extension: at most 18 pairs.

Campaign order: laya b1, laya-typed-decisions b1, [look n=6, both models], laya b2, [look n=12,
laya], laya-typed-decisions b2, [look n=12, laya-typed-decisions], laya b3, laya-typed-decisions
b3, [final look n=18, both]. The n=6 and n=12 looks are futility-only (they can stop R1, never
pass it); the n=18 look is the formal #83 paired gate. analyze.py applies the rule; `next`
prints the next block's runs as "model short long config round" lines, or STOP.
"""

from __future__ import annotations

import argparse
from collections import Counter

# model: (short, long, production configuration). The shapes are the v1.0 commands.
MODELS = {
    "laya": (128, 512, "A"),
    "laya-typed-decisions": (128, 1024, "A"),
}
CONFIGS = ("P", "C", "PB")
BLOCK_ORDERS = {
    1: ("P", "C", "PB", "PB", "C", "P"),
    2: ("PB", "P", "P", "PB"),
    3: ("P", "PB", "PB", "P"),
}
BLOCKS = tuple(BLOCK_ORDERS)
LOOKS = {1: ("futility", 6), 2: ("futility", 12), 3: ("final", 18)}  # after block b: (kind, pairs)
CYCLES = 3  # hetero windows per run, one per cycle
SECONDS = 20.0  # window length of every campaign run
ROUNDS_PER_BLOCK = 2
STOP = "STOP"


def config_name(model: str, config: str) -> str:
    """The run's configuration as recorded: P is the model's production configuration (A)."""
    return MODELS[model][2] if config == "P" else config


def block_runs(model: str, block: int) -> list[tuple[str, str, int]]:
    """(model, recorded configuration, round) of one block's runs, in run order."""
    if block not in BLOCK_ORDERS:
        raise ValueError(f"blocks are {BLOCKS}")
    seen: Counter = Counter()
    out = []
    for c in BLOCK_ORDERS[block]:
        seen[c] += 1
        out.append((model, config_name(model, c), ROUNDS_PER_BLOCK * (block - 1) + seen[c]))
    return out


def blocks(models=None) -> list[tuple[str, int]]:
    """(model, block) in campaign order: the models alternate within each block number."""
    models = list(MODELS) if models is None else list(models)
    return [(m, b) for b in BLOCKS for m in models]


def rounds_through(block: int) -> tuple[int, ...]:
    """The rounds a look after `block` uses: all rounds of blocks 1..block."""
    if block not in BLOCK_ORDERS:
        raise ValueError(f"blocks are {BLOCKS}")
    return tuple(range(1, ROUNDS_PER_BLOCK * block + 1))


def pairs_at(block: int) -> int:
    return len(rounds_through(block)) * CYCLES


def run_file(model: str, config: str, rnd: int) -> str:
    return f"{model}-{config}-r{rnd}.json.gz"


def designed_files(models=None) -> set[str]:
    return {run_file(m, c, r) for m, b in blocks(models) for _, c, r in block_runs(m, b)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("runs", help="print every run in campaign order: model short long config round")
    sub.add_parser("next", help="print the next block's runs (model short long config round), or STOP")
    a = ap.parse_args()
    if a.cmd == "runs":
        todo = [x for m, b in blocks() for x in block_runs(m, b)]
    else:
        import analyze  # the rule needs the data; analyze.py imports this module, not the reverse

        step = analyze.summarise(analyze.RAW)["next"]
        if step is None:
            print(STOP)
            return
        todo = block_runs(step["model"], step["block"])
    for m, c, rnd in todo:
        short, long_, _ = MODELS[m]
        print(m, short, long_, c, rnd)


if __name__ == "__main__":
    main()
