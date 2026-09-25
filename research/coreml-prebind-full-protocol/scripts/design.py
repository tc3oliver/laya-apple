"""The run order and the sequential-look rule of criteria.md (pure, unit-tested).

    uv run python research/coreml-prebind-full-protocol/scripts/design.py runs --stage 1 [--models ...]
    uv run python research/coreml-prebind-full-protocol/scripts/design.py next

Run order. Per model, a block is six runs of the three configurations in one of three
ABBA-style orders; the three orders rotate, so over each stage every configuration takes every
position equally often (mean position 3.5 in every block, and each configuration first and last
once per stage). A configuration's round r is its r-th run of the campaign: block b holds rounds
2b - 1 and 2b of every configuration, so a candidate's round r and P's round r always sit in the
same block, at most five runs apart. Within a stage the models alternate block by block, so each
model's runs are spread over the stage's whole duration.

Stages. Stage s adds blocks 3s - 2 .. 3s (18 runs per model, 6 per configuration, 18 matched
hetero windows per candidate at 3 cycles per run). The looks are at 18, 36 and 54 pairs. `next`
prints the stage and models the preregistered rule requires next (analyze.py applies the rule),
or nothing when R1 has reached its outcome.
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
BLOCK_ORDERS = (
    ("P", "C", "PB", "PB", "C", "P"),
    ("C", "PB", "P", "P", "PB", "C"),
    ("PB", "P", "C", "C", "P", "PB"),
)
BLOCKS_PER_STAGE = 3
MAX_STAGE = 3
CYCLES = 3  # hetero windows per run, one per cycle
SECONDS = 20.0  # window length of every campaign run
ROUNDS_PER_BLOCK = 2


def config_name(model: str, config: str) -> str:
    """The run's configuration as recorded: P is the model's production configuration (A)."""
    return MODELS[model][2] if config == "P" else config


def block_runs(model: str, block: int) -> list[tuple[str, str, int]]:
    """(model, configuration, round) of the six runs of one block (block numbers start at 1)."""
    if block < 1:
        raise ValueError("blocks start at 1")
    order = BLOCK_ORDERS[(block - 1) % len(BLOCK_ORDERS)]
    seen: Counter = Counter()
    out = []
    for c in order:
        seen[c] += 1
        out.append((model, config_name(model, c), ROUNDS_PER_BLOCK * (block - 1) + seen[c]))
    return out


def stage_blocks(stage: int) -> range:
    if not 1 <= stage <= MAX_STAGE:
        raise ValueError(f"stage must be 1..{MAX_STAGE}")
    return range(BLOCKS_PER_STAGE * (stage - 1) + 1, BLOCKS_PER_STAGE * stage + 1)


def stage_runs(stage: int, models) -> list[tuple[str, str, int]]:
    """Every run of one stage in campaign order: the models alternate block by block."""
    out = []
    for b in stage_blocks(stage):
        for m in models:
            out += block_runs(m, b)
    return out


def rounds_through(stage: int) -> tuple[int, ...]:
    """The rounds a look at the end of `stage` uses: all rounds of stages 1..stage."""
    stage_blocks(stage)
    return tuple(range(1, ROUNDS_PER_BLOCK * BLOCKS_PER_STAGE * stage + 1))


def pairs_at(stage: int) -> int:
    return len(rounds_through(stage)) * CYCLES


def run_file(model: str, config: str, rnd: int) -> str:
    return f"{model}-{config}-r{rnd}.json.gz"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("runs", help="print one stage's runs in order: model short long config round")
    r.add_argument("--stage", type=int, required=True)
    r.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    sub.add_parser("next", help="print 'stage model...' the preregistered rule requires next, or nothing")
    a = ap.parse_args()
    if a.cmd == "runs":
        for m, c, rnd in stage_runs(a.stage, a.models):
            short, long_, _ = MODELS[m]
            print(m, short, long_, c, rnd)
        return
    import analyze  # the rule needs the data; analyze.py imports this module, not the reverse

    step = analyze.summarise(analyze.RAW)["next"]
    if step is not None:
        print(step["stage"], *step["models"])


if __name__ == "__main__":
    main()
