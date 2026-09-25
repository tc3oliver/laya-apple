"""The run order and the slow-window classifier of the slow-state trigger test (pure, unit-tested).

    uv run python research/coreml-slow-state-trigger/scripts/design.py next

laya only, L128 / L512. Round 1 runs every cell once, in the fixed order A, PB-R, PB-H, PB-W,
PB-P. analyze.py's `decide` applies the rules of ../criteria.md to the runs so far and names the
next round, if any (at most one: a round-2 PB-R rerun, or a round-2 confirmation of A, PB-R and
the round-1 candidates in the fixed order); there is never a third round. `next` prints the next
round's runs as "cell round" lines, or nothing when the design has reached an end.
"""

from __future__ import annotations

import argparse

MODEL = "laya"
SHORT, LONG = 128, 512
CELLS = ("A", "PB-R", "PB-H", "PB-W", "PB-P")
ROUND1 = CELLS  # fixed order
CANDIDATES = ("PB-H", "PB-W", "PB-P")
CYCLES = 3  # hetero windows per run, one per cycle
SECONDS = 20.0  # window length of every campaign run
SLOW_P99_MS = 13.0  # a hetero window is slow if its short P99 >= this, normal if below


def is_slow(short_p99_ms: float) -> bool:
    return short_p99_ms >= SLOW_P99_MS


def run_file(cell: str, rnd: int) -> str:
    return f"{MODEL}-{cell}-r{rnd}.json.gz"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("next", help='print the next round\'s runs as "cell round" lines, or nothing')
    ap.parse_args()
    import analyze  # the rule needs the data; analyze.py imports this module, not the reverse

    for cell, rnd in analyze.summarise(analyze.RAW)["next"] or []:
        print(cell, rnd)


if __name__ == "__main__":
    main()
