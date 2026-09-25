"""The run order and the slow-window classifier of the async-predict screen (pure, unit-tested).

    uv run python research/coreml-async-predict/scripts/design.py next

laya only, L128 / L512. Round 1 runs A, PB-SYNC, PB-ASYNC once each, in that order. analyze.py's
`decide` applies ../criteria.md to the runs so far: the only possible round 2 is the replication
in the reversed order PB-ASYNC, PB-SYNC, A; there is never a third round. `next` prints the next
round's runs as "cell round" lines, or nothing when the design has reached an end.
"""

from __future__ import annotations

import argparse

MODEL = "laya"
SHORT, LONG = 128, 512
CELLS = ("A", "PB-SYNC", "PB-ASYNC")
ROUND1 = CELLS  # fixed order
ROUND2 = tuple(reversed(CELLS))  # the replication: PB-ASYNC, PB-SYNC, A
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
