"""The fixed run order of the async transient experiment (pure, unit-tested).

    uv run python research/coreml-async-transient/scripts/design.py runs

laya only, L128 / L512, #92's full protocol with 2 cycles of 20 s windows: two hetero transitions
per run (cycle 0 follows solo_long, cycle 1 follows gpu_only). Four runs, one fresh process each,
in this order, all instrumented the same way:

  run  cell      file round
  1    A         1
  2    PB-ASYNC  1
  3    PB-ASYNC  2
  4    A         2

`runs` prints "cell round" lines in that order.
"""

from __future__ import annotations

import argparse

MODEL = "laya"
SHORT, LONG = 128, 512
CELLS = ("A", "PB-ASYNC")
RUNS = (("A", 1), ("PB-ASYNC", 1), ("PB-ASYNC", 2), ("A", 2))
CYCLES = 2
SECONDS = 20.0


def run_file(cell: str, rnd: int) -> str:
    return f"{MODEL}-{cell}-r{rnd}.json.gz"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("runs", help='print every run as "cell round" lines, in order')
    ap.parse_args()
    for cell, rnd in RUNS:
        print(cell, rnd)


if __name__ == "__main__":
    main()
