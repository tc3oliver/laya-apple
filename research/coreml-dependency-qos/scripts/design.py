"""The fixed run order of the dependency QoS screen (pure, unit-tested; ../criteria.md).

    uv run python research/coreml-dependency-qos/scripts/design.py runs 1
    uv run python research/coreml-dependency-qos/scripts/design.py runs 2 [--raw DIR]

laya only, L128 / L512, #92's full protocol with 2 cycles of 20 s windows (two hetero transitions
per run), one fresh process per run, every cell #94's PB-ASYNC with the same instrumentation.

Round 1 runs B, O, Q. Round 2 runs only if round 1 has a valid B and at least one PASS: the passing
candidates in the reverse of their round-1 order, then B (Q, O, B; or O, B; or Q, B). There is no
third round.

`runs 1` prints round 1 as "cell round" lines; `runs 2` judges round 1 from the raw files (through
analyze.py) and prints round 2's lines, or nothing when round 2 does not run.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

MODEL = "laya"
SHORT, LONG = 128, 512
CELLS = ("B", "O", "Q")
CANDIDATES = ("O", "Q")
ROUND1 = tuple((c, 1) for c in CELLS)
CYCLES = 2
SECONDS = 20.0


def round2(passing) -> tuple[tuple[str, int], ...]:
    """Round 2's runs for the candidates that passed round 1 (with a valid B): the passing ones in
    the reverse of their round-1 order, then B; none if nothing passed."""
    passing = set(passing)
    unknown = passing - set(CANDIDATES)
    if unknown:
        raise ValueError(f"not a candidate: {sorted(unknown)}")
    runs = tuple((c, 2) for c in reversed(CELLS) if c in passing)
    return runs + (("B", 2),) if runs else ()


def run_file(cell: str, rnd: int) -> str:
    return f"{MODEL}-{cell}-r{rnd}.json.gz"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("runs", help='print one round\'s runs as "cell round" lines, in order')
    r.add_argument("round", type=int, choices=(1, 2))
    r.add_argument("--raw", type=Path, default=None, help="raw directory (default: the experiment's raw/)")
    a = ap.parse_args()
    if a.round == 1:
        runs = ROUND1
    else:
        spec = importlib.util.spec_from_file_location(
            "dependency_qos_analyze", Path(__file__).resolve().parent / "analyze.py"
        )
        analyze = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(analyze)
        runs = round2(analyze.round2_trigger(a.raw or analyze.RAW))
    for cell, rnd in runs:
        print(cell, rnd)


if __name__ == "__main__":
    main()
