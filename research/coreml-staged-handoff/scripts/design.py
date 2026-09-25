"""The fixed run order of the staged-handoff screen (pure, unit-tested; ../criteria.md).

    uv run python research/coreml-staged-handoff/scripts/design.py runs 1
    uv run python research/coreml-staged-handoff/scripts/design.py runs 2 --leader H32

laya only, L128 / L512, #92's full protocol with 2 cycles of 20 s windows (two hetero transitions
per run: cycle 0 after solo_long, cycle 1 after gpu_only), one fresh process per run.

Round 1 (screen): A, B, H32, H64, twice each, in a mirrored order so no cell is always first or
always after the same cell: H32 A H64 B | B H64 A H32. Round 1 gives every cell 4 transitions.

Round 2 (replication, only when round 1 names a leading candidate): 3 fresh runs of the leader,
each followed by an A run, then one more leader run after A: L A L A L. Round 2 gives the leader
6 more transitions (3 after solo_long, 3 after gpu_only) and A 4 more.
"""

from __future__ import annotations

import argparse

MODEL = "laya"
SHORT, LONG = 128, 512
CELLS = ("A", "B", "H32", "H64")
CANDIDATES = ("H32", "H64")
ROUND1 = (("H32", 1), ("A", 1), ("H64", 1), ("B", 1), ("B", 2), ("H64", 2), ("A", 2), ("H32", 2))
CYCLES = 2
SECONDS = 20.0


def round2(leader: str, fallback: bool = False) -> tuple[tuple[str, int], ...]:
    """Round 2's runs: leader r3, A r3, leader r4, A r4, leader r5. The H64 fallback (criteria.md,
    addendum 1) has its own A runs: H64 r3, A r5, H64 r4, A r6, H64 r5."""
    if leader not in CANDIDATES:
        raise ValueError(f"not a candidate: {leader}")
    if fallback:
        if leader != "H64":
            raise ValueError("only H64 is a fallback")
        return ((leader, 3), ("A", 5), (leader, 4), ("A", 6), (leader, 5))
    return ((leader, 3), ("A", 3), (leader, 4), ("A", 4), (leader, 5))


def run_file(cell: str, rep: int) -> str:
    return f"{MODEL}-{cell}-r{rep}.json.gz"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("runs", help='print one round\'s runs as "cell rep" lines, in order')
    r.add_argument("round", type=int, choices=(1, 2))
    r.add_argument("--leader", choices=CANDIDATES, default=None)
    r.add_argument("--fallback", action="store_true", help="the H64 fallback round 2 (addendum 1)")
    a = ap.parse_args()
    if a.round == 1:
        runs = ROUND1
    else:
        if a.leader is None:
            ap.error("round 2 needs --leader")
        runs = round2(a.leader, a.fallback)
    for cell, rep in runs:
        print(cell, rep)


if __name__ == "__main__":
    main()
