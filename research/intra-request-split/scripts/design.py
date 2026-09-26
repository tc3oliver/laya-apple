"""Run order and pairing (pure; unit-tested in tests/unit/test_intra_request_split.py).

Blocks are ABBA sequences of two labels (criteria.json). Within a block, each run of the
candidate label is paired with the adjacent run of the other label:

    A B B A  ->  (1, 2), (4, 3)          B A A B  ->  (2, 1), (3, 4)

so every pair is two consecutive runs, and each label runs first in half of the pairs.
"""

from __future__ import annotations


def adjacent_pairs(block: list[str], cand: str, other: str) -> list[tuple[int, int]]:
    """[(candidate position, other position)], 1-based, for one block."""
    if sorted(block) != sorted([cand, cand, other, other]) or len(block) != 4:
        raise ValueError(f"a block is two {cand!r} and two {other!r} runs, got {block}")
    taken: set[int] = set()
    pairs = []
    for i, label in enumerate(block):
        if label != cand:
            continue
        for j in (i - 1, i + 1):
            if 0 <= j < len(block) and block[j] == other and j not in taken:
                taken.add(j)
                pairs.append((i + 1, j + 1))
                break
        else:
            raise ValueError(f"no adjacent {other!r} run for position {i + 1} in {block}")
    return pairs


def run_ids(prefix_by_label: dict, blocks: list[list[str]]) -> list[str]:
    """The run ids in execution order: <prefix>-b<block>-p<position>."""
    return [
        f"{prefix_by_label[label]}-b{b}-p{p}" for b, block in enumerate(blocks, 1) for p, label in enumerate(block, 1)
    ]


def mix_futility(ratios: list[float], limit: float) -> bool:
    """Mix fast fail after block 1: every pair's split/base measured-class P99 ratio above `limit`."""
    return bool(ratios) and all(r > limit for r in ratios)
