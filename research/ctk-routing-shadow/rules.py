"""Deterministic keyword rules. Tuned only on data/dev.jsonl; never fitted on eval.

classify(text) -> (label, confidence). Confidence is the score margin, in [0, 1]; the cascade
treats the rule arm as "sure" at conf >= RULE_SURE.
"""

import re

RULE_SURE = 0.5
CUES = {
    "needs-human": r"\b(production|prod|credentials?|rotate|migrat\w+|permissions?|billing data|injection|decide|cutover|secret|schema)\b",
    "lint-fix": r"\b(lint\w*|ruff|eslint|prettier|mypy|clippy|rubocop|formatter|warning|unused|whitespace|E501|F401|W291)\b",
    "docs": r"\b(readme|docs?|docstrings?|documentation|typo|contributing|document)\b",
    "refactor": r"\b(refactor\w*|extract|rename|split|duplicated|async/await|type hints?|callbacks?)\b",
    "small-bugfix": r"\b(throws?|fails?|failing|off-by-one|bug|crash\w*|null|exception|error|skipped)\b",
}
PRIORITY = ["needs-human", "lint-fix", "docs", "refactor", "small-bugfix"]  # tie-break order


def classify(text):
    t = text.lower() if text.isascii() else text
    score = {k: len(re.findall(p, t, re.I)) for k, p in CUES.items()}
    best = max(PRIORITY, key=lambda k: (score[k], -PRIORITY.index(k)))
    if score[best] == 0 or len(text.split()) < 4:
        return "needs-more-info", 0.0
    rest = sorted(score.values())[-2]
    return best, round((score[best] - rest) / score[best], 3)
