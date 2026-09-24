"""Deterministic exact-length requests for benchmarks, parity fixtures and tests.

Same generator as research/phase-0-feasibility/scripts/common.py (generated support-ticket
text, no third-party prompts): the longest prompt row is exactly `length` tokens, built with
the runtime's own prompt builder. It never pads or truncates to fake a length.
"""

from __future__ import annotations

import random

from .prompt import QTYPES, Tokenizer, build_sequence, to_internal

QUESTION_POOL = [
    {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {
            "billing": "charges, invoices, refunds",
            "technical": "errors, outages, broken features",
            "account": "login, profile, permissions",
            "other": "anything else",
        },
    },
    {
        "type": "score",
        "instructions": "How urgent is this ticket?",
        "criteria": ["can wait", "should be handled today", "blocking, handle now"],
    },
    {"type": "noul", "instructions": "Is the customer threatening to cancel the service?"},
    {
        "type": "choice",
        "instructions": "What is the overall sentiment of the message?",
        "criteria": ["positive", "neutral", "negative"],
    },
    {"type": "noul", "instructions": "Does the message mention a specific invoice number?"},
    {
        "type": "score",
        "instructions": "How clearly does the customer describe the problem?",
        "criteria": ["unclear", "partly clear", "clear", "very clear with evidence"],
    },
    {"type": "noul", "instructions": "Should this ticket be escalated to a human supervisor?"},
]

_SUBJECTS = ["invoice", "subscription", "account", "order", "payment", "license", "renewal"]
_VERBS = ["was charged", "was billed", "got an error", "cannot log in", "was refused", "saw a delay"]
_DETAILS = [
    "twice this month",
    "after the latest update",
    "on the annual plan",
    "for the second time",
    "despite the earlier ticket",
    "without any notice",
]
_ASKS = [
    "Please refund the duplicate amount.",
    "Can someone look into this today?",
    "I would like an explanation.",
    "Otherwise we will cancel the plan.",
    "Please confirm once it is fixed.",
]


def filler_sentences(seed: int):
    rng = random.Random(seed)
    while True:
        yield (
            f"The {rng.choice(_SUBJECTS)} {rng.randint(1000, 9999)} {rng.choice(_VERBS)} "
            f"{rng.choice(_DETAILS)}. {rng.choice(_ASKS)}"
        )


def questions_for(count: int, offset: int = 0) -> dict:
    return {f"q{i}": QUESTION_POOL[(offset + i) % len(QUESTION_POOL)] for i in range(count)}


def make_request(tok: Tokenizer, cfg: dict, length: int, n_questions: int = 1, seed: int = 0):
    """(state, questions) whose longest prompt row is exactly `length` tokens, or raise."""
    questions = questions_for(n_questions, offset=seed)
    internal = [to_internal(q) for q in questions.values()]
    mlen, hlen = int(cfg["max_len"]), int(cfg.get("head_max_len", 192))
    if length > mlen:
        raise ValueError(f"length {length} exceeds checkpoint max_len {mlen}")

    def longest(words):
        state = " ".join(words)
        return state, max(len(build_sequence(tok, state, q, mlen, hlen)[0]) for q in internal)

    words: list[str] = []
    stream, sentence = filler_sentences(seed), []
    for _ in range(20000):
        state, n = longest(words)
        if n == length:
            return state, questions
        if n > length:
            break
        if not sentence:
            sentence = next(stream).split()
        words.append(sentence.pop(0))
    words = words[:-1]  # overshot by a multi-token word: back off, grow with 1-token words
    for _ in range(64):
        state, n = longest(words)
        if n == length:
            return state, questions
        if n > length:
            break
        words.append("a")
    raise RuntimeError(f"could not build an exact {length}-token request")


BURST_PERIOD_S = 3.0  # bursty arrivals: every period starts with BURST_ON_S at 3x the rate,
BURST_ON_S = 1.0  # then nothing for the rest of the period (the same mean offered load)


def arrivals(rate: float, seconds: float, seed: int, bursty: bool) -> list[float]:
    """Open-loop arrival offsets (s) in [0, seconds): Poisson at `rate`, or bursty (1 s on at
    3x the rate, 2 s off: the same mean offered load). Bit-identical to
    scripts/bench_concurrency.arrivals, so both drive the same timetable for a seed."""
    rng = random.Random(seed)
    t, out = 0.0, []
    while t < seconds:
        r = rate
        if bursty:
            r = rate * 3 if (t % BURST_PERIOD_S) < BURST_ON_S else 1e-9
        gap = rng.expovariate(r)
        if bursty and (t % BURST_PERIOD_S) >= BURST_ON_S:
            t = (t // BURST_PERIOD_S + 1) * BURST_PERIOD_S
            continue
        t += gap
        out.append(t)
    return [x for x in out if x < seconds]


__all__ = ["BURST_ON_S", "BURST_PERIOD_S", "QUESTION_POOL", "QTYPES", "arrivals", "make_request", "questions_for"]
