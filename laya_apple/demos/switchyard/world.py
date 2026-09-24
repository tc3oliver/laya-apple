"""The switchyard-v1 workload: a pure, seeded timetable of trains and background requests.

Nothing here touches a model, a clock or the filesystem. For a seed and a duration the
schedule (arrival offsets, request classes, train contents and their oracle answers) is
fully determined, and `schedule_sha256` fingerprints it so two rounds can prove they ran
the same timetable.

Random streams, one per concern, so changing one never shifts another:
  arrivals        Random(seed)       laya_apple.workload.arrivals (bursty, 40 req/s nominal)
  class picks     Random(seed + 1)   train / medium_1q / long_1q / short_4q by MIX weight
  train contents  Random(seed + 2)   line, train id, platform states, occupying train ids
The warmup uses its own schedule from seed + WARMUP_SEED_OFFSET.

Each train asks one `choice` question (ROUTE A / B / C). Exactly one platform is clear and
the other two are each occupied or closed: 3 x 2 x 2 = 12 core patterns, and the clear
platform is the oracle answer.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import asdict, dataclass

from ...workload import BURST_ON_S, BURST_PERIOD_S, arrivals  # noqa: F401 (re-exported for result.json)

WORKLOAD_ID = "switchyard-v1"
WORKLOAD_VERSION = 1
MODEL = "laya-typed-decisions"
DEFAULT_SEED = 11
DURATION_S = 60.0
WARMUP_S = 5.0
WARMUP_SEED_OFFSET = 100
RATE_REQ_S = 40.0
BURSTY = True
ARRIVALS = "bursty"  # 1 s on at 3x the rate, 2 s off (scripts/bench_concurrency.arrivals)
MIN_DURATION_S = BURST_PERIOD_S  # a round shorter than one burst cycle is not the workload
DEADLINE_MS = 100.0
THRESHOLDS_MS = (25, 50, 100, 250, 500)
DRAIN_TIMEOUT_S = 600.0

TRAIN = "train"
MIX = ((TRAIN, 0.6), ("medium_1q", 0.2), ("long_1q", 0.1), ("short_4q", 0.1))
# Background classes: (exact length, questions, workload.make_request seed), as v1.0 part B.
BACKGROUND = {"medium_1q": (512, 1, 2), "long_1q": (1024, 1, 3), "short_4q": (128, 4, 4)}
TRAIN_MAX_TOKENS = 128
LENGTHS = {TRAIN: f"<={TRAIN_MAX_TOKENS}", **{k: v[0] for k, v in BACKGROUND.items()}}

LINES = 6
PLATFORMS = ("A", "B", "C")
CLEAR, OCCUPIED, CLOSED = "clear", "occupied", "closed"

# The frozen wording (docs/switchyard.md records every wording tried and its oracle accuracy).
HEADER = "Line {line} inbound. Train {train} is approaching the junction."
SENTENCES = {
    OCCUPIED: "Platform {platform} is occupied by train {other}.",
    CLOSED: "Platform {platform} is closed for maintenance.",
    CLEAR: "Platform {platform} is clear and open.",
}
QUESTION_ID = "platform"
QUESTION = "Train {train} must be routed to the clear platform. Which platform is clear?"
WORDING = {"header": HEADER, "sentences": SENTENCES, "question": QUESTION, "criteria": list(PLATFORMS)}


def _patterns() -> tuple:
    out = []
    for clear in PLATFORMS:
        others = [p for p in PLATFORMS if p != clear]
        for states in itertools.product((OCCUPIED, CLOSED), repeat=2):
            platforms = dict(zip(others, states)) | {clear: CLEAR}
            out.append(tuple(platforms[p] for p in PLATFORMS))
    return tuple(out)


PATTERNS = _patterns()  # 12 tuples of platform states in PLATFORMS order


@dataclass(frozen=True)
class Train:
    line: int
    train_id: int
    pattern: int  # index into PATTERNS
    occupants: tuple  # occupying train id per platform (None unless occupied), PLATFORMS order

    @property
    def platforms(self) -> dict:
        return dict(zip(PLATFORMS, PATTERNS[self.pattern]))

    @property
    def oracle(self) -> str:
        return next(p for p, s in self.platforms.items() if s == CLEAR)


@dataclass(frozen=True)
class Arrival:
    index: int
    offset_s: float  # scheduled arrival, seconds after the round start
    cls: str  # TRAIN or a BACKGROUND class
    train: Train | None = None


def make_train(rng: random.Random, pattern: int) -> Train:
    line = rng.randint(1, LINES)
    train_id = rng.randint(1000, 9999)
    occupants = []
    for state in PATTERNS[pattern]:
        other = None
        if state == OCCUPIED:
            other = rng.randint(1000, 9999)
            while other == train_id or other in occupants:
                other = rng.randint(1000, 9999)
        occupants.append(other)
    return Train(line, train_id, pattern, tuple(occupants))


def schedule(seed: int = DEFAULT_SEED, duration_s: float = DURATION_S, rate: float = RATE_REQ_S) -> list[Arrival]:
    """The timetable for one round: every arrival in [0, duration_s), in order."""
    times = arrivals(rate, duration_s, seed=seed, bursty=BURSTY)
    names, weights = [c for c, _ in MIX], [w for _, w in MIX]
    picks_rng, train_rng = random.Random(seed + 1), random.Random(seed + 2)
    out = []
    for i, at in enumerate(times):
        cls = picks_rng.choices(names, weights)[0]
        train = make_train(train_rng, train_rng.randrange(len(PATTERNS))) if cls == TRAIN else None
        out.append(Arrival(i, at, cls, train))
    return out


def warmup_schedule(seed: int = DEFAULT_SEED) -> list[Arrival]:
    return schedule(seed + WARMUP_SEED_OFFSET, WARMUP_S)


def train_prompt(train: Train, wording: dict | None = None) -> tuple[str, dict]:
    """(context, questions) for one train, in the frozen wording (or another `wording` with
    the same keys as WORDING: scripts/switchyard_oracle.py replays the wordings tried)."""
    w = wording or WORDING
    parts = [w["header"].format(line=train.line, train=train.train_id)]
    for platform, state, other in zip(PLATFORMS, PATTERNS[train.pattern], train.occupants):
        parts.append(w["sentences"][state].format(platform=platform, other=other))
    question = {
        "type": "choice",
        "instructions": w["question"].format(train=train.train_id),
        "criteria": w["criteria"],
    }
    return " ".join(parts), {QUESTION_ID: question}


def oracle_cases(variants: int = 20, seed: int = 0) -> list[Train]:
    """Every core pattern x `variants` surface variants (line, train ids), for the oracle gate."""
    rng = random.Random(seed)
    return [make_train(rng, p) for p in range(len(PATTERNS)) for _ in range(variants)]


def _sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def schedule_sha256(items: list[Arrival]) -> str:
    """Fingerprint of a timetable: offsets (exact float repr), classes and train contents."""
    return _sha256([[a.offset_s, a.cls, asdict(a.train) if a.train else None] for a in items])


def prompt_template_sha256(wording: dict | None = None) -> str:
    w = wording or WORDING
    return _sha256(
        {
            "header": w["header"],
            "sentences": w["sentences"],
            "question_id": QUESTION_ID,
            "question": w["question"],
            "criteria": w["criteria"],
        }
    )


def whole_burst_cycles(duration_s: float) -> bool:
    """At least one burst cycle, and a whole number of them: only then does the realised
    offered rate equal the nominal mean (a partial cycle is all burst or all pause)."""
    cycles = duration_s / BURST_PERIOD_S
    return duration_s >= MIN_DURATION_S and abs(cycles - round(cycles)) < 1e-9


def offered(items: list[Arrival], duration_s: float) -> dict:
    trains = sum(a.cls == TRAIN for a in items)
    return {"requests": len(items), "trains": trains, "req_s": len(items) / duration_s}
