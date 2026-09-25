"""Upstream-compatible Laya prompts, question schema, calibration and answer format.

Adapted from NandhaKishorM/laya@573e5b6 (laya/common.py, laya/agent.py) via
mizorewww/laya-mlx@0a85951 (laya_mlx/common.py, laya_mlx/tokenizer.py), then brought to
NandhaKishorM/laya v0.3.20 (@23a1752); Apache-2.0, see NOTICE. Behaviour is unchanged:
Phase -1 verified that these rules reproduce upstream token ids exactly on every fixture
(research/phase-0-feasibility/parity.md), and the shipped goldens also cover the v0.3.20
changes (left-truncated conversation states, noul labels and key normalisation, non-ASCII
structured instructions, answer_confidence).
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import InvalidRequestError

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}
TEMP_MIN, TEMP_MAX = 0.5, 5.0
_DEFAULT_NOUL_LABELS = {"false": "false", "true": "true"}


# ----------------------------------------------------------------------------- tokenizer


class Tokenizer:
    """The checkpoint's Rust tokenizer, without Transformers or torch."""

    def __init__(self, path):
        from tokenizers import Tokenizer as Backend

        path = Path(path)
        self.backend = Backend.from_file(str(path / "tokenizer.json"))
        self.backend.no_padding()
        self.backend.no_truncation()
        config = json.loads((path / "tokenizer_config.json").read_text())
        for name in ("cls_token", "sep_token", "pad_token", "mask_token"):
            value = config.get(name)
            if isinstance(value, dict):
                value = value.get("content")
            token_id = self.backend.token_to_id(value) if isinstance(value, str) else None
            if token_id is None:
                raise ValueError(f"Tokenizer is missing a valid {name}")
            setattr(self, name, value)
            setattr(self, name + "_id", token_id)

    def encode(self, text: str) -> list[int]:
        return self.backend.encode(text, add_special_tokens=False).ids


# ----------------------------------------------------------------------------- schema


def serialize_state(state) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_criterion(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def normalize_questions(questions) -> dict:
    """Accept upstream's dict schema, or a list of plain strings (each becomes a `noul`).

    An empty dict (or list) is valid and yields no answers, as in upstream v0.3.20.
    """
    if isinstance(questions, (list, tuple)):
        if not all(isinstance(q, str) for q in questions):
            raise InvalidRequestError("a question list must contain strings; use a dict for typed questions")
        if len(set(questions)) != len(questions):
            raise InvalidRequestError("duplicate questions in list")
        questions = {q: {"type": "noul", "instructions": q} for q in questions}
    if not isinstance(questions, dict):
        raise InvalidRequestError("questions must be a dict (or list of strings)")
    return questions


def resolve_noul_labels(labels=None) -> tuple[str, str]:
    """The (false, true) option labels of a noul question, validated as upstream v0.3.20 does."""
    if labels is None:
        labels = _DEFAULT_NOUL_LABELS
    message = "noul labels must map exactly 'false' and 'true' to distinct non-empty strings"
    if not isinstance(labels, dict) or set(labels) != {"false", "true"}:
        raise InvalidRequestError(message)
    false_label, true_label = labels["false"], labels["true"]
    if not isinstance(false_label, str) or not isinstance(true_label, str):
        raise InvalidRequestError(message)
    false_label, true_label = false_label.strip(), true_label.strip()
    if not false_label or not true_label or false_label == true_label:
        raise InvalidRequestError(message)
    return false_label, true_label


def to_internal(qdef) -> dict:
    """Validate one upstream question definition; returns {'t','ins','crit'} (+ 'labels')."""
    if not isinstance(qdef, dict):
        raise InvalidRequestError("each question must be a dictionary")
    kind = qdef.get("type")
    if kind not in QTYPES:
        raise InvalidRequestError(f"unknown question type {kind!r}; expected choice, score or noul")
    if "instructions" not in qdef:
        raise InvalidRequestError("question is missing instructions")
    crit = qdef.get("criteria")
    if kind == "choice":
        if isinstance(crit, list):
            if not all(isinstance(c, str) for c in crit):
                raise InvalidRequestError("choice labels must be strings")
            if len(set(crit)) != len(crit):
                raise InvalidRequestError("choice labels must be unique")
            crit = dict.fromkeys(crit)
        if not isinstance(crit, dict) or not crit:
            raise InvalidRequestError("choice criteria must be a non-empty dict or list")
        if not all(isinstance(k, str) for k in crit):
            raise InvalidRequestError("choice labels must be strings")
    elif kind == "score":
        if not isinstance(crit, list) or not crit:
            raise InvalidRequestError("score criteria must be a non-empty list")
    elif crit is not None and not isinstance(crit, dict):
        raise InvalidRequestError("noul criteria must be a dict with false/true descriptions")
    elif isinstance(crit, dict):
        # Keys are case-insensitive (a Python True reads as "true"). Any other key used to be
        # replaced by the default texts without a word; upstream rejects it since v0.3.20.
        keys = {str(k).lower() for k in crit}
        if not keys <= {"true", "false"}:
            raise InvalidRequestError(
                f"noul criteria must be keyed only 'true'/'false' (either or both), got {sorted(keys)}; "
                "set 'labels' to word the options differently"
            )
        crit = {str(k).lower(): v for k, v in crit.items()}
    if "labels" in qdef:
        if kind != "noul":
            raise InvalidRequestError("'labels' is only supported for noul questions")
        resolve_noul_labels(qdef["labels"])
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins, ensure_ascii=False)
    q = {"t": kind, "ins": ins, "crit": crit}
    if "labels" in qdef:
        q["labels"] = qdef["labels"]
    return q


def render_options(q: dict) -> list[str]:
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        return [k if v is None or v == "" else "%s: %s" % (k, render_criterion(v)) for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, render_criterion(c)) for i, c in enumerate(crit)]
    crit = crit or {}
    false_label, true_label = resolve_noul_labels(q.get("labels"))
    f, t_ = crit.get("false"), crit.get("true")
    return [
        false_label + ": " + (render_criterion(f) if f not in (None, "") else "no, the statement does not hold"),
        true_label + ": " + (render_criterion(t_) if t_ not in (None, "") else "yes, the statement holds"),
    ]


def encode_state(tok: Tokenizer, state) -> list[int]:
    return tok.encode(serialize_state(state).replace(tok.mask_token, " "))


def build_sequence(
    tok: Tokenizer,
    state,
    q: dict,
    max_len: int,
    head_max_len: int,
    *,
    truncate_left: bool = False,
    state_ids: list[int] | None = None,
):
    """[CLS] <type> question: ins [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP].

    `truncate_left` keeps the end of an overflowing state (upstream does this for conversation
    lists, so the newest turn survives); `state_ids` reuses a state tokenized once per request.
    Option texts keep their first 48 tokens: upstream v0.3.20 caps them in the tokenizer
    (truncation=True, max_length=48), which yields the same ids as this slice.
    """
    mask_tok = tok.mask_token
    opts = render_options(q)
    head_ids = tok.encode("%s question: %s" % (q["t"], str(q["ins"]).replace(mask_tok, " ")))
    opt_ids = [[tok.mask_token_id] + tok.encode(" " + o.replace(mask_tok, " "))[:48] for o in opts]
    budget = head_max_len - sum(len(o) for o in opt_ids)
    if budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    if state_ids is None:
        state_ids = encode_state(tok, state)
    # not state_ids[-room:]: with no room left that is the whole state rather than none of it
    st = state_ids[max(0, len(state_ids) - room) :] if truncate_left else state_ids[:room]
    ids = ids + st + [tok.sep_token_id]
    return ids[:max_len], [m for m in markers if m < max_len]


@dataclass
class PreparedRequest:
    question_ids: list
    internal: list  # validated internal question dicts
    items: list  # [{"ids", "markers", "qtype"}]

    @property
    def sequence_length(self) -> int:
        return max((len(it["ids"]) for it in self.items), default=0)

    @property
    def question_count(self) -> int:
        return len(self.items)

    @property
    def input_tokens(self) -> int:
        return sum(len(it["ids"]) for it in self.items)


def prepare(tok: Tokenizer, cfg: dict, state, questions) -> PreparedRequest:
    questions = normalize_questions(questions)
    max_len, head = int(cfg.get("max_len", 512)), int(cfg.get("head_max_len", 192))
    if not questions:  # upstream v0.3.20: empty answers and zero usage, nothing tokenized
        return PreparedRequest([], [], [])
    # Validate every question, then tokenize the shared state once. A conversation list is
    # newest-last, so it is truncated from the left, keeping the newest turns (upstream v0.3.20).
    internal = [to_internal(qdef) for qdef in questions.values()]
    state_ids = encode_state(tok, state)
    items = []
    for qid, q in zip(questions, internal):
        ids, markers = build_sequence(
            tok, state, q, max_len, head, truncate_left=isinstance(state, list), state_ids=state_ids
        )
        if len(markers) != len(render_options(q)):
            raise InvalidRequestError(f"question {qid!r} has too many options for the token budget")
        items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
    return PreparedRequest(list(questions), internal, items)


# ----------------------------------------------------------------------------- calibration


def clamp_temperature(t) -> float:
    try:
        t = float(t)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(t):
        return 1.0
    return min(TEMP_MAX, max(TEMP_MIN, t))


def temp_bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


class Calibration:
    """Checkpoint temperatures, clamped to [0.5, 5.0] exactly as upstream v0.3.20 does.

    An invalid entry (not a number, NaN or infinite) falls back to 1.0 with a warning; it never
    stops the checkpoint from loading.
    """

    def __init__(self, cfg: dict):
        self.raw = cfg.get("temperature", [1.0, 1.0, 1.0])
        self.by_options_raw = cfg.get("temperature_by_options", {})
        self.temperature = [clamp_temperature(t) for t in self.raw]
        self.by_options = {k: clamp_temperature(v) for k, v in self.by_options_raw.items()}
        entries = [(k, v, self.by_options[k]) for k, v in self.by_options_raw.items()]
        entries += [("temperature[%d]" % i, t, self.temperature[i]) for i, t in enumerate(self.raw)]
        rejected = []
        for name, raw, applied in entries:
            try:
                if float(raw) == applied:
                    continue
            except (TypeError, ValueError):
                pass
            rejected.append("%s=%r -> %g" % (name, raw, applied))
        if rejected:
            warnings.warn(
                "laya-apple: checkpoint temperatures that are invalid or outside [0.5, 5.0] are replaced (%s), "
                "as upstream does; treat confidence from those entries as uncalibrated" % ", ".join(rejected),
                RuntimeWarning,
                stacklevel=3,
            )

    def scale(self, qtype: int, k: int) -> float:
        return self.by_options.get(temp_bucket(qtype, k), self.temperature[qtype])

    def probabilities(self, logits, qtype: int, k: int) -> np.ndarray:
        z = np.asarray(logits[:k], np.float64) / self.scale(qtype, k)
        p = np.exp(z - z.max())
        return p / p.sum()


def confidence_from_probs(p: np.ndarray, k: int) -> float:
    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


def answer_confidence(p: np.ndarray, k: int) -> float:
    """Calibrated probability of the reported answer, max(p) (upstream v0.3.20)."""
    if k < 1:
        return 1.0
    return float(np.clip(np.max(p[:k]), 0.0, 1.0))


def format_answers(prep: PreparedRequest, logits, act_logits, calib: Calibration) -> dict:
    """Upstream v0.3.20 answer schema (field order, 4-decimal rounding) from raw logits.

    `confidence` is the normalised-entropy score on choice/score and max(p) on noul;
    `answer_confidence` is max(p) on every type, the quantity temperature scaling fits.
    """
    if not prep.items:
        return {}
    logits = np.asarray(logits, np.float32)
    act = np.asarray(act_logits, np.float64)
    if not np.isfinite(logits).all() or not np.isfinite(act).all():
        raise FloatingPointError("non-finite model outputs")
    act = np.exp(act - act.max(-1, keepdims=True))
    act /= act.sum(-1, keepdims=True)
    answers = {}
    for row, (qid, q, it) in enumerate(zip(prep.question_ids, prep.internal, prep.items)):
        k, qt = len(it["markers"]), it["qtype"]
        p = calib.probabilities(logits[row], qt, k)
        if q["t"] == "choice":
            labels = list(q["crit"])
            ans = {
                "type": "choice",
                "choice": labels[int(p.argmax())],
                "probabilities": {label: round(float(v), 4) for label, v in zip(labels, p)},
                "confidence": round(confidence_from_probs(p, k), 4),
            }
        elif q["t"] == "score":
            ans = {
                "type": "score",
                "score": round(float((np.arange(k) * p).sum()), 4),
                "legend": {str(i): c for i, c in enumerate(q["crit"])},
                "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                "confidence": round(confidence_from_probs(p, k), 4),
            }
        else:
            ans = {
                "type": "noul",
                "noul": round(float(p[1]), 4),
                "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4),
            }
        ans["answer_confidence"] = round(answer_confidence(p, k), 4)
        ans["action"] = {"act_probability": round(float(act[row, 0]), 4)}
        answers[qid] = ans
    return answers
