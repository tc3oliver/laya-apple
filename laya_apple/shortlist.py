"""Opt-in embedding shortlist for high-cardinality choice questions.

Adapted from NandhaKishorM/laya v0.3.20 (laya/shortlist.py), with reference to its MLX port in
mizorewww/laya-mlx@0a85951 (laya_mlx/shortlist.py); Apache-2.0, see NOTICE.

Choice options share one `head_max_len` budget, so a large label set leaves only a few tokens
per label. `predict_shortlist` embeds the state and each option with a caller-supplied
`embed_fn`, keeps the top `k` labels of each choice question, and runs one ordinary
`Laya.predict` on the reduced question set. It is off by default: `Laya.predict` never
shortlists and still scores every criterion it is given. This module does not change the
decision model and does not add a second decision-model pass.

Ranking is cosine similarity on whatever vectors `embed_fn` returns, as upstream. Upstream
issue #102 reports a BANKING77 improvement with a top-20 shortlist; those figures are the
reporter's, and laya-apple has not measured them.

Differences from upstream, on purpose:
- The result is a `Result` whose `extra["shortlist"]` holds upstream's per-question metadata;
  `Result.to_dict()` returns it as the top-level `shortlist` key upstream's dict has.
- Errors follow laya-apple's contract: a malformed question raises `InvalidRequestError` (a
  `ValueError`, where upstream raises `TypeError` or `ValueError`); an invalid `k` or
  `embed_fn` raises `ValueError`.
- A missing state (None) is embedded as the empty string, the text `predict` reads it as.
- `embed_fn_from_laya` mean-pools the MLX encoder of an inline `Laya` (upstream's
  `embed_fn_from_agent` uses the PyTorch agent's encoder).
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Callable, Sequence

import numpy as np

from .errors import BackendUnavailableError, InvalidRequestError
from .prompt import render_options, serialize_state

DEFAULT_SHORTLIST_K = 20

EmbedFn = Callable[[Sequence[str]], Any]


def shortlist_choice(
    state: Any,
    criteria: Any,
    embed_fn: EmbedFn,
    k: int = DEFAULT_SHORTLIST_K,
    *,
    instructions: Any = None,
) -> list:
    """Return the top-`k` choice labels for `state`.

    `embed_fn` maps a list of strings to an array of shape `(len(texts), dim)`. It is called
    once, with the query text first and then one string per option in criteria order. Option
    strings are the ones the model sees for a choice question (`label` or `label: criterion`).

    When `k` is at least the number of labels, every label is returned in its original order
    and `embed_fn` is not called. Ties keep the earlier label. A zero vector scores 0 and does
    not outrank a label that came before it.
    """
    labels, _scores, _passthrough, _n = _rank(state, criteria, embed_fn, k, instructions)
    return labels


def shortlist_questions(state: Any, questions: dict, embed_fn: EmbedFn, k: int = DEFAULT_SHORTLIST_K):
    """(reduced questions, metadata): each choice question cut to its top-`k` labels.

    Non-choice questions are forwarded unchanged. A choice whose label count is `<= k` is
    forwarded unchanged and does not call `embed_fn`. The caller's dict is not mutated.
    `metadata[qid]` holds `labels` (rank order), `scores` (cosine, or None when nothing was
    dropped), `k`, `n` and `passthrough`, as upstream."""
    if not isinstance(questions, dict):
        raise InvalidRequestError("predict_shortlist needs questions as a dict of question id -> definition")
    checked = _check_k(k)
    reduced: dict = {}
    meta: dict = {}
    for qid, qdef in questions.items():
        if not isinstance(qdef, dict) or qdef.get("type") != "choice":
            reduced[qid] = qdef
            continue
        if "criteria" not in qdef:
            raise InvalidRequestError(f"question {qid!r} is a choice but has no criteria")
        labels, scores, passthrough, n = _rank(state, qdef["criteria"], embed_fn, checked, qdef.get("instructions"))
        meta[qid] = {"labels": list(labels), "scores": scores, "k": checked, "n": n, "passthrough": passthrough}
        if passthrough:
            reduced[qid] = qdef
            continue
        updated = dict(qdef)
        updated["criteria"] = _subset_criteria(qdef["criteria"], labels)
        reduced[qid] = updated
    return reduced, meta


def predict_shortlist(
    laya: Any,
    state: Any,
    questions: dict,
    embed_fn: EmbedFn,
    k: int = DEFAULT_SHORTLIST_K,
):
    """Shortlist each choice question, then call `laya.predict` once on the reduced set.

    Returns that `Result` with `extra["shortlist"]` added (see `shortlist_questions`).
    Probabilities on a shortlisted choice are over the kept labels only."""
    reduced, meta = shortlist_questions(state, questions, embed_fn, k)
    result = laya.predict(state, reduced)
    extra = dict(result.extra)
    extra["shortlist"] = meta
    return dataclasses.replace(result, extra=extra)


def embed_fn_from_laya(laya: Any, max_length: int = 512, batch_size: int = 32) -> Callable[[Sequence[str]], np.ndarray]:
    """Mean-pool the MLX encoder already loaded in an inline `Laya`.

    The callable tokenizes with the checkpoint's tokenizer (special tokens added, cut to
    `max_length`) and averages the encoder's last hidden state over the non-padding positions.
    It does not run the decision head and does not download weights. A dedicated bi-encoder
    passed as `embed_fn` will usually shortlist better; this helper is for callers who only
    have the Laya checkpoint in memory. It needs execution="inline" and an MLX backend
    (device "gpu" or "auto"): in workers mode the encoder lives in another process."""
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
        raise ValueError(f"max_length must be a positive integer, got {max_length!r}")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    backend = getattr(laya, "mlx", None)
    model = getattr(backend, "model", None)
    if getattr(laya, "execution", None) != "inline" or model is None:
        raise BackendUnavailableError(
            'embed_fn_from_laya needs an inline Laya with an MLX backend (execution="inline", device "gpu" or "auto")'
        )
    mx = backend.mx
    tok = laya.tokenizer
    encoder = model.encoder
    hidden = int(encoder.config.hidden_size)

    def embed_fn(texts: Sequence[str]) -> np.ndarray:
        rows = ["" if text is None else str(text) for text in texts]
        if not rows:
            return np.zeros((0, hidden), dtype=np.float32)
        parts = []
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            encoded = [tok.backend.encode(text, add_special_tokens=True).ids[:max_length] for text in chunk]
            length = max(1, max(len(ids) for ids in encoded))
            input_ids = np.full((len(encoded), length), tok.pad_token_id, dtype=np.int32)
            attention_mask = np.zeros((len(encoded), length), dtype=np.bool_)
            for i, ids in enumerate(encoded):
                input_ids[i, : len(ids)] = ids
                attention_mask[i, : len(ids)] = True
            with laya._lock, mx.stream(mx.gpu):  # the same model predict uses; one caller at a time
                states = encoder(mx.array(input_ids), mx.array(attention_mask))
                mask = mx.array(attention_mask.astype(np.float32))[:, :, None]
                pooled = (states.astype(mx.float32) * mask).sum(axis=1) / mx.maximum(mask.sum(axis=1), 1.0)
                mx.eval(pooled)
            parts.append(np.asarray(pooled, dtype=np.float32))
        return np.concatenate(parts, axis=0)

    return embed_fn


# ----------------------------------------------------------------------------- internals


def _rank(state, criteria, embed_fn, k, instructions):
    checked = _check_k(k)
    items = _criteria_items(criteria)
    n = len(items)
    keys = [key for key, _value in items]
    if checked >= n:
        return list(keys), None, True, n
    query = _query_text(state, instructions)
    matrix = _embeddings(embed_fn, [query] + _option_texts(items))
    sims = _cosine(matrix[0], matrix[1:])
    order = np.argsort(-sims, kind="mergesort")[:checked]  # stable: ties keep the earlier label
    return [keys[int(i)] for i in order], [float(sims[int(i)]) for i in order], False, n


def _check_k(k) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    return k


def _criteria_items(criteria) -> list:
    if isinstance(criteria, dict):
        items = list(criteria.items())
    elif isinstance(criteria, list):
        items = [(item, None) for item in criteria]
    else:
        raise InvalidRequestError(f"choice criteria must be a dict or list, got {type(criteria).__name__}")
    if not items:
        raise InvalidRequestError("choice criteria must contain at least one option")
    seen = set()
    for key, _value in items:
        if key in seen:
            raise InvalidRequestError(f"choice criteria label {key!r} is duplicated")
        seen.add(key)
    return items


def _option_texts(items) -> list[str]:
    texts = [str(piece) for piece in render_options({"t": "choice", "ins": "", "crit": dict(items)})]
    if len(texts) != len(items):
        raise InvalidRequestError("could not render every choice option")
    return texts


def _query_text(state, instructions) -> str:
    body = serialize_state("" if state is None else state)
    if instructions is None or instructions == "":
        return body
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions, ensure_ascii=False)
    return f"{instructions}\n{body}"


def _subset_criteria(criteria, labels):
    if isinstance(criteria, dict):
        return {label: criteria[label] for label in labels}
    return list(labels)


def _embeddings(embed_fn, texts: Sequence[str]) -> np.ndarray:
    if not callable(embed_fn):
        raise ValueError("embed_fn must be callable")
    raw = embed_fn(list(texts))
    if hasattr(raw, "detach"):  # a torch tensor
        raw = raw.detach().float().cpu().numpy()
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != len(texts) or arr.shape[1] < 1:
        raise ValueError(f"embed_fn must return an array of shape ({len(texts)}, dim), got {tuple(arr.shape)}")
    return np.nan_to_num(arr, copy=True, nan=0.0, posinf=0.0, neginf=0.0)


def _cosine(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
    qn = float(np.linalg.norm(query))
    if qn == 0.0 or docs.shape[0] == 0:
        return np.zeros(docs.shape[0], dtype=np.float64)
    denom = np.linalg.norm(docs, axis=1) * qn
    ok = denom > 0.0
    sims = np.zeros(docs.shape[0], dtype=np.float64)
    if np.any(ok):
        sims[ok] = np.clip(np.dot(docs[ok], query) / denom[ok], -1.0, 1.0)
    return sims
