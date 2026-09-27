"""The opt-in embedding shortlist (laya_apple/shortlist.py), with a table-driven embed_fn and a
fake Laya: ranking, pass-through, the reduced request, the metadata, and argument checks."""

from __future__ import annotations

import numpy as np
import pytest

import laya_apple
from laya_apple import Laya
from laya_apple.errors import BackendUnavailableError, InvalidRequestError
from laya_apple.result import Result, RuntimeInfo
from laya_apple.shortlist import predict_shortlist, shortlist_choice, shortlist_questions


class TableEmbed:
    """Maps each text to a fixed vector; unknown texts get a zero vector. Records every call."""

    def __init__(self, table, dim=2):
        self.table, self.dim, self.calls = table, dim, []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return np.array([self.table.get(t, [0.0] * self.dim) for t in texts], dtype=np.float32)


class FakeLaya:
    def __init__(self):
        self.calls = []

    def predict(self, context=None, questions=None, *, state=None):
        self.calls.append((context, questions))
        runtime = RuntimeInfo(
            backend="mlx",
            device="gpu",
            model="laya",
            model_revision="rev",
            sequence_length=10,
            question_count=len(questions),
            routing_reason="gpu_requested",
            artifact_revision="mlx:abc",
            latency_ms=1.0,
        )
        return Result(answers={q: {} for q in questions}, usage={}, runtime=runtime)


STATE = "I was charged twice for a transfer"
EMBED = {
    STATE: [1.0, 0.0],
    "card_arrival: where is my card": [0.0, 1.0],
    "transfer_fee: fee charged on a transfer": [0.9, 0.1],
    "duplicate_charge: charged twice": [1.0, 0.05],
    "refund: money back": [0.5, 0.5],
}
CRITERIA = {
    "card_arrival": "where is my card",
    "transfer_fee": "fee charged on a transfer",
    "duplicate_charge": "charged twice",
    "refund": "money back",
}


def test_top_k_by_cosine_in_rank_order():
    embed = TableEmbed(EMBED)
    assert shortlist_choice(STATE, CRITERIA, embed, k=2) == ["duplicate_charge", "transfer_fee"]
    assert embed.calls == [[STATE, *(f"{k}: {v}" for k, v in CRITERIA.items())]]


def test_k_covering_every_label_passes_through_without_embedding():
    embed = TableEmbed(EMBED)
    assert shortlist_choice(STATE, CRITERIA, embed, k=4) == list(CRITERIA)
    assert shortlist_choice(STATE, CRITERIA, embed, k=50) == list(CRITERIA)
    assert embed.calls == []


def test_ties_and_zero_vectors_keep_the_earlier_label():
    same = TableEmbed({"s": [1.0, 0.0], "a": [1.0, 0.0], "b": [2.0, 0.0], "c": [3.0, 0.0]})
    assert shortlist_choice("s", ["a", "b", "c"], same, k=2) == ["a", "b"]
    zero_query = TableEmbed({"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 1.0]})  # "s" is unknown: zero
    assert shortlist_choice("s", ["a", "b", "c"], zero_query, k=2) == ["a", "b"]


def test_nan_scores_zero_and_sorts_behind_a_match():
    embed = TableEmbed({"s": [1.0, 0.0], "a": [float("nan"), 1.0], "b": [1.0, 0.0]})
    assert shortlist_choice("s", ["a", "b"], embed, k=1) == ["b"]


def test_instructions_and_dict_state_form_the_query():
    embed = TableEmbed({}, dim=3)
    shortlist_choice({"text": "héllo"}, ["a", "b"], embed, k=1, instructions="Which intent?")
    assert embed.calls[0][0] == 'Which intent?\n{"text": "héllo"}'
    shortlist_choice(None, ["a", "b"], embed, k=1)
    assert embed.calls[1][0] == ""  # a missing state is the empty text predict reads


def test_falsy_criterion_values_stay_in_the_option_text():
    embed = TableEmbed({}, dim=2)
    shortlist_choice("s", {"a": 0, "b": False, "c": None, "d": ""}, embed, k=1)
    assert embed.calls[0][1:] == ["a: 0", "b: false", "c", "d"]


def test_predict_shortlist_reduces_only_large_choices():
    laya, embed = FakeLaya(), TableEmbed(EMBED)
    small = {"type": "choice", "instructions": "yes or no", "criteria": ["yes", "no"]}
    noul = {"type": "noul", "instructions": "urgent?"}
    questions = {
        "intent": {"type": "choice", "instructions": "", "criteria": CRITERIA},
        "small": small,
        "urgent": noul,
    }
    original = {k: dict(v) for k, v in questions.items()}
    result = predict_shortlist(laya, STATE, questions, embed, k=2)
    assert questions == original  # the caller's dict is not mutated
    ((context, sent),) = laya.calls
    assert context == STATE
    assert sent["intent"]["criteria"] == {
        "duplicate_charge": "charged twice",
        "transfer_fee": "fee charged on a transfer",
    }
    assert sent["small"] is small and sent["urgent"] is noul
    meta = result.extra["shortlist"]
    assert set(meta) == {"intent", "small"}  # non-choice questions have no entry
    assert meta["intent"]["labels"] == ["duplicate_charge", "transfer_fee"]
    assert meta["intent"]["k"] == 2 and meta["intent"]["n"] == 4 and not meta["intent"]["passthrough"]
    assert len(meta["intent"]["scores"]) == 2 and meta["intent"]["scores"][0] >= meta["intent"]["scores"][1]
    assert meta["small"] == {"labels": ["yes", "no"], "scores": None, "k": 2, "n": 2, "passthrough": True}
    assert result.to_dict()["shortlist"] == meta


def test_list_criteria_stay_a_list_in_rank_order():
    laya = FakeLaya()
    q = {"intent": {"type": "choice", "instructions": "", "criteria": list(CRITERIA)}}
    embed = TableEmbed({STATE: [1.0, 0.0], "refund": [1.0, 0.0], "card_arrival": [0.5, 0.5]})
    predict_shortlist(laya, STATE, q, embed, k=2)
    assert laya.calls[0][1]["intent"]["criteria"] == ["refund", "card_arrival"]


def test_laya_method_accepts_state_keyword(monkeypatch):
    laya = Laya.__new__(Laya)  # no model: predict is replaced below
    fake = FakeLaya()
    monkeypatch.setattr(laya, "predict", fake.predict, raising=False)
    q = {"intent": {"type": "choice", "instructions": "", "criteria": CRITERIA}}
    result = laya.predict_shortlist(questions=q, state=STATE, embed_fn=TableEmbed(EMBED), k=1)
    assert fake.calls[0][0] == STATE
    assert result.extra["shortlist"]["intent"]["labels"] == ["duplicate_charge"]
    with pytest.raises(InvalidRequestError):
        laya.predict_shortlist("s", q, state="s", embed_fn=TableEmbed(EMBED))


@pytest.mark.parametrize("k", [0, -1, 1.5, True, "3", None])
def test_bad_k_is_rejected(k):
    with pytest.raises(ValueError, match="k must be"):
        shortlist_choice("s", ["a", "b"], TableEmbed({}), k=k)


@pytest.mark.parametrize(
    "questions, message",
    [
        (["a", "b"], "dict of question id"),
        ({"q": {"type": "choice", "instructions": ""}}, "no criteria"),
        ({"q": {"type": "choice", "instructions": "", "criteria": "abc"}}, "dict or list"),
        ({"q": {"type": "choice", "instructions": "", "criteria": []}}, "at least one"),
        ({"q": {"type": "choice", "instructions": "", "criteria": ["a", "a"]}}, "duplicated"),
    ],
)
def test_malformed_questions_raise_invalid_request(questions, message):
    laya = FakeLaya()
    with pytest.raises(InvalidRequestError, match=message):
        shortlist_questions("s", questions, TableEmbed({}), k=1)
    assert laya.calls == []


@pytest.mark.parametrize(
    "embed_fn",
    [
        "not callable",
        lambda texts: np.zeros((len(texts),)),
        lambda texts: np.zeros((len(texts) - 1, 4)),
        lambda texts: np.zeros((len(texts), 0)),
    ],
)
def test_bad_embed_fn_is_rejected_before_predict(embed_fn):
    laya = FakeLaya()
    q = {"q": {"type": "choice", "instructions": "", "criteria": ["a", "b", "c"]}}
    with pytest.raises(ValueError):
        predict_shortlist(laya, "s", q, embed_fn, k=1)
    assert laya.calls == []


def test_exports():
    assert laya_apple.shortlist_choice is shortlist_choice
    assert callable(laya_apple.embed_fn_from_laya)


def test_embed_fn_from_laya_needs_inline_mlx():
    laya = Laya.__new__(Laya)
    laya.execution, laya.mlx = "workers", None
    with pytest.raises(BackendUnavailableError, match="inline"):
        laya_apple.embed_fn_from_laya(laya)
    with pytest.raises(ValueError, match="max_length"):
        laya_apple.embed_fn_from_laya(laya, max_length=0)
