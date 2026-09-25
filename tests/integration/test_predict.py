from __future__ import annotations

import asyncio

import pytest

from laya_apple import Laya
from laya_apple.errors import UnsupportedModelError

pytestmark = pytest.mark.integration

QUESTIONS = {
    "sentiment": {"type": "choice", "instructions": "sentiment", "criteria": ["positive", "negative"]},
    "urgency": {"type": "score", "instructions": "how urgent", "criteria": ["low", "medium", "high"]},
    "escalate": {"type": "noul", "instructions": "should this be escalated?"},
}


def test_model_loads_via_mlx_and_answers_all_question_types(cached_laya, model_name):
    laya = cached_laya(model_name, device="gpu")
    result = laya.predict(context="The customer is upset about a duplicate charge.", questions=QUESTIONS)
    assert set(result.answers) == set(QUESTIONS)
    assert result.answers["sentiment"]["type"] == "choice"
    assert result.answers["urgency"]["type"] == "score"
    assert result.answers["escalate"]["type"] == "noul"
    assert result.runtime.device == "gpu"
    assert result.runtime.backend == "mlx"


def test_predict_via_context_kwarg(laya_gpu):
    r = laya_gpu.predict(context="hello there", questions={"q": {"type": "noul", "instructions": "is it a greeting?"}})
    assert "q" in r.answers


def test_predict_via_state_kwarg(laya_gpu):
    r = laya_gpu.predict(state="hello there", questions={"q": {"type": "noul", "instructions": "is it a greeting?"}})
    assert "q" in r.answers


def test_predict_context_and_state_both_raises(laya_gpu):
    from laya_apple.errors import InvalidRequestError

    with pytest.raises(InvalidRequestError):
        laya_gpu.predict(context="a", state="b", questions={"q": {"type": "noul", "instructions": "x"}})


def test_repeated_predict_is_bitwise_identical(laya_gpu):
    kwargs = dict(context="Repeat this exact request.", questions=QUESTIONS)
    r1 = laya_gpu.predict(**kwargs)
    r2 = laya_gpu.predict(**kwargs)
    assert r1.answers == r2.answers


def test_offline_load_with_local_files_only(model_name, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    laya = Laya.from_pretrained(model_name, device="gpu", local_files_only=True)
    assert laya.spec.name == model_name


def test_from_pretrained_unregistered_model_raises():
    with pytest.raises(UnsupportedModelError):
        Laya.from_pretrained("not-a-registered-model", device="gpu", local_files_only=True)


def test_normalised_outputs_have_input_and_output_token_usage(laya_gpu):
    r = laya_gpu.predict(context="ctx", questions={"q": {"type": "noul", "instructions": "x?"}})
    assert r.usage["input_tokens"] > 0
    assert r.usage["output_tokens"] == 0


def test_empty_questions_return_empty_answers_without_a_device(laya_gpu):
    """Upstream v0.3.20: questions={} answers {} with zero usage; nothing is routed or run."""
    r = laya_gpu.predict(context="ctx", questions={})
    assert r.answers == {}
    assert r.usage == {"input_tokens": 0, "output_tokens": 0}
    assert (r.runtime.backend, r.runtime.device, r.runtime.routing_reason) == ("none", "none", "no_questions")
    assert asyncio.run(laya_gpu.apredict(context="ctx", questions={})).answers == {}
