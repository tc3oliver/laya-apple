"""Language routing in the Python API (laya_apple/router.py) with fake checkpoints: the same
decision as `serve --model auto`, recorded in RuntimeInfo.model_routing and
Result.extra["routing"], and `Laya.from_pretrained("auto")` building a LayaRouter."""

from __future__ import annotations

import asyncio
import dataclasses
from concurrent.futures import Future

import numpy as np
import pytest

from laya_apple import Laya, LayaRouter, router, serve
from laya_apple.result import Result, RuntimeInfo

QUESTIONS = {"next": {"type": "choice", "instructions": "What next?", "criteria": ["run_tests", "ask_user"]}}


class FakeLaya:
    def __init__(self, name):
        self.name, self.calls, self.closed = name, [], False

    def _result(self, questions):
        runtime = RuntimeInfo(
            backend="mlx",
            device="gpu",
            model=self.name,
            model_revision="rev",
            sequence_length=12,
            question_count=len(questions),
            routing_reason="gpu_requested",
            artifact_revision="mlx:abc",
            latency_ms=1.0,
        )
        return Result(answers={q: {"type": "choice"} for q in questions}, usage={}, runtime=runtime)

    def predict(self, context=None, questions=None, *, state=None):
        self.calls.append((context, questions, state))
        return self._result(questions)

    def predict_shortlist(self, context=None, questions=None, *, state=None, embed_fn, k=20):
        from laya_apple.shortlist import predict_shortlist

        return predict_shortlist(self, state if state is not None else context, questions, embed_fn, k)

    def submit(self, context=None, questions=None, *, state=None):
        fut: Future = Future()
        fut.set_result(self.predict(context, questions, state=state))
        return fut

    async def apredict(self, context=None, questions=None, *, state=None):
        return self.predict(context, questions, state=state)

    def wait_for_ane(self, timeout=None):
        return True

    def info(self):
        return {"model": self.name}

    def close(self):
        self.closed = True


def make_router():
    return LayaRouter({name: FakeLaya(name) for name in router.CHECKPOINTS})


CASES = [
    ("Fix the failing parser test and run the suite", "laya", router.LANGUAGE_ENGLISH),
    ("修正失敗的測試，然後重新執行", "laya-multilingual", router.LANGUAGE_NON_LATIN_SCRIPT),
    ({"msg": "Quero cancelar minha assinatura agora mesmo"}, "laya-multilingual", router.LANGUAGE_NOT_ENGLISH),
    ("12345 !!!", "laya", router.NO_TEXT_DEFAULT),
    (None, "laya", router.NO_TEXT_DEFAULT),
]


@pytest.mark.parametrize("state, expected, code", CASES)
def test_predict_routes_by_language_and_records_the_decision(state, expected, code):
    r = make_router()
    result = r.predict(state, QUESTIONS)
    assert result.runtime.model == expected
    assert result.runtime.model_routing == code
    assert result.runtime.routing_reason == "gpu_requested"  # the device decision is untouched
    block = result.extra["routing"]
    assert block["model"] == router.UPSTREAM_KEY[expected]
    assert set(block) == {"model", "repo", "reason", "detection", "workflow"}
    assert r.instances[expected].calls == [(state, QUESTIONS, None)]
    other = next(n for n in router.CHECKPOINTS if n != expected)
    assert r.instances[other].calls == []


@pytest.mark.parametrize("state, expected, code", CASES[:4])
def test_python_api_and_serve_choose_the_same_checkpoint(state, expected, code):
    name, reason, detection = serve.route_by_language(state)
    decision = LayaRouter.route(state, QUESTIONS)
    assert name == decision["checkpoint"] == expected
    assert decision["model_routing"] == code
    assert reason == decision["routing"]["reason"]
    assert detection == decision["routing"]["detection"]


def test_state_keyword_is_routed_on():
    r = make_router()
    result = r.predict(questions=QUESTIONS, state="修正失敗的測試")
    assert result.runtime.model == "laya-multilingual"
    assert r.instances["laya-multilingual"].calls == [(None, QUESTIONS, "修正失敗的測試")]


def test_workflow_is_reported_without_changing_the_checkpoint():
    questions = {q: {"type": "noul", "instructions": q} for q in router.TYPED_DECISION_WORKFLOWS["customer_service"]}
    result = make_router().predict("Refund my order please, it arrived broken", questions)
    assert result.extra["routing"]["workflow"] == "customer_service"
    assert result.runtime.model == "laya"


def test_to_dict_carries_upstream_routing_key():
    d = make_router().predict("Fix the failing parser test and run the suite", QUESTIONS).to_dict()
    assert d["routing"]["model"] == "english"
    assert d["runtime"]["model_routing"] == router.LANGUAGE_ENGLISH


def test_submit_and_apredict_are_annotated():
    r = make_router()
    assert r.submit("修正測試", QUESTIONS).result().runtime.model_routing == router.LANGUAGE_NON_LATIN_SCRIPT
    result = asyncio.run(r.apredict("Fix the failing parser test and run the suite", QUESTIONS))
    assert result.runtime.model_routing == router.LANGUAGE_ENGLISH


def test_submit_propagates_the_checkpoint_error():
    r = make_router()
    failed: Future = Future()
    failed.set_exception(ValueError("bad question"))
    r.instances["laya"].submit = lambda *a, **k: failed
    with pytest.raises(ValueError, match="bad question"):
        r.submit("Fix the failing parser test and run the suite", QUESTIONS).result()


def test_close_closes_every_checkpoint_once():
    r = make_router()
    with r:
        pass
    assert all(laya.closed for laya in r.instances.values())
    r.close()  # idempotent


def test_info_lists_both_checkpoints():
    info = make_router().info()
    assert info["model"] == "auto" and set(info["checkpoints"]) == set(router.CHECKPOINTS)


def test_router_needs_both_checkpoints():
    with pytest.raises(ValueError, match="missing"):
        LayaRouter({"laya": FakeLaya("laya")})


def test_from_pretrained_auto_builds_a_router_with_the_same_arguments(monkeypatch):
    calls = []

    def load(name, device, kwargs):
        calls.append((name, device, kwargs))
        return FakeLaya(name)

    monkeypatch.setattr(router, "_load", load)
    r = Laya.from_pretrained("auto", "gpu", execution="workers", local_files_only=True)
    assert isinstance(r, LayaRouter)
    assert [c[0] for c in calls] == list(router.CHECKPOINTS)
    assert all(c[1] == "gpu" and c[2]["execution"] == "workers" and c[2]["local_files_only"] for c in calls)


def test_a_failed_load_closes_what_already_loaded(monkeypatch):
    loaded = []

    def load(name, device, kwargs):
        if name == "laya-multilingual":
            raise RuntimeError("download failed")
        loaded.append(FakeLaya(name))
        return loaded[-1]

    monkeypatch.setattr(router, "_load", load)
    with pytest.raises(RuntimeError, match="download failed"):
        LayaRouter.from_pretrained()
    assert loaded and all(laya.closed for laya in loaded)


def test_plain_result_is_unchanged():
    result = FakeLaya("laya").predict("s", QUESTIONS)
    assert result.runtime.model_routing is None and "routing" not in result.to_dict()
    assert dataclasses.asdict(result.runtime)["model_routing"] is None


def test_predict_shortlist_runs_on_the_routed_checkpoint():
    labels = [f"label_{i}" for i in range(5)]
    questions = {"intent": {"type": "choice", "instructions": "", "criteria": labels}}
    state = "修正失敗的測試，然後重新執行"
    vectors = {state: [1.0, 0.0], "label_3": [1.0, 0.0], "label_1": [0.7, 0.7]}

    def embed(texts):
        return np.array([vectors.get(t, [0.0, 1.0]) for t in texts])

    r = make_router()
    result = r.predict_shortlist(state, questions, embed_fn=embed, k=2)
    assert result.runtime.model == "laya-multilingual"
    assert result.runtime.model_routing == router.LANGUAGE_NON_LATIN_SCRIPT
    assert result.extra["shortlist"]["intent"]["labels"] == ["label_3", "label_1"]
    assert result.extra["routing"]["model"] == "multilingual"
    ((context, sent, _state),) = r.instances["laya-multilingual"].calls
    assert sent["intent"]["criteria"] == ["label_3", "label_1"]
    assert r.instances["laya"].calls == []
    assert set(result.to_dict()) >= {"shortlist", "routing"}
