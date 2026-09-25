"""`laya-apple serve` on a real checkpoint: the HTTP answers are the runtime's answers, and
a request over the model's max length is flagged as truncated."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from laya_apple import Laya, serve

pytestmark = pytest.mark.integration
MODEL = "laya-typed-decisions"

QUESTIONS = {
    "next_step": {
        "type": "choice",
        "instructions": "What should the agent do next?",
        "criteria": {"run_tests": "run the test suite", "ask_user": "ask the user", "commit": "commit the change"},
    },
    "risk": {"type": "score", "instructions": "How risky is this change?", "criteria": ["none", "low", "high"]},
    "done": {"type": "noul", "instructions": "The task is complete."},
}
STATE = {"task": "fix the failing parser test", "last_tool": "edit parser.py", "tests": "not run yet"}


@pytest.fixture(scope="module")
def http():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    app = serve.create_app(default_model=MODEL, loader=serve.default_loader(offline=True))
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def direct():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    laya = Laya.from_pretrained(MODEL, device="gpu", local_files_only=True)
    yield laya
    laya.close()


def test_http_answers_match_the_runtime(http, direct):
    r = http.post("/v1/systemone", json={"model": "jev-latest", "state": STATE, "questions": QUESTIONS})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["laya_apple"]["model"] == MODEL
    assert body["laya_apple"]["truncated"] is False
    expected = direct.predict(state=STATE, questions=QUESTIONS)
    assert body["usage"] == expected.usage
    assert set(body["answers"]) == set(expected.answers)
    for qid, want in expected.answers.items():
        got = body["answers"][qid]
        assert got.keys() == want.keys()
        assert got.get("choice") == want.get("choice")
        for field in ("score", "noul", "confidence"):
            if field in want:
                assert got[field] == pytest.approx(want[field], abs=0.02)
        assert got["action"]["act_probability"] == pytest.approx(want["action"]["act_probability"], abs=0.02)


def test_long_state_is_flagged_truncated(http):
    state = "log line with some words. " * 1500  # well past 1024 tokens, under 50,000 chars
    body = http.post("/v1/systemone", json={"state": state, "questions": {"done": QUESTIONS["done"]}}).json()
    assert body["laya_apple"]["truncated"] is True
    assert body["laya_apple"]["sequence_length"] == 1024


def test_health_reports_the_loaded_model(http):
    h = http.get("/health").json()
    assert h["loaded"] == [MODEL]
    assert h["ane"][MODEL]["status"] in ("warming", "ready", "unavailable")
