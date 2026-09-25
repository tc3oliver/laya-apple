"""serve.py with a mocked Laya: wire format, upstream limits and status codes, auth, model
mapping, the truncation flag, loopback-only binding, and that two slow requests overlap
(no global inference lock)."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future

import pytest
from fastapi.testclient import TestClient

from laya_apple import routing, serve
from laya_apple.errors import InvalidRequestError, LayaAppleError
from laya_apple.result import Result, RuntimeInfo

QUESTIONS = {"next": {"type": "choice", "instructions": "What next?", "criteria": ["run_tests", "ask_user"]}}


class FakeLaya:
    """Answers after `delay` seconds on its own thread, like a workers-mode Laya."""

    def __init__(self, name, delay=0.0, truncated=False, ane=None):
        self.name, self.delay, self.truncated = name, delay, truncated
        self.calls, self.closed = [], False
        self.active = self.max_active = 0
        self._lock = threading.Lock()
        self.ane = ane
        self.ane_state = routing.AneState(None, (128,)) if ane else routing.AneState(routing.RUNTIME_UNAVAILABLE)

    def submit(self, context=None, questions=None, *, state=None):
        if not isinstance(questions, dict) or any(q.get("type") == "bogus" for q in questions.values()):
            raise InvalidRequestError("unknown question type 'bogus'; expected choice, score or noul")
        self.calls.append((context, questions))
        fut: Future = Future()

        def work():
            with self._lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(self.delay)
            with self._lock:
                self.active -= 1
            fut.set_result(self._result(questions))

        threading.Thread(target=work, daemon=True).start()
        return fut

    def _result(self, questions):
        runtime = RuntimeInfo(
            backend="mlx",
            device="gpu",
            model=self.name,
            model_revision="rev",
            sequence_length=42,
            question_count=len(questions),
            routing_reason="auto_default_gpu",
            artifact_revision="mlx:abc",
            latency_ms=1.5,
            execution="workers",
            request_id=7,
            truncated=self.truncated,
        )
        answers = {
            q: {"type": "choice", "choice": "run_tests", "confidence": 0.9, "action": {"act_probability": 0.8}}
            for q in questions
        }
        return Result(answers=answers, usage={"input_tokens": 42, "output_tokens": 0}, runtime=runtime)

    def close(self):
        self.closed = True


class Loader:
    def __init__(self, **kwargs):
        self.kwargs, self.loaded = kwargs, {}

    def __call__(self, name):
        self.loaded[name] = FakeLaya(name, **self.kwargs)
        return self.loaded[name]


def client(api_key=None, default_model=serve.DEFAULT_MODEL, **kwargs):
    loader = Loader(**kwargs)
    app = serve.create_app(loader=loader, api_key=api_key, default_model=default_model)
    return TestClient(app), loader


def post(c, body, **kw):
    return c.post("/v1/systemone", json=body, **kw)


def test_response_has_upstream_shape_and_laya_apple_block():
    c, loader = client()
    with c:
        r = post(c, {"model": "jev-latest", "state": {"tests": "failing"}, "questions": QUESTIONS})
    assert r.status_code == 200
    body = r.json()
    assert {"model", "answers", "usage", "routing", "laya_apple"} <= set(body)
    assert body["answers"]["next"]["choice"] == "run_tests"
    assert body["usage"] == {"input_tokens": 42, "output_tokens": 0}
    block = body["laya_apple"]
    assert block["model"] == "laya" and block["device"] == "gpu" and block["request_id"] == 7
    assert block["truncated"] is False
    assert float(r.headers["X-Inference-Time-Ms"]) >= 0
    assert r.headers["Server-Timing"].startswith("inference;dur=")
    assert loader.loaded["laya"].calls == [({"tests": "failing"}, QUESTIONS)]


def test_truncation_is_reported():
    c, _ = client(truncated=True)
    with c:
        body = post(c, {"state": "x", "questions": QUESTIONS}).json()
    assert body["laya_apple"]["truncated"] is True


@pytest.mark.parametrize(
    "requested, expected",
    [
        (None, "laya"),
        ("", "laya"),
        ("jev-latest", "laya"),
        ("jev-1", "laya"),
        ("convaiinnovations/laya", "laya"),
        ("english", "laya"),
        ("typed-decisions", "laya-typed-decisions"),
        ("multilingual", "laya-multilingual"),
        ("laya-typed-decisions", "laya-typed-decisions"),
        ("convaiinnovations/laya-typed-decisions", "laya-typed-decisions"),
        ("Typed", "laya-typed-decisions"),
        ("ml", "laya-multilingual"),
        ("default", "laya"),
        (12, "laya"),
    ],
)
def test_model_mapping(requested, expected):
    assert serve.resolve_model(requested, "laya") == expected


def test_named_model_loads_on_first_use():
    c, loader = client()
    with c:
        assert set(loader.loaded) == {"laya", "laya-multilingual"}  # auto loads both at startup
        body = post(c, {"model": "typed-decisions", "state": "s", "questions": QUESTIONS}).json()
        assert body["laya_apple"]["model"] == "laya-typed-decisions"
        assert set(loader.loaded) == {"laya", "laya-multilingual", "laya-typed-decisions"}
    assert all(m.closed for m in loader.loaded.values())


@pytest.mark.parametrize(
    "body, status",
    [
        ({"state": "s"}, 400),
        ([1, 2], 400),
        ({"state": "s", "questions": ["a"]}, 400),
        ({"state": "s", "questions": {f"q{i}": QUESTIONS["next"] for i in range(65)}}, 413),
        ({"state": "x" * 50001, "questions": QUESTIONS}, 413),
        ({"state": "s", "questions": {"q": {"type": "bogus", "instructions": "?"}}}, 422),
    ],
)
def test_limits_and_validation_match_upstream(body, status):
    c, _ = client()
    with c:
        assert post(c, body).status_code == status


def test_invalid_json_is_400_and_oversized_body_is_413():
    c, _ = client()
    with c:
        assert c.post("/v1/systemone", content=b"{not json").status_code == 400
        big = b'{"state": "' + b"x" * (serve.MAX_BODY_BYTES + 1) + b'", "questions": {}}'
        assert c.post("/v1/systemone", content=big).status_code == 413


def test_internal_error_is_500_without_details(monkeypatch, caplog):
    c, loader = client()
    with c:
        monkeypatch.setattr(loader.loaded["laya"], "submit", lambda *a, **k: 1 / 0)
        r = post(c, {"state": "/secret/path", "questions": QUESTIONS})
    assert r.status_code == 500
    assert r.json() == {"detail": "inference failed"}
    assert "ZeroDivisionError" in caplog.text


def test_auth_optional_by_default():
    c, _ = client()
    with c:
        assert post(c, {"state": "s", "questions": QUESTIONS}, headers={"Authorization": "Bearer x"}).status_code == 200
        assert post(c, {"state": "s", "questions": QUESTIONS}).status_code == 200


def test_auth_required_when_key_set(caplog):
    c, _ = client(api_key="s3cret")
    with c:
        ok = {"Authorization": "Bearer s3cret"}
        assert post(c, {"state": "s", "questions": QUESTIONS}, headers=ok).status_code == 200
        assert post(c, {"state": "s", "questions": QUESTIONS}).status_code == 401
        assert (
            post(c, {"state": "s", "questions": QUESTIONS}, headers={"Authorization": "Bearer no"}).status_code == 401
        )
        # a non-ASCII header value is a 401, not a 500 (upstream's compare_digest fix)
        bad = {"Authorization": "Bearer s\xe9cret".encode("latin-1")}
        assert post(c, {"state": "s", "questions": QUESTIONS}, headers=bad).status_code == 401
        assert c.get("/v1/models").status_code == 401
        assert c.get("/health").status_code == 200  # health stays open for supervisors
    assert "s3cret" not in caplog.text


def test_health_reports_ane_state():
    c, _ = client(ane=object(), default_model="laya")
    with c:
        h = c.get("/health").json()
        assert h["status"] == "ok" and h["loaded"] == ["laya"] and h["default_model"] == "laya"
        assert h["ane"]["laya"] == {"status": "ready", "buckets": [128]}
        assert c.get("/healthz").json() == h


@pytest.mark.parametrize(
    "state, status",
    [
        (routing.AneState(routing.ANE_STARTING), "warming"),
        (routing.AneState(routing.PLATFORM_NOT_VALIDATED), "unavailable"),
        (routing.AneState(None, ()), "unavailable"),
    ],
)
def test_ane_status(state, status):
    fake = FakeLaya("laya", ane=object())
    fake.ane_state = state
    assert serve.ane_status(fake)["status"] == status


def test_models_endpoint_lists_supported_checkpoints():
    c, _ = client()
    with c:
        data = c.get("/v1/models").json()["models"]
    ids = {m["name"]: m for m in data}
    assert set(ids) == {"auto", "laya", "laya-multilingual", "laya-typed-decisions"}
    # the Jev SDKs require these three string fields
    assert all(isinstance(m[k], str) for m in data for k in ("name", "description", "release_date"))
    assert ids["auto"]["default"] and ids["laya"]["loaded"] and not ids["laya"]["default"]
    assert not ids["laya-typed-decisions"]["loaded"]


def test_two_slow_requests_overlap():
    """No global lock: two 0.5 s requests finish in well under 1 s and run concurrently."""
    c, loader = client(delay=0.5)
    with c:
        results = []

        def one():
            results.append(post(c, {"state": "s", "questions": QUESTIONS}).status_code)

        threads = [threading.Thread(target=one) for _ in range(2)]
        t0 = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.monotonic() - t0
    assert results == [200, 200]
    assert loader.loaded["laya"].max_active == 2
    assert elapsed < 0.95


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.2"])
def test_loopback_hosts(host):
    assert serve.is_loopback(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "::", "example.com"])
def test_non_loopback_hosts_are_refused_without_flag(host, monkeypatch):
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: pytest.fail("must not start"))
    assert not serve.is_loopback(host)
    with pytest.raises(LayaAppleError, match="loopback-only"):
        serve.run(host=host)


def test_default_bind_is_loopback_and_not_8000(monkeypatch):
    seen = {}
    monkeypatch.delenv("LAYA_APPLE_SERVE_HOST", raising=False)
    monkeypatch.delenv("LAYA_APPLE_SERVE_PORT", raising=False)
    monkeypatch.setattr("uvicorn.run", lambda app, host, port, log_level: seen.update(host=host, port=port))
    assert serve.run() == 0
    assert seen == {"host": "127.0.0.1", "port": serve.DEFAULT_PORT}
    assert serve.DEFAULT_PORT != 8000


def test_invalid_port_is_a_clean_error():
    with pytest.raises(LayaAppleError, match="invalid port"):
        serve.run(port=70000)


def test_cli_parses_serve():
    from laya_apple import cli

    a = cli.build_parser().parse_args(["serve", "--model", "typed-decisions", "--port", "9000"])
    assert a.fn is cli.cmd_serve and a.port == 9000 and a.host is None and not a.allow_remote


def test_default_model_accepts_upstream_names_and_rejects_unknown():
    loader = Loader()
    with TestClient(serve.create_app(default_model="typed-decisions", preload=("english",), loader=loader)):
        assert set(loader.loaded) == {"laya-typed-decisions", "laya"}
    with pytest.raises(LayaAppleError):
        serve.create_app(default_model="jev-latest", loader=Loader())


def test_routing_block_has_upstream_keys():
    c, _ = client()
    with c:
        default = post(c, {"model": "jev-1", "state": "fix the test", "questions": QUESTIONS}).json()["routing"]
        named = post(c, {"model": "typed", "state": "s", "questions": QUESTIONS}).json()["routing"]
    assert set(default) == {"model", "repo", "reason", "detection", "workflow"}
    assert default["model"] == "english" and default["detection"]["is_english"] is True
    assert named["model"] == "typed-decisions" and named["reason"] == "explicit model='typed'"
    assert named["detection"] is None
    assert named["repo"] == "convaiinnovations/laya-typed-decisions"


def test_empty_questions_answer_nothing_like_upstream():
    c, loader = client()
    with c:
        r = post(c, {"state": "s", "questions": {}})
    assert r.status_code == 200
    assert r.json()["answers"] == {} and r.json()["usage"] == {"input_tokens": 0, "output_tokens": 0}
    assert loader.loaded["laya"].calls == []


def test_missing_state_is_serialized_as_upstream_does():
    c, loader = client()
    with c:
        assert post(c, {"questions": QUESTIONS}).status_code == 200
    assert loader.loaded["laya"].calls == [("null", QUESTIONS)]


@pytest.mark.parametrize(
    "state, expected, reason",
    [
        ("Fix the failing parser test and run the suite", "laya", "English Latin text"),
        ("修正失敗的測試，然後重新執行", "laya-multilingual", "non-Latin script (han"),
        ({"msg": "Quero cancelar minha assinatura agora mesmo"}, "laya-multilingual", "looks like 'pt'"),
        ("12345 !!!", "laya", "no letters detected"),
    ],
)
def test_auto_routes_by_language_like_upstream(state, expected, reason):
    c, _ = client()
    with c:
        body = post(c, {"model": "jev-latest", "state": state, "questions": QUESTIONS}).json()
    assert body["laya_apple"]["model"] == expected
    assert reason in body["routing"]["reason"]


def test_pinned_default_model_skips_language_routing():
    c, loader = client(default_model="typed-decisions")
    with c:
        assert set(loader.loaded) == {"laya-typed-decisions"}
        body = post(c, {"model": "jev-latest", "state": "修正測試", "questions": QUESTIONS}).json()
    assert body["laya_apple"]["model"] == "laya-typed-decisions"
    assert body["routing"]["reason"] == "server default model (--model laya-typed-decisions)"


def test_model_env_var_sets_the_default(monkeypatch):
    seen = {}
    monkeypatch.setenv("LAYA_APPLE_SERVE_MODEL", "typed-decisions")
    monkeypatch.setattr(serve, "create_app", lambda **kw: seen.update(kw))
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)
    serve.run()
    assert seen["default_model"] == "typed-decisions"
