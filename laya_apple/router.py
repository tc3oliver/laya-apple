"""Language routing between `laya` and `laya-multilingual`, as upstream's Router does.

    from laya_apple import Laya
    laya = Laya.from_pretrained("auto")          # a LayaRouter: both checkpoints loaded
    result = laya.predict(context="修正失敗的測試", questions={...})
    result.runtime.model                         # "laya-multilingual"
    result.runtime.model_routing                 # "language_non_latin_script"
    result.extra["routing"]                      # upstream's {model, repo, reason, detection, workflow}

The decision is upstream Router's language branch (laya/router.py v0.3.20, `default="english"`,
no language hints) on the state's text (laya_apple/lang.py, vendored from upstream). It is the
same function `laya-apple serve --model auto` uses, so the Python API and the server pick the
same checkpoint for the same state. Each checkpoint is an ordinary `Laya` instance: device
routing inside it is unchanged, and `RuntimeInfo.routing_reason` still names the device
decision. The checkpoint decision goes in `RuntimeInfo.model_routing` (a stable code) and
`Result.extra["routing"]` (upstream's routing block; the `reason` text is not stable).
"""

from __future__ import annotations

import dataclasses
from concurrent.futures import Future
from typing import Any

from .registry import resolve

AUTO = "auto"
# What upstream's Router falls back to when the language is undecided or there is no text.
AUTO_FALLBACK = "laya"
CHECKPOINTS = ("laya", "laya-multilingual")

# Stable `RuntimeInfo.model_routing` codes: why the language router chose the checkpoint.
LANGUAGE_ENGLISH = "language_english"
LANGUAGE_NON_LATIN_SCRIPT = "language_non_latin_script"
LANGUAGE_NOT_ENGLISH = "language_not_english"
LANGUAGE_UNDECIDED_DEFAULT = "language_undecided_default"
NO_TEXT_DEFAULT = "no_text_default"

# Upstream's checkpoint keys (laya.router v0.3.20), used in the `routing` block.
UPSTREAM_KEY = {"laya": "english", "laya-multilingual": "multilingual", "laya-typed-decisions": "typed-decisions"}

# Upstream's typed-decisions workflows (laya/router.py v0.3.20), matched on the exact set of
# question ids. Reported in `routing.workflow`; like upstream with LAYA_AUTO_TASK off, a
# match does not change the checkpoint.
TYPED_DECISION_WORKFLOWS = {
    "agent_trace_observability": {"action", "needs_review", "outcome", "risk", "urgency"},
    "customer_service": {"action", "category", "churn_risk", "needs_human", "urgency"},
    "invoice_processing": {"discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"},
    "security_incidents": {"credential_compromise", "disposition", "severity", "true_positive", "urgency"},
}


def match_workflow(questions: Any) -> str | None:
    try:
        ids = set(questions or {})
    except TypeError:  # not a question set; predict reports the error
        return None
    return next((wf for wf, sig in TYPED_DECISION_WORKFLOWS.items() if ids == sig), None)


def decide_language(state: Any) -> tuple[str, str, str, dict]:
    """(checkpoint, code, reason, detection) for `state`: upstream Router's language branch."""
    from .lang import analyse

    det = analyse(state)
    if det["script"] == "unknown":
        return AUTO_FALLBACK, NO_TEXT_DEFAULT, "no letters detected in state; using default (english)", det
    if det["script"] != "latin":
        reason = "non-Latin script (%s, %.0f%% of letters); the English checkpoint cannot read it" % (
            det["script"],
            100 * float(det["non_latin_fraction"]),
        )
        return "laya-multilingual", LANGUAGE_NON_LATIN_SCRIPT, reason, det
    if not det["is_english"]:
        if det["language"]:
            reason = "Latin script but language looks like %r, not English" % det["language"]
        else:
            reason = (
                "Latin script, language not identified but %.0f%% non-English letters; "
                "not safe for the English checkpoint" % (100 * float(det["diacritic_rate"]))
            )
        return "laya-multilingual", LANGUAGE_NOT_ENGLISH, reason, det
    if det["language_undecided"]:
        reason = "Latin script, language not identified and no non-English letters; using default (english)"
        return AUTO_FALLBACK, LANGUAGE_UNDECIDED_DEFAULT, reason, det
    return "laya", LANGUAGE_ENGLISH, "English Latin text", det


def route_by_language(state: Any) -> tuple[str, str, dict]:
    """(checkpoint, reason, detection): upstream Router's language branch, verbatim in effect
    (laya/router.py v0.3.20, `default="english"`, no lang hints)."""
    name, _code, reason, det = decide_language(state)
    return name, reason, det


def routing_block(name: str, *, reason: str, detection: dict | None = None, workflow: str | None = None) -> dict:
    """Upstream's `routing` keys: {model, repo, reason, detection, workflow}. `repo` names the
    standalone repository the pinned weights come from; stock upstream names its bundle
    repository (convaiinnovations/laya/<subfolder>) instead."""
    spec = resolve(name)
    return {
        "model": UPSTREAM_KEY[spec.name],
        "repo": spec.repo,
        "reason": reason,
        "detection": detection,
        "workflow": workflow,
    }


def _load(name: str, device: str, kwargs: dict):
    from .model import Laya

    return Laya.from_pretrained(name, device, **kwargs)


class LayaRouter:
    """`laya` and `laya-multilingual` behind one `predict`, chosen per request by language.

    Built by `Laya.from_pretrained("auto", ...)` (or `LayaRouter.from_pretrained`), which loads
    both checkpoints with the same arguments. `predict`, `submit`, `apredict`, `close`,
    `wait_for_ane` and the context manager behave as on `Laya`."""

    model = AUTO

    def __init__(self, instances: dict):
        missing = [n for n in CHECKPOINTS if n not in instances]
        if missing:
            raise ValueError(f"LayaRouter needs an instance for each of {CHECKPOINTS}; missing {missing}")
        self.instances = dict(instances)
        self._closed = False

    @classmethod
    def from_pretrained(cls, device: str = "auto", **kwargs) -> "LayaRouter":
        """Load both checkpoints with the arguments `Laya.from_pretrained` takes (except the
        model id). A failure closes whatever already loaded and re-raises."""
        loaded: dict = {}
        try:
            for name in CHECKPOINTS:
                loaded[name] = _load(name, device, kwargs)
        except BaseException:
            for laya in loaded.values():
                try:
                    laya.close()
                except Exception:
                    pass
            raise
        return cls(loaded)

    # ------------------------------------------------------------------ routing

    @staticmethod
    def route(context=None, questions=None, *, state=None) -> dict:
        """The routing decision for a request, without running it: {checkpoint, model_routing,
        routing}. The state is read as `predict` reads it (`state` wins; None is empty)."""
        text = state if state is not None else context
        name, code, reason, det = decide_language("" if text is None else text)
        block = routing_block(name, reason=reason, detection=det, workflow=match_workflow(questions))
        return {"checkpoint": name, "model_routing": code, "routing": block}

    @staticmethod
    def _annotate(result, decision: dict):
        runtime = dataclasses.replace(result.runtime, model_routing=decision["model_routing"])
        extra = dict(result.extra)
        extra["routing"] = decision["routing"]
        return dataclasses.replace(result, runtime=runtime, extra=extra)

    # ------------------------------------------------------------------ inference

    def predict(self, context=None, questions=None, *, state=None):
        decision = self.route(context, questions, state=state)
        result = self.instances[decision["checkpoint"]].predict(context, questions, state=state)
        return self._annotate(result, decision)

    def submit(self, context=None, questions=None, *, state=None) -> Future:
        decision = self.route(context, questions, state=state)
        inner = self.instances[decision["checkpoint"]].submit(context, questions, state=state)
        out: Future = Future()

        def finish(done: Future):
            try:
                result = self._annotate(done.result(), decision)
            except BaseException as e:
                if not out.cancelled():
                    out.set_exception(e)
                return
            if not out.cancelled():
                out.set_result(result)

        inner.add_done_callback(finish)
        return out

    async def apredict(self, context=None, questions=None, *, state=None):
        decision = self.route(context, questions, state=state)
        result = await self.instances[decision["checkpoint"]].apredict(context, questions, state=state)
        return self._annotate(result, decision)

    # ------------------------------------------------------------------ lifecycle

    def wait_for_ane(self, timeout: float | None = None) -> bool:
        """Wait for both checkpoints' ANE start-up; True only if both ANE paths are available."""
        return all([laya.wait_for_ane(timeout) for laya in self.instances.values()])

    def info(self) -> dict:
        return {
            "model": AUTO,
            "router": "language",
            "checkpoints": {name: laya.info() for name, laya in self.instances.items()},
        }

    def close(self):
        if self._closed:
            return
        self._closed = True
        for laya in self.instances.values():
            try:
                laya.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
