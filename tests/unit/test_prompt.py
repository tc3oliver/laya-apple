from __future__ import annotations

import pytest

from laya_apple.errors import InvalidRequestError
from laya_apple.prompt import Calibration, clamp_temperature, format_answers, prepare, render_options, to_internal


@pytest.mark.integration  # tokenizer/config need a downloaded checkpoint
def test_prepared_token_ids_match_goldens_exactly(tokenizer, config, golden):
    """prepare() must reproduce the upstream token ids/markers for every golden case."""
    for case in golden["cases"]:
        prep = prepare(tokenizer, config, case["state"], case["questions"])
        assert prep.items == case["items"], case["name"]


@pytest.mark.parametrize(
    "bad_questions",
    [
        "not a dict or list",
        {"q1": {"instructions": "missing type"}},
        {"q1": {"type": "bogus", "instructions": "x"}},
        {"q1": {"type": "choice", "instructions": "x"}},  # missing criteria
        {"q1": {"type": "choice", "instructions": "x", "criteria": {}}},
        {"q1": {"type": "choice", "instructions": "x", "criteria": ["a", "a"]}},  # dup labels
        {"q1": {"type": "score", "instructions": "x", "criteria": []}},
        {"q1": {"type": "score", "instructions": "x", "criteria": "not a list"}},
        {"q1": "not a dict"},
        ["dup", "dup"],
        [1, 2],
        # upstream v0.3.20: noul criteria keyed other than true/false, and invalid labels
        {"q1": {"type": "noul", "instructions": "x", "criteria": {"yes": "a", "no": "b"}}},
        {"q1": {"type": "noul", "instructions": "x", "criteria": {"true": "a", "maybe": "b"}}},
        {"q1": {"type": "choice", "instructions": "x", "criteria": ["a", "b"], "labels": {"false": "n", "true": "y"}}},
        {"q1": {"type": "score", "instructions": "x", "criteria": ["a", "b"], "labels": {"false": "n", "true": "y"}}},
        {"q1": {"type": "noul", "instructions": "x", "labels": {"false": "same", "true": "same"}}},
        {"q1": {"type": "noul", "instructions": "x", "labels": {"false": " ", "true": "y"}}},
        {"q1": {"type": "noul", "instructions": "x", "labels": {"true": "y"}}},
        {"q1": {"type": "noul", "instructions": "x", "labels": {"false": "n", "true": "y", "other": "z"}}},
        {"q1": {"type": "noul", "instructions": "x", "labels": {"false": 0, "true": 1}}},
        {"q1": {"type": "noul", "instructions": "x", "labels": ["n", "y"]}},
    ],
)
@pytest.mark.integration  # tokenizer/config need a downloaded checkpoint
def test_malformed_questions_raise_invalid_request_error(tokenizer, config, bad_questions):
    with pytest.raises(InvalidRequestError):
        prepare(tokenizer, config, "some context", bad_questions)


@pytest.mark.integration  # tokenizer/config need a downloaded checkpoint
def test_list_of_strings_becomes_noul_questions(tokenizer, config):
    prep = prepare(tokenizer, config, "ctx", ["Is this urgent?", "Did they mention billing?"])
    assert prep.question_count == 2
    assert all(q["t"] == "noul" for q in prep.internal)


@pytest.mark.parametrize(
    "raw,expected",
    [
        (1.0, 1.0),
        (0.1, 0.5),
        (10.0, 5.0),
        (0.5, 0.5),
        (5.0, 5.0),
        ("not a number", 1.0),
        (None, 1.0),
        (float("nan"), 1.0),
        (float("inf"), 1.0),
    ],
)
def test_calibration_temperature_clamp(raw, expected):
    assert clamp_temperature(raw) == expected


@pytest.mark.parametrize("empty", [{}, []])
def test_empty_questions_yield_no_rows_and_no_answers(empty):
    """Upstream v0.3.20: empty answers and zero usage, without tokenizing anything."""
    prep = prepare(None, {}, "any context", empty)  # no tokenizer is touched
    assert (prep.items, prep.question_count, prep.sequence_length, prep.input_tokens) == ([], 0, 0, 0)
    assert format_answers(prep, [], [], Calibration({})) == {}


def test_noul_criteria_keys_are_case_insensitive():
    q = to_internal({"type": "noul", "instructions": "x", "criteria": {"True": "yes it is", "FALSE": "no"}})
    assert q["crit"] == {"true": "yes it is", "false": "no"}
    assert render_options(q) == ["false: no", "true: yes it is"]


def test_noul_labels_replace_the_option_prefixes():
    q = to_internal({"type": "noul", "instructions": "x", "labels": {"false": " B ", "true": "A"}})
    assert render_options(q) == ["B: no, the statement does not hold", "A: yes, the statement holds"]


def test_structured_instructions_keep_non_ascii():
    q = to_internal({"type": "noul", "instructions": {"frage": "Rückerstattung?"}})
    assert q["ins"] == '{"frage": "Rückerstattung?"}'


def test_calibration_warns_and_loads_on_non_numeric_temperatures():
    with pytest.warns(RuntimeWarning, match="uncalibrated"):
        calib = Calibration({"temperature": [1.0, "bad", 9.0], "temperature_by_options": {"choice:2": "x"}})
    assert calib.temperature == [1.0, 1.0, 5.0]
    assert calib.by_options == {"choice:2": 1.0}


def test_calibration_warns_when_checkpoint_values_are_clamped():
    with pytest.warns(RuntimeWarning):
        Calibration({"temperature": [1.0, 1.0, 1.0], "temperature_by_options": {"choice:2": 10.0}})


def test_calibration_no_warning_when_within_range():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Calibration({"temperature": [1.0, 2.0, 3.0]})


@pytest.mark.integration  # tokenizer/config need a downloaded checkpoint
def test_answers_match_upstream_on_every_golden(tokenizer, config, calibration, golden):
    """format_answers on the reference logits reproduces upstream's answers exactly.

    Same fields in the same order (answer_confidence included) and the same rounded values.
    """
    import json

    for case in golden["cases"]:
        prep = prepare(tokenizer, config, case["state"], case["questions"])
        width = max(len(r) for r in case["logits"])
        logits = [r + [-1e4] * (width - len(r)) for r in case["logits"]]
        ours = format_answers(prep, logits, case["action_logits"], calibration)
        assert json.dumps(ours) == json.dumps(case["answers"]), case["name"]


@pytest.mark.integration  # tokenizer/config need a downloaded checkpoint
def test_answer_schema_choice(tokenizer, config, calibration):
    from laya_apple.prompt import format_answers

    prep = prepare(tokenizer, config, "ctx", {"q1": {"type": "choice", "instructions": "pick", "criteria": ["a", "b"]}})
    logits = [[1.0, 0.5] + [-1e4] * 30]
    act = [[0.2, 0.8]]
    answers = format_answers(prep, logits, act, calibration)
    ans = answers["q1"]
    assert list(ans) == ["type", "choice", "probabilities", "confidence", "answer_confidence", "action"]
    assert ans["answer_confidence"] == max(ans["probabilities"].values())
    assert ans["type"] == "choice"
    assert ans["choice"] in ("a", "b")
    for v in ans["probabilities"].values():
        assert round(v, 4) == v
    assert round(ans["action"]["act_probability"], 4) == ans["action"]["act_probability"]


@pytest.mark.integration  # tokenizer/config need a downloaded checkpoint
def test_answer_schema_score(tokenizer, config, calibration):
    from laya_apple.prompt import format_answers

    prep = prepare(
        tokenizer, config, "ctx", {"q1": {"type": "score", "instructions": "rate", "criteria": ["lo", "mid", "hi"]}}
    )
    logits = [[1.0, 0.5, 0.1] + [-1e4] * 29]
    act = [[0.2, 0.8]]
    ans = format_answers(prep, logits, act, calibration)["q1"]
    assert list(ans) == ["type", "score", "legend", "probabilities", "confidence", "answer_confidence", "action"]
    assert ans["answer_confidence"] == max(ans["probabilities"].values())


@pytest.mark.integration  # tokenizer/config need a downloaded checkpoint
def test_answer_schema_noul(tokenizer, config, calibration):
    from laya_apple.prompt import format_answers

    prep = prepare(tokenizer, config, "ctx", {"q1": {"type": "noul", "instructions": "is it?"}})
    logits = [[0.3, 0.7] + [-1e4] * 30]
    act = [[0.9, 0.1]]
    ans = format_answers(prep, logits, act, calibration)["q1"]
    assert list(ans) == ["type", "noul", "confidence", "answer_confidence", "action"]
    assert ans["answer_confidence"] == ans["confidence"]
    assert 0.0 <= ans["noul"] <= 1.0


@pytest.mark.integration  # tokenizer/config need a downloaded checkpoint
def test_answer_probabilities_are_rounded_to_4_decimals(tokenizer, config, calibration):
    from laya_apple.prompt import format_answers

    prep = prepare(
        tokenizer, config, "ctx", {"q1": {"type": "choice", "instructions": "pick", "criteria": ["a", "b", "c"]}}
    )
    logits = [[1.0 / 3, 0.123456789, -0.5] + [-1e4] * 29]
    act = [[0.111111, 0.888889]]
    ans = format_answers(prep, logits, act, calibration)["q1"]
    for v in ans["probabilities"].values():
        assert v == round(v, 4)
    assert ans["action"]["act_probability"] == round(ans["action"]["act_probability"], 4)
    assert ans["confidence"] == round(ans["confidence"], 4)


@pytest.mark.integration  # tokenizer/config need a downloaded checkpoint
def test_answer_non_finite_logits_raise_floating_point_error(tokenizer, config, calibration):
    from laya_apple.prompt import format_answers

    prep = prepare(tokenizer, config, "ctx", {"q1": {"type": "noul", "instructions": "is it?"}})
    logits = [[float("nan"), 0.5] + [-1e4] * 30]
    act = [[0.5, 0.5]]
    with pytest.raises(FloatingPointError):
        format_answers(prep, logits, act, calibration)


@pytest.mark.integration  # tokenizer/config need a downloaded checkpoint
def test_answer_non_finite_action_logits_raise_floating_point_error(tokenizer, config, calibration):
    from laya_apple.prompt import format_answers

    prep = prepare(tokenizer, config, "ctx", {"q1": {"type": "noul", "instructions": "is it?"}})
    logits = [[0.5, 0.5] + [-1e4] * 30]
    act = [[float("inf"), 0.5]]
    with pytest.raises(FloatingPointError):
        format_answers(prep, logits, act, calibration)


class _WordTokenizer:
    """One id per whitespace-separated word; enough to test sequence budgeting offline."""

    mask_token, cls_token_id, sep_token_id, mask_token_id = "[MASK]", 1, 2, 3

    def encode(self, text):
        return [10 + len(w) for w in text.split()]


@pytest.mark.parametrize("state", ["w " * 5, "w " * 200, {"k": "w " * 200}, ["turn " * 50] * 4, ""])
def test_truncation_report_leaves_token_ids_unchanged(state):
    from laya_apple.prompt import build_sequence

    q = {"t": "noul", "ins": "is it done", "crit": None}
    tok = _WordTokenizer()
    plain = build_sequence(tok, state, q, 64, 16)
    ids, markers, cut = build_sequence(tok, state, q, 64, 16, report_truncation=True)
    assert (ids, markers) == plain


def test_truncation_flag_boundary():
    from laya_apple.prompt import build_sequence

    q = {"t": "noul", "ins": "is it done", "crit": None}
    tok = _WordTokenizer()
    ids, _, cut = build_sequence(tok, "", q, 64, 16, report_truncation=True)
    room = 64 - len(ids)  # tokens the state may use: everything but the final [SEP]
    assert not build_sequence(tok, "w " * room, q, 64, 16, report_truncation=True)[2]  # exact fit
    assert build_sequence(tok, "w " * (room + 1), q, 64, 16, report_truncation=True)[2]
    assert not cut
