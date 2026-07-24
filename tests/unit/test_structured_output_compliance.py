"""Structured-output compliance metrics — did the model return the requested shape.

Each is a corpus rate over a string primitive, so they are pinned to the exact count on a small
batch: valid JSON (strict, whole output), JSON present (lenient, extractable from prose), and a
non-empty named tag. The strict/lenient gap is exercised directly since that distinction is the
point of having both.
"""

from __future__ import annotations

import pytest

import batcher as bt

pytestmark = pytest.mark.unit


def _rate(expr, rows: list[str]) -> float:
    return bt.from_pydict({"o": rows}).agg(m=expr).to_pydict()["m"][0]


def test_valid_json_rate_is_strict_whole_output() -> None:
    # Object and array parse; prose-wrapped and plain text do not.
    rows = ['{"a": 1}', "[1, 2, 3]", "here: {\"a\": 1}", "not json"]
    assert _rate(bt.valid_json_rate("o"), rows) == pytest.approx(0.5)


def test_json_present_rate_is_lenient_extraction() -> None:
    # An object is recoverable from the first two; an array has no object; prose has none.
    rows = ['{"a": 1}', "here: {\"a\": 1}", "[1, 2, 3]", "not json"]
    assert _rate(bt.json_present_rate("o"), rows) == pytest.approx(0.5)


def test_strict_is_at_most_lenient_for_objects() -> None:
    # A prose-wrapped object fails strict but passes lenient, so lenient >= strict here.
    rows = ['{"a": 1}', "sure, here: {\"b\": 2}"]
    strict = _rate(bt.valid_json_rate("o"), rows)
    lenient = _rate(bt.json_present_rate("o"), rows)
    assert strict == pytest.approx(0.5)
    assert lenient == pytest.approx(1.0)


def test_tagged_answer_rate_needs_non_empty_tag() -> None:
    rows = ["<answer>Paris</answer>", "no tag here", "<answer></answer>"]
    assert _rate(bt.tagged_answer_rate("o", "answer"), rows) == pytest.approx(1 / 3)


def test_tagged_answer_rate_escapes_the_tag_name() -> None:
    rows = ["<a.b>x</a.b>", "<axb>y</axb>"]
    # The '.' must be literal, so only the first row matches the "a.b" tag.
    assert _rate(bt.tagged_answer_rate("o", "a.b"), rows) == pytest.approx(0.5)


def test_numeric_answer_rate_counts_parseable_numbers() -> None:
    rows = ["the answer is 42", "about -3.5", "no number", "nope"]
    assert _rate(bt.numeric_answer_rate("o"), rows) == pytest.approx(0.5)


def test_choice_answer_rate_counts_standalone_letters() -> None:
    rows = ["The answer is B.", "(C) is right", "no letter", "yes"]
    assert _rate(bt.choice_answer_rate("o"), rows) == pytest.approx(0.5)


def test_boxed_answer_rate_counts_non_empty_boxes() -> None:
    rows = [r"so \boxed{42}", "no box here", r"\boxed{7} done"]
    assert _rate(bt.boxed_answer_rate("o"), rows) == pytest.approx(2 / 3)


def test_boxed_answer_rate_ignores_empty_box() -> None:
    rows = [r"\boxed{}", r"\boxed{5}"]
    assert _rate(bt.boxed_answer_rate("o"), rows) == pytest.approx(0.5)


def test_compliance_metrics_compose_with_group_by() -> None:
    ds = bt.from_pydict(
        {"model": ["a", "a", "b"], "o": ['{"x": 1}', "bad", '{"y": 2}']}
    )
    out = ds.group_by("model").agg(v=bt.valid_json_rate("o")).sort("model").to_pydict()
    assert out["model"] == ["a", "b"]
    assert out["v"] == pytest.approx([0.5, 1.0])
