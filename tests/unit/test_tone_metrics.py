"""Unit tests for the tone/style corpus metrics (`plan.functions.metrics.tone`)."""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.plan.functions.metrics.text import tone

pytestmark = pytest.mark.unit


def _rate(ds: bt.Dataset, expr: object) -> float:
    return ds.agg(m=expr).to_pydict()["m"][0]


def test_question_rate() -> None:
    ds = bt.from_pydict({"o": ["What do you mean?", "The answer is 4.", "Maybe it works?"]})
    assert _rate(ds, tone.question_rate("o")) == pytest.approx(2 / 3)


def test_exclamation_rate() -> None:
    ds = bt.from_pydict({"o": ["Wow, amazing!", "It is fine.", "Great job!"]})
    assert _rate(ds, tone.exclamation_rate("o")) == pytest.approx(2 / 3)


def test_hedge_rate() -> None:
    ds = bt.from_pydict({"o": ["Maybe it works?", "The answer is 4.", "I think so."]})
    assert _rate(ds, tone.hedge_rate("o")) == pytest.approx(2 / 3)


def test_first_person_rate() -> None:
    ds = bt.from_pydict({"o": ["I can help", "The sky is blue", "We are ready", "my cat"]})
    assert _rate(ds, tone.first_person_rate("o")) == pytest.approx(0.75)


def test_politeness_rate() -> None:
    ds = bt.from_pydict({"o": ["Please help me", "Thank you so much", "Just do it.", "No."]})
    assert _rate(ds, tone.politeness_rate("o")) == pytest.approx(0.5)


def test_contains_phrase_rate() -> None:
    ds = bt.from_pydict({"o": ["i cannot assist", "here you go", "i cannot do that"]})
    assert _rate(ds, tone.contains_phrase_rate("o", ["i cannot"])) == pytest.approx(2 / 3)


def test_contains_phrase_rate_is_case_sensitive() -> None:
    ds = bt.from_pydict({"o": ["I cannot", "i cannot", "sure"]})
    assert _rate(ds, tone.contains_phrase_rate("o", ["i cannot"])) == pytest.approx(1 / 3)


def test_group_by() -> None:
    ds = bt.from_pydict({"g": ["a", "a", "b", "b"], "o": ["Why?", "ok.", "Sure!", "How?"]})
    out = ds.group_by("g").agg(m=tone.question_rate("o")).sort("g").to_pydict()
    assert out["g"] == ["a", "b"]
    assert out["m"] == pytest.approx([0.5, 0.5])
