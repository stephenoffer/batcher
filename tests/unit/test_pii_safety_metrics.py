"""Unit tests for the PII/safety corpus metrics (`plan.functions.metrics.pii_safety`)."""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.plan.functions.metrics.text import pii_safety as p

pytestmark = pytest.mark.unit


def _rate(ds: bt.Dataset, expr: object) -> float:
    return ds.agg(m=expr).to_pydict()["m"][0]


def test_email_rate() -> None:
    ds = bt.from_pydict({"o": ["contact me@x.com", "no pii here", "call later"]})
    assert _rate(ds, p.email_rate("o")) == pytest.approx(1 / 3)


def test_phone_rate() -> None:
    ds = bt.from_pydict({"o": ["reach 555-123-4567", "no pii", "hello there"]})
    assert _rate(ds, p.phone_rate("o")) == pytest.approx(1 / 3)


def test_pii_rate() -> None:
    ds = bt.from_pydict({"o": ["mail me@x.com", "see https://x.io", "plain text"]})
    assert _rate(ds, p.pii_rate("o")) == pytest.approx(2 / 3)


def test_contains_any_rate() -> None:
    ds = bt.from_pydict({"o": ["this is spam", "clean reply", "buy now"]})
    assert _rate(ds, p.contains_any_rate("o", ["spam", "buy"])) == pytest.approx(2 / 3)


def test_ssn_like_rate() -> None:
    ds = bt.from_pydict({"o": ["ssn 123-45-6789", "no id", "phone 5551234"]})
    assert _rate(ds, p.ssn_like_rate("o")) == pytest.approx(1 / 3)


def test_credit_card_like_rate() -> None:
    ds = bt.from_pydict({"o": ["card 4111 1111 1111 1111", "no card", "just text"]})
    assert _rate(ds, p.credit_card_like_rate("o")) == pytest.approx(1 / 3)


def test_group_by_composition() -> None:
    ds = bt.from_pydict(
        {
            "g": ["a", "a", "b", "b"],
            "o": ["e@x.com", "none", "reach 555-123-4567", "plain"],
        }
    )
    out = ds.group_by("g").agg(m=p.email_rate("o")).sort("g").to_pydict()
    assert out["g"] == ["a", "b"]
    assert out["m"] == pytest.approx([0.5, 0.0])
