"""Unit tests for the whitespace-hygiene (output-cleanliness) corpus metrics."""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.plan.functions.metrics.text import quality as c

pytestmark = pytest.mark.unit


def _rate(metric, data: list[str]) -> float:
    ds = bt.from_pydict({"o": data})
    return ds.agg(m=metric("o")).to_pydict()["m"][0]


def test_trailing_whitespace_rate() -> None:
    # "space " (space), "tab\t" (tab), "nl\n" (newline all count); "clean", "end." do not.
    data = ["clean", "space ", "tab\t", "nl\n", "end."]
    assert _rate(c.trailing_whitespace_rate, data) == pytest.approx(3 / 5)


def test_leading_whitespace_rate() -> None:
    data = ["clean", " space", "\ttab", "\nnewline", "end"]
    assert _rate(c.leading_whitespace_rate, data) == pytest.approx(3 / 5)


def test_has_tab_rate() -> None:
    data = ["no tab", "a\tb", "plain", "\tlead"]
    assert _rate(c.has_tab_rate, data) == pytest.approx(2 / 4)


def test_double_space_rate() -> None:
    data = ["a  b", "a b", "a   b", "single"]
    assert _rate(c.double_space_rate, data) == pytest.approx(2 / 4)


def test_blank_line_rate() -> None:
    # Two newlines (optionally with spaces/tabs between) count; a single newline does not.
    data = ["a\n\nb", "a\nb", "a\n \nb", "one line"]
    assert _rate(c.blank_line_rate, data) == pytest.approx(2 / 4)


def test_empty_or_whitespace_rate() -> None:
    data = ["real", "", "   ", "\t\n", "x"]
    assert _rate(c.empty_or_whitespace_rate, data) == pytest.approx(3 / 5)


def test_group_by() -> None:
    ds = bt.from_pydict(
        {
            "grp": ["a", "a", "b", "b"],
            "o": ["clean", "trail ", "ok", "bad "],
        }
    )
    out = ds.group_by("grp").agg(m=c.trailing_whitespace_rate("o")).sort("grp").to_pydict()
    assert out["grp"] == ["a", "b"]
    assert out["m"] == pytest.approx([0.5, 0.5])
