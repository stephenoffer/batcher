"""Text-quality and safety monitors — corpus rates over a generated-text column.

Each monitor is a single mergeable aggregate over a string primitive, so the contract to pin is the
exact corpus rate on a small hand-countable batch, plus that the rates compose inside ``group_by``.
"""

from __future__ import annotations

import pytest

import batcher as bt

pytestmark = pytest.mark.unit


def _rate(expr, rows: list[str]) -> float:
    return bt.from_pydict({"o": rows}).agg(m=expr).to_pydict()["m"][0]


def test_all_caps_rate_counts_letter_only_uppercase() -> None:
    rows = ["STOP DOING THAT", "a normal reply", "OK"]
    assert _rate(bt.all_caps_rate("o"), rows) == pytest.approx(2 / 3)


def test_repeated_punctuation_rate() -> None:
    rows = ["really???", "calm.", "wow!!!"]
    assert _rate(bt.repeated_punctuation_rate("o"), rows) == pytest.approx(2 / 3)


def test_non_ascii_rate() -> None:
    assert _rate(bt.non_ascii_rate("o"), ["café", "plain", "naive"]) == pytest.approx(1 / 3)


def test_url_rate() -> None:
    rows = ["see https://example.com", "no link here"]
    assert _rate(bt.url_rate("o"), rows) == pytest.approx(0.5)


def test_code_block_rate() -> None:
    rows = ["use ```print(1)```", "just prose"]
    assert _rate(bt.code_block_rate("o"), rows) == pytest.approx(0.5)


def test_long_output_rate_is_exclusive() -> None:
    # len 5, len 25 ; threshold 10 -> only the second exceeds.
    rows = ["short", "a much longer answer here"]
    assert _rate(bt.long_output_rate("o", min_chars=10), rows) == pytest.approx(0.5)


def test_short_output_rate_is_exclusive() -> None:
    rows = ["ok", "a much longer answer here"]
    assert _rate(bt.short_output_rate("o", max_chars=10), rows) == pytest.approx(0.5)


def test_mean_sentence_count() -> None:
    rows = ["One. Two. Three.", "Just one."]
    assert _rate(bt.mean_sentence_count("o"), rows) == pytest.approx(2.0)


def test_mean_word_length() -> None:
    # "aa bb" -> mean 2.0 ; "cccc dddd" -> mean 4.0 ; corpus mean 3.0.
    assert _rate(bt.mean_word_length("o"), ["aa bb", "cccc dddd"]) == pytest.approx(3.0)


def test_monitors_compose_with_group_by() -> None:
    ds = bt.from_pydict({"model": ["a", "a", "b"], "o": ["SHOUT", "quiet", "ALSO LOUD"]})
    out = ds.group_by("model").agg(caps=bt.all_caps_rate("o")).sort("model").to_pydict()
    assert out["model"] == ["a", "b"]
    assert out["caps"] == pytest.approx([0.5, 1.0])
