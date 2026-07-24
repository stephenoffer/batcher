"""Unit tests for the length-distribution metric functions."""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.plan.functions.metrics import length_dist as ld

pytestmark = pytest.mark.unit


def test_char_length_quantile() -> None:
    ds = bt.from_pydict({"o": ["ab", "abcd", "abcdef"]})  # lengths 2, 4, 6
    m = ds.agg(m=ld.char_length_quantile("o", q=0.5)).to_pydict()["m"][0]
    assert m == pytest.approx(4, abs=1)


def test_word_count_quantile() -> None:
    ds = bt.from_pydict({"o": ["a b", "a b c", "a b c d e"]})  # counts 2, 3, 5
    m = ds.agg(m=ld.word_count_quantile("o", q=0.5)).to_pydict()["m"][0]
    assert m == pytest.approx(3, abs=1)


def test_max_char_length() -> None:
    ds = bt.from_pydict({"o": ["ab", "abcd", "abcdef"]})
    assert ds.agg(m=ld.max_char_length("o")).to_pydict()["m"][0] == 6


def test_min_char_length() -> None:
    ds = bt.from_pydict({"o": ["ab", "abcd", "abcdef"]})
    assert ds.agg(m=ld.min_char_length("o")).to_pydict()["m"][0] == 2


def test_char_length_range() -> None:
    ds = bt.from_pydict({"o": ["ab", "abcd", "abcdef"]})
    assert ds.agg(m=ld.char_length_range("o")).to_pydict()["m"][0] == 4


def test_max_char_length_group_by() -> None:
    ds = bt.from_pydict(
        {"g": ["a", "a", "b"], "o": ["ab", "abcd", "abcdef"]}
    )
    out = ds.group_by("g").agg(m=ld.max_char_length("o")).sort("g").to_pydict()
    assert out == {"g": ["a", "b"], "m": [4, 6]}
