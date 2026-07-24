"""Unit tests for the repetition/degeneration metric module."""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.plan.functions.metrics.text import diversity as rep

pytestmark = pytest.mark.unit


def test_distinct_char_ngram_ratio() -> None:
    # "aaaa" bigrams -> ["aa","aa","aa"], unique 1 / total 3.
    ds = bt.from_pydict({"o": ["aaaa"]})
    val = ds.agg(m=rep.distinct_char_ngram_ratio("o", n=2)).to_pydict()["m"][0]
    assert val == pytest.approx(1 / 3)


def test_char_repetition_rate() -> None:
    ds = bt.from_pydict({"o": ["aaaa"]})
    val = ds.agg(m=rep.char_repetition_rate("o", n=2)).to_pydict()["m"][0]
    assert val == pytest.approx(2 / 3)


def test_char_repetition_complements_distinct() -> None:
    # On a sample with no empty rows, repetition == 1 - distinct per row, so the means agree.
    ds = bt.from_pydict({"o": ["cat sat mat", "cat cat cat cat", "hello world"]})
    out = ds.agg(
        d=rep.distinct_char_ngram_ratio("o", n=3),
        r=rep.char_repetition_rate("o", n=3),
    ).to_pydict()
    assert out["r"][0] == pytest.approx(1 - out["d"][0])


def test_repeated_line_rate() -> None:
    # First output repeats a line; second does not -> 1 of 2.
    ds = bt.from_pydict({"o": ["a\na", "x\ny\nz"]})
    val = ds.agg(m=rep.repeated_line_rate("o")).to_pydict()["m"][0]
    assert val == pytest.approx(0.5)


def test_compression_ratio_proxy() -> None:
    # "aaaa" bigrams -> total 3 / unique 1 = 3.0.
    ds = bt.from_pydict({"o": ["aaaa"]})
    val = ds.agg(m=rep.compression_ratio_proxy("o", n=2)).to_pydict()["m"][0]
    assert val == pytest.approx(3.0)


def test_group_by() -> None:
    ds = bt.from_pydict({"k": ["a", "a", "b"], "o": ["aaaa", "bbbb", "abcd"]})
    out = ds.group_by("k").agg(m=rep.distinct_char_ngram_ratio("o", n=2)).to_pydict()
    by_key = dict(zip(out["k"], out["m"], strict=True))
    assert by_key["a"] == pytest.approx(1 / 3)  # both "aaaa","bbbb" -> 1/3 each
    assert by_key["b"] == pytest.approx(1.0)  # "abcd" -> ["ab","bc","cd"] all distinct
