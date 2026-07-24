"""Unit tests for the corpus-level readability metrics (hand-checked values)."""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.plan.functions.metrics import readability as r

pytestmark = pytest.mark.unit


def test_automated_readability_index() -> None:
    # chars=36, words=9, sents=2: 4.71*(36/9) + 0.5*(9/2) - 21.43 = 18.84 + 2.25 - 21.43.
    ds = bt.from_pydict({"o": ["The cat sat on the mat. It was warm."]})
    got = ds.agg(m=r.automated_readability_index("o")).to_pydict()["m"][0]
    assert got == pytest.approx(-0.34)


def test_mean_words_per_sentence() -> None:
    # words=9, sentences=2 -> 9/2 = 4.5.
    ds = bt.from_pydict({"o": ["The cat sat on the mat. It was warm."]})
    got = ds.agg(m=r.mean_words_per_sentence("o")).to_pydict()["m"][0]
    assert got == pytest.approx(4.5)


def test_mean_chars_per_word() -> None:
    # 26 letters over 9 words -> 26/9 = 2.888...
    ds = bt.from_pydict({"o": ["The cat sat on the mat. It was warm."]})
    got = ds.agg(m=r.mean_chars_per_word("o")).to_pydict()["m"][0]
    assert got == pytest.approx(26 / 9)


def test_long_word_rate() -> None:
    # Row1: 3/3 long (>=7 chars) = 1.0; Row2: 0/3 = 0.0; mean = 0.5.
    ds = bt.from_pydict({"o": ["extraordinary complicated situations", "the cat sat"]})
    got = ds.agg(m=r.long_word_rate("o")).to_pydict()["m"][0]
    assert got == pytest.approx(0.5)


def test_long_word_rate_min_length() -> None:
    # min_length=4: "the"/"cat"/"sat" all len 3, none long -> 0.0.
    ds = bt.from_pydict({"o": ["the cat sat"]})
    got = ds.agg(m=r.long_word_rate("o", min_length=4)).to_pydict()["m"][0]
    assert got == pytest.approx(0.0)


def test_mean_paragraph_count() -> None:
    # Three blank-line-separated blocks -> 3.
    ds = bt.from_pydict({"o": ["First para.\n\nSecond para.\n\nThird."]})
    got = ds.agg(m=r.mean_paragraph_count("o")).to_pydict()["m"][0]
    assert got == pytest.approx(3.0)


def test_empty_string_is_guarded() -> None:
    # No words, no sentences: guarded terms are 0, so ARI is just the -21.43 constant.
    ds = bt.from_pydict({"o": [""]})
    got = ds.agg(m=r.automated_readability_index("o")).to_pydict()["m"][0]
    assert got == pytest.approx(-21.43)
    got_wps = ds.agg(m=r.mean_words_per_sentence("o")).to_pydict()["m"][0]
    assert got_wps == pytest.approx(0.0)


def test_mean_words_per_sentence_compose_with_group_by() -> None:
    # g=a: (4/2=2.0, 2/2=1.0) -> mean 1.5; g=b: (4/2=2.0) -> mean 2.0.
    ds = bt.from_pydict(
        {
            "g": ["a", "a", "b"],
            "o": ["One two. Three four.", "A. B.", "X y z. Q."],
        }
    )
    out = ds.group_by("g").agg(m=r.mean_words_per_sentence("o")).sort("g").to_pydict()
    assert out["g"] == ["a", "b"]
    assert out["m"][0] == pytest.approx(1.5)
    assert out["m"][1] == pytest.approx(2.0)
