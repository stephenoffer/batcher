"""Unit tests for the RAG retrieval-grounding metrics (answer vs context overlap).

These pin the exact corpus scores each metric returns on hand-checkable input, prove the
``unsupported_token_rate == 1 - answer_groundedness`` identity on a non-empty sample, and check the
metrics compose under ``group_by``. The functions are imported directly because ``bt.<name>`` is
not wired onto the public API yet.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.plan.functions.metrics.text import retrieval as rt

pytestmark = pytest.mark.unit


def test_answer_groundedness() -> None:
    # row1 tokens {cat,sat} all in context -> 1.0; row2 {dog,barked,loudly} none -> 0.0; mean 0.5
    ds = bt.from_pydict(
        {"a": ["the cat sat", "a dog barked loudly"], "c": ["the cat sat on the mat"] * 2}
    )
    got = ds.agg(g=rt.answer_groundedness("a", "c")).to_pydict()["g"][0]
    assert got == pytest.approx(0.5)


def test_context_utilization() -> None:
    # answer {cat,sat}; context {cat,sat,on,mat} (only articles stripped) -> 2/4
    ds = bt.from_pydict({"a": ["the cat sat"], "c": ["the cat sat on the mat"]})
    got = ds.agg(u=rt.context_utilization("a", "c")).to_pydict()["u"][0]
    assert got == pytest.approx(0.5)


def test_unsupported_token_rate() -> None:
    # answer {cat,ran,fast}; context {cat,sat,on,mat}; unsupported {ran,fast}=2 / 3
    ds = bt.from_pydict({"a": ["the cat ran fast"], "c": ["the cat sat on the mat"]})
    got = ds.agg(h=rt.unsupported_token_rate("a", "c")).to_pydict()["h"][0]
    assert got == pytest.approx(2.0 / 3.0)


def test_fully_grounded_rate() -> None:
    # row1 fully grounded; row2 {cat,flew} has unsupported {flew} -> not grounded; rate 0.5
    ds = bt.from_pydict({"a": ["the cat sat", "the cat flew"], "c": ["the cat sat on the mat"] * 2})
    got = ds.agg(f=rt.fully_grounded_rate("a", "c")).to_pydict()["f"][0]
    assert got == pytest.approx(0.5)


def test_fully_grounded_rate_empty_answer_not_grounded() -> None:
    # an empty answer has no unsupported tokens but must not count as grounded
    ds = bt.from_pydict({"a": ["", "the cat sat"], "c": ["the cat sat on the mat"] * 2})
    got = ds.agg(f=rt.fully_grounded_rate("a", "c")).to_pydict()["f"][0]
    assert got == pytest.approx(0.5)


def test_citation_rate() -> None:
    # two of three rows carry a bracketed numeric citation
    ds = bt.from_pydict({"t": ["see [1]", "no citation here", "also [12] and [3]"]})
    got = ds.agg(c=rt.citation_rate("t")).to_pydict()["c"][0]
    assert got == pytest.approx(2.0 / 3.0)


def test_unsupported_is_one_minus_groundedness() -> None:
    # on non-empty answers the two are exact complements, row by row and so in the corpus mean
    ds = bt.from_pydict(
        {
            "a": ["the cat ran fast", "the dog sat", "birds fly high"],
            "c": ["the cat sat on the mat"] * 3,
        }
    )
    out = ds.agg(
        g=rt.answer_groundedness("a", "c"),
        h=rt.unsupported_token_rate("a", "c"),
    ).to_pydict()
    assert out["g"][0] + out["h"][0] == pytest.approx(1.0)


def test_group_by_composition() -> None:
    # the metrics are mergeable aggregates, so they work per group
    ds = bt.from_pydict(
        {
            "g": ["x", "x", "y"],
            "a": ["the cat sat", "a dog barked", "the cat sat"],
            "c": ["the cat sat on the mat", "the cat sat on the mat", "the cat sat on the mat"],
        }
    )
    out = ds.group_by("g").agg(m=rt.answer_groundedness("a", "c")).to_pydict()
    by_group = dict(zip(out["g"], out["m"], strict=True))
    # group x: row1 grounded 1.0, row2 {dog,barked} none -> 0.0, mean 0.5; group y: 1.0
    assert by_group["x"] == pytest.approx(0.5)
    assert by_group["y"] == pytest.approx(1.0)
