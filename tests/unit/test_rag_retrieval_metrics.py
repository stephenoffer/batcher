"""Phrase-level grounding and the retrieval-side monitors.

The test that carries the most weight here is the one separating token grounding from phrase
grounding: an answer assembled from the context's own vocabulary into a claim the context never
made is what a confident hallucination looks like, and token overlap scores it perfectly.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.unit


def _score(metric, **columns):
    return bt.from_pydict(columns).agg(m=metric).to_pydict()["m"][0]


# --- phrase grounding --------------------------------------------------------------


def test_a_verbatim_span_is_fully_grounded():
    got = _score(
        bt.phrase_groundedness("a", "c"),
        a=["the cat sat quietly"],
        c=["the cat sat quietly on the mat"],
    )
    assert got == 1.0


def test_a_rearranged_answer_passes_token_grounding_and_fails_phrase_grounding():
    """The whole reason the phrase metric exists."""
    columns = {
        "a": ["mat sat cat quietly the on"],
        "c": ["the cat sat quietly on the mat"],
    }
    assert _score(bt.answer_groundedness("a", "c"), **columns) == 1.0
    assert _score(bt.phrase_groundedness("a", "c"), **columns) < 0.5


def test_an_ungrounded_answer_scores_zero():
    got = _score(
        bt.phrase_groundedness("a", "c"),
        a=["the moon is made of cheese"],
        c=["the moon orbits the earth"],
    )
    assert got == 0.0


def test_the_unsupported_rate_is_the_complement_of_groundedness():
    columns = {"a": ["the cat sat on a warm mat"], "c": ["the cat sat on the mat"]}
    grounded = _score(bt.phrase_groundedness("a", "c"), **columns)
    unsupported = _score(bt.unsupported_phrase_rate("a", "c"), **columns)
    assert grounded + unsupported == pytest.approx(1.0)


def test_a_longer_span_is_a_stricter_test():
    columns = {"a": ["the cat sat down today"], "c": ["the cat sat quietly today"]}
    assert _score(bt.phrase_groundedness("a", "c", n=2), **columns) >= _score(
        bt.phrase_groundedness("a", "c", n=4), **columns
    )


def test_an_empty_answer_scores_zero_rather_than_dividing_by_zero():
    got = _score(bt.phrase_groundedness("a", "c"), a=[""], c=["some context here"])
    assert got == 0.0


@pytest.mark.parametrize("metric", [bt.phrase_groundedness, bt.unsupported_phrase_rate])
def test_the_phrase_metrics_reject_a_span_below_one(metric):
    with pytest.raises(PlanError):
        metric("a", "c", n=0)


# --- retrieval-side monitors -------------------------------------------------------


def test_empty_retrieval_counts_the_queries_that_got_nothing():
    got = _score(bt.empty_retrieval_rate("hits"), hits=[["a"], [], ["b", "c"], []])
    assert got == 0.5


def test_a_list_of_only_nulls_counts_as_an_empty_retrieval():
    """A retriever returning null placeholders has still returned nothing usable."""
    got = _score(bt.empty_retrieval_rate("hits"), hits=[[None, None], ["a"]])
    assert got == 0.5


def test_duplicate_context_catches_a_chunk_returned_twice():
    got = _score(bt.duplicate_context_rate("hits"), hits=[["a", "a", "b"], ["a", "b"]])
    assert got == 0.5


def test_duplicate_context_is_zero_when_every_passage_is_distinct():
    got = _score(bt.duplicate_context_rate("hits"), hits=[["a", "b"], ["c", "d", "e"]])
    assert got == 0.0


def test_mean_retrieved_passages_reports_the_effective_k():
    got = _score(bt.mean_retrieved_passages("hits"), hits=[["a", "b", "c"], ["a"]])
    assert got == 2.0


def test_mean_retrieved_passages_counts_an_empty_retrieval_as_zero():
    got = _score(bt.mean_retrieved_passages("hits"), hits=[["a", "b"], []])
    assert got == 1.0


def test_context_token_estimate_sizes_the_assembled_context():
    got = _score(bt.context_token_estimate("hits"), hits=[["12345678", "1234"]])
    assert got == 3.0  # 12 characters over the default 4 chars per token


def test_context_token_estimate_grows_with_k():
    """The cost regression a raised `k` causes, made visible before the run."""
    small = _score(bt.context_token_estimate("hits"), hits=[["x" * 400]])
    large = _score(bt.context_token_estimate("hits"), hits=[["x" * 400] * 3])
    assert large == pytest.approx(3 * small)


def test_the_retrieval_monitors_break_down_by_group():
    ds = bt.from_pydict({"index": ["old", "old", "new", "new"], "hits": [[], ["a"], ["a"], ["b"]]})
    got = ds.group_by("index").agg(e=bt.empty_retrieval_rate("hits")).to_pydict()
    by_index = dict(zip(got["index"], got["e"], strict=True))
    assert by_index["old"] == 0.5
    assert by_index["new"] == 0.0
