"""The clipped word n-gram metrics — BLEU, ROUGE-N and the primitives under them.

The property that matters is the *clip*: a generation that repeats one correct token must
not score a perfect precision. A set-based intersection cannot see that, so these tests pin
the difference explicitly rather than only checking round numbers.
"""

from __future__ import annotations

import math

import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.unit


def _score(metric, **columns):
    """One aggregate value for a one-row (or few-row) frame."""
    return bt.from_pydict(columns).agg(m=metric).to_pydict()["m"][0]


# --- the primitives ----------------------------------------------------------------


def test_token_ngrams_slides_a_window_of_n_tokens():
    ds = bt.from_pydict({"t": ["the cat sat down"]})
    assert ds.select(g=bt.col("t").str.token_ngrams(2)).to_pydict()["g"][0] == [
        "the cat",
        "cat sat",
        "sat down",
    ]


def test_token_ngrams_of_a_short_text_still_yields_one_gram():
    ds = bt.from_pydict({"t": ["hi"]})
    assert ds.select(g=bt.col("t").str.token_ngrams(4)).to_pydict()["g"][0] == ["hi"]


def test_token_ngrams_null_and_empty_are_distinguished():
    ds = bt.from_pydict({"t": [None, "", "   "]})
    got = ds.select(g=bt.col("t").str.token_ngrams(2)).to_pydict()["g"]
    assert got == [None, [], []]


def test_token_ngrams_rejects_a_size_below_one():
    with pytest.raises(PlanError):
        bt.col("t").str.token_ngrams(0)


def test_multiset_overlap_clips_repeats_at_the_other_side_count():
    ds = bt.from_pydict({"a": [["the", "the", "the"]], "b": [["the", "cat"]]})
    got = ds.select(o=bt.col("a").list.multiset_overlap(bt.col("b"))).to_pydict()["o"]
    assert got == [1.0]


def test_multiset_overlap_differs_from_the_set_intersection_on_repeats():
    """The one behavioural difference that makes the clipped metrics honest."""
    ds = bt.from_pydict({"a": [["x", "x", "x"]], "b": [["x", "x"]]})
    out = ds.select(
        clipped=bt.col("a").list.multiset_overlap(bt.col("b")),
        as_set=bt.col("a").list.set_intersection(bt.col("b")).list.len(),
    ).to_pydict()
    assert out["clipped"] == [2.0]
    assert out["as_set"] == [1]


def test_multiset_overlap_is_null_when_either_row_is_null():
    ds = bt.from_pydict({"a": [None, ["x"]], "b": [["x"], None]})
    got = ds.select(o=bt.col("a").list.multiset_overlap(bt.col("b"))).to_pydict()["o"]
    assert got == [None, None]


def test_multiset_overlap_is_symmetric():
    ds = bt.from_pydict({"a": [["x", "x", "y"]], "b": [["x", "z"]]})
    out = ds.select(
        forward=bt.col("a").list.multiset_overlap(bt.col("b")),
        backward=bt.col("b").list.multiset_overlap(bt.col("a")),
    ).to_pydict()
    assert out["forward"] == out["backward"]


# --- precision / recall / F1 -------------------------------------------------------


def test_precision_is_one_when_every_gram_is_accounted_for():
    assert _score(bt.ngram_precision("p", "r"), p=["the cat sat"], r=["the cat sat down"]) == 1.0


def test_precision_clips_a_degenerate_repetition():
    """A set intersection would score this 1.0; the clip scores it 1/4."""
    assert _score(bt.ngram_precision("p", "r"), p=["cat cat cat cat"], r=["cat sat down"]) == 0.25


def test_recall_measures_coverage_of_the_reference():
    got = _score(bt.ngram_recall("p", "r"), p=["the cat sat"], r=["the cat sat down"])
    assert got == pytest.approx(2 / 3)


def test_f1_is_the_harmonic_mean_of_precision_and_recall():
    kwargs = {"p": ["the cat sat"], "r": ["the cat sat down"]}
    precision = _score(bt.ngram_precision("p", "r"), **kwargs)
    recall = _score(bt.ngram_recall("p", "r"), **kwargs)
    expected = 2 * precision * recall / (precision + recall)
    assert _score(bt.ngram_f1("p", "r"), **kwargs) == pytest.approx(expected)


def test_an_exact_reproduction_scores_one_on_every_order():
    text = ["the quick brown fox jumps"]
    for n in (1, 2, 3):
        assert _score(bt.ngram_precision("p", "r", n=n), p=text, r=text) == 1.0
        assert _score(bt.ngram_recall("p", "r", n=n), p=text, r=text) == 1.0
        assert _score(bt.ngram_f1("p", "r", n=n), p=text, r=text) == 1.0


def test_disjoint_texts_score_zero_not_null():
    assert _score(bt.ngram_precision("p", "r"), p=["alpha beta"], r=["gamma delta"]) == 0.0


def test_an_empty_generation_scores_zero_rather_than_dropping_out_of_the_mean():
    """Two rows, one empty: the mean must be halved, not left at the good row's score."""
    got = _score(bt.ngram_precision("p", "r"), p=["cat sat", ""], r=["cat sat", "cat sat"])
    assert got == 0.5


def test_the_metrics_reject_an_ngram_size_below_one():
    for metric in (bt.ngram_precision, bt.ngram_recall, bt.ngram_f1, bt.ngram_novelty):
        with pytest.raises(PlanError):
            metric("p", "r", n=0)


# --- brevity penalty and BLEU ------------------------------------------------------


def test_brevity_penalty_does_not_punish_a_long_enough_generation():
    assert _score(bt.brevity_penalty("p", "r"), p=["cat sat down"], r=["cat sat"]) == 1.0


def test_brevity_penalty_discounts_a_short_generation_by_the_length_ratio():
    got = _score(bt.brevity_penalty("p", "r"), p=["cat"], r=["cat sat down"])
    assert got == pytest.approx(math.exp(1 - 3 / 1))


def test_brevity_penalty_scores_an_empty_generation_zero():
    assert _score(bt.brevity_penalty("p", "r"), p=[""], r=["cat sat"]) == 0.0


def test_bleu_is_one_for_an_exact_reproduction():
    text = ["the quick brown fox jumps over"]
    assert _score(bt.bleu("p", "r"), p=text, r=text) == 1.0


def test_bleu_is_zero_when_an_order_shares_nothing():
    """Unsmoothed BLEU: one empty order zeroes the product, by definition."""
    assert _score(bt.bleu("p", "r", max_n=2), p=["dog ran"], r=["cat sat"]) == 0.0


def test_bleu_falls_when_the_generation_is_too_short():
    """A prefix of the reference is perfectly precise, so only brevity can penalize it."""
    full = _score(bt.bleu("p", "r", max_n=1), p=["quick brown fox"], r=["quick brown fox"])
    short = _score(bt.bleu("p", "r", max_n=1), p=["quick"], r=["quick brown fox"])
    assert full == 1.0
    assert short < full


def test_bleu_rejects_a_max_order_below_one():
    with pytest.raises(PlanError):
        bt.bleu("p", "r", max_n=0)


# --- diversity and novelty ---------------------------------------------------------


def test_distinct_ngram_ratio_falls_on_a_looping_generation():
    looping = _score(bt.distinct_ngram_ratio("t"), t=["go on go on go on"])
    varied = _score(bt.distinct_ngram_ratio("t"), t=["alpha beta gamma delta"])
    assert looping == pytest.approx(0.4)
    assert varied == 1.0


def test_ngram_novelty_is_zero_for_a_verbatim_copy():
    got = _score(
        bt.ngram_novelty("p", "r"),
        p=["the quick brown fox jumps"],
        r=["the quick brown fox jumps over"],
    )
    assert got == 0.0


def test_ngram_novelty_is_one_when_nothing_is_shared():
    got = _score(
        bt.ngram_novelty("p", "r"),
        p=["alpha beta gamma delta epsilon"],
        r=["zeta eta theta iota"],
    )
    assert got == 1.0


def test_ngram_novelty_is_not_one_minus_precision():
    """Novelty counts each distinct gram once; precision counts every occurrence."""
    columns = {"p": ["one two three four one two three four"], "r": ["one two three four"]}
    novelty = _score(bt.ngram_novelty("p", "r"), **columns)
    precision = _score(bt.ngram_precision("p", "r", n=4), **columns)
    # Four distinct 4-grams, one of them the reference's → 3/4 novel.
    assert novelty == pytest.approx(0.75)
    # Five 4-gram occurrences, one clipped match → 1/5 precise.
    assert precision == pytest.approx(0.2)
    assert novelty != pytest.approx(1 - precision)


# --- grouping ----------------------------------------------------------------------


def test_the_metrics_compose_with_group_by():
    """The averaged (not pooled) form is what lets BLEU be read per category."""
    ds = bt.from_pydict(
        {
            "kind": ["good", "good", "bad"],
            "p": ["cat sat down", "cat sat down", "zebra"],
            "r": ["cat sat down", "cat sat down", "cat sat down"],
        }
    )
    got = ds.group_by("kind").agg(b=bt.ngram_precision("p", "r")).to_pydict()
    scores = dict(zip(got["kind"], got["b"], strict=True))
    assert scores["good"] == 1.0
    assert scores["bad"] == 0.0


# --- ROUGE-L: the metric that reads order ------------------------------------------


def test_lcs_length_counts_an_in_order_match():
    ds = bt.from_pydict({"a": [["the", "cat", "sat"]]})
    assert ds.select(n=bt.col("a").list.lcs_length(bt.col("a"))).to_pydict()["n"] == [3.0]


def test_lcs_length_falls_on_a_reordering():
    """The property that separates it from a bag intersection."""
    ds = bt.from_pydict({"a": [["the", "cat", "sat"]], "b": [["sat", "cat", "the"]]})
    out = ds.select(
        bag=bt.col("a").list.multiset_overlap(bt.col("b")),
        ordered=bt.col("a").list.lcs_length(bt.col("b")),
    ).to_pydict()
    assert out["bag"] == [3.0]
    assert out["ordered"] == [1.0]


def test_lcs_length_allows_a_gap():
    ds = bt.from_pydict({"a": [["a", "x", "b", "y", "c"]], "b": [["a", "b", "c"]]})
    assert ds.select(n=bt.col("a").list.lcs_length(bt.col("b"))).to_pydict()["n"] == [3.0]


def test_rouge_l_scores_a_verbatim_prefix_perfectly_on_precision():
    assert (
        _score(bt.rouge_l_precision("p", "r"), p=["cat sat down"], r=["cat sat down today"]) == 1.0
    )


def test_rouge_l_recall_measures_coverage_of_the_reference():
    got = _score(bt.rouge_l_recall("p", "r"), p=["cat sat"], r=["cat sat down today"])
    assert got == pytest.approx(0.5)


def test_rouge_l_separates_a_reordering_from_a_reproduction():
    """The distinction ROUGE-L exists to make, and the one ROUGE-N cannot."""
    columns = {"p": ["down sat cat"], "r": ["cat sat down"]}
    assert _score(bt.ngram_f1("p", "r"), **columns) == 1.0
    assert _score(bt.rouge_l_f1("p", "r"), **columns) < 0.5


def test_rouge_l_f1_is_one_for_an_exact_reproduction():
    text = ["the quick brown fox jumps"]
    assert _score(bt.rouge_l_f1("p", "r"), p=text, r=text) == 1.0


def test_rouge_l_of_disjoint_texts_is_zero():
    assert _score(bt.rouge_l_f1("p", "r"), p=["alpha beta"], r=["gamma delta"]) == 0.0


def test_rouge_l_f1_is_the_harmonic_mean_of_its_two_halves():
    columns = {"p": ["cat sat on a warm mat"], "r": ["cat sat on the mat"]}
    precision = _score(bt.rouge_l_precision("p", "r"), **columns)
    recall = _score(bt.rouge_l_recall("p", "r"), **columns)
    expected = 2 * precision * recall / (precision + recall)
    assert _score(bt.rouge_l_f1("p", "r"), **columns) == pytest.approx(expected)


def test_an_empty_generation_scores_rouge_l_zero_rather_than_dropping_out():
    got = _score(bt.rouge_l_f1("p", "r"), p=["cat sat", ""], r=["cat sat", "cat sat"])
    assert got == 0.5


def test_rouge_l_never_exceeds_rouge_n_recall_on_the_same_text():
    """An in-order match is a subset of a bag match, so ROUGE-L can only be the stricter one."""
    columns = {"p": ["the fox jumps over the dog"], "r": ["the quick fox jumps over a dog"]}
    assert _score(bt.rouge_l_recall("p", "r"), **columns) <= _score(
        bt.ngram_recall("p", "r"), **columns
    )
