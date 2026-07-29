"""Text-pattern selectivity: the filter of unstructured data, read from its pattern.

`LIKE` already had its pattern read (exact / anchored / substring). `regexp_matches` did
not — every regex got the flat substring prior. That is right for `'error'` and wrong for
the anchored patterns Batcher's *own public API* generates: `is_alpha`, `is_numeric`,
`is_alnum`, `is_space`, `is_url`, and `is_email` all lower to an anchored
`regexp_matches`, and none of them scans for a substring — they classify the whole value.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher import col
from batcher.config import active_config
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.stats.selectivity.patterns import anchored_selectivity

pytestmark = pytest.mark.unit

_CFG = active_config().optimizer.cardinality


def _sel(predicate) -> float:
    """The fraction of rows the estimator thinks `predicate` keeps."""
    ds = bt.from_pydict({"s": [f"v{i}" for i in range(1000)]})
    est = CardinalityEstimator(ds._sources)
    base = est.estimate(ds._plan).rows
    return est.estimate(ds.filter(predicate)._plan).rows / base


# --- a regex is now read the way a LIKE always was --------------------------------


def test_an_unanchored_regex_keeps_the_substring_prior():
    # The case the flat prior was written for, and still the right answer.
    assert _sel(col("s").str.regexp_matches("error")) == pytest.approx(_CFG.substring_selectivity)


@pytest.mark.parametrize("pattern", ["^ERR", "ing$", "^[A-Z]{3}", r"^\d+$"])
def test_an_anchored_regex_is_estimated_as_anchored(pattern):
    assert _sel(col("s").str.regexp_matches(pattern)) == pytest.approx(anchored_selectivity(_CFG))


def test_a_fully_literal_anchored_regex_is_equality():
    # `^foo$` with no metacharacters matches exactly one value, so it gets the equality
    # estimate rather than a pattern prior.
    assert _sel(col("s").str.regexp_matches("^foo$")) == pytest.approx(_CFG.eq_selectivity)


def test_an_escaped_dollar_is_not_an_end_anchor():
    # `\$` is a literal dollar sign. Reading it as an anchor would claim a whole-value
    # match for a pattern that searches for a price anywhere in the text.
    assert _sel(col("s").str.regexp_matches(r"[0-9]+\$")) == pytest.approx(
        _CFG.substring_selectivity
    )


def test_a_bare_anchor_matches_everything_and_claims_nothing():
    # `^` alone constrains nothing; estimating it as an anchored match would invent
    # selectivity out of a pattern that has none.
    assert _sel(col("s").str.regexp_matches("^")) == pytest.approx(_CFG.substring_selectivity)


# --- the public text predicates that lower to anchored regexes --------------------


@pytest.mark.parametrize(
    "predicate",
    [
        lambda c: c.str.is_alpha(),
        lambda c: c.str.is_numeric(),
        lambda c: c.str.is_alnum(),
        lambda c: c.str.is_space(),
        lambda c: c.str.is_url(),
        lambda c: c.str.is_email(),
    ],
)
def test_whole_value_classifiers_are_not_estimated_as_substring_searches(predicate):
    # Every one of these is `^...$`. Estimating a whole-value classification as an
    # anywhere-in-the-text search mis-counts the survivors of the most common text filter
    # in the engine, on the data where cardinality is hardest to recover from later.
    assert _sel(predicate(col("s"))) != pytest.approx(_CFG.substring_selectivity)


# --- LIKE keeps every answer it had ----------------------------------------------


def test_like_is_unchanged_by_the_move():
    # The LIKE reader moved modules; none of its answers may move with it.
    assert _sel(col("s").str.like("%foo%")) == pytest.approx(_CFG.substring_selectivity)
    assert _sel(col("s").str.like("foo%")) == pytest.approx(anchored_selectivity(_CFG))
    assert _sel(col("s").str.like("%foo")) == pytest.approx(anchored_selectivity(_CFG))
    assert _sel(col("s").str.like("foo")) == pytest.approx(_CFG.eq_selectivity)
    assert _sel(col("s").str.like("f_o%")) == pytest.approx(_CFG.substring_selectivity)


def test_anchored_predicates_all_agree_with_each_other():
    # `starts_with`, `ends_with`, an anchored LIKE and an anchored regex are the same
    # shape. Four call sites reading the prior separately is four chances to drift.
    anchored = anchored_selectivity(_CFG)
    assert _sel(col("s").str.starts_with("a")) == pytest.approx(anchored)
    assert _sel(col("s").str.ends_with("a")) == pytest.approx(anchored)
    assert _sel(col("s").str.like("a%")) == pytest.approx(anchored)
    assert _sel(col("s").str.regexp_matches("^a")) == pytest.approx(anchored)
