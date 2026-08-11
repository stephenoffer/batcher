"""A union never hands one column another column's statistics.

`RelStats.columns` is *sparse*: it holds only the columns the estimator could say
something about. `union_columns` aligned branches by position -- which is how a union is
defined -- but indexed that sparse dict by output position, so a branch tracking only its
third column reported that column's bounds for output position 0.

The consequence was not a soft mis-estimate. A branch carrying a literal `weight = 1.0`
made a *string* join key look like the constant `1.0`, so
`infer_join_predicate_from_constant_key` mirrored `key = 1.0` onto the other side and the
query died at execution with `Utf8 == Float64`. Where the types happened to be
compatible, nothing would have raised and the pushed predicate would have returned
silently wrong rows.
"""

from __future__ import annotations

import batcher as bt
from batcher import col
from batcher.kyber.stats.columns import union_columns
from batcher.kyber.stats.estimator import StatsEstimator
from batcher.plan.stats import ColumnStat, Provenance, RelStats


def _shared_union() -> tuple[bt.Dataset, bt.Dataset]:
    """Two branches of one parent whose only tracked column is a trailing literal."""
    shared = bt.from_pydict({"a": [1, 2], "b": [2, 3]}).select(
        src=col("a").cast("string"), dst=col("b").cast("string")
    )
    forward = shared.select("src", "dst").with_columns(weight=bt.lit(1.0))
    reverse = shared.select(src=col("dst"), dst=col("src")).with_columns(weight=bt.lit(1.0))
    return forward, reverse


def test_union_does_not_borrow_another_columns_stats() -> None:
    """The string key must not inherit the float literal's bounds."""
    forward, reverse = _shared_union()
    union = forward.union(reverse)

    # The precondition that made this reachable: each branch tracks only `weight`.
    estimator = StatsEstimator({})
    tracked = set(estimator.estimate(forward._plan).columns)
    assert tracked == {"weight"}

    stats = estimator.estimate(union._plan).columns
    assert "src" not in stats, "a string key must not acquire the float literal's stats"
    assert stats["weight"].min == 1.0
    assert stats["weight"].max == 1.0


def test_join_over_that_union_returns_the_right_rows() -> None:
    """The end-to-end shape: the bogus constant used to become a pushed-down predicate."""
    forward, reverse = _shared_union()
    probe = bt.from_pydict({"src": ["1", "2"], "_l": ["x", "y"]})

    result = forward.union(reverse).join(probe, on="src").sort("src", "dst").to_pydict()

    assert result["src"] == ["1", "2", "2"]
    assert result["dst"] == ["2", "1", "3"]
    assert result["_l"] == ["x", "y", "y"]


def _branch(**columns: tuple[int, int]) -> RelStats:
    """A branch whose tracked columns carry the given (min, max) bounds."""
    return RelStats(
        rows=2.0,
        provenance=Provenance.EXACT,
        columns={
            name: ColumnStat(min=low, max=high, provenance=Provenance.EXACT)
            for name, (low, high) in columns.items()
        },
    )


def test_positional_alignment_uses_declared_names() -> None:
    """Branches align by position against their declared columns, not the stats dict."""
    children = [_branch(a=(1, 2), b=(10, 20)), _branch(c=(3, 4), d=(30, 40))]

    merged = union_columns(children, ["a", "b"], [["a", "b"], ["c", "d"]])

    # Position 0 pairs `a` with `c`, position 1 pairs `b` with `d`.
    assert merged["a"].min == 1
    assert merged["a"].max == 4
    assert merged["b"].min == 10
    assert merged["b"].max == 40


def test_sparse_branch_stats_do_not_shift_positions() -> None:
    """The defect itself: a branch tracking only its *second* column must not fill the first."""
    # Both branches declare (key, weight) but only track `weight`.
    children = [_branch(weight=(1, 1)), _branch(weight=(1, 1))]
    names = [["key", "weight"], ["key", "weight"]]

    merged = union_columns(children, ["key", "weight"], names)

    assert "key" not in merged, "an untracked column must not inherit another's bounds"
    assert merged["weight"].min == 1
    assert merged["weight"].max == 1


def test_missing_branch_names_never_invents_an_estimate() -> None:
    """Without declared names a column no branch reports is skipped, not guessed."""
    children = [_branch(a=(1, 2), b=(10, 20)), _branch(c=(3, 4), d=(30, 40))]

    merged = union_columns(children, ["a", "b"])

    # Only the left branch reports `a`/`b`, so neither can be combined safely.
    assert "a" not in merged
    assert "b" not in merged
