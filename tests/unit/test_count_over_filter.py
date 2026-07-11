"""Plan-shape + semantics unit tests for `count_over_filter_to_count_if`."""

from __future__ import annotations

import batcher as bt
from batcher import col, count
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.fusion import count_over_filter_to_count_if
from batcher.plan.logical import Aggregate, Filter


def _counted():
    """Keyless `COUNT(*)` over a filtered scan (the `SELECT COUNT(*) ... WHERE p` shape)."""
    return (
        bt.from_pydict({"k": [1, 2, 3, 4], "v": [10, 20, 30, 40]})
        .filter(col("v") > 15)
        .agg(n=count())
    )


def test_rule_registered():
    assert "count_over_filter_to_count_if" in {r.name for r in DEFAULT_REGISTRY.rules()}


def test_folds_filter_into_a_masked_count():
    plan = _counted()._plan
    out = count_over_filter_to_count_if(plan, None)
    # The filter is gone (the aggregate now reads the scan directly) and the count
    # became a `count` over the predicate mask — never a `count_star` over a Filter.
    assert isinstance(out, Aggregate)
    assert not isinstance(out.input, Filter)
    assert all(s.agg.func == "count" for s in out.aggregates)


def test_the_mask_is_counted_not_summed():
    """`SUM` over zero rows is NULL; `COUNT` is 0.

    Summing the mask (`count_if`'s desugaring) silently turned
    ``SELECT COUNT(*) FROM empty WHERE p`` into NULL — invisible until the filter's
    input happened to be empty, which is when a count matters most.
    """
    out = count_over_filter_to_count_if(_counted()._plan, None)
    assert all(s.agg.func != "sum" for s in out.aggregates)


def test_count_star_over_an_empty_input_is_zero_not_null():
    """The regression: an empty join under the filter must still count to 0."""
    left = bt.from_pydict({"k": [1]})
    right = bt.from_pydict({"k": [2]})
    empty = left.join(right, on="k").select(x=col("k") + 1).filter(col("x") > 0)
    assert empty.agg(n=count()).to_pydict()["n"] == [0]
    assert empty.count() == 0


def test_no_fire_when_grouped():
    # GROUP BY: a filter also drops all-fail groups that count_if would report as 0.
    plan = (
        bt.from_pydict({"k": [1, 1, 2], "v": [10, 20, 5]})
        .filter(col("v") > 8)
        .group_by("k")
        .agg(n=count())
    )._plan
    assert count_over_filter_to_count_if(plan, None) is None


def test_no_fire_with_noncount_aggregate():
    plan = (
        bt.from_pydict({"k": [1, 2, 3], "v": [10, 20, 30]})
        .filter(col("v") > 8)
        .agg(n=count(), s=col("v").sum())
    )._plan
    assert count_over_filter_to_count_if(plan, None) is None


def test_result_unchanged():
    ds = _counted()
    # v > 15 keeps 20, 30, 40 → count 3, both before and after the rewrite.
    assert ds.to_pydict() == {"n": [3]}
