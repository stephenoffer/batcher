"""A scan's pushed predicate survives a rule that consumes the `Filter` above it.

Source predicates are read off the *optimized* plan, where pushdown parks a residual
`Filter` just above each `Scan`. But the aggregate fusion rewrites ``COUNT(*)`` over
``Filter(p)`` into one ``count_if(CASE WHEN p ...)`` pass over the `Scan` — strictly
faster, and it deletes the only node that extraction knows how to read. The predicate
then reached the source nowhere, so ``SELECT count(*) WHERE p`` pushed nothing and a
lakehouse scan opened every data file in the table instead of the one its log says can
match. `kyber.optimizer._source_predicates` recovers it from the user's plan.

These are plan-shape assertions (does the predicate reach `source_predicates`?); that the
*results* are unchanged is covered by the DuckDB differential suite.
"""

from __future__ import annotations

import batcher as bt
from batcher import col, count
from batcher.plan.logical import Aggregate, Filter, Scan


def _dataset():
    return bt.from_pydict({"day": [1, 1, 2, 3], "x": [10, 20, 30, 40]})


def _optimize(plan, sources):
    from batcher import core, kyber

    return kyber.optimize(plan, sources=sources, hub=core.default_hub())


def test_count_over_filter_still_pushes_its_predicate() -> None:
    """The regression: the fused plan has no `Filter`, but the source must still get one."""
    from batcher.api.terminal.metadata_answer import global_count_plan

    ds = _dataset().filter(col("day") == 1)
    plan = global_count_plan(ds._plan)
    physical = _optimize(plan, ds._sources)

    assert physical.source_predicates, "count(*) over a filter pushed no predicate"
    pushed = physical.source_predicates[0]
    assert pushed["op"] == "eq"
    assert pushed["left"] == {"e": "col", "name": "day"}


def test_the_fusion_really_did_remove_the_filter() -> None:
    """Guards the test above: if fusion stopped happening, it would pass vacuously."""
    from batcher import core, kyber
    from batcher.api.terminal.metadata_answer import global_count_plan

    ds = _dataset().filter(col("day") == 1)
    optimized = kyber.optimize_logical(
        global_count_plan(ds._plan), sources=ds._sources, hub=core.default_hub()
    )
    assert isinstance(optimized, Aggregate)
    assert isinstance(optimized.input, Scan), "expected the Filter to be fused into the aggregate"


def test_a_plain_filter_still_pushes_from_the_optimized_plan() -> None:
    """The ordinary path is untouched: a surviving `Filter` is still the source of truth."""
    ds = _dataset().filter(col("day") == 2)
    physical = _optimize(ds._plan, ds._sources)
    assert physical.source_predicates[0]["op"] == "eq"


def test_an_unfiltered_scan_pushes_nothing() -> None:
    ds = _dataset()
    assert _optimize(ds._plan, ds._sources).source_predicates == {}


def test_a_filter_not_adjacent_to_the_scan_is_not_pushed() -> None:
    """Conservative by design: only a `Filter` sitting *on* a `Scan` constrains it wholly."""
    ds = _dataset().group_by("day").agg(n=count()).filter(col("n") > 1)
    physical = _optimize(ds._plan, ds._sources)
    assert physical.source_predicates == {}


def test_filtered_count_matches_an_unfused_count() -> None:
    """The fused, pushed plan must agree with the plain one — pushdown changed no answer."""
    ds = _dataset()
    assert ds.filter(col("day") == 1).count() == 2
    assert ds.filter(col("day") == 1).collect().num_rows == 2
    assert ds.filter(col("day") == 99).count() == 0


def test_pushed_predicate_is_a_filter_above_a_scan_in_the_users_plan() -> None:
    """The soundness argument, asserted: what we push is exactly what sat on the scan."""
    ds = _dataset().filter(col("day") == 1)
    assert isinstance(ds._plan, Filter)
    assert isinstance(ds._plan.input, Scan)
