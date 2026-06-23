"""Predicate pushdown through Aggregate: key predicates move below grouping (W3)."""

from __future__ import annotations

import batcher as bt
from batcher import col
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.rules.pushdown import (
    push_filter_through_aggregate,
    push_filter_through_sort,
)
from batcher.plan.logical import Aggregate, Filter, Sort


def _t():
    return bt.from_pydict(
        {"dept": ["a", "a", "b", "b", "c"], "region": [1, 1, 2, 2, 1], "sal": [10, 20, 30, 40, 50]}
    )


def test_filter_on_group_key_is_pushed_below_aggregate():
    plan = _t().group_by("dept").agg(total=col("sal").sum()).filter(col("dept") == "a")._plan
    out = push_filter_through_aggregate(plan, None)
    assert isinstance(out, Aggregate)  # aggregate now on top
    assert isinstance(out.input, Filter)  # filter pushed under it
    ir = Optimizer().optimize(plan).ir
    assert ir["op"] == "aggregate"
    assert ir["input"]["op"] == "filter"


def test_filter_on_aggregate_output_is_not_pushed():
    # HAVING-style predicate on the aggregate result must stay above the aggregate.
    plan = _t().group_by("dept").agg(total=col("sal").sum()).filter(col("total") > 25)._plan
    assert push_filter_through_aggregate(plan, None) is None
    ir = Optimizer().optimize(plan).ir
    assert ir["op"] == "filter"  # stays on top


def test_pushdown_aggregate_unit_off_non_aggregate():
    plan = _t().filter(col("sal") > 1)._plan  # filter over a scan
    assert push_filter_through_aggregate(plan, None) is None


# --- push_filter_through_sort -------------------------------------------------


def test_filter_pushed_through_sort():
    plan = _t().sort("sal").filter(col("sal") > 15)._plan
    out = push_filter_through_sort(plan, None)
    assert isinstance(out, Sort)  # sort now on top
    assert isinstance(out.input, Filter)  # filter pushed under it
    ir = Optimizer().optimize(plan).ir
    assert ir["op"] == "sort"
    assert ir["input"]["op"] == "filter"


def test_filter_not_pushed_through_topn_sort():
    # Sort carrying a limit (top-N): filtering first would change which rows
    # survive the limit, so the filter must stay above.
    plan = _t().sort("sal").head(2).filter(col("sal") > 15)._plan
    # The plan is Filter(Limit(Sort)); the rule only matches Filter-over-Sort.
    assert push_filter_through_sort(plan, None) is None
