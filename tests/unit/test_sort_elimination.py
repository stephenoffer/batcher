"""Sort elimination before aggregate: a wasted pre-group-by sort is removed (W3)."""

from __future__ import annotations

import batcher as bt
from batcher import col
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.rules.algebraic import eliminate_sort_before_aggregate
from batcher.plan.logical import Aggregate


def _t():
    return bt.from_pydict({"k": ["a", "a", "b"], "v": [1, 2, 3]})


def test_sort_below_aggregate_is_removed():
    plan = _t().sort("v").group_by("k").agg(total=col("v").sum())._plan
    out = eliminate_sort_before_aggregate(plan, None)
    assert isinstance(out, Aggregate)
    assert out.input is plan.input.input  # the sort was spliced out
    ir = Optimizer().optimize(plan).ir
    assert ir["op"] == "aggregate"
    assert ir["input"]["op"] == "scan"  # no sort between aggregate and scan


def test_topn_sort_below_aggregate_is_kept():
    # Sort with a limit changes which rows are aggregated -> must not be removed.
    plan = _t().sort("v").head(2).group_by("k").agg(total=col("v").sum())._plan
    # plan is Aggregate(Limit(Sort)); the rule matches Aggregate-over-Sort only.
    assert eliminate_sort_before_aggregate(plan, None) is None


def test_eliminate_sort_unit_off_non_sort():
    plan = _t().group_by("k").agg(total=col("v").sum())._plan  # Aggregate(Scan)
    assert eliminate_sort_before_aggregate(plan, None) is None
