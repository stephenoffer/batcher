"""The two plan-level decisions behind a partitioned distributed sort (F095, F096).

`stackable_on_buckets` decides which operators above a sort may run inside every reducer,
and `_buckets_through_limit` decides which ordered buckets a limited sort must read. Both are
pure, so they are pinned here without Ray;
`tests/integration/test_distributed_sort_row_local_above.py` runs them on a cluster.
"""

from __future__ import annotations

import json

import pytest

import batcher as bt
from batcher.dist.executors.plan_analysis import stack_above_ir, stackable_on_buckets
from batcher.dist.executors.sort import _buckets_through_limit
from batcher.plan.ir_specs import task_scan_ir

pytestmark = pytest.mark.unit


def _above_sort(ds):
    """The operators above the first `Sort`, outermost first, as the dispatcher splits them."""
    above, node = [], ds._plan
    while type(node).__name__ != "Sort":
        above.append(node)
        node = node.input
    return above, node


def test_filter_and_project_fold_and_nothing_order_sensitive_does():
    base = bt.from_pydict({"t": [3, 1, 2], "x": [1, 2, 3]}).sort("t")
    above, _ = _above_sort(base.filter(bt.col("x") > 1).select("t"))
    assert stackable_on_buckets(above)
    assert not stackable_on_buckets([])
    limited, _ = _above_sort(base.limit(2).select("t"))
    assert not stackable_on_buckets(limited)  # a Limit reads across buckets


def test_the_stacked_ir_wraps_the_reducer_plan_outermost_last():
    base = bt.from_pydict({"t": [3, 1, 2], "x": [1, 2, 3]}).sort("t")
    above, sort = _above_sort(base.filter(bt.col("x") > 1).select("t"))
    reduce_ir = {**sort.shape_ir(), "input": task_scan_ir()}
    ir = stack_above_ir(above, reduce_ir)
    assert ir["op"] == "project"
    assert ir["input"]["op"] == "filter"
    assert ir["input"]["input"] == reduce_ir
    json.dumps(ir)  # it crosses to the workers as JSON


@pytest.mark.parametrize(
    ("limit", "want"),
    [
        (None, ["a", "b", "c"]),
        (1, ["a"]),
        (4, ["a"]),  # the first bucket already holds four rows
        (5, ["a", "b"]),
        (6, ["a", "b"]),  # an empty bucket is skipped, never read
        (7, ["a", "b", "c"]),
        (100, ["a", "b", "c"]),
    ],
)
def test_a_limit_reads_only_the_leading_buckets(limit, want):
    ordered = [("a", 4), (None, 0), ("b", 2), ("c", 9)]
    assert _buckets_through_limit(ordered, limit) == want
