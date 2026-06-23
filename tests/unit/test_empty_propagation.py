"""Plan-shape unit tests for `propagate_empty_relation`."""

from __future__ import annotations

import batcher as bt
from batcher import col
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.zonemap_pruning import propagate_empty_relation
from batcher.plan.logical import Distinct, Filter, Limit, Sort, Union


def _t():
    return bt.from_pydict({"x": [1, 2, 3], "y": [10, 20, 30]})


def test_rule_registered():
    assert "propagate_empty_relation" in {r.name for r in DEFAULT_REGISTRY.rules()}


def test_filter_over_empty_folds():
    plan = _t().limit(0).filter(col("x") > 1)._plan
    assert isinstance(plan, Filter) and isinstance(plan.input, Limit)
    out = propagate_empty_relation(plan, None)
    assert isinstance(out, Limit) and out.n == 0


def test_filter_over_nonempty_is_noop():
    plan = _t().filter(col("x") > 1)._plan
    assert propagate_empty_relation(plan, None) is None


def test_sort_over_empty_folds():
    plan = _t().limit(0).sort("x")._plan
    assert isinstance(plan, Sort)
    out = propagate_empty_relation(plan, None)
    assert isinstance(out, Limit) and out.n == 0


def test_union_drops_empty_branch():
    plan = Union((_t().limit(0)._plan, _t()._plan))
    out = propagate_empty_relation(plan, None)
    # One non-empty branch survives → the union collapses to it.
    assert not isinstance(out, Union)
    assert isinstance(out, Limit) is False


def test_union_all_empty_stays_empty():
    plan = Union((_t().limit(0)._plan, _t().limit(0)._plan))
    out = propagate_empty_relation(plan, None)
    assert isinstance(out, Limit) and out.n == 0


def test_union_distinct_single_survivor_keeps_dedup():
    plan = Union((_t().limit(0)._plan, _t()._plan), distinct=True)
    out = propagate_empty_relation(plan, None)
    assert isinstance(out, Distinct)


def test_union_no_empty_is_noop():
    plan = Union((_t()._plan, _t()._plan))
    assert propagate_empty_relation(plan, None) is None


def test_full_optimizer_collapses_empty_chain():
    # Filter(Sort(Limit(x, 0))) must optimize to a single empty relation.
    plan = _t().limit(0).sort("x").filter(col("x") > 1)._plan
    ir = Optimizer().optimize(plan).ir
    assert ir["op"] == "limit" and ir["n"] == 0
    assert ir["input"]["op"] == "scan"


def test_optimize_deterministic():
    plan = _t().limit(0).filter(col("x") > 1)._plan
    assert Optimizer().optimize(plan).ir == Optimizer().optimize(plan).ir
