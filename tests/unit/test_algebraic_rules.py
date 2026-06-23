"""Algebraic rewrite rules: plan-shape changes that preserve results (W3)."""

from __future__ import annotations

import batcher as bt
from batcher import col
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.algebraic import (
    combine_limits,
    merge_adjacent_filters,
    prune_true_filter,
    push_filter_into_union,
    remove_redundant_distinct,
)
from batcher.plan.expr_ir import Lit
from batcher.plan.logical import Distinct, Filter, Limit


def _t():
    return bt.from_pydict({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})


def test_rules_registered_in_default_registry():
    names = {r.name for r in DEFAULT_REGISTRY.rules()}
    assert {
        "prune_true_filter",
        "merge_adjacent_filters",
        "remove_redundant_distinct",
        "combine_limits",
    } <= names


# --- merge_adjacent_filters ---------------------------------------------------


def test_adjacent_filters_merge_into_one():
    plan = _t().filter(col("x") > 1).filter(col("y") < 50)._plan
    ir = Optimizer().optimize(plan).ir
    assert ir["op"] == "filter"
    assert ir["input"]["op"] == "scan"  # only ONE filter remains (merged)
    assert ir["predicate"]["op"] == "and"


def test_merge_adjacent_filters_unit_returns_none_when_not_applicable():
    # input is a Scan, not a Filter -> rule does not fire
    plan = _t().filter(col("x") > 1)._plan
    assert merge_adjacent_filters(plan, None) is None


# --- push_filter_into_union ---------------------------------------------------


def test_filter_pushed_into_union_branches():
    a = _t()
    b = _t()
    plan = a.union(b).filter(col("x") > 2)._plan
    ir = Optimizer().optimize(plan).ir
    assert ir["op"] == "union"  # filter pushed below the union
    assert all(child["op"] == "filter" for child in ir["inputs"])


def test_push_filter_into_union_unit_returns_none_off_union():
    plan = _t().filter(col("x") > 1)._plan  # filter over a scan, not a union
    assert push_filter_into_union(plan, None) is None


# --- prune_true_filter --------------------------------------------------------


def test_true_filter_is_pruned():
    # A predicate that folds to TRUE leaves no filter behind.
    plan = _t().filter(bt.lit(1) == bt.lit(1))._plan
    ir = Optimizer().optimize(plan).ir
    assert ir["op"] == "scan"  # the always-true filter was removed


def test_prune_true_filter_unit():
    inner = _t()._plan
    f = Filter(inner, Lit(True))
    assert prune_true_filter(f, None) is inner


# --- remove_redundant_distinct ------------------------------------------------


def test_double_distinct_collapses():
    plan = _t().distinct().distinct()._plan
    ir = Optimizer().optimize(plan).ir
    assert ir["op"] == "distinct"
    assert ir["input"]["op"] == "scan"  # only one distinct remains


def test_distinct_over_aggregate_is_dropped():
    plan = _t().group_by("x").agg(total=col("y").sum()).distinct()._plan
    ir = Optimizer().optimize(plan).ir
    assert ir["op"] == "aggregate"  # distinct removed; aggregate output already unique


def test_remove_redundant_distinct_unit():
    inner = Distinct(_t()._plan)
    assert remove_redundant_distinct(Distinct(inner), None) is inner


# --- combine_limits -----------------------------------------------------------


def test_nested_limits_combine():
    plan = _t().head(4).head(2)._plan
    ir = Optimizer().optimize(plan).ir
    assert ir["op"] == "limit"
    assert ir["input"]["op"] == "scan"  # only one limit remains
    assert ir["n"] == 2


def test_combine_limits_offset_arithmetic_unit():
    base = _t()._plan
    inner = Limit(base, n=10, offset=2)
    outer = Limit(inner, n=3, offset=4)
    out = combine_limits(outer, None)
    assert isinstance(out, Limit)
    assert out.offset == 6  # 2 + 4
    assert out.n == 3  # min(3, 10 - 4)
    assert out.input is base


# --- limit pushdown -----------------------------------------------------------


def test_limit_pushed_through_project():
    plan = _t().select(z=col("x") * col("y")).head(2)._plan
    ir = Optimizer().optimize(plan).ir
    assert ir["op"] == "project"  # projection now on top
    assert ir["input"]["op"] == "limit"  # limit pushed under it
    assert ir["input"]["n"] == 2


def test_limit_pushed_into_union_all():
    a = _t()
    b = _t()
    plan = a.union(b).head(3)._plan  # union(distinct=False)
    ir = Optimizer().optimize(plan).ir
    assert ir["op"] == "limit"
    assert ir["input"]["op"] == "union"
    # each union input is now capped by a limit
    assert all(child["op"] == "limit" for child in ir["input"]["inputs"])


def test_limit_not_pushed_into_distinct_union():
    a = _t()
    b = _t()
    plan = a.union(b, distinct=True).head(3)._plan
    ir = Optimizer().optimize(plan).ir
    # distinct union: inputs must NOT be capped (dedup changes counts)
    union_ir = ir["input"] if ir["op"] == "limit" else ir
    inputs = union_ir["inputs"] if union_ir["op"] == "union" else []
    assert inputs and all(child["op"] != "limit" for child in inputs)
