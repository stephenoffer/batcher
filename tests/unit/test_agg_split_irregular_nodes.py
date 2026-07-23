"""Aggregate-leaf splitting must descend into the irregular `Case`/`MakeStruct` nodes.

`group_by().agg()` accepts a scalar expression *over* aggregates (``sum(x) / sum(y)``):
it hoists each `AggExpr` leaf into a hidden aggregate column and re-evaluates the
surrounding expression in a following projection. That hoist is driven by
`contains_aggregate` / `split_aggregate_leaves`, which walk a node's sub-expressions
via `child_fields`.

`Case` and `MakeStruct` carry their sub-expressions in irregular fields (paired
when/then branches; named struct fields) declared without the `child`/`children`
factories, so `child_fields` cannot see them. Before the fix both functions treated
a CASE / ``struct(...)`` as a childless leaf: an aggregate inside one was invisible,
so ``group_by().agg()`` rejected the (valid) expression as "not an expression over
aggregates", or crashed when the un-hoisted leaf later reached `to_ir`.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.plan.expr_ir.core import Expr
from batcher.plan.expr_ir.nodes import Case, MakeStruct
from batcher.plan.expr_ir.walk import (
    AggregateLeafRegistry,
    contains_aggregate,
    split_aggregate_leaves,
)

col = bt.col
struct = bt.struct
when = bt.when
lit = bt.lit


@pytest.mark.unit
def test_contains_aggregate_sees_into_case() -> None:
    # An aggregate in a CASE condition and in a branch — both must be found.
    c = when(col("x").sum().cast("float64") > 0).then(col("y").sum()).otherwise(lit(0))
    assert isinstance(c, Case)
    assert contains_aggregate(c) is True


@pytest.mark.unit
def test_contains_aggregate_sees_into_make_struct() -> None:
    st = struct(sx=col("x").sum(), sy=col("y").sum())
    assert isinstance(st, MakeStruct)
    assert contains_aggregate(st) is True


@pytest.mark.unit
def test_split_extracts_aggregate_leaves_from_case() -> None:
    c = when(col("x").sum().cast("float64") > 0).then(col("y").sum()).otherwise(lit(0))
    registry = AggregateLeafRegistry()
    out = split_aggregate_leaves(c, registry)
    # Two distinct aggregates (sum(x), sum(y)) hoisted into hidden columns...
    assert len(registry.leaves()) == 2
    # ...and the rewritten expression is a pure scalar Case with no raw aggregate left.
    assert isinstance(out, Case)
    assert contains_aggregate(out) is False
    out.to_ir()  # would raise on any surviving un-hoisted AggExpr


@pytest.mark.unit
def test_split_extracts_aggregate_leaves_from_make_struct() -> None:
    st = struct(sx=col("x").sum(), sy=col("y").sum())
    registry = AggregateLeafRegistry()
    out = split_aggregate_leaves(st, registry)
    assert len(registry.leaves()) == 2
    assert isinstance(out, MakeStruct)
    assert contains_aggregate(out) is False
    out.to_ir()


@pytest.mark.unit
def test_split_dedups_shared_leaf_across_struct_and_outer() -> None:
    # sum(x) appears both inside the struct and outside it; it is one hidden column.
    e: Expr = col("x").sum() + struct(a=col("x").sum()).struct.field("a")
    registry = AggregateLeafRegistry()
    split_aggregate_leaves(e, registry)
    assert len(registry.leaves()) == 1


@pytest.mark.integration
def test_group_by_agg_over_case_and_struct_end_to_end() -> None:
    pytest.importorskip("batcher._native", reason="native engine not built")
    import batcher as bt

    ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 10], "y": [10, 20, 30]})

    got_struct = (
        ds.group_by("g")
        .agg(s=bt.struct(sx=bt.col("x").sum(), sy=bt.col("y").sum()))
        .sort("g")
        .to_pydict()
    )
    assert got_struct == {
        "g": ["a", "b"],
        "s": [{"sx": 3, "sy": 30}, {"sx": 10, "sy": 30}],
    }

    # sum(x): a -> 3, b -> 10; threshold 5 splits them (matches SQL CASE over aggregates).
    got_case = (
        ds.group_by("g")
        .agg(c=bt.when(bt.col("x").sum().cast("float64") > 5).then(bt.lit(1)).otherwise(bt.lit(0)))
        .sort("g")
        .to_pydict()
    )
    assert got_case == {"g": ["a", "b"], "c": [0, 1]}
