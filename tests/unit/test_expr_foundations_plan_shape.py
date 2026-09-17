"""Plan-shape checks for the expression foundations: names, CASE, ``over``, ``col``, order.

The differential tests prove each foundation against DuckDB and Polars on data; these pin
the *construction* -- which name an expression infers, what a finished CASE carries, how
``over`` merges keys, which selector ``col`` returns, and which order a sorted frame lends a
window -- so a regression names the rule it broke rather than a value that moved.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.api.dataset._window import established_order
from batcher.plan.expr_ir import Case, Col, NullIf, WindowExpr
from batcher.plan.expr_ir.func_nodes import ListFilter
from batcher.plan.expr_rewrite.naming import output_name
from batcher.plan.logical import Window

pytestmark = pytest.mark.unit

col = bt.col


@pytest.mark.parametrize(
    ("expr", "name"),
    [
        (col("a") + 1, "a"),
        (bt.lit(1) + col("a"), "literal"),
        (bt.lit(3), "literal"),
        ((col("a") * 2).alias("z"), "z"),
        (col("b").str.upper(), "b"),
        (col("a").sum(), "a"),
        (bt.count(), "count"),
        (col("a").sum().alias("t"), "t"),
        (col("x").sum() / col("y").sum(), "x"),
        (bt.coalesce(col("q"), col("a")), "q"),
        (bt.when(col("a") > 1).then(col("b")).otherwise(col("c")), "b"),
        (col("a").shift(1), "a"),
        (col("a").rank(), "a"),
        (bt.row_number().over(order_by="t"), "t"),
    ],
)
def test_output_name_follows_the_leftmost_leaf(expr, name):
    assert output_name(expr) == name


def test_a_case_without_otherwise_is_null_typed_by_its_first_value():
    case = bt.when(col("a") > 1).then(col("s"))._finish()
    assert isinstance(case, Case)
    assert isinstance(case.otherwise, NullIf)
    assert isinstance(case.otherwise.left, Col) and case.otherwise.left.name == "s"


def test_a_none_then_borrows_the_type_of_a_later_value():
    case = bt.when(col("a") > 1).then(None).otherwise(col("s"))
    assert isinstance(case.branches[0][1], NullIf)
    assert case.branches[0][1].left.name == "s"


def test_the_builder_is_immutable():
    base = bt.when(col("a") > 1).then(1)
    base.when(col("a") < 0).then(2)
    assert len(base._finish().branches) == 1


def test_dangling_when_is_refused():
    with pytest.raises(PlanError, match="dangling"):
        bt.from_pydict({"a": [1]}).select(x=bt.when(col("a") > 1))


def test_over_on_a_row_level_expression_is_the_identity():
    expr = col("a") + 1
    assert expr.over("g") is expr


@pytest.mark.parametrize(
    "build",
    [
        lambda v: bt.coalesce(v.sum(), bt.lit(0)),
        lambda v: v.sum() + v.max(),
        lambda v: v.sum().sqrt(),
        lambda v: ListFilter(v.array_agg(), bt.element().is_not_null()),
    ],
    ids=["coalesce", "binary", "math", "list_filter"],
)
def test_over_binds_every_aggregate_inside_a_composed_expression(build):
    """The generic child rewrite reaches an aggregate under any node, and only aggregates."""
    from batcher.plan.expr_ir import AggExpr
    from batcher.plan.expr_rewrite import subexpressions

    bound = build(col("v")).over(partition_by=["g"], order_by=["t"], frame=(None, None))
    found = [e for e in subexpressions(bound) if isinstance(e, (AggExpr, WindowExpr))]
    assert found, "the composed expression holds an aggregate to bind"
    assert all(isinstance(e, WindowExpr) for e in found)
    assert all(e.partition_by == ["g"] and e.frame == (None, None) for e in found)


def test_over_merges_partitions_and_the_outer_order_wins():
    w = col("x").rank(partition_by="h").over("g", order_by="t")
    assert isinstance(w, WindowExpr)
    assert w.partition_by == ["g", "h"]
    assert w.order_by == ["t"]


def test_over_carries_direction_and_null_placement():
    w = col("x").shift(1).over(order_by=["t", "u"], descending=[True, False], nulls_last=False)
    assert w.order_by == [("t", True, True), ("u", False, True)]


def test_first_over_a_window_lowers_to_a_first_value():
    w = col("x").first().over("g", order_by="t")
    assert isinstance(w, WindowExpr)
    assert w.func == "first_value"
    assert len(w.order_by) == 3


def test_col_returns_a_column_or_a_selector():
    assert isinstance(col("a"), Col)
    assert isinstance(col(["a"]), Col)
    assert type(col("a", "b")).__name__ == "Selector"
    assert type(col("^a.*$")).__name__ == "Selector"
    assert type(col(pa.int64())).__name__ == "Selector"


def test_a_sorted_frame_lends_its_order_to_an_unordered_window():
    ds = bt.from_pydict({"t": [2, 1, 3], "x": [1.0, 2.0, 3.0]}).sort("t", descending=True)
    plan = ds.with_columns(prev=col("x").shift(1))._plan
    assert isinstance(plan, Window)
    assert [k.expr.name for k in plan.order_keys] == ["t"]
    assert plan.order_keys[0].descending
    out = ds.with_columns(prev=col("x").shift(1)).sort("t").to_pydict()
    assert out == {"t": [1, 2, 3], "x": [2.0, 1.0, 3.0], "prev": [1.0, 3.0, None]}


def test_a_rewritten_sort_key_ends_the_established_order():
    ds = bt.from_pydict({"t": [2, 1], "x": [1.0, 2.0]}).sort("t")
    assert established_order(ds.with_columns(t=col("t") * -1)._plan) == ()
    assert [k.expr.name for k in established_order(ds.select("t", "x")._plan)] == ["t"]
    with pytest.raises(PlanError, match="requires order_by"):
        ds.with_columns(t=col("t") * -1).with_columns(prev=col("x").shift(1))
