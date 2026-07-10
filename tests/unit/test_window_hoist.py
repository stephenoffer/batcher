"""`hoist_windows` lifts window expressions out of scalar trees, and the plan shows it.

Result-level parity with DuckDB lives in `tests/differential/test_diff_window_compose.py`.
These are the structural facts that parity depends on: a composed window becomes
``Project(… , Window(…))``; a *bare* window keeps the leaner `Window`-only shape it
always had; inner windows materialize before the outer ones that read them; and a
window that escapes the hoist (a `group_by().agg()` argument, where SQL forbids it too)
raises rather than lowering to invalid IR.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import Col, WindowExpr
from batcher.plan.expr_rewrite import hoist_windows, is_bare_window
from batcher.plan.logical import Project, Window

pytestmark = pytest.mark.unit


@pytest.fixture
def ds():
    return bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 3]})


def test_hoist_leaves_a_window_free_expression_untouched():
    expr = bt.col("x") + 1
    (rewritten,), hoisted = hoist_windows([expr])
    assert rewritten is expr
    assert hoisted == []


def test_hoist_replaces_a_window_with_a_column_reference():
    (rewritten,), hoisted = hoist_windows([bt.col("x") - bt.col("x").shift(1)])
    assert len(hoisted) == 1
    name, window = hoisted[0]
    assert isinstance(window, WindowExpr)
    assert window.func == "lag"
    # The window is gone from the scalar tree, replaced by its synthetic column.
    assert isinstance(rewritten.right, Col)
    assert rewritten.right.name == name


def test_hoist_orders_an_inner_window_before_the_outer_one():
    """`sum(lag(x))` — the outer window reads the column the inner one produces."""
    _, hoisted = hoist_windows([bt.col("x").shift(1).cum_sum()])
    assert [w.func for _, w in hoisted] == ["lag", "sum"]
    inner_name, _ = hoisted[0]
    _, outer = hoisted[1]
    assert isinstance(outer.input, Col) and outer.input.name == inner_name


def test_hoist_numbers_each_window_distinctly():
    _, hoisted = hoist_windows([bt.col("x").shift(1) + bt.col("x").shift(-1)])
    names = [n for n, _ in hoisted]
    assert len(names) == 2 and len(set(names)) == 2


def test_is_bare_window_distinguishes_composed_from_bare():
    assert is_bare_window(bt.col("x").cum_sum())
    assert not is_bare_window(bt.col("x") + 1)
    assert not is_bare_window(bt.col("x").shift(1).cum_sum())  # nested → needs hoisting


def test_bare_window_keeps_the_window_only_plan_shape(ds):
    """The pre-existing fast path: no `Project` is added on top."""
    plan = ds.with_columns(cs=bt.col("x").cum_sum())._plan
    assert isinstance(plan, Window)


def test_composed_window_lowers_to_project_over_window(ds):
    plan = ds.with_columns(d=bt.col("x") - bt.col("x").shift(1))._plan
    assert isinstance(plan, Project)
    assert isinstance(plan.input, Window)


def test_composed_window_drops_its_synthetic_column(ds):
    assert ds.with_columns(d=bt.col("x") - bt.col("x").shift(1)).columns == ["g", "x", "d"]


def test_two_windows_lower_to_two_chained_window_nodes(ds):
    plan = ds.with_columns(z=bt.col("x").shift(1) + bt.col("x").shift(-1))._plan
    assert isinstance(plan, Project)
    assert isinstance(plan.input, Window)
    assert isinstance(plan.input.input, Window)


def test_window_in_a_group_by_aggregate_raises(ds):
    """SQL forbids a window inside an aggregate; so does Batcher, loudly."""
    with pytest.raises(PlanError, match="only valid in select"):
        ds.group_by("g").agg(s=bt.col("x").shift(1).sum()).to_pydict()


def test_window_to_ir_raises_directly():
    with pytest.raises(PlanError, match="is not allowed here"):
        bt.col("x").cum_sum().to_ir()
