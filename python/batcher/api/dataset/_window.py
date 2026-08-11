"""Lowering of window expressions into the relational `Window` operator.

A `WindowExpr` (``col("x").sum().over(...)``, ``col("x").shift(1)``) is an `Expr`
so it composes like any scalar, but the engine has no scalar IR for it: window
functions are computed by the `Window` relational operator. This module is the
bridge. `plan.expr_rewrite.hoist_windows` does the expression half — pulling each
window out of the surrounding tree and leaving a `Col` behind — and the functions
here do the relational half: one `Window` node per hoisted window, chained so each
sees the columns the next one reads, then a `Project` that evaluates the rewritten
scalars and drops the synthetic columns.

That desugaring is why ``col("x") - col("x").shift(1)`` works with no new IR: it
becomes ``Project(x - __bt_win_0, Window(lag(x) AS __bt_win_0))`` — the exact plan
a SQL engine builds for ``x - lag(x) OVER ()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.api._join_helpers import _as_key_expr
from batcher.plan.expr_ir import Col, Expr, WindowExpr
from batcher.plan.expr_ir.walk import broadcast_aggregate_leaves, contains_aggregate
from batcher.plan.expr_rewrite import hoist_windows
from batcher.plan.logical import (
    Filter,
    LogicalPlan,
    Project,
    Projection,
    SortKeySpec,
    Window,
    WindowFrame,
    WindowFuncSpec,
)

if TYPE_CHECKING:
    from batcher.api.dataset.frame import Dataset

__all__ = [
    "build_window_columns",
    "windowed_filter",
    "windowed_project",
]


def _window_node(plan: LogicalPlan, alias: str, we: WindowExpr) -> Window:
    """One `Window` node computing `we` into a new column `alias` beside `plan`'s own."""
    part_keys = tuple(_as_key_expr(k) for k in we.partition_by)
    order_specs: list[SortKeySpec] = []
    for key in we.order_by:
        if isinstance(key, tuple):
            name, descending = key
            order_specs.append(SortKeySpec(_as_key_expr(name), descending=bool(descending)))
        else:
            order_specs.append(SortKeySpec(_as_key_expr(key)))
    frame = WindowFrame(*we.frame) if we.frame is not None else None
    spec = WindowFuncSpec(we.func, we.input, alias, we.offset, frame, we.alpha, we.half_life)
    return Window(plan, part_keys, tuple(order_specs), (spec,))


def build_window_columns(ds: Dataset, items: dict[str, WindowExpr]) -> Dataset:
    """Append bare ``agg.over(...)`` columns — one chained `Window` node each.

    The direct case, where every value is a window and needs no surrounding
    arithmetic: each window is named by its own alias, so no `Project` is needed on
    top. `windowed_project` handles the composed case.
    """
    plan = ds._plan
    for alias, we in items.items():
        plan = _window_node(plan, alias, we)
    return ds._derive(plan)


def _materialize(plan: LogicalPlan, hoisted: list[tuple[str, WindowExpr]]) -> LogicalPlan:
    """Chain one `Window` node per hoisted window, in dependency order."""
    for alias, we in hoisted:
        plan = _window_node(plan, alias, we)
    return plan


def windowed_project(ds: Dataset, items: list[Projection], *, collapse: bool = False) -> Dataset:
    """Project `items`, first materializing any window expression they compose.

    The synthetic window columns exist only between the `Window` nodes and this
    `Project`, so they never reach the output schema.

    An `AggExpr` among the items is not a scalar and has no scalar IR, so it is resolved
    first, in one of the two ways an aggregate can be meant in a row-shaped context:

    * **Every** item is an aggregate and `collapse` is set (a `select`): the projection
      *is* a whole-frame aggregation, so it lowers to `group_by().agg(...)` and returns
      one row. That is what ``ds.select(total=col("x").sum())`` means in Polars and
      pandas, and it used to raise here.
    * Otherwise: each aggregate leaf becomes ``agg.over()`` — the aggregate over the
      whole frame, broadcast to every row — which is the only reading under which
      ``with_columns(share=col("x") / col("x").sum())`` has a row per input row.

    Args:
        ds: The dataset being projected.
        items: The output projections, in order.
        collapse: Whether an all-aggregate projection may collapse to a single row
            (true for `select`, false for `with_columns`, which keeps its input's rows).

    Returns:
        A new `Dataset` with the projection applied.
    """
    if any(contains_aggregate(p.expr) for p in items):
        if collapse and all(contains_aggregate(p.expr) for p in items):
            return ds.group_by().agg(**{p.alias: p.expr for p in items})
        items = [Projection(p.alias, broadcast_aggregate_leaves(p.expr)) for p in items]
    exprs, hoisted = hoist_windows([p.expr for p in items])
    if not hoisted:
        return ds._derive(Project(ds._plan, tuple(items)))
    rewritten = tuple(Projection(p.alias, e) for p, e in zip(items, exprs, strict=True))
    return ds._derive(Project(_materialize(ds._plan, hoisted), rewritten))


def windowed_filter(ds: Dataset, predicate: Expr) -> Dataset:
    """Filter by `predicate`, first materializing any window expression it composes.

    ``filter(col("x") > col("x").mean().over(partition_by=["g"]))`` — keep rows above
    their group mean — lowers to ``Project(cols, Filter(Window(...)))``: the window
    sees every input row, exactly as in the SQL subquery this desugars to. A trailing
    `Project` restores the input schema by dropping the synthetic columns.
    """
    # An aggregate in a predicate is the whole-frame one, broadcast to every row:
    # ``filter(col("x") > col("x").mean())`` keeps the rows above the overall mean, the
    # reading Polars and pandas both give it. Without this it raised.
    if contains_aggregate(predicate):
        predicate = broadcast_aggregate_leaves(predicate)
    (rewritten,), hoisted = hoist_windows([predicate])
    if not hoisted:
        return ds._derive(Filter(ds._plan, predicate))
    keep = tuple(Projection(c, Col(c)) for c in ds._plan.available_columns())
    return ds._derive(Project(Filter(_materialize(ds._plan, hoisted), rewritten), keep))
