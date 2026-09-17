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
from batcher.plan.expr_ir import Col, Expr, WindowExpr, referenced_columns
from batcher.plan.expr_ir.walk import broadcast_aggregate_leaves, contains_aggregate
from batcher.plan.expr_rewrite import contained_types, hoist_windows
from batcher.plan.logical import (
    Filter,
    Limit,
    LogicalPlan,
    Project,
    Projection,
    RowId,
    Sort,
    SortKeySpec,
    Window,
    WindowFrame,
    WindowFuncSpec,
)
from batcher.plan.logical.window import depends_on_row_order, sql_default_frame

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
            # `(key, descending)` or `(key, descending, nulls_first)`, the same shapes
            # `Dataset.window` reads. The two-element unpack used to raise a bare
            # ValueError on the third, so Spark's nulls-first window order (`orderBy`
            # on an ascending key) had no spelling on `over`.
            name, descending, *rest = key
            nulls_first = bool(rest[0]) if rest else False
            order_specs.append(
                SortKeySpec(
                    _as_key_expr(name), descending=bool(descending), nulls_first=nulls_first
                )
            )
        else:
            order_specs.append(SortKeySpec(_as_key_expr(key)))
    frame = WindowFrame(*we.frame) if we.frame is not None else None
    probe = WindowFuncSpec(we.func, we.input, alias, we.offset, frame, we.alpha, we.half_life)
    if not order_specs and depends_on_row_order(probe):
        # Decided before the SQL default frame is applied: that frame is a consequence of
        # having an order, not evidence that the function needs one.
        order_specs.extend(established_order(plan))
    frame = sql_default_frame(we.func, frame, bool(order_specs), we.ignore_nulls)
    spec = WindowFuncSpec(
        we.func, we.input, alias, we.offset, frame, we.alpha, we.half_life, we.ignore_nulls
    )
    return Window(plan, part_keys, tuple(order_specs), (spec,))


def established_order(plan: LogicalPlan) -> tuple[SortKeySpec, ...]:
    """The sort keys `plan`'s rows are already ordered by, when that is certain; else none.

    ``ds.sort("t").with_columns(prev=col("x").shift(1))`` asks for the previous row of a frame
    whose order the query itself established, so the window takes that order as its own
    ``ORDER BY`` rather than refusing. The window then *sorts by the keys itself*, which is
    what keeps the answer the same on a parallel or distributed run, where the physical order
    of rows reaching it is not the sort's.

    Only the nodes that cannot reorder rows or redefine a key are looked through: filters,
    limits, row indexes, earlier windows (which add columns and never replace one), and
    projections that pass every key column through unchanged. Anything else -- a join, an
    aggregate, a projection that rewrites a key -- ends the search with no order.

    Args:
        plan: The plan a window is about to be built over.

    Returns:
        The established sort keys, or an empty tuple.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.api.dataset._window import established_order
            >>> ds = bt.from_pydict({"t": [2, 1], "x": [1, 2]})
            >>> [k.expr.name for k in established_order(ds.sort("t").filter(bt.col("x") > 0)._plan)]
            ['t']
            >>> established_order(ds._plan)
            ()
    """
    node = plan
    kept: list[Project] = []
    while not isinstance(node, Sort):
        if isinstance(node, Project):
            kept.append(node)
        elif not isinstance(node, (Filter, Limit, RowId, Window)):
            return ()
        node = node.input
    keys = node.keys
    needed = set().union(*(referenced_columns(k.expr) for k in keys))
    for project in kept:
        passed = {
            p.alias for p in project.items if isinstance(p.expr, Col) and p.expr.name == p.alias
        }
        if not needed <= passed:
            return ()
    return keys


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

    * **Every** item is an aggregate or a constant, and `collapse` is set (a `select`):
      the projection *is* a whole-frame aggregation, so it lowers to `group_by().agg(...)`
      and returns one row. That is what ``ds.select(total=col("x").sum())`` means in
      Polars and pandas, and ``SELECT sum(x), 1 FROM t`` is one row in DuckDB too. A
      projection of constants alone keeps a row per input row, as DuckDB's
      ``SELECT 1 FROM t`` does (Polars returns one row there; DuckDB is the default).
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
        if collapse and all(_collapses(p.expr) for p in items):
            aggregates = {p.alias: p.expr for p in items if contains_aggregate(p.expr)}
            folded = ds.group_by().agg(**aggregates)
            # The constants ride above the one-row aggregate, in the caller's column order.
            outputs = tuple(
                Projection(p.alias, Col(p.alias) if p.alias in aggregates else p.expr)
                for p in items
            )
            return folded._derive(Project(folded._plan, outputs))
        items = [Projection(p.alias, broadcast_aggregate_leaves(p.expr)) for p in items]
    exprs, hoisted = hoist_windows([p.expr for p in items])
    if not hoisted:
        return ds._derive(Project(ds._plan, tuple(items)))
    rewritten = tuple(Projection(p.alias, e) for p, e in zip(items, exprs, strict=True))
    return ds._derive(Project(_materialize(ds._plan, hoisted), rewritten))


def _collapses(expr: Expr) -> bool:
    """Whether `expr` may sit in a one-row whole-frame aggregation: an aggregate or a constant.

    A constant reads no column and no window, so it has one value for the whole frame and
    rides along beside the aggregates. `group_by().agg()` accepts it as an expression over
    zero aggregates, computed in the projection above the aggregate pass.
    """
    if contains_aggregate(expr):
        return True
    return not ({Col, WindowExpr} & contained_types(expr))


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
