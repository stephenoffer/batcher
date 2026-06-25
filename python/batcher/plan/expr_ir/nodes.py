"""Leaf IR nodes the `Expr` base class does not construct.

`Col` (built by `col()`), `Case`/`CaseBuilder` (built by `when()`), and
`NullIf`/`Greatest`/`Least` (built by the matching free constructors) all subclass
`Expr` but are never created by an `Expr` method, so they live here rather than in
`core` — keeping `core` free of any dependency on this module (the edge points one
way: `nodes` → `core`).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from batcher.plan.expr_ir.core import Expr, IntoExpr, _wrap
from batcher.plan.expr_ir.node_base import IRNode, child, children, expr_node, scalar
from batcher.plan.ir_tags import ExprTag


@expr_node
class Col(IRNode):
    """A reference to an input column by name."""

    tag = ExprTag.COL
    name: str = scalar()


@expr_node
class Case(IRNode):
    """SQL CASE: first true branch wins, else `otherwise`."""

    tag = ExprTag.CASE
    branches: list[tuple[Expr, Expr]]
    otherwise: Expr

    def to_ir(self) -> dict[str, Any]:
        # Irregular shape (paired when/then branches), so to_ir is hand-written.
        return {
            "e": self.tag,
            "branches": [{"when": c.to_ir(), "then": t.to_ir()} for c, t in self.branches],
            "otherwise": self.otherwise.to_ir(),
        }


class CaseBuilder:
    """Fluent builder for CASE: `when(c).then(v).when(c2).then(v2).otherwise(d)`."""

    __slots__ = ("_branches", "_pending")

    def __init__(self) -> None:
        self._branches: list[tuple[Expr, Expr]] = []
        self._pending: Expr | None = None

    def when(self, cond: Expr) -> CaseBuilder:
        self._pending = cond
        return self

    def then(self, value: IntoExpr) -> CaseBuilder:
        if self._pending is None:
            raise ValueError("then() must follow when()")
        self._branches.append((self._pending, _wrap(value)))
        self._pending = None
        return self

    def otherwise(self, value: IntoExpr) -> Case:
        if self._pending is not None:
            raise ValueError("dangling when() without then()")
        return Case(self._branches, _wrap(value))


@expr_node
class NullIf(IRNode):
    """SQL NULLIF: null where `left == right`, else `left`."""

    tag = ExprTag.NULLIF
    left: Expr = child()
    right: Expr = child()


@expr_node
class Greatest(IRNode):
    """The largest argument per row, ignoring nulls (SQL GREATEST)."""

    tag = ExprTag.GREATEST
    inputs: list[Expr] = children()


@expr_node
class Least(IRNode):
    """The smallest argument per row, ignoring nulls (SQL LEAST)."""

    tag = ExprTag.LEAST
    inputs: list[Expr] = children()


@expr_node
class Array(IRNode):
    """An array literal `[e0, e1, …]`: each row becomes a list of the element
    values (coerced to a common type)."""

    tag = ExprTag.ARRAY
    elements: list[Expr] = children()


@expr_node
class Sequence(IRNode):
    """`sequence(start, stop, step)` — each row becomes a list of the integer series
    from ``start`` to ``stop`` inclusive, stepping by ``step`` (Spark ``sequence``).
    → ``List<Int64>``."""

    tag = ExprTag.SEQUENCE
    start: Expr = child()
    stop: Expr = child()
    step: Expr = child()


@expr_node
class MakeStruct(IRNode):
    """Struct construction: each row becomes a struct with the named fields, each
    field's value being the per-row value of its sub-expression (SQL ``struct_pack``;
    Spark ``struct``). Built by ``struct(**fields)`` / ``named_struct(...)``."""

    tag = ExprTag.MAKE_STRUCT
    fields: list[tuple[str, Expr]]

    def to_ir(self) -> dict[str, Any]:
        # Irregular shape (named fields), so to_ir is hand-written.
        return {
            "e": self.tag,
            "fields": [{"name": name, "value": value.to_ir()} for name, value in self.fields],
        }


@expr_node
class ListJoin(IRNode):
    """Concatenate a list column's elements (cast to text, nulls skipped) with a
    separator → text. Backs SQL ``string_agg`` over an ``array_agg`` input."""

    tag = ExprTag.LIST_JOIN
    input: Expr = child()
    separator: str = scalar()


class WindowExpr:
    """A window-function column built via ``agg.over(...)`` (e.g.
    ``col("x").sum().over(partition_by=["g"])``) or a value-function constructor
    (``lag(col("x"), 2).over(order_by=["t"])``).

    This is *not* an `Expr` — it is a control-plane builder consumed by
    `Dataset.with_columns`, which lowers it to the relational `Window` operator
    (SQL ``<fn> OVER (PARTITION BY … ORDER BY …)``). `func` is the engine window-fn
    tag (aggregates ``sum``/``avg``/``min``/``max``/``count``; value functions
    ``lag``/``lead``/``first_value``/``last_value``); `input` is the argument
    expression; `offset` is the lag/lead distance; `frame` is an optional
    ``(start, end)`` ROWS frame (aggregates only).
    """

    __slots__ = ("frame", "func", "input", "offset", "order_by", "partition_by")

    def __init__(
        self,
        func: str,
        input: Expr | None,
        partition_by: list[Any],
        order_by: list[Any],
        frame: tuple[int | None, int | None] | None,
        offset: int = 1,
    ) -> None:
        self.func = func
        self.input = input
        self.partition_by = partition_by
        self.order_by = order_by
        self.frame = frame
        self.offset = offset

    def over(
        self,
        partition_by: Iterable[Any] = (),
        order_by: Iterable[Any] = (),
        frame: tuple[int | None, int | None] | None = None,
    ) -> WindowExpr:
        """Bind this window function to a partition/order (and optional frame).

        Lets a value-function constructor read fluently:
        ``lag(col("x"), 2).over(partition_by=["g"], order_by=["t"])``. Returns a new
        `WindowExpr`; the original is unchanged."""
        return WindowExpr(
            self.func,
            self.input,
            list(partition_by),
            list(order_by),
            frame if frame is not None else self.frame,
            self.offset,
        )


def lag(expr: IntoExpr, n: int = 1) -> WindowExpr:
    """The value `n` rows before the current row in the ordered partition (SQL
    ``LAG``). Bind the window with ``.over(partition_by=…, order_by=…)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [10, 20, 30]})
            >>> ds.with_columns(r=bt.lag(bt.col("x")).over(order_by=["x"])).select("r").to_pydict()
            {'r': [None, 10, 20]}
    """
    return WindowExpr("lag", _wrap(expr), [], [], None, int(n))


def lead(expr: IntoExpr, n: int = 1) -> WindowExpr:
    """The value `n` rows after the current row in the ordered partition (SQL
    ``LEAD``). Bind the window with ``.over(partition_by=…, order_by=…)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [10, 20, 30]})
            >>> ds.with_columns(r=bt.lead(bt.col("x")).over(order_by=["x"])).select("r").to_pydict()
            {'r': [20, 30, None]}
    """
    return WindowExpr("lead", _wrap(expr), [], [], None, int(n))


def first_value(expr: IntoExpr) -> WindowExpr:
    """The first value of the ordered partition (SQL ``FIRST_VALUE``).

    Reads the **whole partition** (``ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED
    FOLLOWING``), not the running frame, so every row of a partition gets the same
    value. Bind with ``.over(partition_by=…, order_by=…)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [10, 20, 30]})
            >>> w = bt.first_value(bt.col("x")).over(order_by=["x"])
            >>> ds.with_columns(r=w).select("r").to_pydict()
            {'r': [10, 10, 10]}

    Args:
        expr: The column (or expression) to read the first value of.
    """
    return WindowExpr("first_value", _wrap(expr), [], [], None)


def last_value(expr: IntoExpr) -> WindowExpr:
    """The last value of the ordered partition (SQL ``LAST_VALUE``).

    Reads the **whole partition** (``ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED
    FOLLOWING``), not the running frame — so this is the partition's final value, the
    same for every row, not a running "last seen so far". Bind with
    ``.over(partition_by=…, order_by=…)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [10, 20, 30]})
            >>> w = bt.last_value(bt.col("x")).over(order_by=["x"])
            >>> ds.with_columns(r=w).select("r").to_pydict()
            {'r': [30, 30, 30]}

    Args:
        expr: The column (or expression) to read the last value of.
    """
    return WindowExpr("last_value", _wrap(expr), [], [], None)


def nth_value(expr: IntoExpr, n: int) -> WindowExpr:
    """The value of the ``n``-th row (1-based) of the ordered partition (SQL
    ``NTH_VALUE``); null if the partition has fewer than ``n`` rows. Bind with
    ``.over(partition_by=…, order_by=…)``.

    Like :func:`first_value`/:func:`last_value`, this reads the **whole partition**
    (equivalent to ``ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING``), not
    the running frame — so the ``n``-th value is the same for every row of a partition.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [10, 20, 30]})
            >>> w = bt.nth_value(bt.col("x"), 2).over(order_by=["x"])
            >>> ds.with_columns(r=w).select("r").to_pydict()
            {'r': [20, 20, 20]}
    """
    if n < 1:
        from batcher._internal.errors import PlanError

        raise PlanError(f"nth_value(n) requires n >= 1, got {n}")
    return WindowExpr("nth_value", _wrap(expr), [], [], None, int(n))


def row_number() -> WindowExpr:
    """Sequential 1-based row number within the ordered partition (SQL
    ``ROW_NUMBER``). Takes no input; bind with ``.over(partition_by=…, order_by=…)``
    — ``order_by`` is required.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [10, 20, 30]})
            >>> ds.with_columns(r=bt.row_number().over(order_by=["x"])).select("r").to_pydict()
            {'r': [1, 2, 3]}
    """
    return WindowExpr("row_number", None, [], [], None)


def rank() -> WindowExpr:
    """Rank within the ordered partition, with gaps after ties (SQL ``RANK``):
    peers share the minimum rank and the next distinct value skips ahead. Takes no
    input; bind with ``.over(partition_by=…, order_by=…)`` — ``order_by`` is
    required.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [10, 10, 30]})
            >>> ds.with_columns(r=bt.rank().over(order_by=["x"])).select("r").to_pydict()
            {'r': [1, 1, 3]}
    """
    return WindowExpr("rank", None, [], [], None)


def dense_rank() -> WindowExpr:
    """Rank within the ordered partition with no gaps after ties (SQL
    ``DENSE_RANK``): peers share a rank and the next distinct value increments by
    one. Takes no input; bind with ``.over(partition_by=…, order_by=…)`` —
    ``order_by`` is required.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [10, 10, 30]})
            >>> ds.with_columns(r=bt.dense_rank().over(order_by=["x"])).select("r").to_pydict()
            {'r': [1, 1, 2]}
    """
    return WindowExpr("dense_rank", None, [], [], None)


def percent_rank() -> WindowExpr:
    """Relative rank within the ordered partition (SQL ``PERCENT_RANK``):
    ``(rank - 1) / (rows - 1)``, in ``[0, 1]``; ``0`` for a single-row partition.
    Takes no input; bind with ``.over(partition_by=…, order_by=…)`` — ``order_by``
    is required.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [10, 20, 30]})
            >>> ds.with_columns(r=bt.percent_rank().over(order_by=["x"])).select("r").to_pydict()
            {'r': [0.0, 0.5, 1.0]}
    """
    return WindowExpr("percent_rank", None, [], [], None)


def cume_dist() -> WindowExpr:
    """Cumulative distribution within the ordered partition (SQL ``CUME_DIST``): the
    fraction of rows at or before the current row's peer group, in ``(0, 1]``. Takes
    no input; bind with ``.over(partition_by=…, order_by=…)`` — ``order_by`` is
    required.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [10, 20, 30]})
            >>> ds.with_columns(r=bt.cume_dist().over(order_by=["x"])).select("r").to_pydict()
            {'r': [0.3333333333333333, 0.6666666666666666, 1.0]}
    """
    return WindowExpr("cume_dist", None, [], [], None)


def ntile(n: int) -> WindowExpr:
    """Distribute the ordered partition into `n` buckets numbered ``1..n`` as evenly
    as possible (SQL ``NTILE(n)``): earlier buckets take the remainder, and with
    fewer rows than buckets each row is its own bucket. Takes no input; bind with
    ``.over(partition_by=…, order_by=…)`` — ``order_by`` is required.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [10, 20, 30, 40]})
            >>> ds.with_columns(r=bt.ntile(2).over(order_by=["x"])).select("r").to_pydict()
            {'r': [1, 1, 2, 2]}
    """
    if n < 1:
        from batcher._internal.errors import PlanError

        raise PlanError(f"ntile(n) requires n >= 1, got {n}")
    return WindowExpr("ntile", None, [], [], None, int(n))
