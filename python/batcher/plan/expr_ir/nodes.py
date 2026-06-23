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
from batcher.plan.ir_tags import ExprTag


class Col(Expr):
    """A reference to an input column by name."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def to_ir(self) -> dict[str, Any]:
        return {"e": ExprTag.COL, "name": self.name}


class Case(Expr):
    """SQL CASE: first true branch wins, else `otherwise`."""

    __slots__ = ("branches", "otherwise")

    def __init__(self, branches: list[tuple[Expr, Expr]], otherwise: Expr) -> None:
        self.branches = branches
        self.otherwise = otherwise

    def to_ir(self) -> dict[str, Any]:
        return {
            "e": ExprTag.CASE,
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


class NullIf(Expr):
    """SQL NULLIF: null where `left == right`, else `left`."""

    __slots__ = ("left", "right")

    def __init__(self, left: Expr, right: Expr) -> None:
        self.left = left
        self.right = right

    def to_ir(self) -> dict[str, Any]:
        return {"e": ExprTag.NULLIF, "left": self.left.to_ir(), "right": self.right.to_ir()}


class Greatest(Expr):
    """The largest argument per row, ignoring nulls (SQL GREATEST)."""

    __slots__ = ("inputs",)

    def __init__(self, inputs: list[Expr]) -> None:
        self.inputs = inputs

    def to_ir(self) -> dict[str, Any]:
        return {"e": ExprTag.GREATEST, "inputs": [e.to_ir() for e in self.inputs]}


class Least(Expr):
    """The smallest argument per row, ignoring nulls (SQL LEAST)."""

    __slots__ = ("inputs",)

    def __init__(self, inputs: list[Expr]) -> None:
        self.inputs = inputs

    def to_ir(self) -> dict[str, Any]:
        return {"e": ExprTag.LEAST, "inputs": [e.to_ir() for e in self.inputs]}


class Array(Expr):
    """An array literal `[e0, e1, …]`: each row becomes a list of the element
    values (coerced to a common type)."""

    __slots__ = ("elements",)

    def __init__(self, elements: list[Expr]) -> None:
        self.elements = elements

    def to_ir(self) -> dict[str, Any]:
        return {"e": ExprTag.ARRAY, "elements": [e.to_ir() for e in self.elements]}


class ListJoin(Expr):
    """Concatenate a list column's elements (cast to text, nulls skipped) with a
    separator → text. Backs SQL ``string_agg`` over an ``array_agg`` input."""

    __slots__ = ("input", "separator")

    def __init__(self, input: Expr, separator: str) -> None:
        self.input = input
        self.separator = separator

    def to_ir(self) -> dict[str, Any]:
        return {"e": ExprTag.LIST_JOIN, "input": self.input.to_ir(), "separator": self.separator}


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
    ``LAG``). Bind the window with ``.over(partition_by=…, order_by=…)``."""
    return WindowExpr("lag", _wrap(expr), [], [], None, int(n))


def lead(expr: IntoExpr, n: int = 1) -> WindowExpr:
    """The value `n` rows after the current row in the ordered partition (SQL
    ``LEAD``). Bind the window with ``.over(partition_by=…, order_by=…)``."""
    return WindowExpr("lead", _wrap(expr), [], [], None, int(n))


def first_value(expr: IntoExpr) -> WindowExpr:
    """The first value in the ordered partition (SQL ``FIRST_VALUE``)."""
    return WindowExpr("first_value", _wrap(expr), [], [], None)


def last_value(expr: IntoExpr) -> WindowExpr:
    """The last value in the ordered partition so far (SQL ``LAST_VALUE``)."""
    return WindowExpr("last_value", _wrap(expr), [], [], None)


def row_number() -> WindowExpr:
    """Sequential 1-based row number within the ordered partition (SQL
    ``ROW_NUMBER``). Takes no input; bind with ``.over(partition_by=…, order_by=…)``
    — ``order_by`` is required."""
    return WindowExpr("row_number", None, [], [], None)


def rank() -> WindowExpr:
    """Rank within the ordered partition, with gaps after ties (SQL ``RANK``):
    peers share the minimum rank and the next distinct value skips ahead. Takes no
    input; bind with ``.over(partition_by=…, order_by=…)`` — ``order_by`` is
    required."""
    return WindowExpr("rank", None, [], [], None)


def dense_rank() -> WindowExpr:
    """Rank within the ordered partition with no gaps after ties (SQL
    ``DENSE_RANK``): peers share a rank and the next distinct value increments by
    one. Takes no input; bind with ``.over(partition_by=…, order_by=…)`` —
    ``order_by`` is required."""
    return WindowExpr("dense_rank", None, [], [], None)


def percent_rank() -> WindowExpr:
    """Relative rank within the ordered partition (SQL ``PERCENT_RANK``):
    ``(rank - 1) / (rows - 1)``, in ``[0, 1]``; ``0`` for a single-row partition.
    Takes no input; bind with ``.over(partition_by=…, order_by=…)`` — ``order_by``
    is required."""
    return WindowExpr("percent_rank", None, [], [], None)


def cume_dist() -> WindowExpr:
    """Cumulative distribution within the ordered partition (SQL ``CUME_DIST``): the
    fraction of rows at or before the current row's peer group, in ``(0, 1]``. Takes
    no input; bind with ``.over(partition_by=…, order_by=…)`` — ``order_by`` is
    required."""
    return WindowExpr("cume_dist", None, [], [], None)


def ntile(n: int) -> WindowExpr:
    """Distribute the ordered partition into `n` buckets numbered ``1..n`` as evenly
    as possible (SQL ``NTILE(n)``): earlier buckets take the remainder, and with
    fewer rows than buckets each row is its own bucket. Takes no input; bind with
    ``.over(partition_by=…, order_by=…)`` — ``order_by`` is required."""
    if n < 1:
        from batcher._internal.errors import PlanError

        raise PlanError(f"ntile(n) requires n >= 1, got {n}")
    return WindowExpr("ntile", None, [], [], None, int(n))
