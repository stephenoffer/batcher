"""Module-level expression constructors (the user-facing entry points).

`col`, `lit`, `when`, `coalesce`, `nullif`, `atan2`, `greatest`, `least`, and
`count` build expression trees out of the node classes in `core`. These are the
free functions users call directly (e.g. `col("x")`, `when(c).then(v)`).
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import (
    AggExpr,
    Coalesce,
    Expr,
    IntoExpr,
    Lit,
    Math2Expr,
    _wrap,
)
from batcher.plan.expr_ir.nodes import Array, CaseBuilder, Col, Greatest, Least, NullIf


def when(cond: Expr) -> CaseBuilder:
    """Begin a CASE expression: `when(cond).then(v).otherwise(d)`."""
    return CaseBuilder().when(cond)


def array(*elements: IntoExpr) -> Array:
    """A list literal built per row from the element expressions (SQL ``ARRAY[...]``).

    Each output row is a list of the per-row element values, coerced to a common
    type: ``array(col("a"), col("b"))`` over ``a=1, b=2`` yields ``[1, 2]``.
    """
    if not elements:
        raise ValueError("array() requires at least one element")
    return Array([_wrap(e) for e in elements])


def coalesce(*exprs: IntoExpr) -> Coalesce:
    """First non-null among the arguments, per row (SQL COALESCE)."""
    if not exprs:
        raise ValueError("coalesce() requires at least one argument")
    return Coalesce([_wrap(e) for e in exprs])


def nullif(left: IntoExpr, right: IntoExpr) -> NullIf:
    """Null where ``left == right``, else ``left`` (SQL NULLIF)."""
    return NullIf(_wrap(left), _wrap(right))


def atan2(y: IntoExpr, x: IntoExpr) -> Math2Expr:
    """Two-argument arctangent of ``y/x`` (→ Float64)."""
    return Math2Expr("atan2", _wrap(y), _wrap(x))


def greatest(*exprs: IntoExpr) -> Greatest:
    """The largest argument per row, ignoring nulls (SQL GREATEST)."""
    if not exprs:
        raise ValueError("greatest() requires at least one argument")
    return Greatest([_wrap(e) for e in exprs])


def least(*exprs: IntoExpr) -> Least:
    """The smallest argument per row, ignoring nulls (SQL LEAST)."""
    if not exprs:
        raise ValueError("least() requires at least one argument")
    return Least([_wrap(e) for e in exprs])


def col(name: str) -> Col:
    """Reference an input column by name."""
    return Col(name)


def count() -> AggExpr:
    """COUNT(*) — number of rows per group."""
    return AggExpr("count_star", None)


def lit(value: int | float | bool | str) -> Lit:
    """A constant literal expression."""
    return Lit(value)
