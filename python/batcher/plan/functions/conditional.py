"""Conditional / null-handling free functions (`iff`, `nanvl`, `ifnull`).

Thin sugar over `when().then().otherwise()` and `coalesce` for the common binary
cases users reach for from DuckDB/Spark. No new IR.
"""

from __future__ import annotations

from batcher.plan.expr_ir.constructors import when
from batcher.plan.expr_ir.core import Coalesce, Expr, IntoExpr, _wrap


def iff(condition: Expr, if_true: IntoExpr, if_false: IntoExpr) -> Expr:
    """``if_true`` where `condition` is true, else ``if_false`` (DuckDB ``IF``/``IFF``).

    The two-branch shorthand for ``when(condition).then(if_true).otherwise(if_false)``.
    """
    return when(condition).then(_wrap(if_true)).otherwise(_wrap(if_false))


def nanvl(value: IntoExpr, fallback: IntoExpr) -> Expr:
    """`value` unless it is NaN, in which case `fallback` (Spark ``nanvl``).

    Distinct from :func:`ifnull` — this replaces IEEE NaN, not NULL. A NULL `value`
    passes through unchanged (NULL is not NaN).
    """
    v = _wrap(value)
    return when(v.is_nan()).then(_wrap(fallback)).otherwise(v)


def ifnull(value: IntoExpr, fallback: IntoExpr) -> Expr:
    """`value` unless it is NULL, in which case `fallback` (DuckDB ``ifnull``).

    The two-argument spelling of `coalesce`.
    """
    return Coalesce([_wrap(value), _wrap(fallback)])
