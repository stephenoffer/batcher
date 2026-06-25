"""Conditional / null-handling free functions (`iff`, `nanvl`).

Thin sugar over `when().then().otherwise()` for the common binary cases users reach
for from DuckDB/Spark. No new IR.
"""

from __future__ import annotations

from batcher.plan.expr_ir.constructors import when
from batcher.plan.expr_ir.core import Expr, IntoExpr, _wrap


def iff(condition: Expr, if_true: IntoExpr, if_false: IntoExpr) -> Expr:
    """``if_true`` where `condition` is true, else ``if_false`` (DuckDB ``IF``/``IFF``).

    The two-branch shorthand for ``when(condition).then(if_true).otherwise(if_false)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [-1, 2]})
            >>> ds.select(s=bt.iff(bt.col("x") > 0, bt.lit("pos"), bt.lit("neg"))).to_pydict()
            {'s': ['neg', 'pos']}
    """
    return when(condition).then(_wrap(if_true)).otherwise(_wrap(if_false))


def nanvl(value: IntoExpr, fallback: IntoExpr) -> Expr:
    """`value` unless it is NaN, in which case `fallback` (Spark ``nanvl``).

    Distinct from `coalesce` — this replaces IEEE NaN, not NULL. A NULL `value`
    passes through unchanged (NULL is not NaN).

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1.0, float("nan")]})
            >>> ds.select(r=bt.nanvl(bt.col("x"), bt.lit(0.0))).to_pydict()
            {'r': [1.0, 0.0]}
    """
    v = _wrap(value)
    return when(v.is_nan()).then(_wrap(fallback)).otherwise(v)
