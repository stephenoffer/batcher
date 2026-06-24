"""Aggregate free functions that compose existing mergeable aggregates.

`count_if` desugars to ``sum(iff(cond, 1, 0))`` — counting the rows where a
predicate holds reuses the mergeable `sum` aggregate, so it stays identical
single-node and distributed with no new engine state.
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import AggExpr, Expr, IntoExpr, Lit, _wrap
from batcher.plan.functions.conditional import iff


def corr(x: IntoExpr, y: IntoExpr) -> AggExpr:
    """Pearson correlation coefficient of two columns (DuckDB/Spark ``corr``).

    Mergeable (6-column sum-of-powers state), so identical single-node and
    distributed. Null when a group has fewer than 2 paired values or either column
    is constant. Symmetric in `x` and `y`."""
    return AggExpr("corr", _wrap(x), input2=_wrap(y))


def covar_pop(x: IntoExpr, y: IntoExpr) -> AggExpr:
    """Population covariance of two columns (DuckDB/Spark ``covar_pop``). Mergeable;
    null when a group has no paired values. Symmetric in `x` and `y`."""
    return AggExpr("covar_pop", _wrap(x), input2=_wrap(y))


def covar_samp(x: IntoExpr, y: IntoExpr) -> AggExpr:
    """Sample covariance of two columns (DuckDB/Spark ``covar_samp``). Mergeable;
    null when a group has fewer than 2 paired values. Symmetric in `x` and `y`."""
    return AggExpr("covar_samp", _wrap(x), input2=_wrap(y))


def count_if(condition: Expr) -> AggExpr:
    """Count the rows in each group where `condition` is true (DuckDB ``count_if``;
    Spark ``count_if``).

    A NULL condition is treated as false (not counted), matching DuckDB. Use inside
    ``group_by(...).agg(...)`` or ``agg(...)``::

        ds.group_by("dept").agg(n_high=count_if(col("salary") > 100_000))
    """
    return iff(condition, Lit(1), Lit(0)).sum()
