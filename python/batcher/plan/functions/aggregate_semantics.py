"""The aggregate parameters that restore another engine's semantics by composition.

Layer 1 (`plan`), neutral. Batcher's aggregates follow DuckDB, the SQL oracle. Where
Polars, Spark, Daft or Ray Data answer the same aggregate differently, the difference is a
*parameter* on Batcher's one spelling (``col("x").sum(empty_value=0)``), never a second
name. Most of those parameters need no engine state of their own: they are an expression
over aggregates the engine already computes, which `group_by().agg()` runs as the same
mergeable pass plus a projection. This module holds those compositions, so the fluent
methods on `Expr` stay small and the reasoning for each composition is written once.

A parameter left at its default never reaches this module: the method returns the plain
`AggExpr`, so an existing plan serializes exactly as it did.

The parameters that do need state (`quantile(interpolation=)`, `skew(bias=True)`,
`mode(all_modes=True)`, `arg_min(ignore_nulls=False)`) are engine kernels instead, and live
beside the aggregate they extend in `bc-runtime`.
"""

from __future__ import annotations

import math

from batcher._internal.errors import PlanError, require_float, require_int
from batcher.plan.expr_ir.constructors import count, lit, when
from batcher.plan.expr_ir.core import AggExpr, Coalesce, Expr, MathExpr
from batcher.plan.expr_ir.func_nodes import ListFilter
from batcher.plan.expr_ir.nodes import NullIf

__all__: list[str] = []

#: What `max(nan_policy=...)` accepts: SQL's total order, where NaN is the greatest float
#: and so wins a `max`, or Polars', where a NaN is skipped unless nothing else is left.
NAN_POLICIES = ("propagate", "ignore")


def with_empty_value(agg: AggExpr, empty_value: object) -> Expr:
    """`agg`, answering `empty_value` for a group with no non-null input instead of null.

    Polars sums an all-null or empty column to ``0``, multiplies one to ``1``, and folds
    ``all``/``any`` over one to ``true``/``false``; DuckDB answers null for each. Every one
    of these aggregates is null *only* in that case, so coalescing the result is exact.
    """
    return Coalesce([agg, lit(empty_value)])  # type: ignore[list-item]


def count_nulls_as_value(agg: AggExpr, column: Expr) -> Expr:
    """`agg` (a distinct count over `column`) that also counts null as one distinct value.

    Polars' `n_unique` and `approx_n_unique` count null as a value; SQL's
    ``COUNT(DISTINCT x)`` skips it. A group holds a null exactly when it has more rows than
    non-null values, so one ``count(*) > count(x)`` beside the distinct count adds it back,
    with both counts as mergeable as the distinct count itself.
    """
    has_null = when(count() > column.count()).then(lit(1)).otherwise(lit(0))  # type: ignore[arg-type]
    return agg + has_null


def nan_ignoring_max(column: Expr, nan_policy: str) -> AggExpr | Expr:
    """`max` of `column` under `nan_policy`: SQL's ``"propagate"`` or Polars' ``"ignore"``.

    Under SQL's total order a NaN is the greatest float, so one NaN decides `max`. Polars
    skips NaN and answers NaN only for a group holding nothing else. That is the maximum
    over the non-NaN values, falling back to the plain maximum when there are none, which is
    NaN precisely when the group held a NaN.
    """
    if nan_policy not in NAN_POLICIES:
        raise PlanError(f"max(nan_policy=...) must be one of {NAN_POLICIES}, got {nan_policy!r}")
    if nan_policy == "propagate":
        return AggExpr("max", column)
    # The null is `nullif(x, x)` rather than a bare null literal so it carries the column's
    # own type: `is_not_nan` is null on a non-float column, and a CASE whose branches are a
    # string and an untyped null does not evaluate. On such a column every value masks to
    # null and the fallback maximum answers, which is right: there was no NaN to skip.
    typed_null = NullIf(column, column)
    without_nan = when(column.is_not_nan()).then(column).otherwise(typed_null)
    return Coalesce([AggExpr("max", without_nan), AggExpr("max", column)])  # type: ignore[list-item]


def entropy_of(column: Expr, base: float, of: str, normalize: bool) -> AggExpr | Expr:
    """Shannon entropy of `column` in `base`, of its value frequencies or of its values.

    ``of="frequencies"`` is DuckDB's `entropy`: the distribution of how often each distinct
    value occurs. ``of="values"`` is Polars' `entropy`: the column itself read as the
    probabilities, normalized to sum to one when `normalize` is set. With ``S = sum(x)``
    that is ``ln S - sum(x ln x) / S``, and ``-sum(x ln x)`` without normalizing, each
    divided by ``ln(base)``. A zero or negative value makes ``x ln x`` NaN, as it does in
    Polars, and a group with no values answers ``0``, as Polars' does.
    """
    base = require_float(base, func="entropy", arg="base")
    if not (base > 0.0 and base != 1.0 and math.isfinite(base)):
        raise PlanError(f"entropy(base=...) must be positive, finite and not 1, got {base}")
    if of == "frequencies":
        if not normalize:
            raise PlanError(
                "entropy(normalize=False) reads the values as probabilities; it needs of='values'"
            )
        bits = AggExpr("entropy", column)
        return bits if base == 2.0 else bits * lit(math.log(2.0) / math.log(base))
    if of != "values":
        raise PlanError(f"entropy(of=...) must be 'frequencies' or 'values', got {of!r}")
    x = column.cast("float64")
    x_ln_x = AggExpr("sum", x * x.ln())
    if normalize:
        total = AggExpr("sum", x)
        nats = MathExpr("ln", total) - x_ln_x / total  # type: ignore[arg-type]
    else:
        nats = -x_ln_x
    return Coalesce([nats / lit(math.log(base)), lit(0.0)])


def variance_ddof(column: Expr, ddof: int, *, sqrt: bool) -> AggExpr | Expr:
    """Variance (or, with `sqrt`, standard deviation) dividing by ``n - ddof``.

    ``ddof=1`` is the sample statistic the engine accumulates, returned untouched. Any
    other ``ddof`` rescales the population co-moment ``covar_pop(x, x)``, which divides by
    ``n`` and is defined from one value up (`var_pop` explains why that form and not a
    rescaled sample variance). A group with ``n <= ddof`` values has no estimate and is null.
    """
    ddof = require_int(ddof, func="std" if sqrt else "var", arg="ddof", minimum=0)
    if ddof == 1:
        return AggExpr("stddev" if sqrt else "var", column)
    population = AggExpr("covar_pop", column, input2=column)
    if ddof == 0:
        variance: AggExpr | Expr = population
    else:
        n = column.count().cast("float64")
        variance = (
            when(column.count() > lit(ddof))
            .then(  # type: ignore[arg-type]
                population * n / (n - lit(float(ddof)))
            )
            .otherwise(lit(None))
        )
    return MathExpr("sqrt", variance) if sqrt else variance  # type: ignore[arg-type]


def array_agg_without_nulls(agg: AggExpr) -> Expr:
    """`agg` (an `array_agg`) with the null elements removed from each collected list.

    Spark's `collect_list`/`array_agg` skip nulls while collecting; DuckDB's keep them.
    Dropping them from the finished list is the same answer, including the empty list a
    group of only nulls collects to in Spark.
    """
    from batcher.plan.functions.collection import element

    return ListFilter(agg, element().is_not_null())  # type: ignore[arg-type]
