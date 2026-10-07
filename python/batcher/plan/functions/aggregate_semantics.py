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
`mode(all_modes=True)`, `min_by(ignore_nulls=False)`) are engine kernels instead, and live
beside the aggregate they extend in `bc-runtime`.
"""

from __future__ import annotations

import math

from batcher._internal.errors import PlanError, require_float, require_int
from batcher.plan.expr_ir.constructors import array, count, lit, when
from batcher.plan.expr_ir.core import AggExpr, Coalesce, Expr, MathExpr
from batcher.plan.expr_ir.func_nodes import ListFilter, ListTransform
from batcher.plan.expr_ir.nodes import NullIf
from batcher.plan.functions.collection import element, struct

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
            .then(population * n / (n - lit(float(ddof))))
            .otherwise(lit(None))
        )
    return MathExpr("sqrt", variance) if sqrt else variance  # type: ignore[arg-type]


def array_agg_without_nulls(agg: AggExpr) -> Expr:
    """`agg` (an `array_agg`) with the null elements removed from each collected list.

    Spark's `collect_list`/`array_agg` skip nulls while collecting; DuckDB's keep them.
    Dropping them from the finished list is the same answer, including the empty list a
    group of only nulls collects to in Spark.
    """
    return ListFilter(agg, element().is_not_null())  # type: ignore[arg-type]


def filter_aggregate(agg: AggExpr, predicate: Expr) -> AggExpr | Expr:
    """`agg` over only the rows where `predicate` is true: SQL ``agg(...) FILTER (WHERE p)``.

    The one lowering both front ends use: `AggExpr.filter` calls it, and the SQL translator
    calls it for every ``FILTER (WHERE ...)`` clause, so the two cannot drift.

    Most aggregates skip a null input, so masking every operand to null where `predicate`
    is not true (false or null, as in a ``WHERE``) leaves exactly the matching rows. Three
    observe a row whose input is null, and each gets the rule that keeps that row out:

    * ``count(*)`` counts a constant that is null off the predicate, so a group with no
      match counts ``0``, not null.
    * ``array_agg`` keeps null elements. The value is packed into a one-field struct that
      is null off the predicate, the null structs are dropped from the collected list and
      the field is unpacked, so a genuine null value survives and a filtered row does not.
      A group with no match is null, as DuckDB answers, rather than ``[]``.
    * ``first``/``last`` with ``ignore_nulls=False`` skip a row only when its order key
      is null, which masking the key as well as the value does.

    Every piece is an existing mergeable aggregate plus a projection over its result, so a
    filtered aggregate is as distributed-safe as the unfiltered one.

    Args:
        agg: The aggregate to restrict.
        predicate: The boolean row condition.

    Returns:
        The restricted aggregate, or an expression over aggregates for ``array_agg``.
    """
    if agg.func == "count_star":
        return AggExpr("count", masked(lit(1), predicate), name=agg._alias)
    if agg.func == "list_agg" and agg.input is not None:
        return _filtered_list(agg, agg.input, predicate)
    return agg.map_operands(lambda operand: masked(operand, predicate))


def filter_aggregate_leaves(expr: AggExpr | Expr, predicate: Expr) -> AggExpr | Expr:
    """`filter_aggregate` applied to every aggregate inside `expr`, for a composite one.

    ``stddev_pop``, ``sum(empty_value=0)`` and the ``regr_*`` family are expressions over
    several aggregates. Restricting each of them to the same rows restricts the whole.

    Args:
        expr: An aggregate, or an expression over aggregates.
        predicate: The boolean row condition.

    Returns:
        The same shape with every aggregate restricted to the matching rows.
    """
    if isinstance(expr, AggExpr):
        return filter_aggregate(expr, predicate)
    from batcher.plan.expr_rewrite.traverse import transform_expr_up

    def rule(node: Expr) -> Expr:
        if isinstance(node, AggExpr):
            return filter_aggregate(node, predicate)  # type: ignore[return-value]
        return node

    return transform_expr_up(expr, rule)


def masked(value: Expr, predicate: Expr) -> Expr:
    """`value` where `predicate` is true, and a null of `value`'s own type elsewhere.

    The row guard every `FILTER (WHERE ...)` lowering is built from, exported so the SQL
    DISTINCT rewrite masks a deduplicated value exactly as `filter_aggregate` masks an
    aggregate's input, and the two expressions compare equal.
    """
    return when(predicate).then(value).otherwise(None)


def _filtered_list(agg: AggExpr, value: Expr, predicate: Expr) -> Expr:
    """`agg`, an `array_agg` of `value`, leaving out the rows `predicate` rejects."""
    packed = masked(struct(v=value), predicate)
    collected = AggExpr("list_agg", packed, order_by=agg.order_by)
    kept = ListTransform(
        ListFilter(collected, element().is_not_null()),  # type: ignore[arg-type]
        element().struct.field("v"),
    )
    matched = AggExpr("count", masked(lit(1), predicate))
    result = when(matched > lit(0)).then(kept).otherwise(None)  # type: ignore[operator]
    return result if agg._alias is None else result.alias(agg._alias)  # type: ignore[return-value]


def quantile_list(column: Expr, qs: list[float], interpolation: str) -> Expr:
    """The quantiles of `column` at each of `qs`, as one list (DuckDB ``quantile_cont(x, [...])``).

    One aggregate per fraction, assembled into a list in the order given, all in the same
    mergeable pass. A group with no non-null value answers a null list, as DuckDB does,
    rather than a list of nulls.

    Args:
        column: The column to summarize.
        qs: The fractions, each in ``[0, 1]``.
        interpolation: How a rank between two values resolves, as for `Expr.quantile`.

    Returns:
        A ``List`` expression over the quantile aggregates.
    """
    if not qs:
        raise PlanError("quantile() needs at least one fraction in its list of q values")
    parts = [column.quantile(q, interpolation) for q in qs]
    return when(column.count() > lit(0)).then(array(*parts)).otherwise(None)


def distinct_array_agg(
    column: Expr, keys: tuple[tuple[Expr, bool, bool], ...], *, drop_nulls: bool
) -> Expr:
    """`array_agg(column)` with each value kept once, sorted by the value itself.

    SQL's ``array_agg(DISTINCT x ORDER BY x)``. The list is collected as usual and then
    deduplicated and sorted, so it is the same mergeable pass as `array_agg` plus a
    projection. The only order key accepted is the value itself, which sets the direction
    and the null placement: among duplicates there is no one row whose key could decide a
    position, so Postgres and DuckDB refuse any other key and so does this. One null is
    kept unless `drop_nulls`.

    Args:
        column: The collected value.
        keys: The normalized ``(expr, descending, nulls_first)`` order keys.
        drop_nulls: Leave the null element out.

    Returns:
        The deduplicated, sorted list expression.
    """
    from batcher.plan.expr_ir.namespaces.collections import _ListNamespace

    if len(keys) > 1 or (keys and repr(keys[0][0]) != repr(column)):
        raise PlanError(
            "array_agg(distinct=True) can only be ordered by the value itself: among "
            "duplicate values there is no single row whose order key decides a position. "
            "Drop order_by, or order by the aggregated column"
        )
    descending, nulls_first = (keys[0][1], keys[0][2]) if keys else (False, False)
    unique = _ListNamespace(AggExpr("list_agg", column)).unique(drop_nulls=drop_nulls)  # type: ignore[arg-type]
    return unique.list.sort(descending=descending, nulls_last=not nulls_first)
