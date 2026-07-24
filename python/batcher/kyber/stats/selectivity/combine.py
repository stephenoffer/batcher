"""Composing leaf selectivities into a whole-predicate estimate.

The top layer of the `selectivity` package: `predicate_selectivity` walks a boolean
expression tree, dispatching each leaf to `leaves` and combining the results —
conjunctions with **exponential backoff** (correlation-robust), disjunctions with
inclusion-exclusion, negation as the (null-aware) complement. Same-column range conjuncts
are recognised and combined as one interval before backoff, because two bounds on one
column carve a single interval rather than two independent predicates.
"""

from __future__ import annotations

from typing import Any

from batcher._internal.mathx import clamp01
from batcher.config import CardinalityConfig
from batcher.kyber.stats.selectivity.leaves import (
    _equality_selectivity,
    _in_list_selectivity,
    _measured_null_fraction,
    _null_mass,
    _range_selectivity,
    _str_func_selectivity,
)
from batcher.kyber.stats.selectivity.scalars import (
    _COMPARISONS,
    _FLIP_OP,
    _fraction_below_bounds,
    _fraction_below_quantiles,
    _mcv_lookup,
    _ordinal,
    _point_mass,
    comparison_col_side,
)
from batcher.plan.expr_ir import (
    Binary,
    Case,
    Coalesce,
    Col,
    Expr,
    InList,
    IsNotNull,
    IsNull,
    Lit,
    Not,
    StrFunc,
)


def predicate_selectivity(
    expr: Expr,
    ndv: dict[str, float],
    cfg: CardinalityConfig,
    quantiles: dict[str, Any] | None = None,
    mcv: dict[str, dict[str, float]] | None = None,
    bounds: dict[str, tuple[Any, Any]] | None = None,
    nulls: dict[str, float] | None = None,
) -> float:
    """Estimate the fraction of rows a predicate keeps, from its structure.

    Conjunctions combine with exponential backoff (the most selective conjunct at
    full weight, each subsequent one dampened) — this lifts the estimate toward
    reality on correlated predicates, where the naive independence product
    underestimates the kept fraction. Disjunctions use inclusion-exclusion;
    negation complements. A leaf `col = literal` uses the literal's measured
    most-common-value frequency when it is a known skew value, else `1/ndv` when the
    distinct count is known; `col < literal` interpolates the fraction below the
    literal from per-column quantile boundaries when known, else from the column's
    exact `min`/`max` `bounds`, else a Selinger range constant. Always clamped to
    `[0, 1]`.

    SQL filters keep only rows where the predicate is **TRUE**, so a NULL operand is
    dropped by `p` *and* by `NOT p`: `sel(p) + sel(NOT p) = 1 - f_null(p)`, not 1. The
    negation and `!=` paths subtract that null mass, using each column's *measured* null
    fraction (`nulls`) when the source declared one.
    """
    sel = _raw_predicate_selectivity(
        expr, ndv, cfg, quantiles or {}, mcv or {}, bounds or {}, nulls or {}
    )
    return clamp01(sel)


def _raw_predicate_selectivity(
    expr: Expr,
    ndv: dict[str, float],
    cfg: CardinalityConfig,
    quantiles: dict[str, Any],
    mcv: dict[str, dict[str, float]],
    bounds: dict[str, tuple[Any, Any]],
    nulls: dict[str, float],
) -> float:
    if isinstance(expr, Binary):
        op = expr.op
        if op == "and":
            conjuncts = _flatten_and(expr)
            sels = _conjunct_selectivities(conjuncts, ndv, cfg, quantiles, mcv, bounds, nulls)
            return _exponential_backoff(sorted(sels))
        if op == "or":
            disjuncts = _flatten_or(expr)
            return _combine_disjuncts(disjuncts, ndv, cfg, quantiles, mcv, bounds, nulls)
        if op == "eq":
            coalesced = _coalesce_equality_selectivity(expr, ndv, cfg, mcv, nulls)
            if coalesced is not None:
                return coalesced
            return _equality_selectivity(expr, ndv, cfg, mcv, bounds)
        if op == "ne":
            # `col != v` is TRUE only where `col` is non-null and unequal. The null rows
            # are dropped, so the complement is taken over `1 - f_null`, not over 1.
            return (1.0 - _null_mass(expr, nulls)) - _equality_selectivity(
                expr, ndv, cfg, mcv, bounds
            )
        if op in _COMPARISONS:
            return _range_selectivity(expr, op, cfg, quantiles, bounds, ndv, mcv)
    if isinstance(expr, Not):
        inner = predicate_selectivity(expr.input, ndv, cfg, quantiles, mcv, bounds, nulls)
        return (1.0 - _null_mass(expr.input, nulls)) - inner
    if isinstance(expr, IsNull):
        measured = _measured_null_fraction(expr, nulls)
        return cfg.null_selectivity if measured is None else measured
    if isinstance(expr, IsNotNull):
        measured = _measured_null_fraction(expr, nulls)
        return 1.0 - (cfg.null_selectivity if measured is None else measured)
    if isinstance(expr, InList):
        return _in_list_selectivity(expr, ndv, cfg, mcv, bounds)
    if isinstance(expr, StrFunc):
        return _str_func_selectivity(expr, ndv, cfg, mcv)
    if isinstance(expr, Lit) and isinstance(expr.value, bool):
        # A constant predicate keeps everything or nothing — never the 0.5 "unknown" default.
        # (The folder usually removes these, but a `CASE` arm reaches here as a literal.)
        return 1.0 if expr.value else 0.0
    if isinstance(expr, Case):
        # SQL `CASE`: the first branch whose `when` is TRUE supplies the value. As a predicate,
        # a row is kept when the *winning* branch's `then` is TRUE, so the branches partition
        # the rows: branch `i` claims `sel(when_i)` of what earlier branches left, and
        # contributes its own `sel(then_i)` of that. With boolean-literal arms this is exact —
        # `CASE WHEN c THEN TRUE ELSE FALSE END` collapses to exactly `sel(c)` — instead of the
        # blunt 0.5 the whole node used to get.
        remaining, total = 1.0, 0.0
        for when, then in expr.branches:
            w = predicate_selectivity(when, ndv, cfg, quantiles, mcv, bounds, nulls)
            t = predicate_selectivity(then, ndv, cfg, quantiles, mcv, bounds, nulls)
            total += remaining * w * t
            remaining *= 1.0 - w
        total += remaining * predicate_selectivity(
            expr.otherwise, ndv, cfg, quantiles, mcv, bounds, nulls
        )
        return min(1.0, total)
    if isinstance(expr, Coalesce) and expr.inputs:
        # A `COALESCE` used *as a predicate* is the null-guarded form of its first argument:
        # `coalesce(p, FALSE)` is TRUE exactly where `p` is TRUE, so its selectivity is
        # **exactly** `sel(p)`. This is the shape `eq_missing`/`IS NOT DISTINCT FROM` desugars
        # to (`coalesce(x = y, FALSE) OR (x IS NULL AND y IS NULL)`), which otherwise collapsed
        # to the blunt 0.5 and made every null-safe comparison a coin flip. With any other fill
        # the fill can only add back the rows where `p` is NULL, so `sel(p) + null_mass(p)` is a
        # safe upper bound.
        first = expr.inputs[0]
        base = predicate_selectivity(first, ndv, cfg, quantiles, mcv, bounds, nulls)
        fill = expr.inputs[1] if len(expr.inputs) > 1 else None
        if isinstance(fill, Lit) and fill.value is False:
            return base
        return min(1.0, base + _null_mass(first, nulls))
    if isinstance(expr, Col):
        # A bare boolean column used as a predicate (`WHERE is_active`) keeps its TRUE rows.
        # With a measured TRUE frequency, use it (a 5%-true flag is far from the 0.5 prior);
        # otherwise a boolean splits ~evenly, which is what `default_filter_selectivity` is.
        freq = _mcv_lookup(mcv.get(expr.name), True)
        if freq is not None:
            return freq
        return cfg.default_filter_selectivity
    return cfg.default_filter_selectivity


def _flatten_and(expr: Binary) -> list[Expr]:
    """Flatten a nested `a AND b AND c …` tree into its conjunct list.

    Splits only on `and`, so each returned conjunct is itself estimated normally
    (an `or`/`not`/comparison subtree is one conjunct). This lets the whole `AND`
    combine with one exponential-backoff pass rather than a left-folded product.
    """
    out: list[Expr] = []
    stack: list[Expr] = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, Binary) and node.op == "and":
            stack.append(node.right)
            stack.append(node.left)
        else:
            out.append(node)
    return out


def _coalesce_equality_selectivity(
    expr: Binary,
    ndv: dict[str, float],
    cfg: CardinalityConfig,
    mcv: dict[str, dict[str, float]],
    nulls: dict[str, float],
) -> float | None:
    """`COALESCE(x, c) = v` selectivity (the `fill_null` data-cleaning shape), else None.

    `COALESCE(x, c)` is `x` where `x` is non-null, else the fill `c`. So it equals `v` where
    `x = v` (over the non-null rows) — the ordinary equality estimate over `x` — **plus**, when
    the fill constant `c` itself equals `v`, the rows where `x` is null (which `COALESCE` maps
    to `c = v`). Adding that measured null mass only *raises* the estimate (the safe direction),
    and lifts it off the blunt `default_filter_selectivity` (0.5) it otherwise hit — a
    `WHERE coalesce(x, 0) = 0` over a sparse column is nowhere near 50%. Returns None unless one
    side is a `COALESCE(Col, ...)` and the other a literal.
    """
    coal, lit = _coalesce_and_literal(expr)
    if coal is None or not coal.inputs or not isinstance(coal.inputs[0], Col):
        return None
    inner = coal.inputs[0]
    base = _equality_selectivity(Binary("eq", inner, lit), ndv, cfg, mcv)
    fill = coal.inputs[1] if len(coal.inputs) > 1 else None
    if isinstance(fill, Lit) and fill.value == lit.value:
        base += nulls.get(inner.name, 0.0)  # x-is-null rows take the fill, which equals v
    return min(1.0, base)


def _coalesce_and_literal(expr: Binary) -> tuple[Coalesce | None, Lit]:
    """`(coalesce, literal)` for `COALESCE(…) = lit` / `lit = COALESCE(…)`, else `(None, …)`."""
    if isinstance(expr.left, Coalesce) and isinstance(expr.right, Lit):
        return expr.left, expr.right
    if isinstance(expr.right, Coalesce) and isinstance(expr.left, Lit):
        return expr.right, expr.left
    return None, Lit(None)


def _flatten_or(expr: Binary) -> list[Expr]:
    """Flatten a nested `a OR b OR c …` tree into its disjunct list (splits only on `or`)."""
    out: list[Expr] = []
    stack: list[Expr] = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, Binary) and node.op == "or":
            stack.append(node.right)
            stack.append(node.left)
        else:
            out.append(node)
    return out


def _combine_disjuncts(
    disjuncts: list[Expr],
    ndv: dict[str, float],
    cfg: CardinalityConfig,
    quantiles: dict[str, Any],
    mcv: dict[str, dict[str, float]],
    bounds: dict[str, tuple[Any, Any]],
    nulls: dict[str, float],
) -> float:
    """Selectivity of an OR, summing same-column equalities and combining the rest by union.

    Two equality/`IN` disjuncts on the **same** column select *disjoint* value sets, so their
    union is the exact **sum** of their selectivities — not the inclusion-exclusion
    `a + b - a·b`, which subtracts a phantom overlap and under-estimates (the dual of
    `_in_list_selectivity`'s own "sum, not `1-(1-1/d)^k`" reasoning; `x = 1 OR x = 2` is just
    `x IN (1, 2)`). Cross-column disjuncts are assumed independent and combined by
    `1 - ∏(1 - sᵢ)` — the exact N-ary generalization of the old pairwise `a + b - a·b`, so a
    disjunction across different columns is unchanged.
    """
    groups: dict[str, float] = {}
    others: list[float] = []
    for d in disjuncts:
        col = _same_column_membership(d)
        sel = predicate_selectivity(d, ndv, cfg, quantiles, mcv, bounds, nulls)
        if col is not None:
            groups[col] = groups.get(col, 0.0) + sel  # disjoint values → sum
        else:
            others.append(sel)
    terms = [min(1.0, g) for g in groups.values()] + others
    product = 1.0
    for s in terms:
        product *= 1.0 - s
    return 1.0 - product


def _same_column_membership(expr: Expr) -> str | None:
    """The column of a `col = literal` or `col IN (...)` disjunct, else None.

    These are the disjuncts whose value sets are provably disjoint from another equality on
    the same column, so an OR of them sums. A range/`OR`/opaque disjunct returns None and is
    combined by the independent-union rule instead.
    """
    if isinstance(expr, InList) and isinstance(expr.input, Col):
        return expr.input.name
    if isinstance(expr, Binary) and expr.op == "eq":
        side = comparison_col_side(expr)
        if side is not None:
            return side[0]
    return None


def _conjunct_selectivities(
    conjuncts: list[Expr],
    ndv: dict[str, float],
    cfg: CardinalityConfig,
    quantiles: dict[str, Any],
    mcv: dict[str, dict[str, float]],
    bounds: dict[str, tuple[Any, Any]],
    nulls: dict[str, float],
) -> list[float]:
    """One selectivity per independent conjunct, with same-column ranges pre-combined.

    Two range predicates on the *same* column (`x >= lo AND x <= hi`, the desugaring of
    `x BETWEEN lo AND hi`) are the tightest possible correlation: they carve a single
    interval out of one distribution, so their joint selectivity is the CDF difference
    `F(hi) - F(lo)`, not the exponential-backoff combination of two "independent" conjuncts.
    Backoff treats them as loosely dependent and roughly *doubles* the estimate — and a
    bounded date/number range is the most common selective filter in analytics (every
    TPC-H date interval, every ClickBench timestamp window), so the error steers real join
    orders and build-side choices.

    Range comparisons (`<`,`<=`,`>`,`>=`) of the form `col OP literal` are grouped by
    column; a column with two or more of them collapses to one interval term
    (`_interval_selectivity`). Everything else — equality, `IN`, `IS NULL`, `OR`/`NOT`
    subtrees, a lone one-sided range — is estimated individually exactly as before. The
    resulting per-term selectivities still combine with exponential backoff in the caller,
    so cross-column correlation is handled as it always was.
    """
    ranges: dict[str, list[Binary]] = {}
    others: list[Expr] = []
    for c in conjuncts:
        col = _range_column(c)
        if col is not None:
            ranges.setdefault(col, []).append(c)  # type: ignore[arg-type]
        else:
            others.append(c)
    sels = [predicate_selectivity(c, ndv, cfg, quantiles, mcv, bounds, nulls) for c in others]
    for col, comps in ranges.items():
        combined = _interval_selectivity(col, comps, quantiles, bounds, ndv, mcv)
        if combined is not None:
            sels.append(combined)
        else:
            sels.extend(
                predicate_selectivity(c, ndv, cfg, quantiles, mcv, bounds, nulls) for c in comps
            )
    return sels


def _range_column(expr: Expr) -> str | None:
    """The column of a `col OP literal` range comparison (`<`,`<=`,`>`,`>=`), else None.

    Only orderable literals qualify — the interval combiner interpolates a CDF, which a
    non-orderable value (a string, a bool) has no position on.
    """
    if not (isinstance(expr, Binary) and expr.op in _COMPARISONS):
        return None
    side = comparison_col_side(expr)
    if side is None or _ordinal(side[1]) is None:
        return None
    return side[0]


def _interval_selectivity(
    col: str,
    comps: list[Binary],
    quantiles: dict[str, Any],
    bounds: dict[str, tuple[Any, Any]],
    ndv: dict[str, float],
    mcv: dict[str, dict[str, float]],
) -> float | None:
    """Selectivity of a set of same-column range comparisons as one interval.

    Combining an AND of range predicates on `col` intersects their half-lines into a
    single interval `(lower, upper)`; its mass is `upper_cdf - lower_cdf` where each bound
    is the tightest of its side (the largest lower CDF, the smallest upper CDF), and strict
    vs. non-strict bounds shift by the boundary's point mass exactly as `_range_selectivity`
    does for one comparison. Returns None — deferring to per-conjunct estimation — when the
    column has fewer than two comparisons or no CDF (neither quantiles nor exact bounds) is
    available to interpolate against, so the previous behaviour is preserved in those cases.
    """
    if len(comps) < 2:
        return None
    q = quantiles.get(col)
    bnd = bounds.get(col)
    if not q and bnd is None:
        return None
    lower_cdf, upper_cdf = 0.0, 1.0
    for c in comps:
        side = comparison_col_side(c)
        if side is None:
            return None
        _, value, col_on_left = side
        eff = c.op if col_on_left else _FLIP_OP[c.op]
        x = _ordinal(value)
        frac = _fraction_below_quantiles(x, q)
        if frac is None:
            frac = _fraction_below_bounds(x, bnd)
        if frac is None:
            return None
        if eff in ("lt", "le"):  # upper bound: keep the smallest (tightest ceiling)
            cdf = frac if eff == "le" else frac - _point_mass(col, value, ndv, mcv)
            upper_cdf = min(upper_cdf, cdf)
        else:  # gt / ge → lower bound: keep the largest (tightest floor)
            cdf = frac - _point_mass(col, value, ndv, mcv) if eff == "ge" else frac
            lower_cdf = max(lower_cdf, cdf)
    return clamp01(upper_cdf - lower_cdf)


def _exponential_backoff(sels: list[float]) -> float:
    """Combine per-conjunct selectivities (sorted ascending) with diminishing
    exponents: `s₁ · s₂^(1/2) · s₃^(1/4) · …`.

    The most selective conjunct carries full weight; each subsequent one is
    dampened, so the result sits between the pure independence product (a lower
    bound, exact only when conjuncts are independent) and the most selective
    conjunct alone (an upper bound, the perfectly-correlated case). This is the
    standard correlation-robust estimator used by production optimizers.
    """
    combined = 1.0
    exponent = 1.0
    for s in sels:
        combined *= s**exponent
        exponent /= 2.0
    return combined
