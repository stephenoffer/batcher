"""Predicate selectivity — the fraction of rows a `Filter` keeps.

Selinger-style structural estimation: conjunctions combine with **exponential
backoff** (not a raw independence product, which badly underestimates the kept
fraction on correlated predicates), disjunctions use inclusion-exclusion,
negation complements. A leaf `col = literal` uses `1/ndv` when the distinct count
is known; `col < literal` interpolates the fraction below the literal from
per-column quantile boundaries when known, else a Selinger range constant. These
feed the row-count estimator; they are *estimates* and never carry `EXACT`
provenance.
"""

from __future__ import annotations

from typing import Any

from batcher.config import CardinalityConfig
from batcher.plan.expr_ir import Binary, Col, Expr, IsNotNull, IsNull, Lit, Not

__all__ = ["predicate_selectivity"]

_COMPARISONS = {"lt", "le", "gt", "ge"}
# Comparison operators flip when the column is on the right (`literal < col` ≡ `col > literal`).
_FLIP_OP = {"lt": "gt", "gt": "lt", "le": "ge", "ge": "le"}


def predicate_selectivity(
    expr: Expr,
    ndv: dict[str, float],
    cfg: CardinalityConfig,
    quantiles: dict[str, Any] | None = None,
) -> float:
    """Estimate the fraction of rows a predicate keeps, from its structure.

    Conjunctions combine with exponential backoff (the most selective conjunct at
    full weight, each subsequent one dampened) — this lifts the estimate toward
    reality on correlated predicates, where the naive independence product
    underestimates the kept fraction. Disjunctions use inclusion-exclusion;
    negation complements. A leaf `col = literal` uses `1/ndv` when the distinct
    count is known; `col < literal` interpolates the fraction below the literal
    from per-column quantile boundaries when known, else a Selinger range constant.
    Always clamped to `[0, 1]`.
    """
    sel = _raw_predicate_selectivity(expr, ndv, cfg, quantiles or {})
    return min(1.0, max(0.0, sel))


def _raw_predicate_selectivity(
    expr: Expr, ndv: dict[str, float], cfg: CardinalityConfig, quantiles: dict[str, Any]
) -> float:
    if isinstance(expr, Binary):
        op = expr.op
        if op == "and":
            conjuncts = _flatten_and(expr)
            sels = sorted(predicate_selectivity(c, ndv, cfg, quantiles) for c in conjuncts)
            return _exponential_backoff(sels)
        if op == "or":
            a = predicate_selectivity(expr.left, ndv, cfg, quantiles)
            b = predicate_selectivity(expr.right, ndv, cfg, quantiles)
            return a + b - a * b
        if op == "eq":
            return _equality_selectivity(expr, ndv, cfg)
        if op == "ne":
            return 1.0 - _equality_selectivity(expr, ndv, cfg)
        if op in _COMPARISONS:
            return _range_selectivity(expr, op, cfg, quantiles)
    if isinstance(expr, Not):
        return 1.0 - predicate_selectivity(expr.input, ndv, cfg, quantiles)
    if isinstance(expr, IsNull):
        return cfg.null_selectivity
    if isinstance(expr, IsNotNull):
        return 1.0 - cfg.null_selectivity
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


def _range_selectivity(
    expr: Binary, op: str, cfg: CardinalityConfig, quantiles: dict[str, Any]
) -> float:
    """`col < literal` (and `<=`/`>`/`>=`) selectivity from per-column quantile
    boundaries when known — the fraction of rows below the literal, interpolated from
    the KLL histogram — else the Selinger range constant."""
    side = comparison_col_side(expr)
    if side is None:
        return cfg.range_selectivity
    col, value, col_on_left = side
    q = quantiles.get(col)
    if not q:
        return cfg.range_selectivity
    try:
        x = float(value)
    except (TypeError, ValueError):
        return cfg.range_selectivity  # non-numeric literal (e.g. a string/date bound)
    frac_le = _fraction_below(x, q.get("probs", []), q.get("values", []))
    if frac_le is None:
        return cfg.range_selectivity
    # Normalize so the column is on the left, then keep below (`<`/`<=`) or above.
    eff = op if col_on_left else _FLIP_OP[op]
    return frac_le if eff in {"lt", "le"} else 1.0 - frac_le


def comparison_col_side(expr: Binary) -> tuple[str, Any, bool] | None:
    """`(column, literal, col_on_left)` for a `col OP literal` / `literal OP col`."""
    if isinstance(expr.left, Col) and isinstance(expr.right, Lit):
        return expr.left.name, expr.right.value, True
    if isinstance(expr.right, Col) and isinstance(expr.left, Lit):
        return expr.right.name, expr.left.value, False
    return None


def _fraction_below(x: float, probs: list[float], values: list[float]) -> float | None:
    """Interpolate the fraction of values ≤ `x` from quantile boundaries (`values` at
    `probs`, both ascending). None if the boundaries are unusable."""
    if len(probs) != len(values) or len(values) < 2:
        return None
    if x <= values[0]:
        return 0.0
    if x >= values[-1]:
        return 1.0
    for i in range(len(values) - 1):
        lo, hi = values[i], values[i + 1]
        if lo <= x <= hi:
            if hi == lo:
                return probs[i]
            return probs[i] + (x - lo) / (hi - lo) * (probs[i + 1] - probs[i])
    return None


def _equality_selectivity(expr: Binary, ndv: dict[str, float], cfg: CardinalityConfig) -> float:
    """`col = literal` keeps ~`1/ndv(col)` of rows when the distinct count is
    known (uniformity assumption), else a small default."""
    col = _column_of_comparison(expr)
    if col is not None and col in ndv and ndv[col] > 0:
        return 1.0 / ndv[col]
    return cfg.eq_selectivity


def _column_of_comparison(expr: Binary) -> str | None:
    """The column name in a `col OP literal` (or `literal OP col`) comparison."""
    if isinstance(expr.left, Col) and isinstance(expr.right, Lit):
        return expr.left.name
    if isinstance(expr.right, Col) and isinstance(expr.left, Lit):
        return expr.right.name
    return None
