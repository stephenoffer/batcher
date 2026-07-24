"""Scalar and column-statistic primitives shared by every selectivity estimator.

The bottom layer of the `selectivity` package: pure helpers that read a literal or a
column's statistics and answer a small numeric question — where a value sits in a
distribution (`_ordinal`, `_fraction_below`), a boundary's point mass (`_point_mass`), a
skewed value's measured frequency (`_mcv_lookup`), whether a value is outside a column's
bounds (`_outside_bounds`). They depend on nothing else in the package, so the leaf
estimators (`leaves`) and the combiner (`combine`) can both build on them.
"""

from __future__ import annotations

import datetime
from collections.abc import Iterator
from decimal import Decimal
from typing import Any

from batcher.kyber.stats.distribution import residual_eq_frequency
from batcher.plan.expr_ir import Binary, Col, Lit

_COMPARISONS = {"lt", "le", "gt", "ge"}
# Comparison operators flip when the column is on the right (`literal < col` ≡ `col > literal`).
_FLIP_OP = {"lt": "gt", "gt": "lt", "le": "ge", "ge": "le"}


def comparison_col_side(expr: Binary) -> tuple[str, Any, bool] | None:
    """`(column, literal, col_on_left)` for a `col OP literal` / `literal OP col`."""
    if isinstance(expr.left, Col) and isinstance(expr.right, Lit):
        return expr.left.name, expr.right.value, True
    if isinstance(expr.right, Col) and isinstance(expr.left, Lit):
        return expr.right.name, expr.left.value, False
    return None


def _column_of_comparison(expr: Binary) -> str | None:
    """The column name in a `col OP literal` (or `literal OP col`) comparison."""
    if isinstance(expr.left, Col) and isinstance(expr.right, Lit):
        return expr.left.name
    if isinstance(expr.right, Col) and isinstance(expr.left, Lit):
        return expr.right.name
    return None


def _dedup(values: tuple) -> list:
    """`values` with duplicates removed, order preserved. Unhashable values pass through."""
    seen: set = set()
    out: list = []
    for v in values:
        try:
            if v in seen:
                continue
            seen.add(v)
        except TypeError:  # pragma: no cover - unhashable literal
            pass
        out.append(v)
    return out


def _mcv_lookup(col_mcv: dict[str, float] | None, value: Any) -> float | None:
    """A most-common-value frequency, tolerant of the literal's numeric type.

    The MCV table is keyed by `str(measured_value)`, so an integer column stores `"5"`
    while a `5.0` literal renders as `"5.0"` — the lookup missed on exactly the skewed
    values the table exists to sharpen. Try every spelling a numerically-equal literal
    could have. `bool` is excluded: it is an `int` subclass, and `True` must not match `1`.
    """
    if not col_mcv:
        return None
    for key in _mcv_keys(value):
        freq = col_mcv.get(key)
        if freq is not None:
            return freq
    return None


def _mcv_keys(value: Any) -> Iterator[str]:
    """Every string spelling under which `value` might be stored in an MCV table."""
    yield str(value)
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        yield str(float(value))
    elif isinstance(value, float) and value.is_integer():
        yield str(int(value))


def _ordinal(value: Any) -> float | None:
    """Map a comparable scalar to a float ordinal, or `None` if it has no linear order.

    Numbers (and `Decimal`) map to themselves; `date`/`datetime` map to their epoch
    offset. Two values only ever get compared against each other after both pass through
    here, so the unit (days vs seconds) is irrelevant — only that it is consistent and
    monotonic. `bool` is excluded: `True` would otherwise interpolate as `1.0`.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        return float(value)
    if isinstance(value, datetime.datetime):
        return value.timestamp()
    if isinstance(value, datetime.date):
        return float(value.toordinal())
    return None


def _point_mass(
    col: str,
    value: Any,
    ndv: dict[str, float],
    mcv: dict[str, dict[str, float]],
) -> float:
    """`P(col = value)` — the probability mass sitting exactly on a range boundary.

    A measured most-common-value frequency when the literal is a known skew value; else the
    *residual* uniform frequency, which is what is left for a value the MCV table does not
    list (`residual_eq_frequency`) rather than the whole column's `1/ndv`. Zero when the
    distinct count is unknown, which degrades the strict/non-strict distinction back to none
    rather than inventing a mass.
    """
    col_mcv = mcv.get(col)
    freq = _mcv_lookup(col_mcv, value)
    if freq is not None:
        return max(0.0, min(1.0, freq))
    d = ndv.get(col)
    if not d or d <= 0:
        return 0.0
    return residual_eq_frequency(d, col_mcv, default=0.0)


def _outside_bounds(value: Any, bound: tuple[Any, Any] | None) -> bool:
    """Whether `value` lies strictly outside a column's `[min, max]` bounds.

    Both the value and the bounds are mapped to a common ordinal, so a `date`/`Decimal`
    literal compares against same-typed bounds. Returns False when there are no bounds or
    the value has no linear order — never claiming emptiness it can't prove.
    """
    if bound is None or bound[0] is None or bound[1] is None:
        return False
    x = _ordinal(value)
    lo, hi = _ordinal(bound[0]), _ordinal(bound[1])
    if x is None or lo is None or hi is None:
        return False
    return x < lo or x > hi


def _fraction_below_quantiles(x: float, q: dict[str, Any] | None) -> float | None:
    """The fraction of rows ≤ `x` interpolated from learned quantile boundaries."""
    if not q:
        return None
    return _fraction_below(x, q.get("probs", []), q.get("values", []))


def _fraction_below_bounds(x: float, bound: tuple[Any, Any] | None) -> float | None:
    """The fraction of rows ≤ `x` assuming values spread uniformly over `[min, max]`."""
    if bound is None:
        return None
    lo, hi = _ordinal(bound[0]), _ordinal(bound[1])
    if lo is None or hi is None:
        return None
    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0
    if hi == lo:
        return 1.0
    return (x - lo) / (hi - lo)


def _fraction_below(x: float, probs: list[float], values: list[float]) -> float | None:
    """Interpolate the fraction of values ≤ `x` from quantile boundaries (`values` at
    `probs`, both ascending). None if the boundaries are unusable.

    The endpoints are *inclusive*: a grid inset from the true extremes (say `probs`
    spanning `[0.1, 0.9]`) means `values[0]` is the 10th percentile, so `x == values[0]`
    is `probs[0] = 0.1` of the rows, not 0.0. Clamping to 0/1 *at* the boundaries — rather
    than strictly outside them — discarded up to `probs[0] + (1 - probs[-1])` of the
    distribution, biasing every predicate that lands on a grid boundary.
    """
    if len(probs) != len(values) or len(values) < 2:
        return None
    if x < values[0]:
        return 0.0
    if x > values[-1]:
        return 1.0
    for i in range(len(values) - 1):
        lo, hi = values[i], values[i + 1]
        if lo <= x <= hi:
            if hi == lo:
                return probs[i]
            return probs[i] + (x - lo) / (hi - lo) * (probs[i + 1] - probs[i])
    return None
