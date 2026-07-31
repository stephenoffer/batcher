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
import math
from collections.abc import Iterator
from typing import Any

from batcher.kyber.stats.distribution import residual_eq_frequency
from batcher.plan.expr_ir import Binary, Col, Lit
from batcher.plan.stats import AXIS_NUMERIC, ordinal_with_axis

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

    The position half of `plan.stats.ordinal_with_axis`, for the comparisons whose two
    operands are *both* literals or bounds and so are already on one axis by construction
    (`_outside_bounds`, `_fraction_below_bounds`). Anything read against a **measured**
    statistic must check the axis instead — see `_fraction_below_quantiles`.
    """
    placed = ordinal_with_axis(value)
    return None if placed is None else placed[1]


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


def _fraction_below_quantiles(value: Any, q: dict[str, Any] | None) -> float | None:
    """The fraction of rows ≤ `value` interpolated from learned quantile boundaries.

    Declines (None, so the caller falls back to the column's bounds) unless the grid was
    measured on the same axis the literal sits on. A grid and a literal on different number
    lines do not merely estimate badly — they put every literal far outside the grid, which
    reads as "no rows match" and collapses the plan built on it.
    """
    placed = ordinal_with_axis(value)
    if placed is None:
        return None
    return _fraction_below_on_axis(placed[1], placed[0], q)


def _fraction_below_on_axis(x: float, axis: str, q: dict[str, Any] | None) -> float | None:
    """`_fraction_below_quantiles` for a position already placed on `axis`.

    For the callers holding an ordinal rather than the literal it came from — a column's own
    `min`/`max` read off its `ColumnStat`. They still must name the axis, because the grid is
    only theirs to read if it was measured on the same one.
    """
    if not q or q.get("axis", AXIS_NUMERIC) != axis:
        return None
    return _fraction_below(x, q.get("probs", []), q.get("values", []))


def _is_discrete(bound: tuple[Any, Any]) -> bool:
    """Whether a column's bounds step through whole units, so its range can be *counted*.

    Decided from the bound values' **types**, not from whether they happen to be whole numbers.
    A `Float64` column whose min and max are exactly `0.0` and `1.0` would pass an
    `is_integer()` test and then be counted as holding two values — badly wrong on a continuous
    column, and wrong at exactly the narrow ranges where counting is meant to help.

    `int` and `date` qualify: their ordinals step by one, so `hi - lo + 1` is the number of
    values the range contains. A `datetime` does not, and excluding it costs nothing — its
    ordinal is in seconds, so the count would be enormous and the two forms agree to within a
    rounding error anyway. `bool` is excluded because `_ordinal` refuses it upstream.
    """
    lo, hi = bound
    if isinstance(lo, bool) or isinstance(hi, bool):
        return False
    if isinstance(lo, int) and isinstance(hi, int):
        return True
    return (
        isinstance(lo, datetime.date)
        and isinstance(hi, datetime.date)
        and not isinstance(lo, datetime.datetime)
        and not isinstance(hi, datetime.datetime)
    )


def discrete_step_mass(bound: tuple[Any, Any] | None) -> float | None:
    """`P(v = x)` implied by a discrete column's own bounds, or ``None``.

    The uniform mass of one step across the `hi - lo + 1` values the range contains — the same
    number `_fraction_below_bounds` divides by, and the same one
    `_date_part_range_selectivity` uses for a bounded field like `month`.

    Used only as a *fallback* where nothing has measured a distinct count. `_point_mass` answers
    0 there, deliberately, so as not to invent a mass — but on a discrete column the mass is not
    invented: it follows from the same uniformity assumption the CDF already rests on. Without it
    the strict and non-strict comparisons cannot separate, and `d < min` came back as a quarter of
    a four-value range rather than the nothing it is.
    """
    if bound is None or not _is_discrete(bound):
        return None
    lo, hi = _ordinal(bound[0]), _ordinal(bound[1])
    if lo is None or hi is None or hi < lo:
        return None
    return 1.0 / (hi - lo + 1)


def _fraction_below_bounds(x: float, bound: tuple[Any, Any] | None) -> float | None:
    """The fraction of rows ≤ `x` assuming values spread uniformly over `[min, max]`.

    **Counted discretely for a discrete column** (see `_is_discrete`). The continuous form
    `(x - lo) / (hi - lo)` answers a flat `0` at `x == lo`, because it cannot tell "below the
    minimum" (genuinely zero) from "equal to the minimum" (every row holding it). That made
    `d <= min` estimate **zero rows** for a predicate matching every row at the column's first
    value — a partition boundary or a `>= min` sentinel, so often a large fraction of the table —
    and a zero-row estimate is the worst kind to be wrong by, since build-side choice, join order,
    broadcast sizing and the adaptive gate all read it as "this subtree is empty".

    The discrete form spreads the rows over the `hi - lo + 1` values the range contains, so
    `F(lo)` is one value's worth rather than none and `F(hi)` is still 1. It rests on the same
    uniformity assumption as the continuous form, applied to the values that actually exist
    rather than to a continuum they do not live on — and it is the form
    `_date_part_range_selectivity` already uses for a bounded field like `month`, so this brings
    the general path in line with it. A float or decimal column keeps the continuous form, where
    the endpoint problem is real but unfixable from bounds alone: there is no "next" value to
    divide by.
    """
    if bound is None:
        return None
    lo, hi = _ordinal(bound[0]), _ordinal(bound[1])
    if lo is None or hi is None:
        return None
    if x < lo:
        return 0.0
    if x >= hi:
        return 1.0
    if hi == lo:
        return 1.0
    if _is_discrete(bound):
        # `floor(x)` because a fractional literal against a discrete column selects the same
        # rows as the whole step below it: `i <= 2.5` is `i <= 2`.
        return (math.floor(x) - lo + 1) / (hi - lo + 1)
    return (x - lo) / (hi - lo)


def fraction_left_below_right(
    left: tuple[Any, Any] | None, right: tuple[Any, Any] | None
) -> float | None:
    """`P(a < b)` for two columns from their `[min, max]` bounds, or None if unusable.

    The two-column analogue of [`_fraction_below_bounds`]: that one interpolates a column
    against a *literal*, this one against another column's span. Both values spread
    uniformly over their own range and are assumed independent, which is the same
    assumption the literal path already makes, applied twice.

    Exact rather than sampled. With `a` uniform on `[a0, a1]` and `b` uniform on
    `[b0, b1]`, `P(a < b)` integrates the length of `b`'s range above `a`::

        P(a < b) = (1 / (La * Lb)) * integral over a of clip(b1 - max(b0, a), 0, Lb)

    and that integrand is piecewise linear: flat at `Lb` while `a <= b0`, falling as
    `b1 - a` across the overlap, zero once `a >= b1`. So the integral is one rectangle
    plus one triangle, both in closed form below.

    Disjoint ranges give a certainty for free — `a1 <= b0` returns 1.0 and `a0 >= b1`
    returns 0.0 — which is the case a flat constant gets most wrong.

    A degenerate range (`min == max`, one distinct value) reduces to the one-sided
    interpolation, and two degenerate ranges to a plain comparison.

    Strict and non-strict are not distinguished: separating them needs `P(a == b)`, which
    needs both columns' distinct counts rather than just their bounds. The difference is
    one distinct value's mass, so callers treat `lt`/`le` alike here — unlike the literal
    path, which does subtract that point mass because it knows the single value involved.
    """
    if left is None or right is None:
        return None
    a0, a1 = _ordinal(left[0]), _ordinal(left[1])
    b0, b1 = _ordinal(right[0]), _ordinal(right[1])
    if a0 is None or a1 is None or b0 is None or b1 is None:
        return None
    if a1 < a0 or b1 < b0:  # a nonsense bound pair; defer rather than invent a number
        return None
    # Certainties first: they are exact, and they need no uniformity assumption at all.
    if a1 <= b0:
        return 1.0
    if a0 >= b1:
        return 0.0
    la, lb = a1 - a0, b1 - b0
    if la == 0.0 and lb == 0.0:
        return 1.0 if a0 < b0 else 0.0
    if la == 0.0:  # `a` is a single value: the fraction of `b` strictly above it
        return min(1.0, max(0.0, (b1 - a0) / lb))
    if lb == 0.0:  # `b` is a single value: the fraction of `a` strictly below it
        return min(1.0, max(0.0, (b0 - a0) / la))
    # Rectangle: the part of `a`'s range wholly below `b`'s, where every `b` is greater.
    below = max(0.0, min(a1, b0) - a0) * lb
    # Triangle: across the overlap, `b1 - a` shrinks linearly as `a` rises.
    lo, hi = max(a0, b0), min(a1, b1)
    overlap = ((b1 - lo) ** 2 - (b1 - hi) ** 2) / 2.0 if hi > lo else 0.0
    return min(1.0, max(0.0, (below + overlap) / (la * lb)))


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
