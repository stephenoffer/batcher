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

import datetime
from collections.abc import Iterator
from decimal import Decimal
from typing import Any

from batcher.config import CardinalityConfig
from batcher.plan.expr_ir import Binary, Col, Expr, InList, IsNotNull, IsNull, Lit, Not, StrFunc

__all__ = ["predicate_selectivity"]

_COMPARISONS = {"lt", "le", "gt", "ge"}
# Comparison operators flip when the column is on the right (`literal < col` ≡ `col > literal`).
_FLIP_OP = {"lt": "gt", "gt": "lt", "le": "ge", "ge": "le"}


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
    return min(1.0, max(0.0, sel))


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
            sels = sorted(
                predicate_selectivity(c, ndv, cfg, quantiles, mcv, bounds, nulls) for c in conjuncts
            )
            return _exponential_backoff(sels)
        if op == "or":
            a = predicate_selectivity(expr.left, ndv, cfg, quantiles, mcv, bounds, nulls)
            b = predicate_selectivity(expr.right, ndv, cfg, quantiles, mcv, bounds, nulls)
            return a + b - a * b
        if op == "eq":
            return _equality_selectivity(expr, ndv, cfg, mcv)
        if op == "ne":
            # `col != v` is TRUE only where `col` is non-null and unequal. The null rows
            # are dropped, so the complement is taken over `1 - f_null`, not over 1.
            return (1.0 - _null_mass(expr, nulls)) - _equality_selectivity(expr, ndv, cfg, mcv)
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
        return _in_list_selectivity(expr, ndv, cfg, mcv)
    if isinstance(expr, StrFunc):
        pattern_sel = _STR_PATTERN_SELECTIVITY.get(expr.fn)
        if pattern_sel is not None:
            return pattern_sel(cfg)
    return cfg.default_filter_selectivity


# Boolean-valued string predicates, by how much of a column each typically matches.
# `like`/`ilike` carry a raw SQL pattern; a leading `%` makes them a substring search,
# otherwise they are anchored like a prefix match.
_STR_PATTERN_SELECTIVITY = {
    "contains": lambda cfg: cfg.substring_selectivity,
    "regexp_matches": lambda cfg: cfg.substring_selectivity,
    "starts_with": lambda cfg: cfg.prefix_selectivity,
    "ends_with": lambda cfg: cfg.prefix_selectivity,
    "like": lambda cfg: cfg.substring_selectivity,
    "ilike": lambda cfg: cfg.substring_selectivity,
}


def _in_list_selectivity(
    expr: InList,
    ndv: dict[str, float],
    cfg: CardinalityConfig,
    mcv: dict[str, dict[str, float]] | None = None,
) -> float:
    """`col IN (v1, …, vk)` — the union of `k` equality predicates.

    `col = v_i` and `col = v_j` are **mutually exclusive** for distinct literals, so the
    union is their *sum*, not an independence union: exactly `k/ndv` under uniformity.
    (`1 - (1 - 1/d)^k` would model `k` independent Bernoulli draws, which this is not.)

    Two refinements over a raw `len(values)/ndv`:

    * literals are **deduplicated** — `x IN (1, 1, 2)` selects two values, not three, and
      a generated `IN` list is a common place for repeats to appear;
    * a literal with a measured most-common-value frequency contributes that frequency
      instead of the uniform `1/ndv`, which is where `1/ndv` is most wrong.

    Falls back to `k * eq_selectivity` when the column's distinct count is unknown.
    """
    distinct = _dedup(expr.values)
    k = len(distinct)
    if k == 0:
        return 0.0  # `x IN ()` matches nothing
    if not isinstance(expr.input, Col):
        return min(1.0, k * cfg.eq_selectivity)
    name = expr.input.name
    column_ndv = ndv.get(name)
    uniform = 1.0 / column_ndv if column_ndv and column_ndv > 0 else cfg.eq_selectivity
    col_mcv = (mcv or {}).get(name)
    total = 0.0
    for value in distinct:
        freq = _mcv_lookup(col_mcv, value)
        total += uniform if freq is None else freq
    return min(1.0, total)


# Predicates that are two-valued: they return TRUE or FALSE for a NULL input and never
# NULL themselves, so `sel(p) + sel(NOT p) == 1` exactly for these.
_TWO_VALUED = (IsNull, IsNotNull)


def _null_mass(expr: Expr, nulls: dict[str, float]) -> float:
    """The fraction of rows on which `expr` evaluates to NULL rather than TRUE/FALSE.

    Only claimed where 3-valued logic is unambiguous *and* the null count was measured:

    * a two-valued predicate (`IS NULL` / `IS NOT NULL`) is never NULL — mass 0;
    * a null-propagating comparison over columns is NULL exactly when some operand is,
      which under the Fréchet lower bound is at least `max_i f_null(col_i)`. The lower
      bound is the conservative choice: it subtracts the least mass, so correlated nulls
      can never make the negation *under*-estimate.

    With **no measured null count** the mass is 0, not the `null_selectivity` prior: an
    unmeasured column is usually null-free, and guessing a 5% null mass would shrink every
    negation's estimate (under-budgeting memory) on no evidence at all. Anything else
    (`AND`/`OR`, whose Kleene tables make the NULL mass depend on the operands' joint
    distribution, and opaque functions) returns 0 — the plain complement, exactly as before.
    """
    if isinstance(expr, _TWO_VALUED):
        return 0.0
    if isinstance(expr, Binary) and expr.op in _NULL_PROPAGATING:
        return _measured_null_fraction(expr, nulls) or 0.0
    return 0.0


# Comparisons propagate NULL: `NULL OP x` is NULL, so the predicate is neither TRUE nor
# FALSE. `and`/`or` do not (Kleene: `FALSE AND NULL` is FALSE), so they are excluded.
_NULL_PROPAGATING = frozenset({"eq", "ne", "lt", "le", "gt", "ge"})


def _measured_null_fraction(expr: Expr, nulls: dict[str, float]) -> float | None:
    """The largest *measured* null fraction among the columns `expr` reads, else `None`."""
    from batcher.plan.expr_ir import referenced_columns

    measured = [nulls[c] for c in referenced_columns(expr) if c in nulls]
    if not measured:
        return None
    return max(0.0, min(1.0, max(measured)))


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
    expr: Binary,
    op: str,
    cfg: CardinalityConfig,
    quantiles: dict[str, Any],
    bounds: dict[str, tuple[Any, Any]],
    ndv: dict[str, float] | None = None,
    mcv: dict[str, dict[str, float]] | None = None,
) -> float:
    """`col < literal` (and `<=`/`>`/`>=`) selectivity, from the sharpest source available.

    In order of precision: the column's learned quantile boundaries (the fraction below
    the literal, interpolated from the KLL histogram); else its EXACT `min`/`max` bounds
    under a uniformity assumption (linear interpolation across `[min, max]`); else the
    Selinger range constant.

    The `min`/`max` fallback matters because footer/source statistics carry exact bounds
    for *every* column from the first query on, while quantiles only appear after the
    learning loop has measured a run. Without it every TPC-H date-range predicate — Q1,
    Q3-Q7, Q12, Q14, Q15, Q20 all filter on a date interval — estimated a flat 1/3
    regardless of how wide the interval was, even though the column's exact date span was
    already known. Values are mapped to a common ordinal first (`_ordinal`), so a
    `datetime.date`/`datetime`/`Decimal` literal interpolates against bounds of the same
    type instead of falling through as "non-numeric"."""
    side = comparison_col_side(expr)
    if side is None:
        return cfg.range_selectivity
    col, value, col_on_left = side
    x = _ordinal(value)
    if x is None:
        return cfg.range_selectivity
    frac_le = _fraction_below_quantiles(x, quantiles.get(col))
    if frac_le is None:
        frac_le = _fraction_below_bounds(x, bounds.get(col))
    if frac_le is None:
        return cfg.range_selectivity
    # Normalize so the column is on the left, then split on the boundary's point mass.
    #
    # With `F(x) = P(v <= x)` and `eq = P(v = x)`, the four comparisons are
    # `le: F`, `lt: F - eq`, `gt: 1 - F`, `ge: 1 - F + eq`.
    # Treating `lt` as `le` (and `ge` as `gt`) drops that point mass entirely, so
    # `x <= 5` and `x < 5` were estimated identically — wrong by a whole distinct value
    # on a low-cardinality integer or date column, which is exactly where a range
    # predicate is most selective. (The Rust `stats.rs` path already subtracts it.)
    eff = op if col_on_left else _FLIP_OP[op]
    eq = _point_mass(col, value, ndv or {}, mcv or {})
    if eff == "le":
        return frac_le
    if eff == "lt":
        return frac_le - eq
    if eff == "gt":
        return 1.0 - frac_le
    return 1.0 - frac_le + eq  # ge


def _point_mass(
    col: str,
    value: Any,
    ndv: dict[str, float],
    mcv: dict[str, dict[str, float]],
) -> float:
    """`P(col = value)` — the probability mass sitting exactly on a range boundary.

    A measured most-common-value frequency when the literal is a known skew value, else
    the uniform `1/ndv`. Zero when the distinct count is unknown, which degrades the
    strict/non-strict distinction back to none rather than inventing a mass.
    """
    freq = _mcv_lookup(mcv.get(col), value)
    if freq is not None:
        return max(0.0, min(1.0, freq))
    d = ndv.get(col)
    return 1.0 / d if d and d > 0 else 0.0


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


def comparison_col_side(expr: Binary) -> tuple[str, Any, bool] | None:
    """`(column, literal, col_on_left)` for a `col OP literal` / `literal OP col`."""
    if isinstance(expr.left, Col) and isinstance(expr.right, Lit):
        return expr.left.name, expr.right.value, True
    if isinstance(expr.right, Col) and isinstance(expr.left, Lit):
        return expr.right.name, expr.left.value, False
    return None


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


def _equality_selectivity(
    expr: Binary,
    ndv: dict[str, float],
    cfg: CardinalityConfig,
    mcv: dict[str, dict[str, float]],
) -> float:
    """`col = literal` selectivity.

    When the literal is a known most-common-value, use its *measured* frequency —
    far sharper than the uniform `1/ndv` on a skewed column, which is exactly where
    `1/ndv` is most wrong. Otherwise keep `~1/ndv(col)` when the distinct count is
    known (uniformity assumption), else a small default.
    """
    side = comparison_col_side(expr)
    if side is not None:
        col, value, _ = side
        freq = _mcv_lookup(mcv.get(col), value)
        if freq is not None:
            return freq
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
