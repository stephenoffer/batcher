"""Leaf predicate selectivity — one estimate per non-composite predicate.

The middle layer of the `selectivity` package: given a single leaf predicate — an equality,
a range, an `IN` list, a string/LIKE test, a date-part test — and the relation's column
statistics, return the fraction of rows it keeps. These never recurse into a boolean tree;
composing `AND`/`OR`/`NOT` is the combiner's job (`combine`). They build on the scalar
primitives in `scalars`.
"""

from __future__ import annotations

import math
from typing import Any

from batcher.config import CardinalityConfig
from batcher.kyber.stats.selectivity.scalars import (
    _FLIP_OP,
    _column_of_comparison,
    _dedup,
    _fraction_below_bounds,
    _fraction_below_quantiles,
    _mcv_lookup,
    _ordinal,
    _outside_bounds,
    _point_mass,
    comparison_col_side,
)
from batcher.plan.expr_ir import (
    Binary,
    Col,
    DateFunc,
    Expr,
    InList,
    IsNotNull,
    IsNull,
    Lit,
    StrFunc,
)

# Boolean-valued string predicates whose match fraction is a fixed prior — the pattern is
# implicit in the function (`contains` is always a substring, `starts_with` always anchored).
# `like`/`ilike` are *not* here: they carry a raw SQL pattern, so their selectivity depends
# on where the wildcards fall and is computed by `_like_selectivity`.
_STR_PATTERN_SELECTIVITY = {
    "contains": lambda cfg: cfg.substring_selectivity,
    "regexp_matches": lambda cfg: cfg.substring_selectivity,
    "starts_with": lambda cfg: cfg.prefix_selectivity,
    "ends_with": lambda cfg: cfg.prefix_selectivity,
}

# SQL `LIKE` wildcards: `%` matches any run, `_` any single character.
_LIKE_WILDCARDS = ("%", "_")


def _str_func_selectivity(
    expr: StrFunc,
    ndv: dict[str, float],
    cfg: CardinalityConfig,
    mcv: dict[str, dict[str, float]],
) -> float:
    """Selectivity of a boolean string predicate, from its function and (for LIKE) pattern."""
    if expr.fn in ("like", "ilike"):
        return _like_selectivity(expr, ndv, cfg, mcv)
    pattern_sel = _STR_PATTERN_SELECTIVITY.get(expr.fn)
    if pattern_sel is not None:
        return pattern_sel(cfg)
    return cfg.default_filter_selectivity


def _like_selectivity(
    expr: StrFunc,
    ndv: dict[str, float],
    cfg: CardinalityConfig,
    mcv: dict[str, dict[str, float]],
) -> float:
    """`col LIKE pattern` selectivity, read from where the wildcards fall.

    A single blunt `substring_selectivity` for every `LIKE` throws away what the pattern
    plainly says — and the shapes it conflates differ by an order of magnitude:

    * **no wildcards** (`col LIKE 'DELIVER IN PERSON'`) is exact *equality*, so it gets the
      equality estimate (`1/ndv`, or a measured skew frequency) — often 10-100x more
      selective than a substring, and a common shape (TPC-H Q19's `l_shipinstruct`, an
      enum column matched to a constant);
    * an **anchored prefix** (`'AIR%'`) or **suffix** (`'%ing'`) matches far fewer rows than
      an unanchored substring, so it gets `prefix_selectivity`;
    * a genuine **substring** (`'%foo%'`, or any pattern with an interior `%`/`_`) keeps
      `substring_selectivity`.

    An unparseable or non-literal pattern falls back to `substring_selectivity`, exactly as
    every `LIKE` did before.
    """
    pat = expr.pattern
    if not isinstance(pat, str):
        return cfg.substring_selectivity
    if not any(w in pat for w in _LIKE_WILDCARDS):
        # No wildcards → `LIKE` is `=`. Reuse the equality estimate over the literal.
        if isinstance(expr.input, Col):
            name = expr.input.name
            freq = _mcv_lookup((mcv or {}).get(name), pat)
            if freq is not None:
                return freq
            d = ndv.get(name)
            if d and d > 0:
                return 1.0 / d
        return cfg.eq_selectivity
    if "_" not in pat:
        body = pat.strip("%")
        # An anchored match: the single wildcard run sits at exactly one end.
        if body and "%" not in body and (pat.startswith("%") ^ pat.endswith("%")):
            return cfg.prefix_selectivity
    return cfg.substring_selectivity


def _in_list_selectivity(
    expr: InList,
    ndv: dict[str, float],
    cfg: CardinalityConfig,
    mcv: dict[str, dict[str, float]] | None = None,
    bounds: dict[str, tuple[Any, Any]] | None = None,
) -> float:
    """`col IN (v1, …, vk)` — the union of `k` equality predicates.

    `col = v_i` and `col = v_j` are **mutually exclusive** for distinct literals, so the
    union is their *sum*, not an independence union: exactly `k/ndv` under uniformity.
    (`1 - (1 - 1/d)^k` would model `k` independent Bernoulli draws, which this is not.)

    Three refinements over a raw `len(values)/ndv`:

    * literals are **deduplicated** — `x IN (1, 1, 2)` selects two values, not three, and
      a generated `IN` list is a common place for repeats to appear;
    * literals **outside the column's bounds** are dropped — they provably match nothing, so
      `x IN (5, 999)` over `x ∈ [0, 10]` selects one value, not two;
    * a literal with a measured most-common-value frequency contributes that frequency
      instead of the uniform `1/ndv`, which is where `1/ndv` is most wrong.

    Falls back to `k * eq_selectivity` when the column's distinct count is unknown.
    """
    distinct = _dedup(expr.values)
    k = len(distinct)
    if k == 0:
        return 0.0  # `x IN ()` matches nothing
    dom = _date_part_domain(expr.input)
    if dom is not None:  # `month(d) IN (6,7,8)` → k of n equiprobable field values
        return min(1.0, k / float(dom[1] - dom[0] + 1))
    if not isinstance(expr.input, Col):
        return min(1.0, k * cfg.eq_selectivity)
    name = expr.input.name
    bound = (bounds or {}).get(name)
    if bound is not None:
        distinct = [v for v in distinct if not _outside_bounds(v, bound)]
        if not distinct:
            return 0.0
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

# Comparisons propagate NULL: `NULL OP x` is NULL, so the predicate is neither TRUE nor
# FALSE. `and`/`or` do not (Kleene: `FALSE AND NULL` is FALSE), so they are excluded.
_NULL_PROPAGATING = frozenset({"eq", "ne", "lt", "le", "gt", "ge"})


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
    # `col IN (...)` and `col LIKE p` are NULL when the column is NULL, so `NOT IN` / `NOT
    # LIKE` drop those rows too — the same 3-valued complement as a comparison. Without this
    # a `NOT IN` over a 30%-null column was estimated to keep the null rows it actually drops.
    if isinstance(expr, (InList, StrFunc)):
        return _measured_null_fraction(expr, nulls) or 0.0
    return 0.0


def _measured_null_fraction(expr: Expr, nulls: dict[str, float]) -> float | None:
    """The largest *measured* null fraction among the columns `expr` reads, else `None`."""
    from batcher.plan.expr_ir import referenced_columns

    measured = [nulls[c] for c in referenced_columns(expr) if c in nulls]
    if not measured:
        return None
    return max(0.0, min(1.0, max(measured)))


# Date/time field extractions with a small, bounded, ~uniform domain `[lo, hi]` (inclusive).
# A predicate on the extracted field — `month(d) = 6`, `hour(ts) < 9`, `dayofweek(d) IN (0,6)`
# — is otherwise blunt: `month(d) = 6` fell to the flat `eq_selectivity` (0.1) when the true
# fraction is 1/12, and `month(d) < 6` to the flat `range_selectivity` (1/3) when it is ~1/2.
# These are the recurring, non-sargable temporal filters the year/decade sargable rewrite
# deliberately leaves alone, so estimating them here is the only place they get sharpened.
# `year`/`decade`/`century`/`epoch`/`iso_year` are unbounded and absent on purpose.
_DATE_PART_DOMAIN: dict[str, tuple[int, int]] = {
    "month": (1, 12),
    "monthname": (1, 12),
    "quarter": (1, 4),
    "day": (1, 31),
    "day_of_week": (0, 6),
    "dayname": (0, 6),
    "isodow": (1, 7),
    "day_of_year": (1, 366),
    "hour": (0, 23),
    "minute": (0, 59),
    "second": (0, 59),
    "week": (1, 53),
    "days_in_month": (28, 31),
}


def _date_part_domain(expr: Expr) -> tuple[int, int] | None:
    """`(lo, hi)` if `expr` is a bounded-domain date-part extraction, else None."""
    if isinstance(expr, DateFunc):
        return _DATE_PART_DOMAIN.get(expr.fn)
    return None


def _date_part_cardinality(expr: Binary) -> float | None:
    """The domain size of a `date_part(col) OP literal` comparison, for `1/n` equality."""
    for dom in (_date_part_domain(expr.left), _date_part_domain(expr.right)):
        if dom is not None:
            return float(dom[1] - dom[0] + 1)
    return None


def _date_part_range_selectivity(expr: Binary, op: str) -> float | None:
    """`date_part(col) OP literal` range selectivity over the field's discrete uniform domain.

    The domain is the integers `[lo, hi]` (12 months, 7 weekdays, 24 hours…), so the CDF is
    `P(v <= x) = (floor(x) - lo + 1)/n` and the boundary point mass is `1/n` — applied per
    comparison exactly as `_range_selectivity` does for a real column. Returns None when
    neither side is a bounded date part or the literal has no linear order (a `monthname`
    string), so those defer to the normal path.
    """
    if isinstance(expr.left, DateFunc) and isinstance(expr.right, Lit):
        dom, value, col_on_left = _date_part_domain(expr.left), expr.right.value, True
    elif isinstance(expr.right, DateFunc) and isinstance(expr.left, Lit):
        dom, value, col_on_left = _date_part_domain(expr.right), expr.left.value, False
    else:
        return None
    if dom is None:
        return None
    x = _ordinal(value)
    if x is None:
        return None
    lo, hi = dom
    n = float(hi - lo + 1)
    xf = math.floor(x)
    if xf < lo:
        frac_le = 0.0
    elif xf >= hi:
        frac_le = 1.0
    else:
        frac_le = (xf - lo + 1) / n
    eff = op if col_on_left else _FLIP_OP[op]
    eq = 1.0 / n
    if eff == "le":
        return frac_le
    if eff == "lt":
        return frac_le - eq
    if eff == "gt":
        return 1.0 - frac_le
    return 1.0 - frac_le + eq  # ge


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
    date_part = _date_part_range_selectivity(expr, op)
    if date_part is not None:
        return date_part
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


def _equality_selectivity(
    expr: Binary,
    ndv: dict[str, float],
    cfg: CardinalityConfig,
    mcv: dict[str, dict[str, float]],
    bounds: dict[str, tuple[Any, Any]] | None = None,
) -> float:
    """`col = literal` (or `col = col`) selectivity.

    A literal **outside the column's `[min, max]` bounds** matches nothing: bounds are a
    valid (if loose) superset of the actual values, so a value beyond them provably cannot
    appear. This is the equality analogue of what the range path already does — `col > max`
    is already estimated at 0 — and closes a real inconsistency: a stale/partition-mismatched
    equality (`WHERE d = '2025-01-01'` over data ending in 2024) kept `1/ndv` of the table
    instead of ~nothing, and the `!=` complement then dropped that mass off a full scan.

    When the literal is a known most-common-value, use its *measured* frequency —
    far sharper than the uniform `1/ndv` on a skewed column, which is exactly where
    `1/ndv` is most wrong. Otherwise keep `~1/ndv(col)` when the distinct count is
    known (uniformity assumption), else a small default.

    A **column = column** equality (a residual filter comparing two columns of one
    relation — a self-join residual, a de-normalized cross-column check) is the Selinger
    containment case: under uniformity the match fraction is ``1 / max(d_a, d_b)``. With
    only one side's distinct count known, ``max >= d_known`` so ``1/d_known`` over-estimates
    — the safe direction (over-budget, never an under-sized hash table). Without this it
    fell through to the flat `eq_selectivity`, a ~10x over-estimate on a low-cardinality
    join-key-shaped column.
    """
    side = comparison_col_side(expr)
    if side is not None:
        col, value, _ = side
        if _outside_bounds(value, (bounds or {}).get(col)):
            return 0.0
        freq = _mcv_lookup(mcv.get(col), value)
        if freq is not None:
            return freq
    # `month(d) = 6` etc.: one of ~n equiprobable field values, whatever the literal's type.
    period = _date_part_cardinality(expr)
    if period is not None:
        return 1.0 / period
    if isinstance(expr.left, Col) and isinstance(expr.right, Col):
        counts = [ndv[c] for c in (expr.left.name, expr.right.name) if c in ndv and ndv[c] > 0]
        if counts:
            return 1.0 / max(counts)
    col = _column_of_comparison(expr)
    if col is not None and col in ndv and ndv[col] > 0:
        return 1.0 / ndv[col]
    return cfg.eq_selectivity
