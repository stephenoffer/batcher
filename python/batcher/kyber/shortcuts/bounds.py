"""Bound shortcuts — a column's extremes, and the facts that follow from them.

A columnar footer records each column's min and max. That single pair answers `min()` and
`max()` outright, and — once you notice a bound is an *actual value present in the column* —
several more questions besides: a column whose min equals its max is constant; the width of
its range is `max - min`; and every "is every value above X" question is decided by the min.

`minimum` and `maximum` read the gated facets straight from `Facts`. Everything *derived*
from a pair of bounds goes through `orderable`, which refuses a float column that may hold a
NaN — NaN has no place in an arithmetic derivation, and a bound that silently dropped one
describes a column that isn't there.
"""

from __future__ import annotations

import math
from typing import Any

from batcher.kyber.shortcuts.facts import ColumnFacts, Facts
from batcher.plan.stats import ambiguous_float_bound

__all__ = [
    "abs_max",
    "bounds",
    "constant_value",
    "is_constant",
    "maximum",
    "midpoint",
    "minimum",
    "orderable",
    "value_range",
]


def orderable(facts: Facts, column: str) -> ColumnFacts | None:
    """`column`'s facts iff a derivation over its bounds is sound, else None.

    The one gate for every *derived* bound fact (range, midpoint, the `checks` family, the
    join-disjointness proof). It exists because a float bound and a float comparison do not
    always mean the same thing, and a derivation that assumes they do returns a wrong answer.

    It refuses a float column in three cases:

    * the source does not declare NaN-aware bounds (`Facts.nan_safe`) — a footer's max is
      the largest *non-NaN* value, so the true maximum may be a NaN nobody recorded;
    * a bound **is** NaN — a NaN is present, and the engine orders NaN above every number
      while Python orders it below nothing, so no comparison derived from the bounds
      describes what executing would produce; and
    * a bound **is a zero of either sign** — this is where `-0.0` lives, and the engine
      distinguishes `-0.0` from `0.0` in a comparison (it compares floats on their total
      order) while Python treats them as equal. A `min` of `-0.0` would let us "prove" that
      every value is non-negative, and executing `WHERE f < 0` would then return the very row
      we proved away.

    Integers, decimals, temporals, booleans, and strings have neither NaN nor a signed zero,
    so they pass straight through and answer from metadata as they always did.

    NOTE: the second and third refusals are conservative *because* the engine's float
    comparisons currently disagree with DuckDB (`WHERE f = 0.0` misses `-0.0`; `WHERE f > 1`
    matches NaN — see `docs/internals/bug_hunt_ledger.md`). Declining is sound under either
    semantics: it costs a scan, never an answer. If the engine's comparisons are moved to
    IEEE, these refusals can be revisited — but not before.
    """
    col = facts.col(column)
    if not col.is_float:
        return col
    if not facts.nan_safe:
        return None
    if _is_ambiguous_float(col.min) or _is_ambiguous_float(col.max):
        return None
    return col


def _is_nan(value: Any) -> bool:
    """True iff `value` is a floating NaN."""
    return isinstance(value, float) and math.isnan(value)


def _is_ambiguous_float(value: Any) -> bool:
    """Whether this bound sits where the engine's float order and Python's disagree.

    Exactly the two places: a NaN (the engine ranks it above every number, Python ranks it
    nowhere) and a zero (the engine separates `-0.0` from `0.0`, Python does not).
    """
    return ambiguous_float_bound(value)


def minimum(facts: Facts, column: str) -> Any | None:
    """The exact `min(column)` from a footer/manifest bound, or None when not provable.

    Sound for floats without any NaN gate: NaN is the greatest value in SQL's total order,
    so a bound that dropped one can never have dropped the minimum.
    """
    return facts.col(column).min


def maximum(facts: Facts, column: str) -> Any | None:
    """The exact `max(column)` from a footer/manifest bound, or None when not provable.

    Populated only when the source's bounds account for NaN (`Facts.nan_safe`); a footer's
    max is the largest *non-NaN* value, which is not the answer SQL gives.
    """
    return facts.col(column).max


def bounds(facts: Facts, column: str) -> tuple[Any, Any] | None:
    """The exact `(min, max)` pair, or None unless **both** are provable."""
    col = facts.col(column)
    return (col.min, col.max) if col.has_bounds else None


def value_range(facts: Facts, column: str) -> Any | None:
    """The width of the column's range (`max - min`), or None when not provable.

    Numeric columns only — a range over strings has no meaning, and a temporal one would
    return a duration whose unit the caller did not ask for.
    """
    col = orderable(facts, column)
    if col is None or not col.is_numeric or not col.has_bounds:
        return None
    return col.max - col.min


def midpoint(facts: Facts, column: str) -> float | None:
    """The midpoint of the column's range (`(min + max) / 2`), or None when not provable.

    The centre of the *range*, not the mean of the values and not the median — a column of
    `[0, 0, 0, 100]` has a midpoint of 50. Useful for choosing a split point (a range
    partition, a binary-search probe) without looking at the data.
    """
    col = orderable(facts, column)
    if col is None or not col.is_numeric or not col.has_bounds:
        return None
    return (float(col.min) + float(col.max)) / 2.0


def abs_max(facts: Facts, column: str) -> float | None:
    """The largest absolute value the column can hold (`max(|min|, |max|)`), or None.

    What a numeric column needs to be *stored* in — the fact that decides whether an
    ``int64`` column fits in an ``int32``, or whether a feature needs scaling — read from
    the bounds rather than from the values.
    """
    col = orderable(facts, column)
    if col is None or not col.is_numeric or not col.has_bounds:
        return None
    return max(abs(float(col.min)), abs(float(col.max)))


def is_constant(facts: Facts, column: str) -> bool | None:
    """Whether every non-null value of `column` is the same, or None when not provable.

    True when the bounds coincide: a min and a max are values that *occur*, so `min == max`
    means one distinct value occurs and no other. A column with no non-null value at all has
    no bounds, so it is not provable here and falls back to execution.
    """
    col = orderable(facts, column)
    if col is None or not col.has_bounds:
        return None
    return bool(col.min == col.max)


def constant_value(facts: Facts, column: str) -> Any | None:
    """The single value `column` holds when it is constant, else None.

    None means either "not constant" or "not provable" — ask `is_constant` to tell them
    apart. A caller wanting the value of a column it already knows to be constant (a
    partition key, a materialized literal) gets it here for free.
    """
    return facts.col(column).min if is_constant(facts, column) else None
