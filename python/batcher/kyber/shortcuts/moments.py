"""Additive shortcuts — a column's total and its average, when something already recorded them.

Unlike a bound, a sum is not something a Parquet footer carries. It comes from a source that
computed it: an immutable in-memory relation (which caches an exact `SUM`/`AVG` per column the
first time either is asked for, so the *second* query is free), or a catalog/materialized view
that maintains one. When nothing recorded it, these return None and a real aggregate runs —
that is the normal case, not a failure.

The one derivation worth having: a recorded total plus an exact non-null count gives the mean,
so a source that records only a sum still answers `avg()`.
"""

from __future__ import annotations

from batcher.kyber.shortcuts.facts import Facts
from batcher.kyber.shortcuts.nulls import non_null_count

__all__ = ["average", "total"]


def total(facts: Facts, column: str) -> float | int | None:
    """The exact `sum(column)` over its non-null values, or None when not provable.

    SQL's `SUM` over a relation with no non-null value is NULL, not 0 — so an empty relation
    is deliberately *not* answered here (a recorded total of 0 would be a wrong answer, not a
    conservative one). It falls through to execution, which returns NULL correctly.
    """
    if facts.rows == 0:
        return None
    return facts.col(column).total_sum


def average(facts: Facts, column: str) -> float | None:
    """The exact `avg(column)` over its non-null values, or None when not provable.

    Read from a recorded mean when the source has one, else derived from a recorded total and
    the exact non-null count. An all-null or empty column has no average (SQL returns NULL),
    and dividing by its zero non-null count would be the wrong answer twice over — so it is
    left to execution.
    """
    col = facts.col(column)
    if col.mean is not None:
        return float(col.mean)
    if col.total_sum is None:
        return None
    non_null = non_null_count(facts, column)
    if not non_null:  # None (not provable) or 0 (SQL NULL, not a division)
        return None
    return float(col.total_sum) / non_null
