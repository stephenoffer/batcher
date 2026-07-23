"""Null-shape shortcuts — how much of a column is missing, and which columns are complete.

Parquet and ORC record a per-chunk null count, and a lakehouse manifest records it per
file, so "how many nulls are in this column" is a footer read, not a scan. Every answer
here needs an `EXACT` null count; the ones phrased against the whole relation
(`non_null_count`, `all_null`, `is_complete`) additionally need the exact row count, since
a null count means nothing without a total to measure it against.
"""

from __future__ import annotations

from batcher.kyber.shortcuts.facts import Facts

__all__ = [
    "all_null",
    "columns_with_nulls",
    "complete_columns",
    "has_nulls",
    "is_complete",
    "no_nulls",
    "non_null_count",
    "null_count",
    "null_counts",
    "null_fraction",
]


def null_count(facts: Facts, column: str) -> int | None:
    """The exact number of nulls in `column`, or None when not provable."""
    return facts.col(column).null_count


def non_null_count(facts: Facts, column: str) -> int | None:
    """The exact number of non-null values in `column` (SQL ``COUNT(col)``), or None.

    `rows - null_count`, so it needs both to be exact — a filtered relation has neither.
    """
    nulls = facts.col(column).null_count
    if facts.rows is None or nulls is None:
        return None
    return facts.rows - nulls


def null_fraction(facts: Facts, column: str) -> float | None:
    """The exact fraction of `column` that is null, in ``[0, 1]``, or None when not provable.

    An empty relation has no rows to be null, and is reported as ``0.0`` rather than a
    division by zero.
    """
    nulls = facts.col(column).null_count
    if facts.rows is None or nulls is None:
        return None
    return 0.0 if facts.rows == 0 else nulls / facts.rows


def has_nulls(facts: Facts, column: str) -> bool | None:
    """Whether `column` contains at least one null, or None when not provable."""
    nulls = facts.col(column).null_count
    return None if nulls is None else nulls > 0


def no_nulls(facts: Facts, column: str) -> bool | None:
    """Whether `column` contains no null at all, or None when not provable."""
    nulls = facts.col(column).null_count
    return None if nulls is None else nulls == 0


def all_null(facts: Facts, column: str) -> bool | None:
    """Whether every value of `column` is null, or None when not provable.

    An **empty** relation is not reported all-null: it has no values, and calling it
    all-null would make `all_null` and `no_nulls` both true, which is a contradiction a
    caller would be right to trust and wrong to act on.
    """
    nulls = facts.col(column).null_count
    if facts.rows is None or nulls is None:
        return None
    return facts.rows > 0 and nulls == facts.rows


def null_counts(facts: Facts) -> dict[str, int] | None:
    """Every column's exact null count, or None unless **all** of them are provable.

    All-or-nothing on purpose: a partial map reads as "these columns have nulls and the
    rest do not", which is a different (and false) statement from "I only know these".
    """
    out: dict[str, int] = {}
    for name, col in facts.columns.items():
        if col.null_count is None:
            return None
        out[name] = col.null_count
    return out


def columns_with_nulls(facts: Facts) -> list[str] | None:
    """The columns that contain at least one null, or None unless all are provable."""
    counts = null_counts(facts)
    return None if counts is None else [name for name, n in counts.items() if n > 0]


def complete_columns(facts: Facts) -> list[str] | None:
    """The columns that contain no null at all, or None unless all are provable."""
    counts = null_counts(facts)
    return None if counts is None else [name for name, n in counts.items() if n == 0]


def is_complete(facts: Facts) -> bool | None:
    """Whether the relation has no null anywhere, or None unless every column is provable."""
    counts = null_counts(facts)
    return None if counts is None else all(n == 0 for n in counts.values())
