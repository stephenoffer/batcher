"""Relation-shape shortcuts — how many rows, how many columns, is there anything there.

The cheapest family: a Parquet footer, an ORC stripe header, a lakehouse manifest and a
warehouse catalog all record a row count, so `count()` and `is_empty()` are the two
questions a scan should never be started to answer. Everything here reads `Facts.rows`,
which is populated only from an `EXACT` provenance — an estimate never answers a count.
"""

from __future__ import annotations

from batcher.kyber.shortcuts.facts import Facts

__all__ = [
    "count",
    "estimated_rows",
    "is_empty",
    "is_not_empty",
    "num_columns",
    "row_count_is_exact",
    "shape",
]


def count(facts: Facts) -> int | None:
    """The exact number of result rows, or None when it is not provable from metadata."""
    return facts.rows


def is_empty(facts: Facts) -> bool | None:
    """Whether the relation has no rows, or None when not provable from metadata."""
    return None if facts.rows is None else facts.rows == 0


def is_not_empty(facts: Facts) -> bool | None:
    """Whether the relation has at least one row, or None when not provable."""
    return None if facts.rows is None else facts.rows > 0


def shape(facts: Facts) -> tuple[int, int] | None:
    """The `(rows, columns)` shape, or None when the row count is not provable.

    The column count is always free (it is the plan's output schema); the row count is
    what has to be earned, so the pair is answerable exactly when `count` is.
    """
    return None if facts.rows is None else (facts.rows, len(facts.columns))


def num_columns(facts: Facts) -> int:
    """The number of output columns — always free, it is the plan's schema."""
    return len(facts.columns)


def estimated_rows(facts: Facts) -> float:
    """The estimated row count — **always** available, and never exact.

    The cost model's number: it may come from a footer (and then be exact anyway), from a
    sketch, from a learned prior, or from a Selinger default. It answers "roughly how big
    is this?" for sizing and planning; it must never answer `count()`.
    """
    return facts.estimated_rows


def row_count_is_exact(facts: Facts) -> bool:
    """Whether `count()` can be answered from metadata — i.e. whether it is free."""
    return facts.rows_known
