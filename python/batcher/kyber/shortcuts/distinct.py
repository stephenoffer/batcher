"""Cardinality shortcuts — how many distinct values, and whether a column is a key.

The distinct count is the statistic with two provenances, and the distinction is the whole
family. An **exact** ndv (an immutable in-memory relation computes one; some catalogs record
one) answers `n_unique()` and settles uniqueness outright. A **sketched** ndv (an HLL, a
Parquet `distinct_count`, which the format explicitly does not guarantee) answers nothing
exactly — it informs cost, and it backs the opt-in `approx_*` answers, and that is all.

`Facts` has already separated them: `ndv` is exact or absent, `approx_ndv` is whatever is
known. A function here that reads `approx_ndv` says `approx` in its name.
"""

from __future__ import annotations

from batcher.kyber.shortcuts.facts import Facts
from batcher.kyber.shortcuts.nulls import non_null_count

__all__ = [
    "approx_cardinality_ratio",
    "approx_n_unique",
    "duplicate_count",
    "has_duplicates",
    "is_key",
    "is_low_cardinality",
    "is_unique",
    "n_unique",
]


def n_unique(facts: Facts, column: str) -> int | None:
    """The exact number of distinct non-null values, or None when not provable.

    Never answered from a sketch: an HLL estimate that is 0.5% off is a *wrong* answer to
    `COUNT(DISTINCT)`, however useful it is to the cost model. Use `approx_n_unique` when an
    approximate count is what you actually want.
    """
    return facts.col(column).ndv


def approx_n_unique(facts: Facts, column: str) -> int | None:
    """An approximate distinct count from whatever sketch is known, or None if none is.

    Explicitly approximate: this reads the ndv at *any* provenance, including an HLL sketch
    and a Parquet `distinct_count`. It must only ever back an `approx_*` answer.
    """
    return facts.col(column).approx_ndv


def is_unique(facts: Facts, column: str) -> bool | None:
    """Whether every non-null value of `column` occurs once, or None when not provable.

    True iff the distinct count equals the non-null count — which needs an exact ndv *and*
    an exact row and null count. A column of all nulls is vacuously unique (no value repeats).
    """
    ndv = facts.col(column).ndv
    non_null = non_null_count(facts, column)
    if ndv is None or non_null is None:
        return None
    return ndv == non_null


def has_duplicates(facts: Facts, column: str) -> bool | None:
    """Whether some non-null value of `column` occurs more than once, or None if not provable."""
    unique = is_unique(facts, column)
    return None if unique is None else not unique


def duplicate_count(facts: Facts, column: str) -> int | None:
    """How many non-null values are repeats of a value already seen, or None if not provable.

    `count(col) - count(distinct col)` — the number of rows a `DISTINCT` on this column
    would remove. Zero exactly when the column is unique.
    """
    ndv = facts.col(column).ndv
    non_null = non_null_count(facts, column)
    if ndv is None or non_null is None:
        return None
    return non_null - ndv


def is_key(facts: Facts, column: str) -> bool | None:
    """Whether `column` is a primary key — unique *and* never null — or None if not provable.

    The fact a join planner wants: a key column on the build side makes the join
    cardinality-preserving, and knowing it without a scan is what lets the plan be chosen
    before the first byte is read.
    """
    unique = is_unique(facts, column)
    nulls = facts.col(column).null_count
    if unique is None or nulls is None:
        return None
    return unique and nulls == 0


def is_low_cardinality(facts: Facts, column: str, max_distinct: int = 128) -> bool | None:
    """Whether `column` has at most `max_distinct` distinct values, or None if not provable.

    The dictionary-encoding / one-hot / `GROUP BY`-fits-in-cache question. Exact only: a
    sketch could put a column on the wrong side of the threshold, and the caller is about to
    make a representation decision on the answer.
    """
    ndv = facts.col(column).ndv
    return None if ndv is None else ndv <= max_distinct


def approx_cardinality_ratio(facts: Facts, column: str) -> float | None:
    """The approximate share of rows that hold a distinct value (`ndv / rows`), or None.

    Near 1.0 the column is key-like; near 0 it is categorical. Explicitly approximate — it
    reads the estimated row count and a sketched ndv, so it describes a column, it does not
    answer a query.
    """
    ndv = facts.col(column).approx_ndv
    rows = facts.estimated_rows
    if ndv is None or rows <= 0:
        return None
    return min(1.0, ndv / rows)
