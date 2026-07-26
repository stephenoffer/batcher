"""Array subscripting is 1-based or 0-based per dialect, and sqlglot says which.

sqlglot rewrites `l[2]` to a 0-based index for the dialects whose subscript is 1-based
(duckdb, postgres) and leaves `offset` unset. Where it cannot rewrite — Spark's
`element_at(a, 2)`, which is 1-based while Spark's own `a[2]` is 0-based — it keeps the
written index and records the base in `Bracket.offset`.

The translator ignored `offset`, so every such subscript returned the *next* element:
`element_at(array(1, 2, 3), 2)` answered `3`. Found by running Spark's own documented
examples through `bt.sql(dialect="spark")`.

These are unit tests rather than differential ones because the divergence is in *parsing*
a dialect DuckDB cannot read, so DuckDB is not the oracle here — Spark's own documented
output is, and it is quoted in each case.
"""

from __future__ import annotations

import pytest

import batcher as bt


@pytest.mark.unit
@pytest.mark.parametrize(
    ("query", "dialect", "expected"),
    [
        # Spark: `element_at` is 1-based (Spark's documented answer is 2).
        ("SELECT element_at(array(1, 2, 3), 2) AS r", "spark", 2),
        ("SELECT try_element_at(array(1, 2, 3), 2) AS r", "spark", 2),
        # Spark: the bare subscript is 0-based, so the same index is the third element.
        ("SELECT array(1, 2, 3)[2] AS r", "spark", 3),
        # DuckDB: 1-based subscript and 1-based `list_extract`/`element_at`.
        ("SELECT ([1, 2, 3])[2] AS r", "duckdb", 2),
        ("SELECT list_extract([1, 2, 3], 2) AS r", "duckdb", 2),
        ("SELECT element_at([1, 2, 3], 2) AS r", "duckdb", 2),
        # Postgres: 1-based subscript.
        ("SELECT (ARRAY[1, 2, 3])[2] AS r", "postgres", 2),
    ],
)
def test_subscript_indexing_follows_the_dialect(query, dialect, expected):
    assert bt.sql(query, dialect=dialect).to_pydict()["r"][0] == expected
