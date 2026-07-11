"""Differential tests for SQL null ordering and `SELECT *` star modifiers.

Both were previously *silently* wrong: `ORDER BY x NULLS FIRST` dropped the null
clause and `SELECT * EXCLUDE (c)` returned every column. A silent wrong answer is
the worst failure mode, so each construct is pinned against DuckDB here.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError


@pytest.fixture
def nulls(duck):
    t = pa.table({"x": [3, None, 1, None, 2], "g": ["a", "b", "a", "b", "a"]})
    duck.register("t", t)
    return t


@pytest.fixture
def wide(duck):
    u = pa.table({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    duck.register("u", u)
    return u


@pytest.mark.differential
@pytest.mark.parametrize(
    "query",
    [
        # Explicit null placement, both directions.
        "SELECT x FROM t ORDER BY x NULLS FIRST",
        "SELECT x FROM t ORDER BY x NULLS LAST",
        "SELECT x FROM t ORDER BY x DESC NULLS FIRST",
        "SELECT x FROM t ORDER BY x DESC NULLS LAST",
        # Implicit default must stay NULLS LAST for both directions (DuckDB parity).
        "SELECT x FROM t ORDER BY x",
        "SELECT x FROM t ORDER BY x DESC",
        # Positional ORDER BY carries the null clause too.
        "SELECT x FROM t ORDER BY 1 NULLS FIRST",
        # Per-key null placement in a multi-key sort.
        "SELECT g, x FROM t ORDER BY g ASC, x DESC NULLS FIRST",
        # Null ordering through an aggregate output.
        "SELECT g, MAX(x) AS m FROM t GROUP BY g ORDER BY m NULLS FIRST",
    ],
)
def test_null_ordering_vs_duckdb(duck, nulls, query):
    from conftest import assert_same_ordered

    out = bt.sql(query, t=nulls).collect()
    assert_same_ordered(out, duck.sql(query))


@pytest.mark.differential
@pytest.mark.parametrize(
    "query",
    [
        "SELECT * EXCLUDE (b) FROM u",
        "SELECT * EXCLUDE (a, b) FROM u",
        "SELECT * REPLACE (a * 10 AS a) FROM u",
        "SELECT * REPLACE (a + b AS a, c * 2 AS c) FROM u",
        "SELECT * RENAME (a AS z) FROM u",
        "SELECT * EXCLUDE (c) REPLACE (b + 1 AS b) FROM u",
        "SELECT * FROM u",
    ],
)
def test_star_modifiers_vs_duckdb(duck, wide, query):
    from conftest import assert_same

    out = bt.sql(query, u=wide).collect()
    duck_table = duck.sql(query).to_arrow_table()
    # Column *order* is part of the star-modifier contract, not just the set.
    assert out.column_names == duck_table.column_names
    assert_same(out, duck.sql(query))


@pytest.mark.differential
def test_star_except_is_the_ansi_spelling_of_exclude(wide):
    """`EXCEPT` and `EXCLUDE` are the same star modifier (DuckDB accepts only the latter)."""
    assert bt.sql("SELECT * EXCEPT (b) FROM u", u=wide).to_pydict() == {
        "a": [1, 2],
        "c": [5, 6],
    }


@pytest.mark.differential
@pytest.mark.parametrize(
    ("query", "message"),
    [
        ("SELECT * FROM nosuch", "unknown table"),
        ("SELECT a FROM u ORDER BY 9", "ORDER BY position 9 is out of range"),
        ("SELECT a, COUNT(*) AS n FROM u GROUP BY 7", "GROUP BY position 7 is out of range"),
        ("SELECT * EXCLUDE (zz) FROM u", "unknown column"),
        ("SELECT * REPLACE (1 AS zz) FROM u", "unknown column"),
        ("SELECT * RENAME (zz AS q) FROM u", "unknown column"),
        ("SELECT * ILIKE 'a%' FROM u", "ILIKE is not supported"),
    ],
)
def test_rejects_with_typed_plan_error(wide, query, message):
    """Unsupported/invalid SQL raises the typed `PlanError`, never a bare KeyError/IndexError."""
    with pytest.raises(PlanError, match=message):
        bt.sql(query, u=wide).collect()
