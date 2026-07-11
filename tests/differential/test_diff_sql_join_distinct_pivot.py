"""SQL parity fixes for the four silent-wrong-answer cases the DuckDB audit found.

NATURAL JOIN is now implemented (join on shared columns) and must match DuckDB;
DISTINCT ON, PIVOT/UNPIVOT, and a window-frame EXCLUDE clause are rejected with a
clear error rather than silently returning the wrong rows.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt


@pytest.fixture
def ab(duck):
    a = pa.table({"k": [1, 2, 3], "g": ["x", "y", "x"], "va": [10, 20, 30]})
    b = pa.table({"k": [2, 3, 4], "g": ["y", "x", "z"], "vb": [200, 300, 400]})
    duck.register("a", a)
    duck.register("b", b)
    return a, b


@pytest.mark.differential
@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM a NATURAL JOIN b",
        "SELECT * FROM a NATURAL LEFT JOIN b",
        "SELECT * FROM a NATURAL FULL JOIN b",
    ],
)
def test_natural_join_matches_duckdb(duck, ab, query):
    """NATURAL JOIN joins on every shared column name — matches DuckDB (not a cross join)."""
    from conftest import assert_same

    a, b = ab
    out = bt.sql(query, a=a, b=b).collect()
    # Shared columns (k, g) appear once, in the left's order — same as DuckDB.
    assert out.column_names == duck.sql(query).to_arrow_table().column_names
    assert_same(out, duck.sql(query))


@pytest.mark.differential
def test_natural_join_without_shared_columns_raises(ab):
    from batcher._internal.errors import PlanError

    a = bt.from_pydict({"p": [1, 2]})
    b = bt.from_pydict({"q": [3, 4]})
    with pytest.raises(PlanError, match="needs at least one shared column"):
        bt.sql("SELECT * FROM a NATURAL JOIN b", a=a, b=b).collect()


@pytest.mark.differential
@pytest.mark.parametrize(
    ("query", "message"),
    [
        ("SELECT DISTINCT ON (g) g, va FROM a ORDER BY va", "DISTINCT ON"),
        ("SELECT * FROM a PIVOT (SUM(va) FOR g IN ('x', 'y'))", "PIVOT"),
        (
            "SELECT * FROM (SELECT 1 AS x, 2 AS y) UNPIVOT (val FOR col IN (x, y))",
            "UNPIVOT",
        ),
        (
            "SELECT SUM(va) OVER (ORDER BY k ROWS BETWEEN 1 PRECEDING AND CURRENT ROW "
            "EXCLUDE TIES) FROM a",
            "EXCLUDE",
        ),
    ],
)
def test_unsupported_constructs_raise_not_silent(ab, query, message):
    """Each formerly-silent-wrong construct now raises with a clear message."""
    a, b = ab
    with pytest.raises(NotImplementedError, match=message):
        bt.sql(query, a=a, b=b).collect()
