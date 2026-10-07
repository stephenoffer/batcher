"""The DB-API adapter vs DuckDB's own DB-API cursor, on the same statements and parameters.

Both are PEP 249 drivers with ``qmark`` parameters, so the same client code runs against
each. Every query here is ordered, and the rows are compared as ordered lists of tuples:
the cursor is the thing under test, and a cursor that reorders rows is wrong.
"""

from __future__ import annotations

import datetime as dt

import pytest

import batcher as bt
from batcher import dbapi

duckdb = pytest.importorskip("duckdb")

pytestmark = pytest.mark.differential

_ROWS = {
    "id": [1, 2, 3, 4, 5, 6],
    "name": ["ann", "o'brien", "x'; DROP TABLE t; --", None, "ann", ""],
    "score": [0.5, 1.5, 3.5, None, 2.5, -0.0],
    "day": [dt.date(2024, 1, d) for d in (1, 2, 3, 4, 5, 6)],
}

_CASES = [
    ("SELECT id, name FROM t WHERE score > ? ORDER BY id", [1.0]),
    ("SELECT id FROM t WHERE name = ? ORDER BY id", ["x'; DROP TABLE t; --"]),
    ("SELECT id FROM t WHERE name = ? ORDER BY id", ["o'brien"]),
    ("SELECT id FROM t WHERE name IS NULL ORDER BY id", None),
    ("SELECT id FROM t WHERE day >= ? AND id <> ? ORDER BY id", [dt.date(2024, 1, 3), 5]),
    ("SELECT name, COUNT(*) AS n FROM t GROUP BY name ORDER BY name NULLS LAST", None),
    ("SELECT id FROM t WHERE id > ? ORDER BY id", [100]),  # empty
    ("SELECT id, score FROM t WHERE id = ? ORDER BY id", [6]),  # one row, -0.0
    ("SELECT ? AS v", [None]),
]


@pytest.fixture(scope="module")
def cursors():
    session = bt.Session()
    session.register("t", bt.from_pydict(_ROWS))
    con = duckdb.connect()
    con.register("t_src", bt.from_pydict(_ROWS).to_arrow())
    con.execute("CREATE TABLE t AS SELECT * FROM t_src")
    return dbapi.connect(session).cursor(), con.cursor()


@pytest.mark.parametrize(("query", "params"), _CASES)
def test_fetchall_matches_duckdb(cursors, query, params) -> None:
    ours, theirs = cursors
    got = ours.execute(query, params).fetchall()
    want = theirs.execute(query, params).fetchall()
    assert got == want
    assert [d[0] for d in ours.description] == [d[0] for d in theirs.description]


def test_fetchmany_pages_like_duckdb(cursors) -> None:
    ours, theirs = cursors
    query = "SELECT id FROM t ORDER BY id"
    ours.execute(query)
    theirs.execute(query)
    for size in (2, 3, 5):
        assert ours.fetchmany(size) == theirs.fetchmany(size)
    assert ours.fetchone() is None
    assert theirs.fetchone() is None
