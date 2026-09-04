"""`ORDER BY` over a **grouped expression**, in every spelling SQL allows.

After ``GROUP BY <expr>`` the relation holds the group keys and the aggregate outputs and
nothing else — the input columns the expression was computed from are gone. A sort key that
still spells the expression therefore has nothing to resolve against, and three of the four
ordinary spellings failed with ``sort key references unknown column(s)``:

    SELECT h % 4 AS k, min(v) FROM t GROUP BY h % 4 ORDER BY 1        -- failed
    SELECT h % 4 AS k, min(v) FROM t GROUP BY h % 4 ORDER BY h % 4    -- failed
    SELECT h % 4 AS k, min(v) FROM t GROUP BY 1     ORDER BY 1        -- failed
    SELECT h % 4 AS k, min(v) FROM t GROUP BY h % 4 ORDER BY k        -- worked

The fourth worked only because the alias happens to name a column that survives the
aggregate, which is what made the gap read as a quirk of the other three rather than as a
missing resolution step. DuckDB accepts all four, and so does the standard: an ORDER BY term
matching a select-list item refers to that item's output.

Row order is asserted **positionally**, not through `assert_same`: every query here has an
explicit ORDER BY, so order is the property under test and an order-independent comparison
would pass for an engine that ignored the clause entirely.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.differential

_ROWS = 60


@pytest.fixture(scope="module")
def table() -> pa.Table:
    return pa.table(
        {
            "h": pa.array([i % 13 for i in range(_ROWS)], pa.int64()),
            "v": pa.array([float(i) for i in range(_ROWS)], pa.float64()),
        }
    )


@pytest.fixture(scope="module")
def session(table):
    s = bt.Session()
    s.register("t", bt.from_arrow(table))
    return s


def _rows(tbl: pa.Table) -> list[tuple]:
    return [tuple(tbl.column(c)[i].as_py() for c in tbl.column_names) for i in range(tbl.num_rows)]


_QUERIES = [
    # the three that failed
    "SELECT h % 4 AS k, min(v) AS a FROM t GROUP BY h % 4 ORDER BY 1",
    "SELECT h % 4 AS k, min(v) AS a FROM t GROUP BY h % 4 ORDER BY h % 4",
    "SELECT h % 4 AS k, min(v) AS a FROM t GROUP BY 1 ORDER BY 1",
    # the one that worked, kept so a fix cannot regress it
    "SELECT h % 4 AS k, min(v) AS a FROM t GROUP BY h % 4 ORDER BY k",
    # neighbours that must keep working: plain group key, ordinal onto the aggregate,
    # a descending grouped expression, sorting by an aggregate that is not in the key,
    # and a compound expression
    "SELECT h AS k, min(v) AS a FROM t GROUP BY h ORDER BY 1",
    "SELECT h AS k, min(v) AS a FROM t GROUP BY h ORDER BY h",
    "SELECT h % 4 AS k, min(v) AS a FROM t GROUP BY h % 4 ORDER BY 2",
    "SELECT h % 4 AS k, min(v) AS a FROM t GROUP BY h % 4 ORDER BY 1 DESC",
    "SELECT h % 4 AS k, sum(v) AS a FROM t GROUP BY h % 4 ORDER BY sum(v) DESC",
    "SELECT h % 4 + 1 AS k, count(*) AS a FROM t GROUP BY h % 4 + 1 ORDER BY h % 4 + 1",
]


@pytest.mark.parametrize("query", _QUERIES)
def test_the_ordered_result_matches_duckdb_row_for_row(query, session, table):
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.register("t", table)
    assert _rows(session.sql(query).collect()) == [tuple(r) for r in con.sql(query).fetchall()]


def test_every_spelling_of_the_same_sort_gives_the_same_rows(session):
    """The four spellings of "sort by the grouped expression" are one request.

    This is the property the positional and structural forms were missing, stated without
    reference to DuckDB: whichever way the sort key is written, the answer is the same.
    """
    spellings = [
        "SELECT h % 4 AS k, min(v) AS a FROM t GROUP BY h % 4 ORDER BY 1",
        "SELECT h % 4 AS k, min(v) AS a FROM t GROUP BY h % 4 ORDER BY h % 4",
        "SELECT h % 4 AS k, min(v) AS a FROM t GROUP BY 1 ORDER BY 1",
        "SELECT h % 4 AS k, min(v) AS a FROM t GROUP BY h % 4 ORDER BY k",
    ]
    answers = [_rows(session.sql(q).collect()) for q in spellings]
    assert all(a == answers[0] for a in answers), "the four spellings disagree"
    # A negative control: this must be a real sort, not four copies of an unsorted answer.
    keys = [r[0] for r in answers[0]]
    assert keys == sorted(keys) and len(keys) > 1
