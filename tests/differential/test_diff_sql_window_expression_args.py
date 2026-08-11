"""A window function's argument may be any expression, not only a bare column.

`ds.window` binds each function to a column *name*, and the translator required the SQL to
be written that way — anything else was refused with *"window aggregate supports a single
plain column argument only"*. That rejected ordinary analytics SQL, not a corner case:

    SELECT sum(price * qty) OVER (ORDER BY d) FROM sales          -- a running revenue total
    SELECT sum(CASE WHEN ok THEN 1 ELSE 0 END) OVER (ORDER BY d)  -- a running conditional count
    SELECT avg(a + b) OVER (PARTITION BY g) FROM t

The argument is an ordinary scalar, so it is computed into a hidden column before the
window runs; the window operator itself is unchanged. The hidden columns carry the `__bc_`
prefix that a star expansion filters out, which the last test pins — an internal column
reaching a user's ``SELECT *`` is the failure mode this kind of hoisting invites.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential


def _t() -> pa.Table:
    return pa.table(
        {
            "id": pa.array([1, 2, 3, 4, 5], pa.int64()),
            "g": pa.array(["a", "a", "b", "b", "b"]),
            "x": pa.array([1, 2, 3, 4, None], pa.int64()),
            "y": pa.array([10, 20, 30, 40, 50], pa.int64()),
        }
    )


@pytest.fixture
def tables(duck):
    duck.register("t", _t())
    return {"t": bt.from_arrow(_t())}


@pytest.mark.parametrize(
    "query",
    [
        "SELECT id, sum(x * 2) OVER (ORDER BY id) AS s FROM t",
        "SELECT id, sum(x + y) OVER (PARTITION BY g) AS s FROM t",
        "SELECT id, avg(x * 1.0) OVER (PARTITION BY g) AS s FROM t",
        "SELECT id, max(x - y) OVER () AS s FROM t",
        "SELECT id, min(y - x) OVER (PARTITION BY g ORDER BY id) AS s FROM t",
        # The conditional running total — a CASE as the aggregate's argument.
        "SELECT id, sum(CASE WHEN x > 2 THEN x ELSE 0 END) OVER (ORDER BY id) AS s FROM t",
        # A NULL in the argument must aggregate the way it does outside a window.
        "SELECT id, count(x + 1) OVER (ORDER BY id) AS s FROM t",
        # Value functions take the same argument slot.
        "SELECT id, lag(x * 10) OVER (ORDER BY id) AS s FROM t",
        "SELECT id, lead(x + y) OVER (ORDER BY id) AS s FROM t",
        "SELECT id, first_value(x + 1) OVER (ORDER BY id) AS s FROM t",
        # An explicit frame, so the hoisted column is read per frame rather than per row.
        (
            "SELECT id, sum(x * 2) OVER "
            "(ORDER BY id ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS s FROM t"
        ),
        # Two windows over different expressions in one query.
        (
            "SELECT id, sum(x * 2) OVER (ORDER BY id) AS a, "
            "avg(y * 2) OVER (PARTITION BY g) AS b FROM t"
        ),
        # Nested inside a larger projection, which takes the hoisted-window path.
        "SELECT id, sum(x * 2) OVER (ORDER BY id) + 1 AS s FROM t",
        # A plain column argument must keep working unchanged.
        "SELECT id, sum(x) OVER (ORDER BY id) AS s FROM t",
        "SELECT id, count(*) OVER () AS s FROM t",
    ],
)
def test_window_expression_arguments_match_duckdb(tables, duck, query):
    assert_same(bt.sql(query, **tables).collect(), duck.sql(query))


def test_hoisted_argument_column_does_not_reach_the_output(tables, duck):
    query = "SELECT *, sum(x * 2) OVER (ORDER BY id) AS s FROM t"
    got = bt.sql(query, **tables).collect()
    assert got.column_names == list(duck.sql(query).df().columns)
    assert not any(c.startswith("__bc_") for c in got.column_names)


def test_star_does_not_expand_the_window_output_column(tables, duck):
    """`*` expands what the query selects *from*, not what this SELECT list produces."""
    query = "SELECT *, rank() OVER (ORDER BY x) AS r FROM t"
    got = bt.sql(query, **tables).collect()
    assert got.column_names == list(duck.sql(query).df().columns)


def test_qualify_helper_column_does_not_reach_a_star_output(tables, duck):
    """QUALIFY materializes its window under a hidden name; it used to leak as `__qualify0`."""
    query = "SELECT * FROM t QUALIFY row_number() OVER (PARTITION BY g ORDER BY id) = 1"
    got = bt.sql(query, **tables).collect()
    assert got.column_names == list(duck.sql(query).df().columns)
    assert_same(got, duck.sql(query))
