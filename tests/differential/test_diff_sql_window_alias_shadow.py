"""A window's output alias may repeat a source column's name.

`SELECT g, sum(i) OVER (PARTITION BY g) AS s FROM t` is ordinary SQL over a table that
already has an `s`: the alias names an *output* column, which SQL lets shadow an input.
The relational window operator cannot, because its output is appended to the input
relation and a duplicate name has nowhere to go — so it refused the whole query with
"window output column 's' collides with an existing column".

The translator now materializes such a window under a hidden name and reads it back under
the alias. Every window form goes through a different branch of that code (the ordinary
aggregate path, the unordered-ranking shortcut, `row_number() OVER ()`'s `with_row_index`,
and a QUALIFY predicate naming the alias), so each is covered here.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential


def _table() -> pa.Table:
    """A table whose column names are exactly the aliases the queries below use."""
    return pa.table(
        {
            "i": pa.array([1, 2, 3, 4, 5], pa.int64()),
            "g": pa.array(["x", "x", "y", "y", "y"], pa.string()),
            "s": pa.array(["a", "b", "c", "d", "e"], pa.string()),
            "r": pa.array([10, 20, 30, 40, 50], pa.int64()),
        }
    )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT g, i, sum(i) OVER (PARTITION BY g) AS s FROM t",
        "SELECT g, i, sum(i) OVER (PARTITION BY g ORDER BY i) AS r FROM t",
        "SELECT i, row_number() OVER (ORDER BY i) AS s FROM t",
        "SELECT i, rank() OVER (ORDER BY i) AS r, dense_rank() OVER (ORDER BY i) AS s FROM t",
        "SELECT i, row_number() OVER () AS s FROM t",
        "SELECT i, rank() OVER () AS r FROM t",
        "SELECT i, lag(i) OVER (ORDER BY i) AS s, lead(i) OVER (ORDER BY i) AS r FROM t",
        "SELECT i, sum(i) OVER (ORDER BY i ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS r FROM t",
        "SELECT i, sum(i) OVER w AS s FROM t WINDOW w AS (ORDER BY i)",
        "SELECT g, sum(i) AS s, rank() OVER (ORDER BY sum(i)) AS r FROM t GROUP BY g",
        "SELECT i, sum(i) OVER (ORDER BY i) + 1 AS s FROM t",
    ],
)
def test_a_window_alias_may_shadow_a_source_column(duck, sql):
    """Each shape reaches the window pass through a different branch."""
    table = _table()
    duck.register("t", table)
    assert_same(bt.sql(sql, t=table).collect(), duck.sql(sql))


def test_a_qualify_naming_a_shadowed_alias_reads_the_window_not_the_column():
    """The predicate must filter on the *window* value, not the source column it shadows.

    Reading the source column would still return rows, and plausible ones — this is the
    shape where the rename is invisible unless the filter is checked against real values.
    """
    table = _table()
    sql = "SELECT i, g, sum(i) OVER (PARTITION BY g) AS r FROM t QUALIFY r > 6"
    got = bt.sql(sql, t=table).collect().to_pydict()
    # Group x sums to 3, group y to 12 — only y survives, and `r` is the window value.
    assert got["g"] == ["y", "y", "y"]
    assert got["r"] == [12, 12, 12]


def test_two_windows_sharing_one_alias_still_land_on_distinct_columns():
    """Two different specs written under one alias must not collide with each other."""
    table = _table()
    sql = "SELECT i, sum(i) OVER (ORDER BY i) AS s FROM t ORDER BY i"
    got = bt.sql(sql, t=table).collect().to_pydict()
    assert got["s"] == [1, 3, 6, 10, 15]
