"""`ORDER BY <window function>` — sorting by a value the SELECT list never names.

`SELECT i FROM t ORDER BY row_number() OVER (ORDER BY i DESC)` has no window anywhere in
its projection, so it reached the plain SELECT path and failed with
`unsupported SQL expression: Window`: the scalar lowering has no node for a window, and
there was nothing to build the column the sort needs.

The fix is the ORDER BY twin of the QUALIFY hoist -- each window becomes a hidden
`__bc_ordwin<n>` column, the sort names that column, and the projection drops it. The
tests below hold both halves: the ordering matches DuckDB, *and* the hidden column never
reaches the output, which a result comparison alone would not catch under `SELECT *`.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same_ordered

pytestmark = pytest.mark.differential


def _table() -> pa.Table:
    return pa.table(
        {
            "g": pa.array(["a", "a", "b", "b", "c", None], pa.string()),
            "i": pa.array([1, 2, 3, 4, 5, 6], pa.int64()),
            "v": pa.array([10.0, 20.0, 30.0, None, 50.0, 60.0], pa.float64()),
        }
    )


_QUERIES = [
    # The shape that failed: no window in the SELECT list at all.
    "SELECT i FROM %s ORDER BY row_number() OVER (ORDER BY i DESC)",
    # A partitioned ranking, with a tiebreak so the expected order is total.
    "SELECT i, g FROM %s ORDER BY rank() OVER (PARTITION BY g ORDER BY i DESC), i",
    # An aggregate window as the sort key, again with a tiebreak.
    "SELECT i FROM %s ORDER BY sum(v) OVER (PARTITION BY g) DESC NULLS LAST, i",
    # A window in the SELECT *and* a different one in the ORDER BY: both must be computed
    # over the same relation, in one pass.
    "SELECT i, sum(v) OVER (PARTITION BY g) AS s FROM %s"
    " ORDER BY row_number() OVER (ORDER BY i DESC)",
]


@pytest.mark.parametrize("sql", _QUERIES, ids=range(len(_QUERIES)))
def test_ordering_by_a_window_function_matches_duckdb(duck, sql):
    t = _table()
    duck.register("t", t)
    assert_same_ordered(bt.from_arrow(t).sql(sql % "self").collect(), duck.sql(sql % "t"))


def test_the_hidden_sort_column_never_reaches_the_output(duck):
    """`SELECT *` must not grow a `__bc_ordwin0` column nobody asked for.

    This is the failure the ordering comparison cannot see: the rows would be in the right
    order and simply carry an extra column. The `__bc_` prefix is what keeps it out of the
    star expansion, and it is the same prefix `_qualify_windows` relies on -- so this test
    is really pinning that prefix, not the star.
    """
    t = _table()
    duck.register("t", t)
    sql = "SELECT * FROM %s ORDER BY row_number() OVER (ORDER BY i DESC)"
    out = bt.from_arrow(t).sql(sql % "self").collect()
    assert out.schema.names == ["g", "i", "v"]
    assert_same_ordered(out, duck.sql(sql % "t"))
