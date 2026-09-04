"""The SQL-standard column alias list — ``AS t(a, b)`` — against DuckDB.

A table alias may rename the relation's output columns positionally: `FROM (SELECT ...) AS
c_orders(c_custkey, c_count)`. It is how the standard lets a derived table name a column the
inner query left unnamed, it is legal on a plain table and on a CTE as well as on a
subquery, and none of the three were applied — the alias's identifier was read and its
column list silently dropped.

The visible cost was **TPC-H q13**, which is written exactly that way:

    SELECT c_count, count(*) FROM (
        SELECT c_custkey, count(o_orderkey)
        FROM customer LEFT OUTER JOIN orders ON ... GROUP BY c_custkey
    ) AS c_orders (c_custkey, c_count)
    GROUP BY c_count

The inner aggregate kept its derived name (`count(o_orderkey)`), so the outer `GROUP BY
c_count` could not resolve and the query was refused. It is the only one of the 22 TPC-H
queries that did not run; it runs and matches now.

`VALUES` already read the list, because it has no column names of its own to rename and had
to take them from somewhere — which is why that one case worked and hid the gap.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same, assert_same_ordered

pytestmark = pytest.mark.differential


@pytest.fixture
def t(duck):
    table = pa.table(
        {
            "k": pa.array([1, 1, 2, None], pa.int64()),
            "v": pa.array([10, 20, 30, 40], pa.int64()),
            "s": pa.array(["a", "b", "a", None], pa.string()),
        }
    )
    duck.register("t", table)
    return table


def _run(t, duck, sql, *, ordered=True):
    got = bt.sql(sql, t=bt.from_arrow(t)).collect()
    (assert_same_ordered if ordered else assert_same)(got, duck.sql(sql))


@pytest.mark.parametrize(
    "sql",
    [
        # A derived table naming an unnamed aggregate — the TPC-H q13 shape.
        "SELECT b FROM (SELECT k, sum(v) FROM t GROUP BY k) AS s (a, b) ORDER BY b",
        "SELECT a, b FROM (SELECT k, sum(v) FROM t GROUP BY k) AS s (a, b) ORDER BY a NULLS LAST",
        # A subquery whose columns already had names — renamed anyway.
        "SELECT a FROM (SELECT k FROM t) AS s (a) ORDER BY a NULLS LAST",
        # Fewer aliases than columns renames a prefix and leaves the rest.
        "SELECT a, v FROM (SELECT k, v FROM t) AS s (a) ORDER BY a NULLS LAST, v",
        # A plain table carries the same list.
        "SELECT a, b FROM t AS x (a, b) ORDER BY a NULLS LAST, b",
        # Qualified by the alias.
        "SELECT x.a, x.b FROM t AS x (a, b) ORDER BY x.a NULLS LAST, x.b",
        # A CTE.
        "WITH s (a, b) AS (SELECT k, v FROM t) SELECT a, b FROM s ORDER BY a NULLS LAST, b",
        # A CTE referenced twice, which takes the materializing path.
        "WITH s (a, b) AS (SELECT k, v FROM t) "
        "SELECT l.a, r.b FROM s AS l JOIN s AS r ON l.a = r.a ORDER BY l.a, r.b",
        # The renamed column drives a grouped aggregate, as q13's does.
        "SELECT b, count(*) AS n FROM (SELECT k, sum(v) FROM t GROUP BY k) AS s (a, b) "
        "GROUP BY b ORDER BY b NULLS LAST",
        # VALUES, which already worked — pinned so the shared path keeps it working.
        "SELECT a, b FROM (VALUES (1, 2), (3, 4)) AS s (a, b) ORDER BY a",
    ],
)
def test_column_alias_list_matches_duckdb(t, duck, sql):
    _run(t, duck, sql)


def test_a_renamed_column_is_reachable_from_every_clause(t, duck):
    """WHERE, GROUP BY, HAVING and ORDER BY all resolve against the new name."""
    _run(
        t,
        duck,
        "SELECT a, sum(b) AS total FROM (SELECT k, v FROM t) AS s (a, b) "
        "WHERE b > 10 GROUP BY a HAVING sum(b) > 20 ORDER BY a NULLS LAST",
    )


def test_the_old_name_is_gone_after_renaming(t):
    """A rename replaces the name; the inner one must not still resolve."""
    from batcher._internal.errors import ColumnNotFoundError

    with pytest.raises(ColumnNotFoundError):
        bt.sql("SELECT k FROM (SELECT k FROM t) AS s (a)", t=bt.from_arrow(t)).collect()


def test_too_many_aliases_is_refused(t):
    """DuckDB and PostgreSQL both reject it; answering with a partial rename would be
    a silently different relation."""
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError, match="column alias list"):
        bt.sql("SELECT * FROM (SELECT k, v FROM t) AS s (a, b, c)", t=bt.from_arrow(t)).collect()
