"""`ORDER BY` and `LIMIT` on a ROLLUP/CUBE/GROUPING SETS query belong above the union.

A multi-level GROUP BY expands into a UNION ALL over grouping levels
(`_sql.parser.grouping_sets`), and each level is a copy of the *whole* SELECT node. The
copy carried the query's `ORDER BY`, `LIMIT` and `OFFSET` down into every branch, so both
were applied per level and neither was applied to the union:

* the limit became per-level — `ROLLUP(a, b) ... LIMIT 7` returned 7 + 5 + 1 = 13 rows;
* the sort became per-level, so the unioned output was not in `ORDER BY` order at all.

TPC-DS q5, q14, q18 and q22 all failed the DuckDB oracle on row count for the first
reason (q18 and q22 returned 401 rows against DuckDB's 100).

The row-count half is what `assert_same` can see. The *ordering* half it cannot — the
harness comparison is order-independent by design, which is exactly how a per-level sort
stays invisible — so the ordering assertions here compare the row sequence directly
against DuckDB's, as `.claude/rules/testing.md` requires of any sort.
"""

from __future__ import annotations

import duckdb
import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

# Deliberately uneven level sizes: 5 distinct `a` x 4 distinct `b`, so the three ROLLUP
# levels hold 20 / 5 / 1 rows. A limit smaller than the total but larger than the smallest
# level is what makes the per-level bug arithmetic (7 + 5 + 1 = 13) rather than a wash.
_ROWS = 60
_A, _B = 5, 4


@pytest.fixture
def tables():
    t = pa.table(
        {
            "a": pa.array([f"a{i % _A}" for i in range(_ROWS)]),
            "b": pa.array([f"b{i % _B}" for i in range(_ROWS)]),
            "v": pa.array(range(_ROWS), pa.int64()),
        }
    )
    d = duckdb.connect()
    d.register("t", t)
    sess = bt.Session()
    sess.register("t", t)
    return sess, d


_LIMITED = [
    (
        "rollup-alias-order",
        "SELECT a, b, sum(v) AS s FROM t GROUP BY ROLLUP(a, b) "
        "ORDER BY s NULLS FIRST, a NULLS FIRST, b NULLS FIRST LIMIT 7",
    ),
    (
        "cube-desc",
        "SELECT a, b, sum(v) AS s FROM t GROUP BY CUBE(a, b) "
        "ORDER BY s DESC, a NULLS FIRST, b NULLS FIRST LIMIT 5",
    ),
    (
        "grouping-sets-positional",
        "SELECT a, b, sum(v) AS s FROM t "
        "GROUP BY GROUPING SETS ((a), (b), ()) "
        "ORDER BY 3, 1 NULLS FIRST, 2 NULLS FIRST LIMIT 4",
    ),
    (
        "rollup-offset",
        "SELECT a, sum(v) AS s FROM t GROUP BY ROLLUP(a) ORDER BY s NULLS FIRST LIMIT 3 OFFSET 2",
    ),
    (
        "rollup-count-star",
        "SELECT a, b, count(*) AS n FROM t GROUP BY ROLLUP(a, b) "
        "ORDER BY n, a NULLS FIRST, b NULLS FIRST LIMIT 9",
    ),
]


@pytest.mark.differential
@pytest.mark.parametrize(("label", "sql"), _LIMITED, ids=[c[0] for c in _LIMITED])
def test_grouping_sets_limit_applies_to_the_union(tables, label, sql):
    """The limit is the query's, not each level's.

    This is the regression: before the fix every case here returned strictly more rows
    than DuckDB, because each grouping level was limited on its own.
    """
    sess, duck = tables
    assert_same(sess.sql(sql).collect(), duck.sql(sql))


@pytest.mark.differential
@pytest.mark.parametrize(("label", "sql"), _LIMITED, ids=[c[0] for c in _LIMITED])
def test_grouping_sets_order_is_global(tables, label, sql):
    """The unioned levels come back in one global ORDER BY order.

    `assert_same` is order-independent, so it cannot see a per-level sort. Compare the
    row *sequence* to DuckDB's. Every ORDER BY above is a total order over the level's
    output, so there are no ties for the two engines to break differently.
    """
    sess, duck = tables
    got = [tuple(r.values()) for r in sess.sql(sql).collect().to_pylist()]
    want = [tuple(r) for r in duck.sql(sql).fetchall()]
    assert got == want, f"{label}: rows are not in ORDER BY order across the grouping levels"


@pytest.mark.differential
def test_grouping_sets_without_limit_is_unchanged(tables):
    """The un-limited shape keeps every level's rows — the fix must not drop any."""
    sql = "SELECT a, b, sum(v) AS s FROM t GROUP BY ROLLUP(a, b)"
    sess, duck = tables
    assert_same(sess.sql(sql).collect(), duck.sql(sql))
