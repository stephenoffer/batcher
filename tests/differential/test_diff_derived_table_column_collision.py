"""Two derived tables exposing the same column name must not collapse onto one column.

`core_utils._disambiguate_columns` renames colliding columns so the alias-blind resolver
sees distinct names. It only ever considered `exp.Table` sources, so *derived* tables were
skipped entirely — and two of them sharing a column name collapsed onto one physical
column. A comma join's `WHERE a.r = b.r` then degenerated to `r = r`, true for every pair,
and the query silently returned the **cartesian product**:

    FROM (SELECT k AS r FROM t) a, (SELECT k AS r FROM t) b WHERE a.r = b.r
    -- DuckDB: 5 rows.  Batcher, before the fix: 25.

TPC-DS q44 is exactly this shape — two ranked relations joined on `rnk` — and returned 100
rows (10 x 10, capped by `LIMIT 100`) where DuckDB returns 10.

The wrong answer is a *row multiset*, not an error, so these assertions are what catch it.
`assert_same` is order-independent but not count-blind, so it does see a cross product.

Two cases are deliberately here and would not be caught by the comma-join case alone: the
q44 shape hides the collision two levels down behind `SELECT *`, so it is only found by
expanding the star against the inner FROM; and the outer-join case pins that a *missing*
right row still yields NULL rather than a wrongly-matched value.

Column *naming* is deliberately not asserted. Batcher collapses two projections that share
an output name (`SELECT x.k, y.k` yields one `k`) for base and derived tables alike; that
predates this fix, is unchanged by it, and is a separate defect. The queries below alias
their projections so the two concerns stay separate.
"""

from __future__ import annotations

import duckdb
import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

_K = 5


@pytest.fixture
def tables():
    t = pa.table(
        {
            "k": pa.array(range(1, _K + 1), pa.int64()),
            "v": pa.array([10 * i for i in range(1, _K + 1)], pa.int64()),
        }
    )
    d = duckdb.connect()
    d.register("t", t)
    sess = bt.Session()
    sess.register("t", t)
    return sess, d


_CASES = [
    (
        "comma-join",
        "SELECT a.r AS ar, b.r AS br FROM (SELECT k AS r FROM t) a, (SELECT k AS r FROM t) b "
        "WHERE a.r = b.r",
    ),
    (
        "nested-star",  # the TPC-DS q44 shape: the collision is behind a `SELECT *`
        "SELECT a.r AS ar, b.r AS br "
        "FROM (SELECT * FROM (SELECT k AS r FROM t) u) a, "
        "(SELECT * FROM (SELECT k AS r FROM t) w) b "
        "WHERE a.r = b.r",
    ),
    (
        "explicit-join-on",
        "SELECT a.r AS ar, b.r AS br FROM (SELECT k AS r FROM t) a "
        "JOIN (SELECT k AS r FROM t) b ON a.r = b.r",
    ),
    (
        "left-join-keeps-nulls",
        "SELECT a.r AS ar, b.r AS br FROM (SELECT k AS r FROM t) a "
        "LEFT JOIN (SELECT k AS r FROM t WHERE k < 3) b ON a.r = b.r",
    ),
    (
        "set-op-branch",
        "SELECT a.r AS ar, b.r AS br "
        "FROM (SELECT k AS r FROM t UNION ALL SELECT k AS r FROM t) a, (SELECT k AS r FROM t) b "
        "WHERE a.r = b.r",
    ),
    (
        "three-derived",
        "SELECT a.r AS ar, b.r AS br, c.r AS cr FROM (SELECT k AS r FROM t) a, "
        "(SELECT k AS r FROM t) b, (SELECT k AS r FROM t) c WHERE a.r = b.r AND b.r = c.r",
    ),
    (
        "derived-and-base",
        "SELECT a.k AS ak, t.v AS tv FROM (SELECT k FROM t) a, t WHERE a.k = t.k",
    ),
    (
        "base-tables-unchanged",  # the path this fix must not disturb
        "SELECT x.k AS xk, y.k AS yk FROM t x, t y WHERE x.k = y.k",
    ),
]


@pytest.mark.differential
@pytest.mark.parametrize(("label", "sql"), _CASES, ids=[c[0] for c in _CASES])
def test_derived_tables_sharing_a_column_name(tables, label, sql):
    """The join condition applies, so the answer is DuckDB's and not a cross product."""
    sess, duck = tables
    assert_same(sess.sql(sql).collect(), duck.sql(sql))


@pytest.mark.differential
def test_the_join_is_not_a_cartesian_product(tables):
    """The row count itself, stated separately from the oracle.

    `_K` distinct keys joined on equality give `_K` rows; the collapse gave `_K * _K`. This
    asserts the number directly so a future change that reintroduces the collapse fails
    with an unmistakable message rather than a multiset diff.
    """
    sess, _ = tables
    sql = (
        "SELECT a.r AS ar, b.r AS br FROM (SELECT k AS r FROM t) a, "
        "(SELECT k AS r FROM t) b WHERE a.r = b.r"
    )
    n = sess.sql(sql).collect().num_rows
    assert n == _K, (
        f"{n} rows for a {_K}-key equi-join between two derived tables — "
        f"the shared column name collapsed and the predicate became a tautology "
        f"({_K * _K} rows is the full cartesian product)"
    )
