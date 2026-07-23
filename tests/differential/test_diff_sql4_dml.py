"""SQL DML (INSERT/DELETE/UPDATE) and ordered-set aggregates vs DuckDB.

The DML statements mutate a `Session` catalog by rebinding the target table to a
new lazy `Dataset`; we compare the resulting table *state* against DuckDB running
the same statement over a real table. The ordered-set aggregates
(`percentile_cont`/`mode() WITHIN GROUP`) are compared as ordinary query results.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same


# --------------------------------------------------------------------------
# Ordered-set aggregates: `agg(...) WITHIN GROUP (ORDER BY x)`.
#
# Regression: the WITHIN GROUP clause parses as WithinGroup(this=agg, ORDER BY),
# and the aggregate registrar walked past the wrapper to the inner
# PercentileCont(this=<fraction>) — silently dropping the ORDER BY column and
# treating the fraction as the value column (then erroring). These must now match.
# --------------------------------------------------------------------------
@pytest.fixture
def nums(duck):
    t = pa.table(
        {
            "g": [1, 1, 1, 2, 2, 2, 2],
            "x": [1, 2, 3, 4, 5, 5, 5],
        }
    )
    duck.register("t", t)
    return t


@pytest.mark.parametrize(
    "query",
    [
        "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY x) AS m FROM t",
        "SELECT percentile_cont(0.25) WITHIN GROUP (ORDER BY x) AS m FROM t",
        "SELECT percentile_cont(0.0) WITHIN GROUP (ORDER BY x) AS m FROM t",
        "SELECT percentile_cont(1.0) WITHIN GROUP (ORDER BY x) AS m FROM t",
        "SELECT mode() WITHIN GROUP (ORDER BY x) AS m FROM t",
        "SELECT g, percentile_cont(0.5) WITHIN GROUP (ORDER BY x) AS m "
        "FROM t GROUP BY g ORDER BY g",
        "SELECT g, mode() WITHIN GROUP (ORDER BY x) AS m FROM t GROUP BY g ORDER BY g",
    ],
)
def test_ordered_set_aggregate_matches_duckdb(nums, duck, query):
    assert_same(bt.sql(query, t=nums).to_arrow(), duck.sql(query))


def test_percentile_disc_within_group_clean_error(nums):
    # No discrete quantile in the engine — a clean error, never a wrong result.
    with pytest.raises(NotImplementedError):
        bt.sql(
            "SELECT percentile_disc(0.5) WITHIN GROUP (ORDER BY x) AS m FROM t", t=nums
        ).to_arrow()


# --------------------------------------------------------------------------
# DML: INSERT / DELETE / UPDATE compared as table state after the statement.
# --------------------------------------------------------------------------
_DDL = ("CREATE TABLE t(x BIGINT, y BIGINT)", "INSERT INTO t VALUES (1,10),(2,20),(3,30)")


def _base_table() -> pa.Table:
    return pa.table({"x": pa.array([1, 2, 3], pa.int64()), "y": pa.array([10, 20, 30], pa.int64())})


@pytest.mark.parametrize(
    "dml",
    [
        "INSERT INTO t VALUES (4, 40)",
        "INSERT INTO t VALUES (5, 50), (6, 60)",
        "INSERT INTO t SELECT x, y FROM t",
        "INSERT INTO t (x) VALUES (7)",
        "INSERT INTO t (y, x) VALUES (80, 8)",
        "DELETE FROM t WHERE x > 1",
        "DELETE FROM t WHERE x > 100",
        "DELETE FROM t",
        "UPDATE t SET y = 99 WHERE x = 1",
        "UPDATE t SET y = y + 1",
        "UPDATE t SET y = 0 WHERE x > 100",
        "UPDATE t SET x = x * 10, y = y - 5 WHERE x <> 2",
    ],
)
def test_dml_table_state_matches_duckdb(duck, dml):
    for stmt in _DDL:
        duck.execute(stmt)
    duck.execute(dml)

    s = bt.Session()
    s.register("t", _base_table())
    s.sql(dml)

    assert_same(s.table("t").to_arrow(), duck.sql("SELECT * FROM t"))


def test_delete_null_predicate_keeps_null_rows(duck):
    # Three-valued logic: DELETE removes rows where the predicate is TRUE; a row
    # whose predicate is NULL (x IS NULL, so `x > 1` is unknown) must survive.
    t = pa.table({"x": pa.array([1, None, 3], pa.int64()), "y": pa.array([10, 20, 30], pa.int64())})
    duck.execute("CREATE TABLE t(x BIGINT, y BIGINT)")
    duck.execute("INSERT INTO t VALUES (1,10),(NULL,20),(3,30)")
    duck.execute("DELETE FROM t WHERE x > 1")

    s = bt.Session()
    s.register("t", t)
    s.sql("DELETE FROM t WHERE x > 1")

    assert_same(s.table("t").to_arrow(), duck.sql("SELECT * FROM t"))


def test_update_null_predicate_leaves_row_unchanged(duck):
    t = pa.table({"x": pa.array([1, None, 3], pa.int64()), "y": pa.array([10, 20, 30], pa.int64())})
    duck.execute("CREATE TABLE t(x BIGINT, y BIGINT)")
    duck.execute("INSERT INTO t VALUES (1,10),(NULL,20),(3,30)")
    duck.execute("UPDATE t SET y = 99 WHERE x > 1")

    s = bt.Session()
    s.register("t", t)
    s.sql("UPDATE t SET y = 99 WHERE x > 1")

    assert_same(s.table("t").to_arrow(), duck.sql("SELECT * FROM t"))


def test_insert_cte_body(duck):
    t = pa.table({"x": pa.array([1], pa.int64()), "y": pa.array([10], pa.int64())})
    duck.execute("CREATE TABLE t(x BIGINT, y BIGINT)")
    duck.execute("INSERT INTO t VALUES (1, 10)")
    stmt = "INSERT INTO t WITH src AS (SELECT 9 AS x, 90 AS y) SELECT x, y FROM src"
    duck.execute(stmt)

    s = bt.Session()
    s.register("t", t)
    s.sql(stmt)

    assert_same(s.table("t").to_arrow(), duck.sql("SELECT * FROM t"))


@pytest.mark.parametrize(
    "dml",
    [
        "INSERT INTO t VALUES (1)",  # too few columns
        "INSERT INTO t (x) VALUES (1, 2)",  # arity mismatch
        "INSERT INTO t (nope) VALUES (1)",  # unknown column
        "INSERT INTO t VALUES (1, 1) ON CONFLICT DO NOTHING",  # unsupported clause
        "INSERT INTO t VALUES (1, 1) RETURNING x",  # unsupported clause
        "UPDATE t SET z = 1",  # unknown column
        "INSERT INTO missing VALUES (1, 1)",  # unknown table
    ],
)
def test_dml_bad_statements_raise_clean(dml):
    from batcher._internal.errors import PlanError

    s = bt.Session()
    s.register("t", _base_table())
    with pytest.raises((PlanError, NotImplementedError)):
        s.sql(dml)
