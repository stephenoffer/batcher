"""Differential tests for SQL CTEs and window functions vs DuckDB."""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher._sql.parser import sql as bsql


@pytest.fixture
def emp(duck):
    tbl = pa.table(
        {
            "dept": ["a", "a", "a", "b", "b", "b"],
            "name": list("uvwxyz"),
            "salary": [100, 300, 200, 150, 250, 250],
        }
    )
    duck.register("emp", tbl)
    return tbl


@pytest.fixture
def dept_info(duck):
    tbl = pa.table(
        {
            "dept": ["a", "b"],
            "region": ["west", "east"],
        }
    )
    duck.register("dept_info", tbl)
    return tbl


# --- CTEs ------------------------------------------------------------------
def test_simple_cte(duck, emp):
    from conftest import assert_same

    q = """
        WITH high AS (SELECT dept, name, salary FROM emp WHERE salary > 150)
        SELECT dept, name, salary FROM high
    """
    out = bsql(q, emp=emp).collect()
    assert_same(out, duck.sql(q))


def test_chained_multi_cte(duck, emp):
    from conftest import assert_same

    q = """
        WITH a AS (SELECT dept, salary FROM emp WHERE salary >= 150),
             b AS (SELECT dept, salary FROM a WHERE salary < 300)
        SELECT dept, salary FROM b
    """
    out = bsql(q, emp=emp).collect()
    assert_same(out, duck.sql(q))


def test_cte_used_in_join(duck, emp, dept_info):
    from conftest import assert_same

    q = """
        WITH e AS (SELECT dept, name, salary FROM emp WHERE salary > 100)
        SELECT e.dept AS dept, e.name AS name, dept_info.region AS region
        FROM e JOIN dept_info ON e.dept = dept_info.dept
    """
    out = bsql(q, emp=emp, dept_info=dept_info).collect()
    assert_same(out, duck.sql(q))


# --- ranking window functions ----------------------------------------------
def test_row_number(duck, emp):
    from conftest import assert_same

    q = """
        SELECT dept, salary,
               row_number() OVER (PARTITION BY dept ORDER BY salary DESC) AS rn
        FROM emp
    """
    out = bsql(q, emp=emp).collect()
    assert_same(out, duck.sql(q))


def test_rank_and_dense_rank(duck, emp):
    from conftest import assert_same

    q = """
        SELECT dept, salary,
               rank() OVER (PARTITION BY dept ORDER BY salary DESC) AS rk,
               dense_rank() OVER (PARTITION BY dept ORDER BY salary DESC) AS dr
        FROM emp
    """
    out = bsql(q, emp=emp).collect()
    assert_same(out, duck.sql(q))


# --- whole-partition aggregate window functions ----------------------------
def test_sum_over_partition(duck, emp):
    from conftest import assert_same

    q = "SELECT dept, salary, SUM(salary) OVER (PARTITION BY dept) AS tot FROM emp"
    out = bsql(q, emp=emp).collect()
    assert_same(out, duck.sql(q))


def test_avg_min_max_count_over_partition(duck, emp):
    from conftest import assert_same

    q = """
        SELECT dept, salary,
               AVG(salary) OVER (PARTITION BY dept) AS av,
               MIN(salary) OVER (PARTITION BY dept) AS lo,
               MAX(salary) OVER (PARTITION BY dept) AS hi,
               COUNT(salary) OVER (PARTITION BY dept) AS n
        FROM emp
    """
    out = bsql(q, emp=emp).collect()
    assert_same(out, duck.sql(q))


def test_multiple_window_functions_mixed(duck, emp):
    from conftest import assert_same

    q = """
        SELECT dept, salary,
               row_number() OVER (PARTITION BY dept ORDER BY salary DESC) AS rn,
               SUM(salary) OVER (PARTITION BY dept) AS tot
        FROM emp
    """
    out = bsql(q, emp=emp).collect()
    assert_same(out, duck.sql(q))


# --- running (cumulative) aggregate frames ---------------------------------
@pytest.mark.parametrize("fn", ["SUM", "AVG", "MIN", "MAX", "COUNT"])
def test_running_aggregate_window_vs_duckdb(duck, emp, fn):
    from conftest import assert_same

    # ORDER BY salary has ties (250,250 in dept b), exercising RANGE peer-sharing.
    q = (
        f"SELECT dept, name, salary, "
        f"{fn}(salary) OVER (PARTITION BY dept ORDER BY salary) AS r FROM emp"
    )
    assert_same(bsql(q, emp=emp).collect(), duck.sql(q))


# --- QUALIFY (filter on window results) ------------------------------------
def test_qualify_row_number(duck, emp):
    from conftest import assert_same

    # Tiebreak by name so row_number is deterministic over the tied salaries.
    q = (
        "SELECT dept, name, salary, "
        "row_number() OVER (PARTITION BY dept ORDER BY salary DESC, name) rn "
        "FROM emp QUALIFY rn <= 2"
    )
    assert_same(bsql(q, emp=emp).collect(), duck.sql(q))


def test_qualify_rank_filter(duck, emp):
    from conftest import assert_same

    q = (
        "SELECT dept, salary, rank() OVER (PARTITION BY dept ORDER BY salary) rk "
        "FROM emp QUALIFY rk = 1"
    )
    assert_same(bsql(q, emp=emp).collect(), duck.sql(q))
