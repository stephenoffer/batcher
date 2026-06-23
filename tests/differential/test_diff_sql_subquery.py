"""Subquery SQL frontend differential tests vs DuckDB.

Covers the uncorrelated subquery forms added to the translator:

* ``x IN (SELECT ...)``       → semi-join
* ``x NOT IN (SELECT ...)``   → anti-join
* scalar subquery in WHERE    → ``x > (SELECT AVG(...) ...)``
* scalar subquery in SELECT   → ``(SELECT AVG(...) ...) AS a``
* ``EXISTS`` / ``NOT EXISTS`` (uncorrelated)

Correlated subqueries are out of scope and must raise NotImplementedError.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt


@pytest.fixture
def tables(duck):
    emp = pa.table(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["alice", "bob", "carol", "dave", "alan"],
            "dept_id": [10, 20, 10, 30, 20],
            "salary": [100, 200, 150, 300, 250],
        }
    )
    dept = pa.table(
        {
            "dept_id": [10, 20, 40],
            "dept_name": ["eng", "sales", "ops"],
            "budget": [1000, 2000, 500],
        }
    )
    duck.register("emp", emp)
    duck.register("dept", dept)
    return emp, dept


@pytest.mark.parametrize(
    "query",
    [
        # IN (subquery) → semi-join
        "SELECT id, dept_id FROM emp WHERE dept_id IN (SELECT dept_id FROM dept)",
        "SELECT id FROM emp WHERE dept_id IN (SELECT dept_id FROM dept WHERE budget > 800)",
        # NOT IN (subquery) → anti-join
        "SELECT id, dept_id FROM emp WHERE dept_id NOT IN (SELECT dept_id FROM dept)",
        "SELECT id FROM emp WHERE dept_id NOT IN (SELECT dept_id FROM dept WHERE budget > 800)",
        # multiple IN-subqueries chained by AND
        "SELECT id FROM emp WHERE dept_id IN (SELECT dept_id FROM dept) "
        "AND dept_id IN (SELECT dept_id FROM dept WHERE budget > 800)",
        # IN-subquery combined with an ordinary predicate (AND)
        "SELECT id FROM emp WHERE dept_id IN (SELECT dept_id FROM dept) AND salary > 120",
        # scalar subquery in WHERE
        "SELECT id, salary FROM emp WHERE salary > (SELECT AVG(salary) FROM emp)",
        "SELECT id FROM emp WHERE salary >= (SELECT MIN(budget) FROM dept)",
        # scalar subquery in SELECT
        "SELECT id, (SELECT AVG(salary) FROM emp) AS avg_sal FROM emp",
        "SELECT id, salary - (SELECT MIN(salary) FROM emp) AS diff FROM emp",
        # EXISTS / NOT EXISTS (uncorrelated)
        "SELECT id FROM emp WHERE EXISTS (SELECT 1 FROM dept WHERE budget > 800)",
        "SELECT id FROM emp WHERE EXISTS (SELECT 1 FROM dept WHERE budget > 9999)",
        "SELECT id FROM emp WHERE NOT EXISTS (SELECT 1 FROM dept WHERE budget > 9999)",
        "SELECT id FROM emp WHERE NOT EXISTS (SELECT 1 FROM dept WHERE budget > 800)",
    ],
)
def test_sql_subquery_vs_duckdb(duck, tables, query):
    from conftest import assert_same

    emp, dept = tables
    out = bt.sql(query, emp=emp, dept=dept).collect()
    assert_same(out, duck.sql(query))


def test_correlated_subquery_raises(tables):
    emp, dept = tables
    query = (
        "SELECT id FROM emp WHERE dept_id IN "
        "(SELECT dept_id FROM dept WHERE dept.budget > emp.salary)"
    )
    with pytest.raises(NotImplementedError):
        bt.sql(query, emp=emp, dept=dept)


def test_correlated_scalar_subquery_vs_duckdb(duck, tables):
    from conftest import assert_same

    emp, dept = tables
    # Correlated scalar subquery is now supported and matches DuckDB.
    query = (
        "SELECT id, (SELECT AVG(budget) FROM dept WHERE dept.dept_id = emp.dept_id) AS b FROM emp"
    )
    assert_same(bt.sql(query, emp=emp, dept=dept).collect(), duck.sql(query))
