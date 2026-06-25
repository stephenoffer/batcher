"""Subquery SQL frontend differential tests vs DuckDB.

Covers the uncorrelated subquery forms added to the translator:

* ``x IN (SELECT ...)``       → semi-join
* ``x NOT IN (SELECT ...)``   → anti-join
* scalar subquery in WHERE    → ``x > (SELECT AVG(...) ...)``
* scalar subquery in SELECT   → ``(SELECT AVG(...) ...) AS a``
* ``EXISTS`` / ``NOT EXISTS`` (uncorrelated)

Equi-correlated subqueries decorrelate to semi/anti/LEFT joins, including when the
outer column is referenced **unqualified** (the TPC-H style, e.g.
``WHERE l_orderkey = o_orderkey``). Non-equi correlations still raise
NotImplementedError.
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


@pytest.mark.parametrize(
    "query",
    [
        # HAVING aggregate vs a scalar subquery that itself aggregates (TPC-H Q11 shape):
        # the subquery's aggregate must not clobber the outer query's aggregate state.
        "SELECT dept_id, SUM(salary) AS v FROM emp GROUP BY dept_id "
        "HAVING SUM(salary) > (SELECT SUM(salary) * 0.3 FROM emp)",
        # scalar subquery (aggregating) alongside an outer aggregate in the SELECT list
        "SELECT dept_id, SUM(salary) AS v, (SELECT SUM(salary) FROM emp) AS tot "
        "FROM emp GROUP BY dept_id",
        # subquery aggregate inside an aggregate query's projection arithmetic
        "SELECT dept_id, SUM(salary) - (SELECT MIN(salary) FROM emp) AS adj "
        "FROM emp GROUP BY dept_id",
    ],
)
def test_aggregating_scalar_subquery_in_aggregate_query(duck, tables, query):
    """A scalar subquery that aggregates must not corrupt the enclosing aggregate query."""
    from conftest import assert_same

    emp, dept = tables
    assert_same(bt.sql(query, emp=emp, dept=dept).collect(), duck.sql(query))


@pytest.fixture
def tpch_like(duck):
    """Distinct-column-name tables so unqualified outer references are unambiguous."""
    orders = pa.table({"o_orderkey": [1, 2, 3, 4], "o_priority": ["hi", "lo", "hi", "lo"]})
    lineitem = pa.table(
        {
            "l_orderkey": [1, 1, 3],
            "l_partkey": [10, 20, 10],
            "l_commit": [5, 9, 5],
            "l_receipt": [9, 5, 9],
            "l_quantity": [10.0, 20.0, 30.0],
        }
    )
    part = pa.table({"p_partkey": [10, 20], "p_brand": ["A", "B"]})
    duck.register("orders", orders)
    duck.register("lineitem", lineitem)
    duck.register("part", part)
    return orders, lineitem, part


@pytest.mark.parametrize(
    "query",
    [
        # correlated EXISTS, outer column referenced unqualified (TPC-H Q4 shape)
        "SELECT o_priority, count(*) c FROM orders WHERE EXISTS "
        "(SELECT * FROM lineitem WHERE l_orderkey = o_orderkey AND l_commit < l_receipt) "
        "GROUP BY o_priority",
        # correlated NOT EXISTS, unqualified
        "SELECT o_orderkey FROM orders WHERE NOT EXISTS "
        "(SELECT * FROM lineitem WHERE l_orderkey = o_orderkey)",
        # correlated IN, unqualified outer reference
        "SELECT o_orderkey FROM orders WHERE o_orderkey IN "
        "(SELECT l_orderkey FROM lineitem WHERE l_commit < l_receipt)",
        # correlated scalar subquery, unqualified (TPC-H Q17 shape)
        "SELECT sum(l_quantity) s FROM part, lineitem WHERE p_partkey = l_partkey "
        "AND l_quantity < (SELECT 2 * avg(l_quantity) FROM lineitem WHERE l_partkey = p_partkey)",
    ],
)
def test_unqualified_correlated_subquery_vs_duckdb(duck, tpch_like, query):
    from conftest import assert_same

    orders, lineitem, part = tpch_like
    out = bt.sql(query, orders=orders, lineitem=lineitem, part=part).collect()
    assert_same(out, duck.sql(query))
