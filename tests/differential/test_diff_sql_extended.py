"""Extended SQL frontend differential tests vs DuckDB.

Covers the constructs added to the translator: IN / NOT IN, BETWEEN, LIKE
(prefix/suffix/contains/exact), NOT, SELECT DISTINCT, FROM-subquery, COALESCE,
and UNION / UNION ALL.
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
            # nullable column to exercise COALESCE / NOT NULL
            "bonus": [None, 50, None, 25, None],
        }
    )
    other = pa.table(
        {
            "id": [6, 7, 3],
            "name": ["zoe", "yan", "carol"],
            "dept_id": [10, 40, 10],
            "salary": [400, 120, 150],
            "bonus": [10, None, None],
        }
    )
    duck.register("emp", emp)
    duck.register("other", other)
    return emp, other


@pytest.mark.parametrize(
    "query",
    [
        # IN / NOT IN
        "SELECT id, dept_id FROM emp WHERE dept_id IN (10, 30)",
        "SELECT id FROM emp WHERE id IN (1, 3, 5)",
        "SELECT id, dept_id FROM emp WHERE dept_id NOT IN (10, 30)",
        "SELECT name FROM emp WHERE name IN ('alice', 'bob')",
        # BETWEEN
        "SELECT id, salary FROM emp WHERE salary BETWEEN 150 AND 250",
        "SELECT id FROM emp WHERE id BETWEEN 2 AND 4",
        # LIKE prefix / suffix / contains / exact
        "SELECT id, name FROM emp WHERE name LIKE 'al%'",
        "SELECT id, name FROM emp WHERE name LIKE '%e'",
        "SELECT id, name FROM emp WHERE name LIKE '%a%'",
        "SELECT id, name FROM emp WHERE name LIKE 'bob'",
        "SELECT id, name FROM emp WHERE name NOT LIKE 'al%'",
        # NOT
        "SELECT id FROM emp WHERE NOT (salary > 200)",
        "SELECT id FROM emp WHERE NOT (dept_id = 10)",
        # SELECT DISTINCT
        "SELECT DISTINCT dept_id FROM emp",
        "SELECT DISTINCT dept_id FROM emp WHERE salary > 100",
        # FROM-subquery
        "SELECT s.id FROM (SELECT id, salary FROM emp WHERE salary > 150) AS s",
        "SELECT id, salary FROM (SELECT id, salary FROM emp WHERE dept_id = 10) AS s "
        "WHERE salary < 200",
        # COALESCE
        "SELECT id, COALESCE(bonus, 0) AS b FROM emp",
        "SELECT id, COALESCE(bonus, salary, 0) AS b FROM emp",
        # UNION / UNION ALL
        "SELECT id, name FROM emp UNION ALL SELECT id, name FROM other",
        "SELECT dept_id FROM emp UNION SELECT dept_id FROM other",
    ],
)
def test_sql_extended_vs_duckdb(duck, tables, query):
    from conftest import assert_same

    emp, other = tables
    out = bt.sql(query, emp=emp, other=other).collect()
    assert_same(out, duck.sql(query))


@pytest.mark.parametrize(
    "query",
    [
        # interior '%' and '_' wildcards are now supported (LIKE → regex)
        "SELECT id FROM emp WHERE name LIKE 'a%b'",
        "SELECT id FROM emp WHERE name LIKE 'a_c'",
        "SELECT id FROM emp WHERE name LIKE '%a%'",
    ],
)
def test_like_interior_wildcards_vs_duckdb(duck, tables, query):
    from conftest import assert_same

    emp, other = tables
    assert_same(bt.sql(query, emp=emp, other=other).collect(), duck.sql(query))


@pytest.mark.parametrize(
    "q",
    [
        "SELECT dept_id FROM emp EXCEPT SELECT dept_id FROM other",
        "SELECT dept_id FROM emp INTERSECT SELECT dept_id FROM other",
    ],
)
def test_except_intersect_vs_duckdb(duck, tables, q):
    from conftest import assert_same

    emp, other = tables
    duck.register("emp", emp)
    duck.register("other", other)
    assert_same(bt.sql(q, emp=emp, other=other).collect(), duck.sql(q))
