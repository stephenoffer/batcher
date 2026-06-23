"""SQL frontend differential tests vs DuckDB."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt


@pytest.fixture
def tables(duck):
    emp = pa.table(
        {
            "id": [1, 2, 3, 4, 5],
            "name": list("abcde"),
            "dept_id": [10, 20, 10, 30, 20],
            "salary": [100, 200, 150, 300, 250],
        }
    )
    dept = pa.table({"dept_id": [10, 20, 30], "dept": ["eng", "sales", "ops"]})
    duck.register("emp", emp)
    duck.register("dept", dept)
    return emp, dept


@pytest.mark.parametrize(
    "query",
    [
        "SELECT id, salary FROM emp WHERE salary > 150",
        "SELECT name, salary * 2 AS double FROM emp WHERE id <= 3",
        "SELECT dept_id, SUM(salary) AS total, COUNT(*) AS n, AVG(salary) AS a, "
        "MAX(salary) AS hi FROM emp GROUP BY dept_id",
        "SELECT dept_id, SUM(salary) s FROM emp GROUP BY dept_id HAVING SUM(salary) > 250",
        "SELECT dept_id, SUM(salary) s FROM emp GROUP BY dept_id ORDER BY s DESC",
        "SELECT id, CASE WHEN salary >= 200 THEN 'high' ELSE 'low' END AS band FROM emp",
        "SELECT name, dept FROM emp JOIN dept USING (dept_id)",
        "SELECT dept, SUM(salary) total FROM emp JOIN dept USING (dept_id) "
        "GROUP BY dept ORDER BY total DESC",
        "SELECT id FROM emp ORDER BY salary DESC LIMIT 2",
        "SELECT SUM(salary*2) AS s, COUNT(*) c FROM emp WHERE salary > 100",
        "SELECT dept_id, SUM(salary)/COUNT(*) AS avg_manual FROM emp GROUP BY dept_id",
        "SELECT * FROM emp WHERE dept_id = 10 ORDER BY id",
        "SELECT name FROM emp LEFT JOIN dept USING (dept_id) WHERE dept IS NOT NULL",
    ],
)
def test_sql_vs_duckdb(duck, tables, query):
    from conftest import assert_same

    emp, dept = tables
    out = bt.sql(query, emp=emp, dept=dept).collect()
    assert_same(out, duck.sql(query))
