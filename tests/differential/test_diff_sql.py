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


@pytest.mark.parametrize(
    "query",
    [
        # comma self-join with a non-equi filter across the two aliases
        "SELECT a.id AS aid, b.id AS bid FROM emp a, emp b "
        "WHERE a.dept_id = b.dept_id AND a.id < b.id",
        # explicit JOIN ON self-join
        "SELECT a.id AS aid, b.id AS bid FROM emp a JOIN emp b "
        "ON a.dept_id = b.dept_id AND a.id <> b.id",
        # self-join with aggregation referencing both aliases
        "SELECT a.dept_id AS d, COUNT(*) c, SUM(b.salary) s FROM emp a, emp b "
        "WHERE a.dept_id = b.dept_id GROUP BY a.dept_id",
        # a bare (unaliased) projection of a self-join column keeps its output name
        "SELECT a.name FROM emp a, emp b WHERE a.id = b.id AND b.salary > 150",
    ],
)
def test_self_join_vs_duckdb(duck, tables, query):
    """Self-joins (same table aliased twice) are disambiguated and match DuckDB."""
    from conftest import assert_same

    emp, dept = tables
    out = bt.sql(query, emp=emp, dept=dept).collect()
    assert_same(out, duck.sql(query))


@pytest.mark.parametrize(
    "query",
    [
        # decimal literal arithmetic folds exactly (0.06 + 0.01 == 0.07), so the
        # boundary rows are selected the same as DuckDB (TPC-H Q6 shape).
        "SELECT COUNT(*) c FROM disc WHERE d BETWEEN 0.06 - 0.01 AND 0.06 + 0.01",
        "SELECT SUM(d) s FROM disc WHERE d <= 0.05 + 0.02",
        "SELECT COUNT(*) c FROM disc WHERE d > 0.1 - 0.03",
        # integer arithmetic keeps integer semantics
        "SELECT COUNT(*) c FROM disc WHERE q < 5 + 10",
    ],
)
def test_decimal_literal_arithmetic_vs_duckdb(duck, query):
    """SQL decimal literals fold with exact (Decimal) semantics, matching DuckDB."""
    from conftest import assert_same

    disc = pa.table(
        {
            "d": pa.array([0.04, 0.05, 0.06, 0.07, 0.08, 0.05, 0.06, 0.07], type=pa.float64()),
            "q": pa.array([10, 12, 14, 16, 18, 11, 13, 15], type=pa.int64()),
        }
    )
    duck.register("disc", disc)
    assert_same(bt.sql(query, disc=disc).collect(), duck.sql(query))
