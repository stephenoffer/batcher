"""Simple CASE, COUNT(*) OVER, bool_and/bool_or vs DuckDB."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt


@pytest.fixture
def emp(duck):
    t = pa.table(
        {
            "id": [1, 2, 3, 4, 5],
            "dept": ["x", "y", "x", "y", "z"],
            "sal": [10, 20, 30, 40, 50],
            "ok": [True, True, False, True, False],
        }
    )
    duck.register("emp", t)
    return t


@pytest.mark.parametrize(
    "q",
    [
        "SELECT CASE dept WHEN 'x' THEN 1 WHEN 'y' THEN 2 ELSE 0 END c FROM emp",
        "SELECT CASE dept WHEN 'x' THEN 1 END c FROM emp",
        "SELECT CASE id WHEN 1 THEN 100 WHEN 2 THEN 200 ELSE -1 END c FROM emp",
        "SELECT CASE WHEN sal > 25 THEN 1 ELSE 0 END c FROM emp",  # searched still works
    ],
)
def test_simple_case(duck, emp, q):
    from conftest import assert_same

    assert_same(bt.sql(q, emp=emp).collect(), duck.sql(q))


@pytest.mark.parametrize(
    "q",
    [
        "SELECT id, COUNT(*) OVER () n FROM emp",
        "SELECT id, COUNT(*) OVER (PARTITION BY dept) n FROM emp",
        "SELECT id, COUNT(sal) OVER (PARTITION BY dept) n FROM emp",
        "SELECT id, COUNT(*) OVER (PARTITION BY dept) c, "
        "SUM(sal) OVER (PARTITION BY dept) s FROM emp",
    ],
)
def test_count_star_over(duck, emp, q):
    from conftest import assert_same

    assert_same(bt.sql(q, emp=emp).collect(), duck.sql(q))


@pytest.mark.parametrize(
    "q",
    [
        "SELECT dept, bool_and(sal > 20) b FROM emp GROUP BY dept",
        "SELECT dept, bool_or(sal > 35) b FROM emp GROUP BY dept",
        "SELECT dept, bool_and(ok) b FROM emp GROUP BY dept",
        "SELECT bool_or(sal > 45) b FROM emp",
        "SELECT bool_and(sal > 5) b FROM emp",
    ],
)
def test_bool_aggregates(duck, emp, q):
    from conftest import assert_same

    assert_same(bt.sql(q, emp=emp).collect(), duck.sql(q))
