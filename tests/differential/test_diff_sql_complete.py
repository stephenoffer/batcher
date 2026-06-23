"""SQL completeness: cross join, positional GROUP BY/ORDER BY, qualified star."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt


@pytest.fixture
def tables(duck):
    emp = pa.table(
        {
            "id": [1, 2, 3, 4],
            "dept": ["x", "x", "y", "y"],
            "sal": [10, 20, 30, 40],
        }
    )
    dept = pa.table({"dept": ["x", "y"], "loc": ["NY", "SF"]})
    duck.register("emp", emp)
    duck.register("dept", dept)
    return emp, dept


@pytest.mark.parametrize(
    "q",
    [
        "SELECT emp.id, dept.loc FROM emp, dept",
        "SELECT emp.id, dept.loc FROM emp CROSS JOIN dept",
    ],
)
def test_cross_join(duck, tables, q):
    from conftest import assert_same

    emp, dept = tables
    assert_same(bt.sql(q, emp=emp, dept=dept).collect(), duck.sql(q))


@pytest.mark.parametrize(
    "q",
    [
        "SELECT dept, COUNT(*) n FROM emp GROUP BY 1",
        "SELECT sal % 20 b, COUNT(*) n FROM emp GROUP BY 1",
        "SELECT dept, sal % 20 b, COUNT(*) n FROM emp GROUP BY 1, 2",
    ],
)
def test_group_by_position(duck, tables, q):
    from conftest import assert_same

    emp, _ = tables
    assert_same(bt.sql(q, emp=emp).collect(), duck.sql(q))


@pytest.mark.parametrize(
    "q",
    [
        "SELECT id, sal FROM emp ORDER BY 2 DESC",
        "SELECT dept, id FROM emp ORDER BY 1, 2",
        "SELECT id, sal FROM emp ORDER BY 2 DESC, 1",
    ],
)
def test_order_by_position(duck, tables, q):
    from conftest import assert_same_ordered

    emp, _ = tables
    assert_same_ordered(bt.sql(q, emp=emp).collect(), duck.sql(q))


@pytest.mark.parametrize(
    "q",
    [
        "SELECT emp.* FROM emp",
        "SELECT emp.* FROM emp WHERE sal > 10",
    ],
)
def test_qualified_star(duck, tables, q):
    from conftest import assert_same

    emp, _ = tables
    assert_same(bt.sql(q, emp=emp).collect(), duck.sql(q))


def test_cross_join_then_filter(duck, tables):
    from conftest import assert_same

    emp, dept = tables
    q = "SELECT emp.id, dept.loc FROM emp, dept WHERE dept.loc = 'NY'"
    assert_same(bt.sql(q, emp=emp, dept=dept).collect(), duck.sql(q))
