"""Correlated subqueries (EXISTS / IN / scalar) decorrelate to joins, vs DuckDB."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt


@pytest.fixture
def tables(duck):
    emp = pa.table(
        {
            "id": [1, 2, 3, 4],
            "dept": ["x", "y", "x", "z"],
            "sal": [10, 20, 30, 40],
            "mgr": [None, 1, 1, 2],
        }
    )
    proj = pa.table(
        {
            "eid": [1, 1, 3, 2],
            "d": ["x", "y", "x", "w"],
            "hrs": [5, 10, 20, 15],
        }
    )
    duck.register("emp", emp)
    duck.register("proj", proj)
    return emp, proj


@pytest.mark.parametrize(
    "q",
    [
        "SELECT id FROM emp e WHERE EXISTS (SELECT 1 FROM proj p WHERE p.eid = e.id)",
        "SELECT id FROM emp e WHERE NOT EXISTS (SELECT 1 FROM proj p WHERE p.eid = e.id)",
        "SELECT id FROM emp e WHERE EXISTS (SELECT 1 FROM proj p WHERE p.eid = e.id AND p.hrs > 8)",
        "SELECT id FROM emp e WHERE EXISTS (SELECT 1 FROM emp m WHERE m.mgr = e.id)",
        "SELECT id FROM emp e WHERE EXISTS "
        "(SELECT 1 FROM proj p WHERE p.eid = e.id AND p.d = e.dept)",
    ],
)
def test_correlated_exists(duck, tables, q):
    from conftest import assert_same

    emp, proj = tables
    assert_same(bt.sql(q, emp=emp, proj=proj).collect(), duck.sql(q))


@pytest.mark.parametrize(
    "q",
    [
        "SELECT id FROM emp e WHERE e.dept IN (SELECT p.d FROM proj p WHERE p.eid = e.id)",
        "SELECT id FROM emp e WHERE e.dept NOT IN (SELECT p.d FROM proj p WHERE p.eid = e.id)",
        "SELECT id FROM emp e WHERE e.dept IN "
        "(SELECT p.d FROM proj p WHERE p.eid = e.id AND p.hrs > 8)",
    ],
)
def test_correlated_in(duck, tables, q):
    from conftest import assert_same

    emp, proj = tables
    assert_same(bt.sql(q, emp=emp, proj=proj).collect(), duck.sql(q))


@pytest.mark.parametrize(
    "q",
    [
        "SELECT id, (SELECT MAX(p.hrs) FROM proj p WHERE p.eid = e.id) m FROM emp e",
        "SELECT id, (SELECT SUM(p.hrs) FROM proj p WHERE p.eid = e.id) s FROM emp e",
        "SELECT id, (SELECT COUNT(*) FROM proj p WHERE p.eid = e.id) c FROM emp e",
        "SELECT id, (SELECT COUNT(hrs) FROM proj p WHERE p.eid = e.id) c FROM emp e",
        "SELECT id FROM emp e WHERE sal > (SELECT SUM(p.hrs) FROM proj p WHERE p.eid = e.id)",
        "SELECT id, (SELECT MAX(p.hrs) FROM proj p WHERE p.eid = e.id AND p.hrs < 20) m FROM emp e",
        "SELECT id, (SELECT MAX(p.hrs) FROM proj p WHERE p.eid = e.id) m, "
        "(SELECT COUNT(*) FROM proj p WHERE p.eid = e.id) c FROM emp e",
    ],
)
def test_correlated_scalar(duck, tables, q):
    from conftest import assert_same

    emp, proj = tables
    assert_same(bt.sql(q, emp=emp, proj=proj).collect(), duck.sql(q))
