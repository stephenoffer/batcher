"""Advanced SQL: GREATEST/LEAST/NULLIF/concat/ILIKE/EXTRACT, named windows, no-FROM."""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt


@pytest.fixture
def emp(duck):
    t = pa.table(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Al", "bob", "CARol", "dave", "EVE"],
            "dept": ["x", "y", "x", "y", "z"],
            "sal": [10, 40, 30, 20, 50],
        }
    )
    duck.register("emp", t)
    return t


@pytest.mark.parametrize(
    "q",
    [
        "SELECT GREATEST(id, sal) g FROM emp",
        "SELECT LEAST(id, sal) g FROM emp",
        "SELECT GREATEST(id, sal, 25) g FROM emp",
        "SELECT NULLIF(dept, 'x') r FROM emp",
        "SELECT concat(name, '!') r FROM emp",
        "SELECT concat(name, '-', dept) r FROM emp",
        "SELECT concat_ws('-', name, dept) r FROM emp",
    ],
)
def test_sql_scalar_extras(duck, emp, q):
    from conftest import assert_same

    assert_same(bt.sql(q, emp=emp).collect(), duck.sql(q))


@pytest.mark.parametrize(
    "q",
    [
        "SELECT id FROM emp WHERE name ILIKE 'a%'",
        "SELECT id FROM emp WHERE name ILIKE '%o%'",
        "SELECT id FROM emp WHERE name NOT ILIKE 'e%'",
    ],
)
def test_sql_ilike(duck, emp, q):
    from conftest import assert_same

    assert_same(bt.sql(q, emp=emp).collect(), duck.sql(q))


@pytest.fixture
def ts(duck):
    t = pa.table(
        {
            "d": pa.array(
                [dt.datetime(2021, 3, 15, 14, 30, 45), dt.datetime(2020, 12, 1, 9, 5, 1)],
                pa.timestamp("us"),
            )
        }
    )
    duck.register("ts", t)
    return t


@pytest.mark.parametrize(
    "part", ["year", "month", "day", "hour", "minute", "second", "quarter", "week", "dow", "doy"]
)
def test_sql_extract(duck, ts, part):
    from conftest import assert_same

    q = f"SELECT extract({part} FROM d) r FROM ts"
    assert_same(bt.sql(q, ts=ts).collect(), duck.sql(q))


@pytest.mark.parametrize(
    "q",
    [
        "SELECT id, row_number() OVER w r FROM emp WINDOW w AS (PARTITION BY dept ORDER BY sal)",
        "SELECT id, row_number() OVER w rn, lag(sal) OVER w lg "
        "FROM emp WINDOW w AS (PARTITION BY dept ORDER BY sal)",
        "SELECT id, SUM(sal) OVER w s FROM emp WINDOW w AS (PARTITION BY dept)",
    ],
)
def test_sql_named_window(duck, emp, q):
    from conftest import assert_same

    assert_same(bt.sql(q, emp=emp).collect(), duck.sql(q))


@pytest.mark.parametrize(
    "q",
    [
        "SELECT 1 + 1 AS r",
        "SELECT 'a' || 'b' AS r",
        "SELECT upper('hi') AS r",
        "SELECT extract(year FROM DATE '2021-03-15') AS y",
        "SELECT GREATEST(3, 7, 2) AS g",
    ],
)
def test_sql_no_from(duck, q):
    from conftest import assert_same

    assert_same(bt.sql(q).collect(), duck.sql(q))


@pytest.mark.parametrize(
    "q",
    [
        "SELECT id FROM nl WHERE mgr IS DISTINCT FROM 1",
        "SELECT id FROM nl WHERE mgr IS NOT DISTINCT FROM 1",
        "SELECT id FROM nl WHERE mgr IS DISTINCT FROM NULL",
        "SELECT id FROM nl WHERE mgr IS NOT DISTINCT FROM NULL",
        "SELECT id, mgr IS DISTINCT FROM 2 AS d FROM nl",
    ],
)
def test_sql_is_distinct_from(duck, q):
    from conftest import assert_same

    nl = pa.table({"id": [1, 2, 3, 4, 5], "mgr": [None, 1, 1, 2, 2]})
    duck.register("nl", nl)
    assert_same(bt.sql(q, nl=nl).collect(), duck.sql(q))


@pytest.mark.parametrize(
    "q",
    [
        "SELECT id FROM lk WHERE s LIKE 'a_c'",
        "SELECT id FROM lk WHERE s LIKE 'a%c'",
        "SELECT id FROM lk WHERE s LIKE 'a_%'",
        "SELECT id FROM lk WHERE s NOT LIKE 'a_c'",
        "SELECT id FROM lk WHERE s LIKE 'a.c'",
        "SELECT id FROM lk WHERE s LIKE 'a!%c' ESCAPE '!'",
        "SELECT id FROM lk WHERE s LIKE 'a!_c' ESCAPE '!'",
        "SELECT id FROM lk WHERE s ILIKE 'A_C'",
    ],
)
def test_sql_like_wildcards(duck, q):
    from conftest import assert_same

    lk = pa.table({"id": [1, 2, 3, 4, 5, 6], "s": ["abc", "aXc", "abbc", "a_c", "a%c", "xyz"]})
    duck.register("lk", lk)
    assert_same(bt.sql(q, lk=lk).collect(), duck.sql(q))


@pytest.mark.parametrize(
    "q",
    [
        "SELECT dept, SUM(sal) s FROM gs GROUP BY ROLLUP(dept)",
        "SELECT dept, role, SUM(sal) s FROM gs GROUP BY ROLLUP(dept, role)",
        "SELECT dept, role, SUM(sal) s FROM gs GROUP BY CUBE(dept, role)",
        "SELECT dept, role, SUM(sal) s FROM gs GROUP BY GROUPING SETS ((dept, role), (dept), ())",
        "SELECT dept, role, COUNT(*) n, SUM(sal) s FROM gs GROUP BY ROLLUP(dept, role)",
        "SELECT dept, role, MIN(sal) mn, MAX(sal) mx FROM gs GROUP BY CUBE(dept, role)",
    ],
)
def test_sql_grouping_sets(duck, q):
    from conftest import assert_same

    gs = pa.table(
        {
            "dept": ["x", "x", "y", "y", "z"],
            "role": ["a", "b", "a", "a", "b"],
            "sal": [10, 20, 30, 40, 50],
        }
    )
    duck.register("gs", gs)
    assert_same(bt.sql(q, gs=gs).collect(), duck.sql(q))
