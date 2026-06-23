"""SQL scalar functions (math / string / date) dispatch to the engine, vs DuckDB."""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt


@pytest.fixture
def nums(duck):
    t = pa.table({"x": [1.0, 4.0, 9.0, 16.0]})
    duck.register("n", t)
    return t


@pytest.fixture
def strs(duck):
    t = pa.table({"s": ["Hello", "ab", "WORLD"]})
    duck.register("s", t)
    return t


@pytest.fixture
def dates(duck):
    t = pa.table(
        {
            "d": pa.array(
                [dt.datetime(2021, 3, 15, 14, 30, 45), dt.datetime(2020, 12, 1, 9, 5, 1)],
                pa.timestamp("us"),
            )
        }
    )
    duck.register("d", t)
    return t


@pytest.mark.parametrize(
    "fn",
    [
        "ln",
        "log10",
        "log2",
        "log",
        "exp",
        "sqrt",
        "abs",
        "sin",
        "cos",
        "tan",
        "floor",
        "ceil",
        "sign",
        "cbrt",
        "trunc",
        "degrees",
        "radians",
    ],
)
def test_sql_math_functions(duck, nums, fn):
    from conftest import assert_same

    q = f"SELECT {fn}(x) AS r FROM n"
    assert_same(bt.sql(q, n=nums).collect(), duck.sql(q))


def test_sql_round(duck, nums):
    from conftest import assert_same

    assert_same(
        bt.sql("SELECT round(x) AS r FROM n", n=nums).collect(),
        duck.sql("SELECT round(x) AS r FROM n"),
    )


@pytest.mark.parametrize(
    "q",
    [
        "SELECT upper(s) AS r FROM s",
        "SELECT lower(s) AS r FROM s",
        "SELECT length(s) AS r FROM s",
        "SELECT reverse(s) AS r FROM s",
        "SELECT trim(s) AS r FROM s",
        "SELECT substring(s, 1, 2) AS r FROM s",
        "SELECT lpad(s, 7, '*') AS r FROM s",
        "SELECT rpad(s, 7, '*') AS r FROM s",
        "SELECT position('l' IN s) AS r FROM s",
    ],
)
def test_sql_string_functions(duck, strs, q):
    from conftest import assert_same

    assert_same(bt.sql(q, s=strs).collect(), duck.sql(q))


@pytest.mark.parametrize(
    "fn", ["year", "month", "day", "hour", "minute", "second", "quarter", "week"]
)
def test_sql_date_functions(duck, dates, fn):
    from conftest import assert_same

    q = f"SELECT {fn}(d) AS r FROM d"
    assert_same(bt.sql(q, d=dates).collect(), duck.sql(q))


def test_sql_function_in_where_and_nested(duck, strs):
    from conftest import assert_same

    q = "SELECT upper(s) AS u FROM s WHERE length(s) > 3"
    assert_same(bt.sql(q, s=strs).collect(), duck.sql(q))


def test_sql_nested_functions(duck, nums):
    from conftest import assert_same

    q = "SELECT sqrt(abs(x)) AS r FROM n"
    assert_same(bt.sql(q, n=nums).collect(), duck.sql(q))


@pytest.fixture
def emp(duck):
    t = pa.table(
        {
            "name": ["alice", "bob", "carol", "dave", "erin"],
            "dept": ["eng", "eng", "sales", "sales", "eng"],
            "salary": [100, 250, 150, 300, 175],
        }
    )
    duck.register("emp", t)
    return t


@pytest.mark.parametrize(
    "q",
    [
        "SELECT upper(dept) d, SUM(salary) s FROM emp GROUP BY upper(dept)",
        "SELECT salary % 100 b, COUNT(*) n FROM emp GROUP BY salary % 100",
        "SELECT dept, salary % 200 b, COUNT(*) n FROM emp GROUP BY dept, salary % 200",
        "SELECT length(name) l, COUNT(*) n FROM emp GROUP BY length(name)",
    ],
)
def test_sql_group_by_expression(duck, emp, q):
    from conftest import assert_same

    assert_same(bt.sql(q, emp=emp).collect(), duck.sql(q))


def test_sql_order_by_expression(duck, emp):
    from conftest import assert_same_ordered

    # ORDER BY a function; unique lengths/names avoid tie ambiguity.
    q = "SELECT name FROM emp ORDER BY length(name), name"
    assert_same_ordered(bt.sql(q, emp=emp).collect(), duck.sql(q))


def test_sql_order_by_aggregate(duck, emp):
    from conftest import assert_same_ordered

    # ORDER BY an aggregate (resolved to its output column); salaries are distinct.
    q = "SELECT dept, SUM(salary) s FROM emp GROUP BY dept ORDER BY SUM(salary)"
    assert_same_ordered(bt.sql(q, emp=emp).collect(), duck.sql(q))
