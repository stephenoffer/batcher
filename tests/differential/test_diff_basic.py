"""Differential tests for filter/project/aggregate against DuckDB."""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col, count


def test_filter_project_vs_duckdb(duck):
    from conftest import assert_same

    data = {"x": [1, 2, 3, 4, 5, 6], "y": [10, 20, 30, 40, 50, 60]}
    t = pa.table(data)
    duck.register("t", t)

    out = bt.from_arrow(t).filter(col("x") > 2).select("x", xy=col("x") * col("y")).collect()
    expected = duck.sql("SELECT x, x * y AS xy FROM t WHERE x > 2")
    assert_same(out, expected)


def test_group_by_sum_vs_duckdb(duck):
    from conftest import assert_same

    t = pa.table(
        {
            "dept": ["eng", "eng", "sales", "sales", "eng", "ops"],
            "salary": [100, 200, 150, 50, 300, 75],
        }
    )
    duck.register("t", t)

    out = (
        bt.from_arrow(t)
        .group_by("dept")
        .agg(total=col("salary").sum(), n=count(), hi=col("salary").max(), avg=col("salary").mean())
        .collect()
    )
    expected = duck.sql(
        "SELECT dept, SUM(salary) AS total, COUNT(*) AS n, "
        "MAX(salary) AS hi, AVG(salary) AS avg FROM t GROUP BY dept"
    )
    assert_same(out, expected)


def test_global_aggregate_vs_duckdb(duck):
    from conftest import assert_same

    t = pa.table({"x": [1, 2, 3, 4], "y": [1.5, 2.5, 3.5, 4.5]})
    duck.register("t", t)

    out = (
        bt.from_arrow(t).group_by().agg(sx=col("x").sum(), my=col("y").mean(), n=count()).collect()
    )
    expected = duck.sql("SELECT SUM(x) AS sx, AVG(y) AS my, COUNT(*) AS n FROM t")
    assert_same(out, expected)


def test_two_key_group_by_vs_duckdb(duck):
    from conftest import assert_same

    t = pa.table(
        {
            "a": ["x", "x", "y", "y", "x"],
            "b": [1, 1, 2, 2, 2],
            "v": [10, 20, 30, 40, 50],
        }
    )
    duck.register("t", t)

    out = bt.from_arrow(t).group_by("a", "b").agg(s=col("v").sum()).collect()
    expected = duck.sql("SELECT a, b, SUM(v) AS s FROM t GROUP BY a, b")
    assert_same(out, expected)
