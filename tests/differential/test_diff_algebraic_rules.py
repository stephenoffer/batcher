"""Algebraic rewrite rules preserve results vs DuckDB (W3)."""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col


def _data():
    return pa.table(
        {
            "x": [3, 1, 4, 1, 5, 9, 2, 6, 5, 3],
            "y": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        }
    )


def test_adjacent_filters_vs_duckdb(duck):
    from conftest import assert_same

    t = _data()
    duck.register("t", t)
    out = bt.from_arrow(t).filter(col("x") > 1).filter(col("y") < 90).collect()
    expected = duck.sql("SELECT * FROM t WHERE x > 1 AND y < 90")
    assert_same(out, expected)


def test_double_distinct_vs_duckdb(duck):
    from conftest import assert_same

    t = _data()
    duck.register("t", t)
    out = bt.from_arrow(t).select("x").distinct().distinct().collect()
    expected = duck.sql("SELECT DISTINCT x FROM t")
    assert_same(out, expected)


def test_distinct_over_aggregate_vs_duckdb(duck):
    from conftest import assert_same

    t = _data()
    duck.register("t", t)
    out = bt.from_arrow(t).group_by("x").agg(total=col("y").sum()).distinct().collect()
    expected = duck.sql("SELECT x, SUM(y) AS total FROM t GROUP BY x")
    assert_same(out, expected)


def test_nested_limits_vs_duckdb(duck):
    from conftest import assert_same

    # Order first so the row window is deterministic for the comparison.
    t = _data()
    duck.register("t", t)
    out = bt.from_arrow(t).sort("y").limit(6).limit(3, offset=1).collect()
    expected = duck.sql("SELECT * FROM (SELECT * FROM t ORDER BY y LIMIT 6) LIMIT 3 OFFSET 1")
    assert_same(out, expected)


def test_limit_through_project_vs_duckdb(duck):
    from conftest import assert_same

    # Distinct x so ORDER BY x LIMIT 2 has no tie ambiguity.
    t = pa.table({"x": [5, 3, 1, 4, 2], "y": [50, 30, 10, 40, 20]})
    duck.register("t", t)
    out = bt.from_arrow(t).sort("x").select(z=col("x") * col("y")).limit(2).collect()
    expected = duck.sql("SELECT x * y AS z FROM t ORDER BY x LIMIT 2")
    assert_same(out, expected)


def test_sort_before_aggregate_vs_duckdb(duck):
    from conftest import assert_same

    t = _data()
    duck.register("t", t)
    # A sort feeding a group-by is wasted; removing it must not change the result.
    out = bt.from_arrow(t).sort("y").group_by("x").agg(total=col("y").sum()).collect()
    expected = duck.sql("SELECT x, SUM(y) AS total FROM t GROUP BY x")
    assert_same(out, expected)


def test_filter_through_sort_vs_duckdb(duck):
    from conftest import assert_same

    t = _data()
    duck.register("t", t)
    out = bt.from_arrow(t).sort("x").filter(col("x") > 3).collect()
    expected = duck.sql("SELECT * FROM t WHERE x > 3 ORDER BY x")
    assert_same(out, expected)


def test_filter_through_aggregate_vs_duckdb(duck):
    from conftest import assert_same

    t = pa.table(
        {
            "dept": ["a", "a", "b", "b", "c", "c", "a"],
            "sal": [10, 20, 30, 40, 50, 60, 70],
        }
    )
    duck.register("t", t)
    # Filter on the group key: pushed below the aggregate; result matches DuckDB.
    out = (
        bt.from_arrow(t)
        .group_by("dept")
        .agg(total=col("sal").sum())
        .filter(col("dept") != "b")
        .collect()
    )
    expected = duck.sql("SELECT dept, SUM(sal) AS total FROM t WHERE dept <> 'b' GROUP BY dept")
    assert_same(out, expected)


def test_merged_projections_vs_duckdb(duck):
    from conftest import assert_same

    t = _data()
    duck.register("t", t)
    # with_columns then select then with_columns -> stacked Projects collapsed.
    out = (
        bt.from_arrow(t)
        .with_columns(z=col("x") * 2)
        .select("x", "z")
        .with_columns(w=col("z") + col("x"))
        .collect()
    )
    expected = duck.sql("SELECT x, x * 2 AS z, (x * 2) + x AS w FROM t")
    assert_same(out, expected)


def test_filter_through_projection_vs_duckdb(duck):
    from conftest import assert_same

    t = _data()
    duck.register("t", t)
    # rename x->a then filter on a: the optimizer pushes the filter below the
    # projection; result must still match DuckDB.
    out = bt.from_arrow(t).select(a=col("x"), b=col("y")).filter(col("a") > 3).collect()
    expected = duck.sql("SELECT x AS a, y AS b FROM t WHERE x > 3")
    assert_same(out, expected)


def test_filter_into_union_all_vs_duckdb(duck):
    from conftest import assert_same

    t = _data()
    duck.register("t", t)
    out = (
        bt.from_arrow(t)
        .select("x")
        .union(bt.from_arrow(t).select("x"))
        .filter(col("x") > 3)
        .collect()
    )
    expected = duck.sql("SELECT x FROM (SELECT x FROM t UNION ALL SELECT x FROM t) WHERE x > 3")
    assert_same(out, expected)


def test_limit_into_union_all_preserves_multiset_vs_duckdb(duck):
    from conftest import assert_same

    # Non-truncating limit: validates the pushdown drops no rows of the UNION ALL
    # (the top-N of an *unordered* union isn't deterministic, so don't truncate).
    t = _data()
    duck.register("t", t)
    out = bt.from_arrow(t).select("x").union(bt.from_arrow(t).select("x")).limit(100).collect()
    expected = duck.sql("SELECT x FROM t UNION ALL SELECT x FROM t LIMIT 100")
    assert_same(out, expected)
