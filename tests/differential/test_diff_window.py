"""Window-function differential tests vs DuckDB."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "dept": ["a", "a", "a", "b", "b", "b"],
            "name": list("uvwxyz"),
            "salary": [100, 300, 200, 150, 250, 250],
        }
    )
    duck.register("t", tbl)
    return tbl


def test_ranking_functions(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .window(
            partition_by=["dept"],
            order_by=[("salary", True)],
            functions={"rn": "row_number", "rk": "rank", "dr": "dense_rank"},
        )
        .collect()
    )
    expected = duck.sql("""
        SELECT *, row_number() OVER w rn, rank() OVER w rk, dense_rank() OVER w dr
        FROM t WINDOW w AS (PARTITION BY dept ORDER BY salary DESC)
    """)
    assert_same(out, expected)


def test_partition_aggregates(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .window(
            partition_by=["dept"],
            functions={"tot": ("sum", "salary"), "avg": ("avg", "salary"), "hi": ("max", "salary")},
        )
        .collect()
    )
    expected = duck.sql("""
        SELECT *, SUM(salary) OVER w tot, AVG(salary) OVER w avg, MAX(salary) OVER w hi
        FROM t WINDOW w AS (PARTITION BY dept)
    """)
    assert_same(out, expected)


def test_running_aggregates(duck, t):
    # Aggregates WITH an ORDER BY are cumulative (running) over the ordered
    # partition, with RANGE peer semantics — tied rows (dept 'b' has two 250s)
    # share the end-of-peer-group value, matching SQL's default frame.
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .window(
            partition_by=["dept"],
            order_by=[("salary", False)],
            functions={
                "rs": ("sum", "salary"),
                "ra": ("avg", "salary"),
                "rmin": ("min", "salary"),
                "rmax": ("max", "salary"),
                "rc": ("count", "salary"),
            },
        )
        .collect()
    )
    expected = duck.sql("""
        SELECT *,
          SUM(salary) OVER w rs, AVG(salary) OVER w ra,
          MIN(salary) OVER w rmin, MAX(salary) OVER w rmax, COUNT(salary) OVER w rc
        FROM t WINDOW w AS (PARTITION BY dept ORDER BY salary)
    """)
    assert_same(out, expected)


def test_running_sum_no_partition(duck, t):
    # A single running aggregate over the whole relation ordered by a key.
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .window(order_by=[("salary", False)], functions={"rs": ("sum", "salary")})
        .collect()
    )
    expected = duck.sql("SELECT *, SUM(salary) OVER (ORDER BY salary) rs FROM t")
    assert_same(out, expected)


def test_global_window_no_partition(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .window(order_by=[("salary", False)], functions={"rn": "row_number"})
        .collect()
    )
    expected = duck.sql("SELECT *, row_number() OVER (ORDER BY salary) rn FROM t")
    assert_same(out, expected)


def test_window_then_filter_uses_pushdown(duck, t):
    # A filter after a window exercises the pushdown passes' Window branch.
    from batcher import col
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .window(partition_by=["dept"], order_by=[("salary", True)], functions={"rk": "rank"})
        .filter(col("rk") == 1)
        .select("dept", "salary", "rk")
        .collect()
    )
    expected = duck.sql("""
        SELECT dept, salary, rk FROM (
          SELECT *, rank() OVER (PARTITION BY dept ORDER BY salary DESC) rk FROM t
        ) WHERE rk = 1
    """)
    assert_same(out, expected)


def test_rows_frame_trailing_sum_vs_duckdb(duck):
    from conftest import assert_same

    tbl = pa.table(
        {
            "dept": ["a", "a", "a", "a", "b", "b", "b"],
            "ts": [1, 2, 3, 4, 1, 2, 3],
            "amount": [10, 20, 30, 40, 5, 15, 25],
        }
    )
    duck.register("w", tbl)
    # ROWS BETWEEN 2 PRECEDING AND CURRENT ROW — a trailing 3-row rolling sum.
    out = (
        bt.from_arrow(tbl)
        .window(
            partition_by=["dept"],
            order_by=["ts"],
            functions={"roll": ("sum", "amount")},
            frame=(-2, 0),
        )
        .collect()
    )
    expected = duck.sql("""
        SELECT *, sum(amount) OVER (
            PARTITION BY dept ORDER BY ts
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS roll
        FROM w
    """)
    assert_same(out, expected)


def test_rows_frame_centered_avg_min_max_count_vs_duckdb(duck):
    from conftest import assert_same

    tbl = pa.table(
        {
            "dept": ["a", "a", "a", "a", "a"],
            "ts": [1, 2, 3, 4, 5],
            "v": [5, 1, 9, 3, 7],
        }
    )
    duck.register("wc", tbl)
    # ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING — a centered 3-row window.
    out = (
        bt.from_arrow(tbl)
        .window(
            partition_by=["dept"],
            order_by=["ts"],
            functions={
                "a": ("avg", "v"),
                "mn": ("min", "v"),
                "mx": ("max", "v"),
                "c": ("count", "v"),
            },
            frame=(-1, 1),
        )
        .collect()
    )
    expected = duck.sql("""
        SELECT *,
            avg(v)   OVER w a,
            min(v)   OVER w mn,
            max(v)   OVER w mx,
            count(v) OVER w c
        FROM wc
        WINDOW w AS (PARTITION BY dept ORDER BY ts ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING)
    """)
    assert_same(out, expected)


def test_rows_frame_unbounded_following_vs_duckdb(duck):
    from conftest import assert_same

    tbl = pa.table({"g": [1, 1, 1, 1], "ts": [1, 2, 3, 4], "v": [10, 20, 30, 40]})
    duck.register("wu", tbl)
    # CURRENT ROW .. UNBOUNDED FOLLOWING — a "remaining suffix" sum.
    out = (
        bt.from_arrow(tbl)
        .window(
            partition_by=["g"],
            order_by=["ts"],
            functions={"suffix": ("sum", "v")},
            frame=(0, None),
        )
        .collect()
    )
    expected = duck.sql("""
        SELECT *, sum(v) OVER (
            PARTITION BY g ORDER BY ts ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
        ) AS suffix
        FROM wu
    """)
    assert_same(out, expected)
