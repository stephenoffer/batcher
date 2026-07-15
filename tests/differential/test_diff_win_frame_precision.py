"""Differential coverage for numeric stability + overflow in sliding window frames.

A `ROWS` frame aggregate slides in one pass. The naive slide keeps a running
accumulator and *subtracts* the value leaving the frame — which is catastrophically
unstable on floats (``1e16 + 1 - 1e16 == 0``, losing the real window sum) and can
silently wrap an i64 SUM. These check the sliding SUM/AVG against DuckDB on inputs
that expose both failure modes.
"""

from __future__ import annotations

import pytest

import batcher as bt

pytestmark = pytest.mark.differential


def test_sliding_float_sum_is_exact_over_large_magnitudes(duck):
    from conftest import assert_same

    # A huge value followed by tiny ones: once the huge value leaves the trailing
    # 2-row frame, the sum must recover the exact 1+1=2, not 1e16+1-1e16=0.
    ds = bt.from_pydict({"i": [1, 2, 3, 4], "v": [1e16, 1.0, 1.0, 1.0]})
    duck.register("t", ds.collect())
    got = ds.window(
        order_by=["i"], functions={"s": ("sum", "v"), "a": ("avg", "v")}, frame=(-1, 0)
    ).collect()
    want = duck.sql(
        "SELECT *, "
        "sum(v) OVER (ORDER BY i ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS s, "
        "avg(v) OVER (ORDER BY i ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS a "
        "FROM t"
    )
    assert_same(got, want)


def test_rolling_sum_float_matches_duckdb(duck):
    from conftest import assert_same

    ds = bt.from_pydict({"i": [1, 2, 3, 4, 5], "v": [1e15, -1e15, 2.0, 3.0, 1e15]})
    duck.register("t", ds.collect())
    got = ds.with_columns(r=bt.col("v").rolling_sum(2, order_by=["i"])).collect()
    want = duck.sql(
        "SELECT *, "
        "sum(v) OVER (ORDER BY i ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS r "
        "FROM t"
    )
    assert_same(got, want)


def test_sliding_partitioned_float_sum(duck):
    from conftest import assert_same

    # Two partitions, each with a large-magnitude leader, exercises the per-partition
    # reset of the sliding accumulator.
    ds = bt.from_pydict(
        {
            "g": ["a", "a", "a", "b", "b", "b"],
            "i": [1, 2, 3, 1, 2, 3],
            "v": [1e16, 1.0, 2.0, 5.0, 1e16, 3.0],
        }
    )
    duck.register("t", ds.collect())
    got = ds.window(
        partition_by=["g"], order_by=["i"], functions={"s": ("sum", "v")}, frame=(-1, 0)
    ).collect()
    want = duck.sql(
        "SELECT *, sum(v) OVER "
        "(PARTITION BY g ORDER BY i ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS s FROM t"
    )
    assert_same(got, want)
