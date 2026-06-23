"""Sketch-driven cardinality end-to-end: measure stats, learn, results unchanged (W2)."""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col


def test_column_statistics_measures_distinct_counts():
    from batcher import core

    t = pa.table(
        {
            "a": [1, 1, 2, 2, 3, 3, 3],  # 3 distinct
            "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],  # 7 distinct, numeric
        }
    )
    ndv, quants, avg_bytes = core.column_statistics(t.to_batches(), ["a", "b"])
    # HLL is exact-ish at these tiny cardinalities (linear-counting regime).
    assert abs(ndv["a"] - 3) <= 1
    assert abs(ndv["b"] - 7) <= 1
    # Numeric column gets quantile boundaries (ascending), non-trivial range.
    assert quants["b"]["values"][0] <= quants["b"]["values"][-1]
    # Both columns report a positive measured per-row byte width.
    assert avg_bytes["a"] > 0
    assert avg_bytes["b"] > 0


def test_learning_path_keeps_results_correct(duck):
    from conftest import assert_same

    t = pa.table({"k": [1, 1, 2, 3, 3, 3], "v": [10, 20, 30, 40, 50, 60]})
    duck.register("t", t)
    ds = bt.from_arrow(t)
    # First run measures + records column stats into the (singleton) hub; the second
    # run plans with the learned ndv. Both must match DuckDB — stats change the plan,
    # never the answer.
    expected = duck.sql("SELECT * FROM t WHERE k = 3")
    out_cold = ds.filter(col("k") == 3).collect()
    out_warm = ds.filter(col("k") == 3).collect()
    assert_same(out_cold, expected)
    assert_same(out_warm, expected)


def test_learned_join_cardinality_keeps_results_correct(duck):
    from conftest import assert_same

    # A join where learned key ndv refines the cardinality estimate; result must
    # stay identical to DuckDB.
    left = pa.table({"id": [1, 2, 3, 4], "x": [10, 20, 30, 40]})
    right = pa.table({"id": [2, 3, 3, 5], "y": [200, 300, 301, 500]})
    duck.register("l", left)
    duck.register("r", right)
    out = bt.from_arrow(left).join(bt.from_arrow(right), on="id").collect()
    expected = duck.sql("SELECT * FROM l JOIN r USING (id)")
    assert_same(out, expected)
