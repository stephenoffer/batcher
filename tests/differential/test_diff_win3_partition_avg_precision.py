"""Differential coverage for whole-partition window `avg` numeric precision.

A whole-partition ``AVG(i) OVER (PARTITION BY k)`` over 64-bit integers must sum the
values EXACTLY before dividing. Accumulating each value through f64 as it is added
loses precision above 2^53, so the average drifts by ±1 from DuckDB (which sums in
128-bit before converting to double). See ``window_partition_agg::grouped_i64``.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential


def test_partition_avg_i64_above_2_53(duck):
    from conftest import assert_same

    # avg([2^53+1, 1]) is exactly 2^52+1 = 4503599627370497, not 2^52 that a running
    # `sum += v as f64` produced. A second partition of ordinary values guards the path.
    t = pa.table(
        {
            "k": pa.array([1, 1, 2, 2], type=pa.int64()),
            "i": pa.array([2**53 + 1, 1, 10, 20], type=pa.int64()),
        }
    )
    out = bt.from_arrow(t).with_columns(a=col("i").mean().over(partition_by=["k"])).collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT *, AVG(i) OVER (PARTITION BY k) AS a FROM t"))


def test_partition_avg_i64_sum_overflows_i64(duck):
    from conftest import assert_same

    # avg([2^62, 2^62]) = 2^62: the sum overflows i64 but DuckDB (128-bit sum) still
    # returns a finite average, so the exact accumulator must not overflow either.
    t = pa.table(
        {
            "k": pa.array([1, 1], type=pa.int64()),
            "i": pa.array([2**62, 2**62], type=pa.int64()),
        }
    )
    out = bt.from_arrow(t).with_columns(a=col("i").mean().over(partition_by=["k"])).collect()
    duck.register("t", t)
    assert_same(out, duck.sql("SELECT *, AVG(i) OVER (PARTITION BY k) AS a FROM t"))


def test_running_avg_i64_above_2_53(duck):
    from conftest import assert_same_ordered

    # `AVG(i) OVER ()` drives a *separate* accumulator (`running_numeric_i64` in window.rs)
    # than the whole-partition PARTITION-BY path above — all rows are peers of one frame, so
    # every row gets the whole-frame average. It must sum in 128-bit too: the running
    # `sum += v as f64` lost the low bit and returned 4503599627370496.0 for every row
    # instead of the exact 2^52+1 = 4503599627370497.0.
    t = pa.table(
        {
            "j": pa.array([1, 2], type=pa.int64()),
            "i": pa.array([2**53 + 1, 1], type=pa.int64()),
        }
    )
    duck.register("t", t)
    out = bt.sql(
        "SELECT j, AVG(i) OVER () AS a FROM t ORDER BY j",
        t=bt.from_arrow(t),
    ).collect()
    assert_same_ordered(
        out, duck.sql("SELECT j, AVG(i) OVER () AS a FROM t ORDER BY j")
    )
