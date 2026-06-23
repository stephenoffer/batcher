"""Differential coverage for expression window functions — `agg.over(...)` vs DuckDB.

`col("x").sum().over(partition_by=["g"])` lowers to the relational `Window` operator
and must match SQL ``SUM(x) OVER (PARTITION BY g)``.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential


def _t():
    return pa.table(
        {
            "g": pa.array(["a", "a", "b", "b", "b"]),
            "v": pa.array([10, 20, 30, 40, 50], type=pa.int64()),
        }
    )


def test_partition_sum_over(duck):
    from conftest import assert_same

    out = bt.from_arrow(_t()).with_columns(gsum=col("v").sum().over(partition_by=["g"])).collect()
    duck.register("t", _t())
    assert_same(out, duck.sql("SELECT *, SUM(v) OVER (PARTITION BY g) AS gsum FROM t"))


def test_partition_mean_min_max_over(duck):
    from conftest import assert_same

    out = (
        bt.from_arrow(_t())
        .with_columns(
            gm=col("v").mean().over(partition_by=["g"]),
            gmin=col("v").min().over(partition_by=["g"]),
            gmax=col("v").max().over(partition_by=["g"]),
        )
        .collect()
    )
    duck.register("t", _t())
    assert_same(
        out,
        duck.sql(
            "SELECT *, AVG(v) OVER (PARTITION BY g) AS gm, "
            "MIN(v) OVER (PARTITION BY g) AS gmin, MAX(v) OVER (PARTITION BY g) AS gmax FROM t"
        ),
    )


def test_running_sum_over_order(duck):
    from conftest import assert_same

    out = (
        bt.from_arrow(_t())
        .with_columns(run=col("v").sum().over(partition_by=["g"], order_by=["v"]))
        .collect()
    )
    duck.register("t", _t())
    assert_same(
        out,
        duck.sql(
            "SELECT *, SUM(v) OVER (PARTITION BY g ORDER BY v "
            "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS run FROM t"
        ),
    )


def test_global_window_no_partition(duck):
    from conftest import assert_same

    out = bt.from_arrow(_t()).with_columns(total=col("v").sum().over()).collect()
    duck.register("t", _t())
    assert_same(out, duck.sql("SELECT *, SUM(v) OVER () AS total FROM t"))
