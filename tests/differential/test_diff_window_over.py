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


def _ordered():
    # A unique order key makes the running-window result deterministic vs DuckDB.
    return pa.table(
        {
            "g": pa.array(["a", "a", "b", "a", "b"]),
            "ord": pa.array([1, 2, 3, 4, 5], type=pa.int64()),
            "v": pa.array([10, 20, 30, 40, 50], type=pa.int64()),
        }
    )


def test_cum_sum_matches_duckdb(duck):
    from conftest import assert_same

    out = bt.from_arrow(_ordered()).with_columns(
        cs=col("v").cum_sum(order_by=["ord"]),
        cmx=col("v").cum_max(order_by=["ord"]),
        cmn=col("v").cum_min(order_by=["ord"]),
        cc=col("v").cum_count(order_by=["ord"]),
    )
    duck.register("t", _ordered())
    assert_same(
        out.collect(),
        duck.sql(
            "SELECT *, "
            "SUM(v) OVER w cs, MAX(v) OVER w cmx, MIN(v) OVER w cmn, COUNT(v) OVER w cc "
            "FROM t WINDOW w AS (ORDER BY ord ROWS UNBOUNDED PRECEDING)"
        ),
    )


def test_grouped_cum_sum_matches_duckdb(duck):
    from conftest import assert_same

    out = bt.from_arrow(_ordered()).with_columns(
        cs=col("v").cum_sum(partition_by=["g"], order_by=["ord"])
    )
    duck.register("t", _ordered())
    assert_same(
        out.collect(),
        duck.sql(
            "SELECT *, SUM(v) OVER (PARTITION BY g ORDER BY ord ROWS UNBOUNDED PRECEDING) cs FROM t"
        ),
    )


def test_shift_matches_duckdb_lag_lead(duck):
    from conftest import assert_same

    # The rows are already in `ord` order, so positional shift == LAG/LEAD over ord.
    out = bt.from_arrow(_ordered()).with_columns(
        s1=col("v").shift(1),
        sm1=col("v").shift(-1),
    )
    duck.register("t", _ordered())
    assert_same(
        out.collect(),
        duck.sql(
            "SELECT *, LAG(v, 1) OVER (ORDER BY ord) s1, LEAD(v, 1) OVER (ORDER BY ord) sm1 FROM t"
        ),
    )
