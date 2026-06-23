"""Differential coverage for fluent window-ranking functions vs DuckDB.

``row_number()``/``rank()``/``dense_rank()`` are no-input window functions bound
with ``.over(partition_by=…, order_by=…)``; they lower to the relational ``Window``
operator and must match SQL ``ROW_NUMBER()``/``RANK()``/``DENSE_RANK() OVER (...)``.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, dense_rank, rank, row_number

pytestmark = pytest.mark.differential


def _t():
    # Ties on (g, v) exercise the gap/no-gap difference between rank and dense_rank.
    return pa.table(
        {
            "g": pa.array(["a", "a", "a", "b", "b", "b"]),
            "v": pa.array([10, 10, 20, 30, 30, 40], type=pa.int64()),
        }
    )


def test_row_number_partitioned(duck):
    from conftest import assert_same

    out = (
        bt.from_arrow(_t())
        .with_columns(rn=row_number().over(partition_by=["g"], order_by=["v"]))
        .collect()
    )
    duck.register("t", _t())
    assert_same(
        out,
        duck.sql("SELECT *, ROW_NUMBER() OVER (PARTITION BY g ORDER BY v) AS rn FROM t"),
    )


def test_rank_and_dense_rank_with_ties(duck):
    from conftest import assert_same

    out = (
        bt.from_arrow(_t())
        .with_columns(
            rk=rank().over(partition_by=["g"], order_by=["v"]),
            dr=dense_rank().over(partition_by=["g"], order_by=["v"]),
        )
        .collect()
    )
    duck.register("t", _t())
    assert_same(
        out,
        duck.sql(
            "SELECT *, RANK() OVER (PARTITION BY g ORDER BY v) AS rk, "
            "DENSE_RANK() OVER (PARTITION BY g ORDER BY v) AS dr FROM t"
        ),
    )


def test_rank_global_no_partition(duck):
    from conftest import assert_same

    out = bt.from_arrow(_t()).with_columns(rn=row_number().over(order_by=["v"])).collect()
    duck.register("t", _t())
    assert_same(out, duck.sql("SELECT *, ROW_NUMBER() OVER (ORDER BY v) AS rn FROM t"))


def test_rank_descending_order(duck):
    from conftest import assert_same

    out = (
        bt.from_arrow(_t())
        .with_columns(rk=rank().over(partition_by=["g"], order_by=[("v", True)]))
        .collect()
    )
    duck.register("t", _t())
    assert_same(
        out,
        duck.sql("SELECT *, RANK() OVER (PARTITION BY g ORDER BY v DESC) AS rk FROM t"),
    )


def test_dense_rank_combined_with_aggregate_window(duck):
    from conftest import assert_same

    out = (
        bt.from_arrow(_t())
        .with_columns(
            dr=dense_rank().over(partition_by=["g"], order_by=["v"]),
            gsum=col("v").sum().over(partition_by=["g"]),
        )
        .collect()
    )
    duck.register("t", _t())
    assert_same(
        out,
        duck.sql(
            "SELECT *, DENSE_RANK() OVER (PARTITION BY g ORDER BY v) AS dr, "
            "SUM(v) OVER (PARTITION BY g) AS gsum FROM t"
        ),
    )
