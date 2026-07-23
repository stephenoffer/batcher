"""Differential + mergeability coverage for product / bit_and / bit_or / bit_xor.

Each is a 1-column-state associative aggregate, so it must match DuckDB *and* be
identical single-node vs multi-partition (the mergeable-algebra invariant).
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col

pytestmark = pytest.mark.differential


def _data():
    return pa.table(
        {
            "g": ["a", "a", "a", "b", "b", "c"],
            "v": pa.array([2, 3, 4, 5, 7, None], type=pa.int64()),
        }
    )


def test_product_bitwise_match_duckdb(duck):
    duck.register("t", _data())
    out = (
        bt.from_arrow(_data())
        .group_by("g")
        .agg(
            p=col("v").product(),
            ba=col("v").bit_and(),
            bo=col("v").bit_or(),
            bx=col("v").bit_xor(),
        )
        .collect()
    )
    assert_same(
        out,
        duck.sql(
            "SELECT g, product(v) AS p, bit_and(v) AS ba, bit_or(v) AS bo, "
            "bit_xor(v) AS bx FROM t GROUP BY g"
        ),
    )


def test_product_bitwise_single_node_equals_distributed():
    ds = bt.from_arrow(_data()).group_by("g").agg(p=col("v").product(), bx=col("v").bit_xor())
    sd = ds.collect().to_pydict()
    single = {g: (p, x) for g, p, x in zip(sd["g"], sd["p"], sd["bx"], strict=True)}
    dist = ds.collect(distributed=True, num_workers=3).to_pydict()
    multi = {g: (p, x) for g, p, x in zip(dist["g"], dist["p"], dist["bx"], strict=True)}
    assert single == multi


def test_groupby_shortcut_reducers_match_duckdb(duck):
    """The new `GroupBy.product/skewness/kurtosis/mode` shortcuts equal DuckDB's aggregates."""
    tbl = pa.table(
        {
            "g": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "x": pa.array([1.0, 2.0, 3.0, 4.0, 10.0, 10.0, 20.0, 40.0], type=pa.float64()),
        }
    )
    duck.register("t", tbl)
    ds = bt.from_arrow(tbl)
    assert_same(
        ds.group_by("g").product().collect(), duck.sql("SELECT g, product(x) x FROM t GROUP BY g")
    )
    assert_same(
        ds.group_by("g").skewness().collect(), duck.sql("SELECT g, skewness(x) x FROM t GROUP BY g")
    )
    assert_same(
        ds.group_by("g").kurtosis().collect(), duck.sql("SELECT g, kurtosis(x) x FROM t GROUP BY g")
    )


def test_groupby_array_agg_and_mode(duck):
    tbl = pa.table({"g": ["a", "a", "a", "b"], "x": pa.array([5, 5, 7, 9], type=pa.int64())})
    duck.register("t", tbl)
    ds = bt.from_arrow(tbl)
    # array_agg collects into a list (order-independent multiset comparison via assert_same).
    assert_same(
        ds.group_by("g").array_agg().collect(),
        duck.sql("SELECT g, array_agg(x) x FROM t GROUP BY g"),
    )
    assert_same(
        ds.group_by("g").mode().collect(), duck.sql("SELECT g, mode(x) x FROM t GROUP BY g")
    )
