"""Differential + mergeability coverage for product / bit_and / bit_or / bit_xor.

Each is a 1-column-state associative aggregate, so it must match DuckDB *and* be
identical single-node vs multi-partition (the mergeable-algebra invariant).
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
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
    from conftest import assert_same

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
