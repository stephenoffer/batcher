"""Differential coverage for higher-order list ops (DuckDB list_transform/list_filter).

`element()` is the per-element placeholder; the sub-expression is evaluated columnarly
over the list's flattened child, so empty and null lists pass through unchanged.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, element

pytestmark = pytest.mark.differential


def _data():
    return pa.table(
        {"a": pa.array([[1, 2, 3], [4, 5], [], None, [-1, 0, 7]], type=pa.list_(pa.int64()))}
    )


def test_transform_matches_duckdb(duck):
    from conftest import assert_same

    duck.register("t", _data())
    got = bt.from_arrow(_data()).select(r=col("a").list.transform(element() * 2 + 1)).collect()
    assert_same(got, duck.sql("SELECT list_transform(a, x -> x * 2 + 1) AS r FROM t"))


def test_filter_matches_duckdb(duck):
    from conftest import assert_same

    duck.register("t", _data())
    got = bt.from_arrow(_data()).select(r=col("a").list.filter(element() > 2)).collect()
    assert_same(got, duck.sql("SELECT list_filter(a, x -> x > 2) AS r FROM t"))


def test_transform_then_filter_chains():
    # transform doubles, filter keeps > 5 — element() rebinds in each stage.
    got = (
        bt.from_arrow(_data())
        .select(r=col("a").list.transform(element() * 2).list.filter(element() > 5))
        .collect()
        .to_pydict()["r"]
    )
    assert got == [[6], [8, 10], [], None, [14]]
