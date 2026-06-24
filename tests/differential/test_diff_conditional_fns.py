"""nullif / greatest / least differential tests vs DuckDB."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, greatest, least, nullif


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "a": pa.array([1, 5, 3, None, 7], type=pa.int64()),
            "b": pa.array([1, 2, 3, 4, None], type=pa.int64()),
            "c": pa.array([9, None, 1, 2, 8], type=pa.int64()),
        }
    )
    duck.register("t", tbl)
    return tbl


def test_nullif_vs_duckdb(duck, t):
    from conftest import assert_same

    out = bt.from_arrow(t).select(n=nullif(col("a"), col("b"))).collect()
    assert_same(out, duck.sql("SELECT nullif(a, b) n FROM t"))


def test_greatest_least_vs_duckdb(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .select(
            g=greatest(col("a"), col("b"), col("c")),
            l=least(col("a"), col("b"), col("c")),
        )
        .collect()
    )
    assert_same(out, duck.sql("SELECT greatest(a, b, c) g, least(a, b, c) l FROM t"))


def test_greatest_least_with_literal(duck, t):
    from conftest import assert_same

    out = bt.from_arrow(t).select(g=greatest(col("a"), 4), l=least(col("a"), 4)).collect()
    assert_same(out, duck.sql("SELECT greatest(a, 4) g, least(a, 4) l FROM t"))


def test_sum_horizontal_ignores_nulls():
    ds = bt.from_pydict(
        {
            "a": [1.0, 2.0, None, None],
            "b": [10.0, None, 30.0, None],
            "c": [100.0, 200.0, 300.0, None],
        }
    )
    out = ds.with_columns(s=bt.sum_horizontal(bt.col("a"), bt.col("b"), bt.col("c"))).collect()
    assert out.column("s").to_pylist() == [111.0, 202.0, 330.0, 0.0]


def test_mean_horizontal_ignores_nulls():
    ds = bt.from_pydict(
        {
            "a": [1.0, 2.0, None, None],
            "b": [10.0, None, 30.0, None],
            "c": [100.0, 200.0, 300.0, None],
        }
    )
    out = ds.with_columns(m=bt.mean_horizontal(bt.col("a"), bt.col("b"), bt.col("c"))).collect()
    got = out.column("m").to_pylist()
    assert got[:3] == [37.0, 101.0, 165.0]
    assert got[3] is None  # all-null row -> null (no division by zero)
