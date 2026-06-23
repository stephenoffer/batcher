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
