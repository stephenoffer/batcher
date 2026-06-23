"""str.split → list and list.get tests (split vs DuckDB; get structural)."""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col


def test_split_then_len_vs_duckdb(duck):
    from conftest import assert_same

    tbl = pa.table({"s": pa.array(["a,b,c", "x", "p,q", "", None])})
    duck.register("t", tbl)
    out = bt.from_arrow(tbl).select(n=col("s").str.split(",").list.len()).collect()
    # DuckDB: len(string_split(s, ',')). null input → null.
    assert_same(out, duck.sql("SELECT len(string_split(s, ',')) n FROM t"))


def test_split_first_element():
    tbl = pa.table({"s": pa.array(["a,b,c", "x", "p,q", None])})
    out = bt.from_arrow(tbl).select(first=col("s").str.split(",").list.get(0)).collect().to_pydict()
    assert out["first"] == ["a", "x", "p", None]


def test_list_get_structural():
    tbl = pa.table({"a": pa.array([[10, 20, 30], [40], [], None], type=pa.list_(pa.int64()))})
    out = (
        bt.from_arrow(tbl)
        .select(g0=col("a").list.get(0), g1=col("a").list.get(1))
        .collect()
        .to_pydict()
    )
    assert out["g0"] == [10, 40, None, None]  # empty/null → null
    assert out["g1"] == [20, None, None, None]  # out of range → null
