"""Differential/structural tests for `.str.translate` and `.list.median`."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "s": pa.array(["abcdef", "cabbage", "naïve", "ABC", "", None]),
        }
    )
    duck.register("t", tbl)
    return tbl


def test_translate_vs_duckdb(duck, t):
    from conftest import assert_same

    out = bt.from_arrow(t).select(tr=col("s").str.translate("abc", "xyz")).collect()
    expected = duck.sql("SELECT translate(s, 'abc', 'xyz') tr FROM t")
    assert_same(out, expected)


def test_translate_deletion_vs_duckdb(duck, t):
    # `to` shorter than `from`: 'b' and 'c' (beyond 'x') are deleted.
    from conftest import assert_same

    out = bt.from_arrow(t).select(tr=col("s").str.translate("abc", "x")).collect()
    expected = duck.sql("SELECT translate(s, 'abc', 'x') tr FROM t")
    assert_same(out, expected)


def test_translate_unicode_vs_duckdb(duck, t):
    # Unicode + null rows: 'ï' → 'i'.
    from conftest import assert_same

    out = bt.from_arrow(t).select(tr=col("s").str.translate("ï", "i")).collect()
    expected = duck.sql("SELECT translate(s, 'ï', 'i') tr FROM t")
    assert_same(out, expected)


def test_list_median_structural():
    # Hand-computed: [[3,1,2],[4,4],[],None,[10]] → [2.0, 4.0, None, None, 10.0].
    ds = bt.from_arrow(
        pa.table({"a": pa.array([[3, 1, 2], [4, 4], [], None, [10]], type=pa.list_(pa.int64()))})
    )
    out = ds.select(m=col("a").list.median()).collect().to_pydict()
    assert out["m"] == [2.0, 4.0, None, None, 10.0]
