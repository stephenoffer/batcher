"""String transform (`.str`) differential tests vs DuckDB, incl. reverse."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "s": pa.array(["  Hello ", "WORLD", "Çafé", "  mixED  ", "", "naïve"]),
        }
    )
    duck.register("t", tbl)
    return tbl


def test_str_transforms_vs_duckdb(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .select(
            up=col("s").str.upper(),
            lo=col("s").str.lower(),
            tr=col("s").str.trim(),
            lt=col("s").str.lstrip(),
            rt=col("s").str.rstrip(),
            rev=col("s").str.reverse(),
            n=col("s").str.len(),
        )
        .collect()
    )
    expected = duck.sql(
        "SELECT upper(s) up, lower(s) lo, trim(s) tr, ltrim(s) lt, rtrim(s) rt, "
        "reverse(s) rev, length(s) n FROM t"
    )
    assert_same(out, expected)


def test_str_reverse_roundtrip():
    # reverse(reverse(x)) == x — a structural property independent of DuckDB.
    tbl = pa.table({"s": pa.array(["abc", "résumé", "", "a"])})
    out = bt.from_arrow(tbl).select(r=col("s").str.reverse().str.reverse()).collect().to_pydict()
    assert out["r"] == ["abc", "résumé", "", "a"]
