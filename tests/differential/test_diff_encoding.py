"""base64 / hex encode-decode (`.str`) tests vs DuckDB, plus round-trips."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "s": pa.array(["Hello", "Çafé", "", "naïve 🦀"]),
        }
    )
    duck.register("t", tbl)
    return tbl


def test_base64_encode_vs_duckdb(duck, t):
    # DuckDB `to_base64(encode(s))` is standard base64 of the UTF-8 bytes.
    from conftest import assert_same

    out = bt.from_arrow(t).select(b=col("s").str.base64()).collect()
    expected = duck.sql("SELECT to_base64(encode(s)) b FROM t")
    assert_same(out, expected)


def test_base64_roundtrip():
    # base64 then from_base64 recovers the original (incl. unicode/empty/null).
    tbl = pa.table({"s": pa.array(["Hello", "Çafé", "", "naïve 🦀", None])})
    out = bt.from_arrow(tbl).select(r=col("s").str.base64().str.from_base64()).collect().to_pydict()
    assert out["r"] == ["Hello", "Çafé", "", "naïve 🦀", None]


def test_unhex_roundtrip_vs_duckdb(duck, t):
    # hex (existing) then unhex recovers the original; DuckDB agrees via
    # decode(unhex(hex(s))).
    from conftest import assert_same

    out = bt.from_arrow(t).select(r=col("s").str.hex().str.unhex()).collect()
    expected = duck.sql("SELECT decode(unhex(hex(s))) r FROM t")
    assert_same(out, expected)


def test_hex_unhex_roundtrip():
    tbl = pa.table({"s": pa.array(["abc", "résumé", "", "🦀", None])})
    out = bt.from_arrow(tbl).select(r=col("s").str.hex().str.unhex()).collect().to_pydict()
    assert out["r"] == ["abc", "résumé", "", "🦀", None]


def test_from_base64_invalid_is_null():
    # Invalid base64 → null (DuckDB raises here, so this is structural).
    tbl = pa.table({"s": pa.array(["!!!", "abc", "====", None])})
    out = bt.from_arrow(tbl).select(r=col("s").str.from_base64()).collect().to_pydict()
    assert out["r"] == [None, None, None, None]


def test_unhex_invalid_is_null():
    # Odd length, non-hex digits, and non-UTF-8 bytes all → null.
    tbl = pa.table({"s": pa.array(["xyz", "abc", "ff", "c3", None])})
    # "ff" and "c3" are valid hex but decode to non-UTF-8 single bytes → null.
    out = bt.from_arrow(tbl).select(r=col("s").str.unhex()).collect().to_pydict()
    assert out["r"] == [None, None, None, None, None]
