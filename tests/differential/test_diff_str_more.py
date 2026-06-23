"""Differential tests for additional `.str` functions vs DuckDB.

Covers `octet_length`, `bit_length`, and `hex` against DuckDB, plus `initcap`
against an expected literal (DuckDB has no `initcap`). Includes unicode, empty,
and null rows.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "s": pa.array(["abc", "çafé", "naïve", "A", "", None, "Hello WORLD"]),
        }
    )
    duck.register("t", tbl)
    return tbl


def test_octet_bit_hex_vs_duckdb(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .select(
            ob=col("s").str.octet_length(),
            bl=col("s").str.bit_length(),
            hx=col("s").str.hex(),
        )
        .collect()
    )
    # DuckDB 1.5.x: `strlen` is the UTF-8 byte count (octet_length only takes
    # BIT/BLOB); `bit_length` on VARCHAR is bytes*8; `hex` returns UPPERCASE hex.
    expected = duck.sql("SELECT strlen(s) ob, bit_length(s::VARCHAR) bl, hex(s) hx FROM t")
    assert_same(out, expected)


def test_initcap_expected():
    # DuckDB has no `initcap`; check against an explicit expectation. A word is a
    # maximal run of alphanumerics; the rest is lowercased.
    tbl = pa.table(
        {
            "s": pa.array(
                [
                    "hello world",
                    "foo-BAR baz",
                    "çafé NAÏVE",
                    "a1b2 c3",
                    "",
                    None,
                    "  multiple   spaces  ",
                ]
            ),
        }
    )
    out = bt.from_arrow(tbl).select(ic=col("s").str.initcap()).collect().to_pydict()
    assert out["ic"] == [
        "Hello World",
        "Foo-Bar Baz",
        "Çafé Naïve",
        "A1b2 C3",
        "",
        None,
        "  Multiple   Spaces  ",
    ]


def test_hex_uppercase_unicode():
    # DuckDB `hex('çafé')` == 'C3A76166C3A9' (uppercase, UTF-8 bytes).
    tbl = pa.table({"s": pa.array(["çafé", "abc", "", None])})
    out = bt.from_arrow(tbl).select(h=col("s").str.hex()).collect().to_pydict()
    assert out["h"] == ["C3A76166C3A9", "616263", "", None]
