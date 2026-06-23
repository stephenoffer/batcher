"""Regexp string functions vs DuckDB (patterns without backref-syntax divergence)."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col


@pytest.fixture
def t(duck):
    tbl = pa.table({"s": pa.array(["abc123", "hello world", "2021-06-19", "no digits", "", None])})
    duck.register("t", tbl)
    return tbl


def test_regexp_matches_vs_duckdb(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .select(
            d=col("s").str.regexp_matches(r"\d+"),
            w=col("s").str.regexp_matches(r"^\w+$"),
        )
        .collect()
    )
    expected = duck.sql(r"SELECT regexp_matches(s, '\d+') d, regexp_matches(s, '^\w+$') w FROM t")
    assert_same(out, expected)


def test_regexp_replace_vs_duckdb(duck, t):
    from conftest import assert_same

    # Replacement without backreferences (DuckDB \1 vs regex $1 differ).
    out = bt.from_arrow(t).select(r=col("s").str.regexp_replace(r"\d+", "#")).collect()
    assert_same(out, duck.sql(r"SELECT regexp_replace(s, '\d+', '#') r FROM t"))


def test_regexp_extract_vs_duckdb(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .select(
            e0=col("s").str.regexp_extract(r"\d+"),
            e1=col("s").str.regexp_extract(r"(\d+)-(\d+)", 1),
        )
        .collect()
    )
    expected = duck.sql(
        r"SELECT regexp_extract(s, '\d+') e0, regexp_extract(s, '(\d+)-(\d+)', 1) e1 FROM t"
    )
    assert_same(out, expected)
