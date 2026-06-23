"""String pad/repeat (`.str`) differential tests vs DuckDB."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col


@pytest.fixture
def t(duck):
    tbl = pa.table({"s": pa.array(["ab", "hello", "5", "", "x", None])})
    duck.register("t", tbl)
    return tbl


def test_repeat_vs_duckdb(duck, t):
    from conftest import assert_same

    out = bt.from_arrow(t).select(r3=col("s").str.repeat(3), r0=col("s").str.repeat(0)).collect()
    assert_same(out, duck.sql("SELECT repeat(s, 3) r3, repeat(s, 0) r0 FROM t"))


def test_lpad_rpad_vs_duckdb(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .select(
            lp=col("s").str.lpad(5, "xy"),
            rp=col("s").str.rpad(5, "xy"),
            lz=col("s").str.lpad(3, "0"),
            # width shorter than the string → truncation
            trunc=col("s").str.rpad(2, "."),
        )
        .collect()
    )
    expected = duck.sql(
        "SELECT lpad(s, 5, 'xy') lp, rpad(s, 5, 'xy') rp, lpad(s, 3, '0') lz, "
        "rpad(s, 2, '.') trunc FROM t"
    )
    assert_same(out, expected)
