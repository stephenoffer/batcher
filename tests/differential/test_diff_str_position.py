"""String position (`.str.position`) differential tests vs DuckDB strpos."""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col


@pytest.fixture
def t(duck):
    tbl = pa.table({"s": pa.array(["hello", "ababab", "café", "x", "", None])})
    duck.register("t", tbl)
    return tbl


def test_position_vs_duckdb(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .select(
            p_ll=col("s").str.position("ll"),
            p_a=col("s").str.position("a"),
            p_z=col("s").str.position("z"),
        )
        .collect()
    )
    expected = duck.sql(
        "SELECT strpos(s, 'll') p_ll, strpos(s, 'a') p_a, strpos(s, 'z') p_z FROM t"
    )
    assert_same(out, expected)
