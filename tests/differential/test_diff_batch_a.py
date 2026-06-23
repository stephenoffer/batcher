"""str.right/ascii + binary math (pow/atan2/round-digits) vs DuckDB."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import atan2, col


@pytest.fixture
def s(duck):
    tbl = pa.table({"s": pa.array(["hello", "AB", "café", "x", "", None])})
    duck.register("s", tbl)
    return tbl


@pytest.fixture
def n(duck):
    rng = np.random.default_rng(4)
    tbl = pa.table(
        {
            "x": rng.uniform(0.1, 10.0, 150),
            "y": rng.uniform(-5.0, 5.0, 150),
            "p": rng.uniform(-3.14, 3.14, 150),
        }
    )
    duck.register("n", tbl)
    return tbl


def test_str_right_ascii_vs_duckdb(duck, s):
    from conftest import assert_same

    out = bt.from_arrow(s).select(r2=col("s").str.right(2), a=col("s").str.ascii()).collect()
    assert_same(out, duck.sql("SELECT right(s, 2) r2, ascii(s) a FROM s"))


def test_pow_round_atan2_vs_duckdb(duck, n):
    from conftest import assert_same

    out = (
        bt.from_arrow(n)
        .select(
            pw=col("x").pow(col("y")),
            po=col("x") ** 2,
            rd=col("x").round(2),
            atn=atan2(col("y"), col("p")),
        )
        .collect()
    )
    expected = duck.sql("SELECT pow(x, y) pw, pow(x, 2) po, round(x, 2) rd, atan2(y, p) atn FROM n")
    assert_same(out, expected)
