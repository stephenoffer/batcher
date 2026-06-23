"""Unary math function differential tests vs DuckDB."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col


@pytest.fixture
def t(duck):
    rng = np.random.default_rng(0)
    tbl = pa.table(
        {
            "x": rng.uniform(0.1, 100.0, 200),  # positive (for ln/log/sqrt)
            "y": rng.uniform(-50.0, 50.0, 200),  # signed (for sign/trunc/trig)
        }
    )
    duck.register("t", tbl)
    return tbl


def test_positive_domain_math_vs_duckdb(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .select(
            ln=col("x").ln(),
            l10=col("x").log10(),
            l2=col("x").log2(),
            sq=col("x").sqrt(),
            cb=col("x").cbrt(),
        )
        .collect()
    )
    expected = duck.sql("SELECT ln(x) ln, log10(x) l10, log2(x) l2, sqrt(x) sq, cbrt(x) cb FROM t")
    assert_same(out, expected)


def test_signed_domain_math_vs_duckdb(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .select(
            sg=col("y").sign(),
            tr=col("y").trunc(),
            si=col("y").sin(),
            co=col("y").cos(),
            ex=col("y").exp(),
        )
        .collect()
    )
    expected = duck.sql("SELECT sign(y) sg, trunc(y) tr, sin(y) si, cos(y) co, exp(y) ex FROM t")
    assert_same(out, expected)


def test_sign_of_zero():
    out = (
        bt.from_arrow(pa.table({"v": pa.array([0.0, -3.0, 7.0])}))
        .select(s=col("v").sign())
        .collect()
        .to_pydict()
    )
    assert out["s"] == [0.0, -1.0, 1.0]
