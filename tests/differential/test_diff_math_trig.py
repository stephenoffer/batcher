"""Trig / hyperbolic math differential tests vs DuckDB."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col


@pytest.fixture
def t(duck):
    rng = np.random.default_rng(1)
    tbl = pa.table(
        {
            "u": rng.uniform(-1.0, 1.0, 200),  # domain of asin/acos
            "y": rng.uniform(-3.0, 3.0, 200),
        }
    )
    duck.register("t", tbl)
    return tbl


def test_inverse_trig_vs_duckdb(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .select(asn=col("u").asin(), acs=col("u").acos(), atn=col("y").atan())
        .collect()
    )
    assert_same(out, duck.sql("SELECT asin(u) asn, acos(u) acs, atan(y) atn FROM t"))


def test_hyperbolic_and_angle_vs_duckdb(duck, t):
    from conftest import assert_same

    out = (
        bt.from_arrow(t)
        .select(
            sh=col("y").sinh(),
            ch=col("y").cosh(),
            th=col("y").tanh(),
            dg=col("y").degrees(),
            rd=col("y").radians(),
            ct=col("y").cot(),
        )
        .collect()
    )
    expected = duck.sql(
        "SELECT sinh(y) sh, cosh(y) ch, tanh(y) th, degrees(y) dg, radians(y) rd, cot(y) ct FROM t"
    )
    assert_same(out, expected)
