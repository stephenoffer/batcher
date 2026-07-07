"""Arithmetic algebraic rewrites must match DuckDB after the full optimizer runs.

Each rule in `kyber.rules.extra.arith_algebra` is result-preserving only if the
engine's wrapping i64 arithmetic and null propagation agree with DuckDB. These run
each rewritten shape end to end through the full optimizer (via `.collect()`) and
assert equality vs DuckDB — over null rows and empty input, and over float columns
that the rules must NOT touch. A separate Batcher-only check pins the wrapping
associativity the integer rules rely on, at the ``INT64_MAX`` boundary (where DuckDB
raises on overflow and so cannot serve as the oracle).
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
import batcher.kyber.rules.extra.arith_algebra  # registers the rules into DEFAULT_REGISTRY
from batcher import col
from conftest import assert_same

_INT64_MAX = 2**63 - 1


def _wrap_i64(k: int) -> int:
    return ((k + 2**63) % 2**64) - 2**63


@pytest.fixture
def t(duck):
    tbl = pa.table({"a": [1, 20, 3, None], "b": [4, 5, 60, 7], "f": [1.5, 2.5, 3.5, None]})
    duck.register("t", tbl)
    return tbl


@pytest.fixture
def empty(duck):
    tbl = pa.table(
        {
            "a": pa.array([], type=pa.int64()),
            "b": pa.array([], type=pa.int64()),
            "f": pa.array([], type=pa.float64()),
        }
    )
    duck.register("t", tbl)
    return tbl


# --- fold_add_sub_constants -------------------------------------------------


def test_add_add(duck, t):
    out = bt.from_arrow(t).select(r=(col("a") + 5) + 2).collect()
    assert_same(out, duck.sql("SELECT (a + 5) + 2 AS r FROM t"))


def test_add_sub_cancels(duck, t):
    out = bt.from_arrow(t).select(r=(col("a") + 5) - 5).collect()
    assert_same(out, duck.sql("SELECT (a + 5) - 5 AS r FROM t"))


def test_sub_sub(duck, t):
    out = bt.from_arrow(t).select(r=(col("a") - 5) - 2).collect()
    assert_same(out, duck.sql("SELECT (a - 5) - 2 AS r FROM t"))


def test_add_sub_in_filter(duck, t):
    out = bt.from_arrow(t).filter(((col("a") + 5) + 2) > 10).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE (a + 5) + 2 > 10"))


def test_add_sub_empty(duck, empty):
    out = bt.from_arrow(empty).select(r=(col("a") + 5) + 2).collect()
    assert_same(out, duck.sql("SELECT (a + 5) + 2 AS r FROM t"))


# --- fold_mul_constants -----------------------------------------------------


def test_mul_mul(duck, t):
    out = bt.from_arrow(t).select(r=(col("a") * 3) * 4).collect()
    assert_same(out, duck.sql("SELECT (a * 3) * 4 AS r FROM t"))


def test_mul_mul_cancels(duck, t):
    out = bt.from_arrow(t).select(r=(col("a") * -1) * -1).collect()
    assert_same(out, duck.sql("SELECT (a * -1) * -1 AS r FROM t"))


# --- fold_const_minus_sum ---------------------------------------------------


def test_const_minus_sum(duck, t):
    out = bt.from_arrow(t).select(r=10 - (col("a") + 3)).collect()
    assert_same(out, duck.sql("SELECT 10 - (a + 3) AS r FROM t"))


def test_const_minus_diff(duck, t):
    out = bt.from_arrow(t).select(r=10 - (col("a") - 3)).collect()
    assert_same(out, duck.sql("SELECT 10 - (a - 3) AS r FROM t"))


# --- fold_neg_sub -----------------------------------------------------------


def test_neg_sub(duck, t):
    out = bt.from_arrow(t).select(r=0 - (col("a") - col("b"))).collect()
    assert_same(out, duck.sql("SELECT 0 - (a - b) AS r FROM t"))


def test_double_negation(duck, t):
    neg_a = -col("a")
    out = bt.from_arrow(t).select(r=-neg_a).collect()
    assert_same(out, duck.sql("SELECT -(-a) AS r FROM t"))


# --- factor_common_mul ------------------------------------------------------


def test_factor_add(duck, t):
    out = bt.from_arrow(t).select(r=col("a") * 3 + col("a") * 4).collect()
    assert_same(out, duck.sql("SELECT a * 3 + a * 4 AS r FROM t"))


def test_factor_sub(duck, t):
    out = bt.from_arrow(t).select(r=col("a") * 5 - col("a") * 2).collect()
    assert_same(out, duck.sql("SELECT a * 5 - a * 2 AS r FROM t"))


# --- floats must NOT be rewritten (result still matches DuckDB) -------------


def test_float_add_sub_unchanged(duck, t):
    out = bt.from_arrow(t).select(r=(col("f") + 5) + 2).collect()
    assert_same(out, duck.sql("SELECT (f + 5) + 2 AS r FROM t"))


def test_float_mul_unchanged(duck, t):
    out = bt.from_arrow(t).select(r=(col("f") * 3) * 4).collect()
    assert_same(out, duck.sql("SELECT (f * 3) * 4 AS r FROM t"))


# --- wrapping associativity at the i64 boundary (Batcher-only oracle) -------


def test_add_wrapping_boundary():
    # (a + INT64_MAX) + 1 folds to a single wrapping add; the result must equal the
    # value two's-complement arithmetic defines. DuckDB overflow-errors here, so the
    # oracle is the ring computation itself.
    a = [1, 20, 3, None]
    tbl = pa.table({"a": pa.array(a, type=pa.int64())})
    out = bt.from_arrow(tbl).select(r=(col("a") + _INT64_MAX) + 1).collect()
    expected = [None if v is None else _wrap_i64(v + _INT64_MAX + 1) for v in a]
    assert out.column("r").to_pylist() == expected
