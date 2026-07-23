"""Differential tests vs DuckDB for the sargable-predicate normalization rules.

Every query runs through the full Batcher optimizer (so the `sarg_*` rewrites fire) and
its result is asserted equal to DuckDB evaluating the *un*-rewritten predicate. Covers the
selective-filter case (the rewritten predicate keeps exactly the same rows), NULLs (an
arithmetic-wrapped null stays null and is dropped identically), empty input, negative
coefficients, and the overflow guard (a folded literal that would leave i64 must not be
rewritten, yet still match DuckDB).

Importing the module registers the rules into `DEFAULT_REGISTRY` for the full Optimizer.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
import batcher.kyber.rules.extra.sargable as _sargable  # noqa: F401
from _harness import assert_same
from batcher import col

pytestmark = pytest.mark.differential

_INT64_MAX = 2**63 - 1


@pytest.fixture
def t(duck):
    tbl = pa.table({"x": pa.array([1, 2, 3, 100, 400, -2, None], type=pa.int64())})
    duck.register("t", tbl)
    return tbl


@pytest.fixture
def big(duck):
    tbl = pa.table({"x": pa.array([0, 1_000_000_000, 2_000_000_000, None], type=pa.int64())})
    duck.register("big", tbl)
    return tbl


def test_add_eq_selective_and_nulls(duck, t):
    out = bt.from_arrow(t).filter((col("x") + 100) == 500).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE x + 100 = 500"))


def test_add_ne_and_commuted(duck, t):
    out = bt.from_arrow(t).filter((100 + col("x")) != 500).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE 100 + x <> 500"))


def test_sub_eq(duck, t):
    out = bt.from_arrow(t).filter((col("x") - 3) == 97).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE x - 3 = 97"))


def test_rsub_eq(duck, t):
    out = bt.from_arrow(t).filter((5 - col("x")) == 3).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE 5 - x = 3"))


def test_unary_minus_eq(duck, t):
    out = bt.from_arrow(t).filter((-col("x")) == 2).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE -x = 2"))


def test_mul_eq_odd_exact_divide(duck, t):
    out = bt.from_arrow(t).filter((col("x") * 3) == 9).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE x * 3 = 9"))


def test_mul_negative_coefficient(duck, t):
    out = bt.from_arrow(t).filter((col("x") * -1) == -400).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE x * -1 = -400"))


def test_mul_non_divisible_left_alone(duck, t):
    # 7 is not a multiple of 3: the rule declines; the engine's own eval must still agree
    # with DuckDB (no row satisfies it here).
    out = bt.from_arrow(t).filter((col("x") * 3) == 7).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE x * 3 = 7"))


def test_mul_even_coefficient_left_alone(duck, t):
    out = bt.from_arrow(t).filter((col("x") * 2) == 200).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE x * 2 = 200"))


def test_xor_eq(duck, t):
    out = bt.from_arrow(t).filter(col("x").bitwise_xor(5) == 7).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE xor(x, 5) = 7"))


def test_large_magnitude_no_overflow(duck, big):
    # Folded literal 3e9 is well within i64: the rewrite fires and stays exact at scale.
    out = bt.from_arrow(big).filter((col("x") + 1_000_000_000) == 3_000_000_000).collect()
    assert_same(out, duck.sql("SELECT * FROM big WHERE x + 1000000000 = 3000000000"))


def test_overflow_guard_declines_but_correct(duck, t):
    # lit + 5 = INT64_MAX + 5 overflows i64 -> the rule must NOT fire. The un-rewritten
    # predicate still evaluates correctly on the (non-overflowing) column data.
    out = bt.from_arrow(t).filter((col("x") - 5) == _INT64_MAX).collect()
    assert_same(out, duck.sql(f"SELECT * FROM t WHERE x - 5 = {_INT64_MAX}"))


def test_empty_input(duck):
    tbl = pa.table({"x": pa.array([], type=pa.int64())})
    duck.register("e", tbl)
    out = bt.from_arrow(tbl).filter((col("x") + 100) == 500).collect()
    assert_same(out, duck.sql("SELECT * FROM e WHERE x + 100 = 500"))
