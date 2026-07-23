"""The `arith_extra` rewrites must match DuckDB after the full optimizer runs.

Every rule in `kyber.rules.extra.arith_extra` rewrites an expression that the data plane
would otherwise evaluate, so the only proof that it is result-preserving is the oracle:
run the rewritten shape end to end (`.collect()` optimizes) and compare with DuckDB on the
same input — over NULLs, negative values, NaN / ±inf / `-0.0` floats, and empty input.

The unsound shapes get a differential test too (`x * 0`, `x ^ x`, `-(-f)`): the rules must
leave them alone, and DuckDB says what the untouched answer is.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
import batcher.kyber.rules.extra.arith_extra  # registers the rules into DEFAULT_REGISTRY
from _harness import assert_same
from batcher import col, lit


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "x": [1, -5, 0, None],
            "y": [6, 3, -1, 7],
            # DuckDB refuses to left-shift a negative number, so the shift tests need a
            # non-negative column to compare against the oracle.
            "u": [1, 5, 0, None],
            "f": [1.5, -2.5, 0.0, None],
            "b": [True, False, True, None],
        }
    )
    duck.register("t", tbl)
    return tbl


@pytest.fixture
def edge(duck):
    """The float edges every "obvious" algebraic identity dies on."""
    tbl = pa.table(
        {
            "f": [float("nan"), float("inf"), float("-inf"), -0.0, 0.0, None],
            "x": [1, 2, 3, 4, 5, None],
        }
    )
    duck.register("t", tbl)
    return tbl


@pytest.fixture
def empty(duck):
    tbl = pa.table(
        {
            "x": pa.array([], type=pa.int64()),
            "y": pa.array([], type=pa.int64()),
            "f": pa.array([], type=pa.float64()),
            "b": pa.array([], type=pa.bool_()),
        }
    )
    duck.register("t", tbl)
    return tbl


# --- abs_of_negation --------------------------------------------------------


def test_abs_of_negation_int(duck, t):
    out = bt.from_arrow(t).select(r=abs(-col("x"))).collect()
    assert_same(out, duck.sql("SELECT abs(-x) AS r FROM t"))


def test_abs_of_negation_float(duck, t):
    out = bt.from_arrow(t).select(r=abs(-col("f"))).collect()
    assert_same(out, duck.sql("SELECT abs(-f) AS r FROM t"))


def test_abs_of_negation_float_edges(duck, edge):
    out = bt.from_arrow(edge).select(r=abs(-col("f"))).collect()
    assert_same(out, duck.sql("SELECT abs(-f) AS r FROM t"))


def test_abs_of_negation_empty(duck, empty):
    out = bt.from_arrow(empty).select(r=abs(-col("x"))).collect()
    assert_same(out, duck.sql("SELECT abs(-x) AS r FROM t"))


# --- math-function collapsing -----------------------------------------------


def test_double_abs(duck, t):
    out = bt.from_arrow(t).select(r=abs(abs(col("f")))).collect()
    assert_same(out, duck.sql("SELECT abs(abs(f)) AS r FROM t"))


def test_double_abs_float_edges(duck, edge):
    out = bt.from_arrow(edge).select(r=abs(abs(col("f")))).collect()
    assert_same(out, duck.sql("SELECT abs(abs(f)) AS r FROM t"))


def test_double_sign(duck, edge):
    out = bt.from_arrow(edge).select(r=col("f").sign().sign()).collect()
    assert_same(out, duck.sql("SELECT sign(sign(f)) AS r FROM t"))


def test_nested_rounding(duck, t):
    out = bt.from_arrow(t).select(r=col("f").floor().ceil()).collect()
    assert_same(out, duck.sql("SELECT ceil(floor(f)) AS r FROM t"))


def test_nested_rounding_float_edges(duck, edge):
    out = bt.from_arrow(edge).select(r=col("f").trunc().round()).collect()
    assert_same(out, duck.sql("SELECT round(trunc(f)) AS r FROM t"))


def test_rounding_of_int_is_cast(duck, t):
    out = bt.from_arrow(t).select(r=col("x").floor(), q=col("x").round()).collect()
    assert_same(out, duck.sql("SELECT floor(x) AS r, round(x) AS q FROM t"))


def test_rounding_of_int_empty(duck, empty):
    out = bt.from_arrow(empty).select(r=col("x").ceil()).collect()
    assert_same(out, duck.sql("SELECT ceil(x) AS r FROM t"))


# --- literal folding --------------------------------------------------------


def test_fold_math_of_int_literal(duck, t):
    out = (
        bt.from_arrow(t)
        .select(a=abs(lit(-5)), b=lit(-7).sign(), c=lit(5).floor(), d=lit(4).sqrt())
        .collect()
    )
    assert_same(
        out,
        duck.sql("SELECT abs(-5) AS a, sign(-7) AS b, floor(5) AS c, sqrt(4) AS d FROM t"),
    )


def test_fold_math_of_large_int_literal(duck, t):
    # Beyond 2^53 the int→double promotion rounds; the fold must round identically.
    big = 2**60 + 1
    out = bt.from_arrow(t).select(a=lit(big).floor(), b=lit(big).sqrt()).collect()
    assert_same(out, duck.sql(f"SELECT floor({big}) AS a, sqrt({big}) AS b FROM t"))


def test_abs_of_int64_min_literal_saturates(duck, t):
    # i64::MIN has no positive i64 image, so `abs` saturates to i64::MAX. This is a
    # Batcher-only check — DuckDB raises "Overflow on abs(INT64_MIN)" and cannot serve as
    # the oracle. The contract that matters is that an absolute value is never *negative*
    # (it once returned i64::MIN, and panicked in a debug build); the interpreter and the
    # Cranelift JIT saturate identically, so the two tiers stay bit-for-bit equal.
    out = bt.from_arrow(t).select(r=abs(lit(-(2**63)))).collect()
    assert out.column("r").to_pylist() == [2**63 - 1] * t.num_rows


def test_fold_bitwise_literals(duck, t):
    out = (
        bt.from_arrow(t)
        .select(
            a=lit(6).bitwise_and(lit(3)),
            b=lit(6).bitwise_or(lit(3)),
            c=lit(6).bitwise_xor(lit(3)),
            d=lit(-2).bitwise_and(lit(-3)),
        )
        .collect()
    )
    assert_same(
        out,
        duck.sql("SELECT 6 & 3 AS a, 6 | 3 AS b, xor(6, 3) AS c, (-2) & (-3) AS d FROM t"),
    )


# --- bitwise identity elements ----------------------------------------------


def test_bit_or_zero(duck, t):
    out = bt.from_arrow(t).select(r=col("x").bitwise_or(0)).collect()
    assert_same(out, duck.sql("SELECT x | 0 AS r FROM t"))


def test_bit_xor_zero(duck, t):
    out = bt.from_arrow(t).select(r=col("x") ^ 0).collect()
    assert_same(out, duck.sql("SELECT xor(x, 0) AS r FROM t"))


def test_bit_and_minus_one(duck, t):
    out = bt.from_arrow(t).select(r=col("x").bitwise_and(-1)).collect()
    assert_same(out, duck.sql("SELECT x & (-1) AS r FROM t"))


def test_shift_by_zero(duck, t):
    out = bt.from_arrow(t).select(a=col("u") << 0, b=col("u") >> 0).collect()
    assert_same(out, duck.sql("SELECT u << 0 AS a, u >> 0 AS b FROM t"))


def test_bit_and_self_and_or_self(duck, t):
    out = (
        bt.from_arrow(t)
        .select(a=col("x").bitwise_and(col("x")), b=col("x").bitwise_or(col("x")))
        .collect()
    )
    assert_same(out, duck.sql("SELECT x & x AS a, x | x AS b FROM t"))


def test_bit_identities_keep_nulls(duck, t):
    # The whole NULL story in one query: every identity must leave the NULL row NULL.
    out = bt.from_arrow(t).filter(col("x").bitwise_or(0).is_null()).select("y").collect()
    assert_same(out, duck.sql("SELECT y FROM t WHERE (x | 0) IS NULL"))


def test_bit_identities_empty(duck, empty):
    out = bt.from_arrow(empty).select(r=col("x").bitwise_and(-1)).collect()
    assert_same(out, duck.sql("SELECT x & (-1) AS r FROM t"))


def test_bit_identity_on_a_float_is_still_an_int(duck, t):
    # `f | 0` casts to Int64 — the rule must NOT drop it (that would leave a Float64).
    out = bt.from_arrow(t).select(r=col("f").bitwise_or(0)).collect()
    assert out.schema.field("r").type == pa.int64()
    # DuckDB has no float bitwise operator, so this is a Batcher-only check: the values are
    # the engine's Float64→Int64 coercion of `f`, which dropping the `| 0` would not give.
    assert out.column("r").to_pylist() == [1, -2, 0, None]


# --- the shapes the rules must NOT touch ------------------------------------


def test_mul_zero_keeps_nulls(duck, t):
    out = bt.from_arrow(t).select(r=col("x") * 0).collect()
    assert_same(out, duck.sql("SELECT x * 0 AS r FROM t"))


def test_bit_and_zero_keeps_nulls(duck, t):
    out = bt.from_arrow(t).select(r=col("x").bitwise_and(0)).collect()
    assert_same(out, duck.sql("SELECT x & 0 AS r FROM t"))


def test_xor_self_keeps_nulls(duck, t):
    out = bt.from_arrow(t).select(r=col("x") ^ col("x")).collect()
    assert_same(out, duck.sql("SELECT xor(x, x) AS r FROM t"))


def test_sub_self_keeps_nulls(duck, t):
    out = bt.from_arrow(t).select(r=col("x") - col("x")).collect()
    assert_same(out, duck.sql("SELECT x - x AS r FROM t"))


def test_float_double_negation_is_not_folded_away(duck):
    # Unary minus desugars to `0 - x`, so `-(-(-0.0))` evaluates to `0.0 - (0.0 - -0.0)` =
    # `+0.0`. Folding it to `f` would hand back `-0.0` instead — a different sign bit. The
    # rule is refused, so the engine's own `+0.0` must survive. (`assert_same` canonicalizes
    # ±0.0, so the sign is asserted directly here.)
    tbl = pa.table({"f": [-0.0]})
    negated = -col("f")
    out = bt.from_arrow(tbl).select(r=-negated).collect()
    assert math.copysign(1.0, out.column("r").to_pylist()[0]) == 1.0


def test_nan_survives_mul_zero(duck):
    tbl = pa.table({"f": [float("nan")]})
    duck.register("t", tbl)
    out = bt.from_arrow(tbl).select(r=col("f") * 0).collect()
    assert_same(out, duck.sql("SELECT f * 0 AS r FROM t"))
