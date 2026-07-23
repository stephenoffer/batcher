"""Differential coverage for scalar numeric arithmetic edge cases vs DuckDB.

These pin the fixes for a family of `bc-expr` defects:

* integer `+`/`-`/`*` overflow **wraps** (Polars / Rust-release semantics), matching
  the Cranelift JIT's `iadd/isub/imul` bit-for-bit — the hard interpreter == JIT
  invariant (CLAUDE.md #6). (DuckDB errors here; matching that is deferred because it
  would require the compiled tier to also error, an ABI/SIMD change — the hard
  cross-tier gate wins over the softer oracle-match.)
* integer `%` / `/` by zero returns **NULL** (DuckDB), not a raised error;
* `gcd`/`lcm`/`bit_count`/`factorial` are **integer** functions — exact above 2^53
  and correctly typed, not routed through f64;
* right shift by a negative or ``>= 64`` amount is **0** (DuckDB), not arrow's
  masked ``wrapping_shr`` value.

The interpreter is the oracle; these run through the full Python → engine path, so
they also lock the public output dtype.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col, gcd, lcm

pytestmark = pytest.mark.differential


def test_integer_overflow_wraps_two_s_complement():
    # Scalar i64 `+`/`-`/`*` overflow WRAPS (two's complement), the bit-for-bit match
    # for the Cranelift JIT's `iadd/isub/imul`. This keeps the interpreter and the
    # compiled tier identical (CLAUDE.md #6). DuckDB instead raises Out-of-Range; that
    # divergence is a deliberate, documented trade — see the module docstring.
    mask = (1 << 64) - 1

    def wrap(v: int) -> int:
        v &= mask
        return v - (1 << 64) if v >= (1 << 63) else v

    t = pa.table({"i": pa.array([2**63 - 1], type=pa.int64())})
    got = bt.from_arrow(t).select(a=col("i") + 7, m=col("i") * 7).collect().to_pydict()
    assert got["a"] == [wrap((2**63 - 1) + 7)]
    assert got["m"] == [wrap((2**63 - 1) * 7)]


def test_integer_arithmetic_in_range_matches_duckdb(duck):
    t = pa.table({"i": pa.array([1, -2, 100, 0], type=pa.int64())})
    duck.register("t", t)
    out = bt.from_arrow(t).select(a=col("i") + 7, s=col("i") - 3, m=col("i") * 4).collect()
    assert_same(out, duck.sql("SELECT i + 7 AS a, i - 3 AS s, i * 4 AS m FROM t"))


def test_integer_mod_div_by_zero_is_null(duck):
    # DuckDB: integer `%0` is NULL; a divisor column with a zero nulls that row.
    t = pa.table(
        {
            "i": pa.array([7, -7, 5, 9], type=pa.int64()),
            "j": pa.array([0, 0, 2, 3], type=pa.int64()),
        }
    )
    duck.register("t", t)
    out = bt.from_arrow(t).select(m=col("i") % col("j")).collect()
    assert_same(out, duck.sql("SELECT i % j AS m FROM t"))
    # Modulo by a literal zero is likewise NULL, not a raised error.
    out0 = bt.from_arrow(t).select(m=col("i") % 0).collect()
    assert out0.column("m").to_pylist() == [None, None, None, None]


def test_gcd_bit_count_exact_and_integer_above_2_pow_53(duck):
    # 2^53 + 1 is not representable in f64: the old f64 route gave gcd→1, bit_count→1.
    t = pa.table(
        {
            "a": pa.array([2**53 + 1, 48, 0], type=pa.int64()),
            "b": pa.array([3, 36, 5], type=pa.int64()),
        }
    )
    duck.register("t", t)
    out = bt.from_arrow(t).select(g=gcd(col("a"), col("b")), bc=col("a").bit_count()).collect()
    assert_same(out, duck.sql("SELECT gcd(a, b) AS g, bit_count(a) AS bc FROM t"))
    # The public schema is integer, not double.
    assert pa.types.is_integer(out.schema.field("g").type)
    assert pa.types.is_integer(out.schema.field("bc").type)


def test_lcm_and_factorial_integer_typed(duck):
    t = pa.table(
        {"a": pa.array([4, 6, 0], type=pa.int64()), "b": pa.array([6, 8, 5], type=pa.int64())}
    )
    duck.register("t", t)
    out = bt.from_arrow(t).select(l=lcm(col("a"), col("b"))).collect()
    assert_same(out, duck.sql("SELECT lcm(a, b) AS l FROM t"))
    assert pa.types.is_integer(out.schema.field("l").type)

    # factorial stays integer and terminates (no hang) on values it can hold.
    tf = pa.table({"a": pa.array([0, 5, 20], type=pa.int64())})
    duck.register("tf", tf)
    fout = bt.from_arrow(tf).select(f=col("a").factorial()).collect()
    assert_same(fout, duck.sql("SELECT factorial(a::INTEGER) AS f FROM tf"))
    assert pa.types.is_integer(fout.schema.field("f").type)


def test_factorial_of_huge_value_terminates():
    # A previous f64 loop `(1..=n)` hung for a huge n; it must now error fast, not hang.
    t = pa.table({"a": pa.array([2**63 - 1], type=pa.int64())})
    with pytest.raises(Exception):  # noqa: B017 - overflow error
        bt.from_arrow(t).select(f=col("a").factorial()).collect()


def test_right_shift_out_of_range_is_zero(duck):
    # DuckDB: `i >> s` for s < 0 or s >= 64 is 0; arrow's wrapping_shr masked it instead
    # (`-7 >> -1` gave -1). In-range shifts are arithmetic (sign-extending).
    t = pa.table(
        {
            "i": pa.array([-7, -7, -7, -100, 42], type=pa.int64()),
            "s": pa.array([-1, 64, 1, -6, 2], type=pa.int64()),
        }
    )
    duck.register("t", t)
    out = bt.from_arrow(t).select(r=col("i").bitwise_right_shift(col("s"))).collect()
    assert_same(out, duck.sql("SELECT i >> s AS r FROM t"))
