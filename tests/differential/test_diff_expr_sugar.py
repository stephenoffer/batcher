"""Differential coverage for Expr sugar (clip, is_nan) and the CASE null-WHEN fix.

`clip` lowers to a null-preserving conditional, and a `when(...)` whose condition is
NULL must fall through to the ELSE branch (SQL/DuckDB semantics) — the engine
previously let a null mask pick the THEN branch.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, when

pytestmark = pytest.mark.differential

_NAN = float("nan")


def _floats():
    return pa.table({"x": pa.array([1.0, 5.0, 10.0, None], type=pa.float64())})


def test_alias_positional_select_matches_duckdb(duck):
    from conftest import assert_same

    t = pa.table({"a": [1, 2, 3], "x": [10, 20, 30], "y": [1, 2, 3]})
    duck.register("t2", t)
    # Positional column ref + aliased derived expression.
    out = bt.from_arrow(t).select(col("a"), (col("x") + col("y")).alias("s")).collect()
    assert_same(out, duck.sql("SELECT a, x + y AS s FROM t2"))


def test_alias_is_ir_transparent():
    # alias() is purely a projection-layer name: the wrapped expression's IR is
    # unchanged, so it never alters semantics.
    e = (col("x") + col("y")).alias("s")
    assert e.to_ir() == (col("x") + col("y")).to_ir()


def test_clip_matches_duckdb_case(duck):
    from conftest import assert_same

    out = bt.from_arrow(_floats()).select(c=col("x").clip(2.0, 8.0)).collect()
    duck.register("t", _floats())
    # CLIP as a null-preserving CASE (LEAST/GREATEST would drop nulls to a bound).
    assert_same(
        out,
        duck.sql("SELECT CASE WHEN x < 2.0 THEN 2.0 WHEN x > 8.0 THEN 8.0 ELSE x END AS c FROM t"),
    )


def test_clip_lower_only(duck):
    from conftest import assert_same

    out = bt.from_arrow(_floats()).select(c=col("x").clip(lower=3.0)).collect()
    duck.register("t", _floats())
    assert_same(out, duck.sql("SELECT CASE WHEN x < 3.0 THEN 3.0 ELSE x END AS c FROM t"))


def test_case_null_when_falls_through(duck):
    from conftest import assert_same

    out = (
        bt.from_arrow(_floats())
        .select(r=when(col("x") < 2.0).then(99.0).otherwise(col("x")))
        .collect()
    )
    duck.register("t", _floats())
    assert_same(out, duck.sql("SELECT CASE WHEN x < 2.0 THEN 99.0 ELSE x END AS r FROM t"))


def test_is_nan_matches_duckdb(duck):
    from conftest import assert_same

    data = pa.table({"x": pa.array([1.0, _NAN, 3.0], type=pa.float64())})
    out = bt.from_arrow(data).select(n=col("x").is_nan()).collect()
    duck.register("t", data)
    assert_same(out, duck.sql("SELECT isnan(x) AS n FROM t"))


def test_is_not_nan_matches_duckdb(duck):
    from conftest import assert_same

    data = pa.table({"x": pa.array([1.0, _NAN, 3.0, -2.5], type=pa.float64())})
    out = bt.from_arrow(data).select(n=col("x").is_not_nan()).collect()
    duck.register("t", data)
    assert_same(out, duck.sql("SELECT NOT isnan(x) AS n FROM t"))


def test_is_nan_and_not_nan_with_nulls_matches_duckdb(duck):
    from conftest import assert_same

    # All three float states + null. is_nan/is_not_nan are dedicated ops (not the
    # `x != x` trick), so NaN is flagged correctly whether or not the column has
    # nulls — null → null, NaN → true/false, normal → false/true.
    data = pa.table({"x": pa.array([1.0, _NAN, None, -2.5], type=pa.float64())})
    out = (
        bt.from_arrow(data).select(isnan=col("x").is_nan(), notnan=col("x").is_not_nan()).collect()
    )
    duck.register("t", data)
    assert_same(out, duck.sql("SELECT isnan(x) AS isnan, NOT isnan(x) AS notnan FROM t"))


def test_float_comparisons_on_nan_match_duckdb(duck):
    from conftest import assert_same

    # The engine's comparison operators use total ordering (NaN == NaN, and NaN
    # sorts greater than every non-NaN value), matching DuckDB. The Cranelift JIT
    # must agree with this, not fall back to IEEE (where NaN compares unordered).
    data = pa.table(
        {
            "x": pa.array([1.0, _NAN, 3.0, _NAN], type=pa.float64()),
            "y": pa.array([2.0, 2.0, _NAN, _NAN], type=pa.float64()),
        }
    )
    out = (
        bt.from_arrow(data)
        .select(
            eq=(col("x") == col("y")),
            ne=(col("x") != col("y")),
            lt=(col("x") < col("y")),
            le=(col("x") <= col("y")),
            gt=(col("x") > col("y")),
            ge=(col("x") >= col("y")),
        )
        .collect()
    )
    duck.register("t", data)
    assert_same(
        out,
        duck.sql(
            "SELECT x=y AS eq, x<>y AS ne, x<y AS lt, x<=y AS le, x>y AS gt, x>=y AS ge FROM t"
        ),
    )
