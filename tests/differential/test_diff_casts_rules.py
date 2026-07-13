"""The `casts` rewrites must match DuckDB after the full optimizer runs.

A wrong cast rule does not crash — it returns a value of the wrong type, hides an error, or
manufactures a NULL. So every rule is run end to end (`.collect()` optimizes) against the
DuckDB oracle, over NULLs, floats (NaN / ±inf / `-0.0`), an unparseable string (the case
`TRY_CAST` exists for), the exact/inexact int→double boundary, and empty input. Where a
rewrite could move a type, the result *schema* is asserted directly — `assert_same`
tolerates int↔float, so it cannot see a type change on its own.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
import batcher.kyber.rules.extra.casts  # registers the rules into DEFAULT_REGISTRY
from batcher import col, lit, when
from batcher.plan.expr_ir import Binary
from conftest import assert_same


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "x": [1, -5, 0, None],
            "f": [1.5, -2.5, 0.0, None],
            "s": ["a", "b", "c", None],
            "n": ["1", "22", "-3", None],  # parseable strings
            "b": [True, False, True, None],
        }
    )
    duck.register("t", tbl)
    return tbl


@pytest.fixture
def edge(duck):
    tbl = pa.table({"f": [float("nan"), float("inf"), float("-inf"), -0.0, 0.0, None]})
    duck.register("t", tbl)
    return tbl


@pytest.fixture
def empty(duck):
    tbl = pa.table(
        {
            "x": pa.array([], type=pa.int64()),
            "f": pa.array([], type=pa.float64()),
            "s": pa.array([], type=pa.string()),
            "b": pa.array([], type=pa.bool_()),
        }
    )
    duck.register("t", tbl)
    return tbl


# --- drop_cast_to_inferred_type / canonicalize_cast_dtype_alias -------------


def test_identity_cast_over_arithmetic(duck, t):
    out = bt.from_arrow(t).select(r=(col("x") + col("x")).cast("int64")).collect()
    assert out.schema.field("r").type == pa.int64()
    assert_same(out, duck.sql("SELECT CAST(x + x AS BIGINT) AS r FROM t"))


def test_identity_cast_alias(duck, t):
    # 'double' canonicalizes to 'float64', which then *is* `f`'s type — both rules fire.
    out = bt.from_arrow(t).select(r=col("f").cast("double")).collect()
    assert out.schema.field("r").type == pa.float64()
    assert_same(out, duck.sql("SELECT CAST(f AS DOUBLE) AS r FROM t"))


def test_identity_cast_float_edges(duck, edge):
    out = bt.from_arrow(edge).select(r=col("f").cast("float64")).collect()
    assert_same(out, duck.sql("SELECT CAST(f AS DOUBLE) AS r FROM t"))


def test_identity_cast_empty(duck, empty):
    out = bt.from_arrow(empty).select(r=col("x").cast("long")).collect()
    assert_same(out, duck.sql("SELECT CAST(x AS BIGINT) AS r FROM t"))


def test_a_real_narrowing_cast_still_narrows(duck, t):
    # The rules must not touch a cast that does convert: int64 → int32 stays.
    out = bt.from_arrow(t).select(r=col("x").cast("int32")).collect()
    assert out.schema.field("r").type == pa.int32()
    assert_same(out, duck.sql("SELECT CAST(x AS INTEGER) AS r FROM t"))


# --- fold_cast_of_literal ---------------------------------------------------


def test_fold_cast_of_literal(duck, t):
    out = (
        bt.from_arrow(t)
        .select(
            a=lit(-5).cast("string"),
            b=lit(True).cast("string"),
            c=lit(False).cast("string"),
            d=lit(True).cast("int64"),
            e=lit(5).cast("float64"),
        )
        .collect()
    )
    assert out.schema.field("a").type == pa.string()
    assert out.schema.field("d").type == pa.int64()
    assert out.schema.field("e").type == pa.float64()
    assert_same(
        out,
        duck.sql(
            "SELECT CAST(-5 AS VARCHAR) a, CAST(TRUE AS VARCHAR) b, CAST(FALSE AS VARCHAR) c, "
            "CAST(TRUE AS BIGINT) d, CAST(5 AS DOUBLE) e FROM t"
        ),
    )


def test_folded_literal_equals_the_engines_own_kernel(duck, t):
    # The fold is only sound if it computes what the *engine's* cast kernel computes on the
    # same value — so fold a literal and cast a column of that value in the same query.
    out = bt.from_arrow(t).select(folded=lit(True).cast("string"), kernel=col("b")).collect()
    per_row = bt.from_arrow(t).select(r=col("b").cast("string")).collect()
    assert per_row.column("r").to_pylist()[0] == out.column("folded").to_pylist()[0] == "true"


def test_unfoldable_literal_casts_still_match(duck, t):
    # Refused folds (float source, string parse, narrow target) — the engine does them.
    out = (
        bt.from_arrow(t)
        .select(a=lit(2.5).cast("int64"), b=lit("12").cast("int64"), c=lit(5).cast("int32"))
        .collect()
    )
    # `2.5::DOUBLE` — a bare `2.5` is a DECIMAL in DuckDB, and DECIMAL→BIGINT rounds
    # half-*away*-from-zero while DOUBLE→BIGINT rounds half-to-even (which is what a Batcher
    # float literal is, and what the engine's cast kernel implements).
    assert_same(
        out,
        duck.sql(
            "SELECT CAST(2.5::DOUBLE AS BIGINT) a, CAST('12' AS BIGINT) b, "
            "CAST(5 AS INTEGER) c FROM t"
        ),
    )


def test_large_int_literal_to_double(duck, t):
    # Beyond 2^53 the fold is refused — the engine's kernel rounds it.
    big = 2**53 + 1
    out = bt.from_arrow(t).select(r=lit(big).cast("float64")).collect()
    assert_same(out, duck.sql(f"SELECT CAST({big} AS DOUBLE) AS r FROM t"))


# --- push_cast_into_case_literal_branches -----------------------------------


def test_push_cast_into_case(duck, t):
    expr = when(col("x") > 0).then(1).otherwise(2).cast("float64")
    out = bt.from_arrow(t).select(r=expr).collect()
    assert out.schema.field("r").type == pa.float64()
    assert_same(out, duck.sql("SELECT CAST(CASE WHEN x > 0 THEN 1 ELSE 2 END AS DOUBLE) r FROM t"))


def test_push_cast_into_case_to_string(duck, t):
    expr = when(col("x") > 0).then(1).otherwise(2).cast("string")
    out = bt.from_arrow(t).select(r=expr).collect()
    assert out.schema.field("r").type == pa.string()
    assert_same(out, duck.sql("SELECT CAST(CASE WHEN x > 0 THEN 1 ELSE 2 END AS VARCHAR) r FROM t"))


def test_case_with_a_column_arm_is_untouched(duck, t):
    expr = when(col("x") > 0).then(col("x")).otherwise(2).cast("float64")
    out = bt.from_arrow(t).select(r=expr).collect()
    assert out.schema.field("r").type == pa.float64()
    assert_same(out, duck.sql("SELECT CAST(CASE WHEN x > 0 THEN x ELSE 2 END AS DOUBLE) r FROM t"))


def test_case_type_join_is_preserved(duck, t):
    # `CASE WHEN … THEN 1 ELSE 2.5 END` is a DOUBLE; a fold that narrowed an arm would make
    # it an INT. Cast it to BIGINT and check the result is still what DuckDB computes.
    expr = when(col("x") > 0).then(1).otherwise(2.5).cast("int64")
    out = bt.from_arrow(t).select(r=expr).collect()
    assert out.schema.field("r").type == pa.int64()
    assert_same(
        out,
        duck.sql("SELECT CAST(CASE WHEN x > 0 THEN 1 ELSE 2.5::DOUBLE END AS BIGINT) r FROM t"),
    )


# --- try_cast_to_strict_when_infallible -------------------------------------


def test_try_cast_int_to_double(duck, t):
    out = bt.from_arrow(t).select(r=col("x").try_cast("float64")).collect()
    assert out.schema.field("r").type == pa.float64()
    assert_same(out, duck.sql("SELECT TRY_CAST(x AS DOUBLE) AS r FROM t"))


def test_try_cast_numeric_to_string(duck, t):
    ds = bt.from_arrow(t)
    out = ds.select(a=col("f").try_cast("string"), b=col("b").try_cast("int64")).collect()
    assert_same(out, duck.sql("SELECT TRY_CAST(f AS VARCHAR) a, TRY_CAST(b AS BIGINT) b FROM t"))


def test_try_cast_of_an_unparseable_string_still_nulls(duck, t):
    # The case TRY_CAST exists for: 'a' cannot parse as an integer, so the row is NULL — the
    # rule must NOT rewrite this to a strict cast (which would abort the query).
    out = bt.from_arrow(t).select(r=col("s").try_cast("int64")).collect()
    assert out.column("r").to_pylist() == [None, None, None, None]
    assert_same(out, duck.sql("SELECT TRY_CAST(s AS BIGINT) AS r FROM t"))


def test_try_cast_of_a_parseable_string(duck, t):
    out = bt.from_arrow(t).select(r=col("n").try_cast("int64")).collect()
    assert_same(out, duck.sql("SELECT TRY_CAST(n AS BIGINT) AS r FROM t"))


def test_try_cast_float_to_int_is_kept(duck, edge):
    # ±inf/NaN cannot become an integer: TRY_CAST must keep NULLing them.
    out = bt.from_arrow(edge).select(r=col("f").try_cast("int64")).collect()
    assert_same(out, duck.sql("SELECT TRY_CAST(f AS BIGINT) AS r FROM t"))


# --- drop_infallible_cast_in_null_check -------------------------------------


def test_null_check_over_an_infallible_cast(duck, t):
    out = bt.from_arrow(t).filter(col("x").cast("float64").is_null()).select("s").collect()
    assert_same(out, duck.sql("SELECT s FROM t WHERE CAST(x AS DOUBLE) IS NULL"))


def test_not_null_check_over_a_string_cast(duck, t):
    out = bt.from_arrow(t).select(r=col("x").cast("string").is_not_null()).collect()
    assert_same(out, duck.sql("SELECT CAST(x AS VARCHAR) IS NOT NULL AS r FROM t"))


def test_null_check_over_a_try_cast_that_makes_nulls(duck, t):
    # `is_null(try_cast(s, int64))` is TRUE on every non-null row too — the cast manufactures
    # the null. Dropping it (which the rule must not do) would flag only the one real NULL.
    out = bt.from_arrow(t).select(r=col("s").try_cast("int64").is_null()).collect()
    assert out.column("r").to_pylist() == [True, True, True, True]
    assert_same(out, duck.sql("SELECT TRY_CAST(s AS BIGINT) IS NULL AS r FROM t"))


def test_null_check_empty(duck, empty):
    out = bt.from_arrow(empty).select(r=col("x").cast("float64").is_null()).collect()
    assert_same(out, duck.sql("SELECT CAST(x AS DOUBLE) IS NULL AS r FROM t"))


# --- drop_string_cast_in_concat ---------------------------------------------


def test_concat_of_a_cast_int(duck, t):
    out = bt.from_arrow(t).select(r=Binary("concat", col("x").cast("string"), lit("!"))).collect()
    assert out.schema.field("r").type == pa.string()
    assert_same(out, duck.sql("SELECT CAST(x AS VARCHAR) || '!' AS r FROM t"))


def test_concat_of_two_cast_operands(duck, t):
    expr = Binary("concat", col("f").cast("string"), col("b").cast("string"))
    out = bt.from_arrow(t).select(r=expr).collect()
    assert_same(out, duck.sql("SELECT CAST(f AS VARCHAR) || CAST(b AS VARCHAR) AS r FROM t"))


def test_concat_via_sql(duck, t):
    ds = bt.from_arrow(t)
    out = bt.sql("SELECT CAST(x AS VARCHAR) || '!' AS r FROM tt", tt=ds).collect()
    assert_same(out, duck.sql("SELECT CAST(x AS VARCHAR) || '!' AS r FROM t"))


# --- drop_numeric_cast_in_float_predicate -----------------------------------


def test_is_nan_over_a_cast(duck, edge):
    out = bt.from_arrow(edge).select(r=col("f").cast("float64").is_nan()).collect()
    assert_same(out, duck.sql("SELECT isnan(CAST(f AS DOUBLE)) AS r FROM t"))


def test_is_inf_over_a_cast(duck, edge):
    out = bt.from_arrow(edge).select(r=col("f").cast("float64").is_infinite()).collect()
    assert_same(out, duck.sql("SELECT isinf(CAST(f AS DOUBLE)) AS r FROM t"))


def test_is_nan_over_a_cast_int(duck, t):
    out = bt.from_arrow(t).select(r=col("x").cast("float64").is_nan()).collect()
    assert_same(out, duck.sql("SELECT isnan(CAST(x AS DOUBLE)) AS r FROM t"))
