"""The null-algebra and numeric-range rewrites must match DuckDB after optimization.

Every rule in `kyber.rules.nulls` and `kyber.rules.math_algebra` replaces an expression
the data plane would otherwise evaluate, so the oracle is the only proof that the
replacement is result-preserving. Each case runs end to end through `.collect()` (which
optimizes) and is compared against DuckDB evaluating the *original* spelling.

The fixtures are chosen to hit exactly the values the rules' correctness arguments turn
on: NULLs everywhere, NaN and both infinities, `-0.0`, the i64 boundaries, negative
dividends for the floored-division buckets, and an empty table.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
import batcher.kyber.rules.math_algebra
import batcher.kyber.rules.nulls
from _harness import assert_same
from batcher import col, lit

_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1


@pytest.fixture
def t(duck):
    """Mixed nulls across an integer, a float, a string, a timestamp and a list."""
    tbl = pa.table(
        {
            "i": pa.array([1, -5, 0, None, 12, -13], type=pa.int64()),
            "j": pa.array([3, 3, 3, 3, None, 3], type=pa.int64()),
            "f": pa.array([1.5, -2.5, 0.0, None, 3.0, -3.0], type=pa.float64()),
            "s": pa.array(["a", "BB", "", None, "ccc", " d "], type=pa.string()),
            "ts": pa.array(
                [
                    dt.datetime(2020, 1, 1),
                    dt.datetime(2021, 6, 2, 3, 4, 5),
                    dt.datetime(1970, 1, 1),
                    None,
                    dt.datetime(2024, 2, 29),
                    dt.datetime(1999, 12, 31, 23, 59, 59),
                ],
                type=pa.timestamp("us"),
            ),
        }
    )
    duck.register("t", tbl)
    return tbl


@pytest.fixture
def edge(duck):
    """The float and integer edges every numeric identity dies on."""
    tbl = pa.table(
        {
            "f": pa.array(
                [float("nan"), float("inf"), float("-inf"), -0.0, 0.0, None, 2.0],
                type=pa.float64(),
            ),
            "i": pa.array([_INT64_MIN, _INT64_MAX, -1, 0, 1, None, 2], type=pa.int64()),
        }
    )
    duck.register("t", tbl)
    return tbl


@pytest.fixture
def empty(duck):
    tbl = pa.table(
        {
            "i": pa.array([], type=pa.int64()),
            "j": pa.array([], type=pa.int64()),
            "f": pa.array([], type=pa.float64()),
            "s": pa.array([], type=pa.string()),
        }
    )
    duck.register("t", tbl)
    return tbl


# --- null strictness --------------------------------------------------------

_STRICTNESS_CASES = [
    ("math_is_null", lambda: col("f").sin().is_null(), "sin(f) IS NULL"),
    ("math_is_not_null", lambda: col("f").abs().is_not_null(), "abs(f) IS NOT NULL"),
    ("str_is_null", lambda: col("s").str.upper().is_null(), "upper(s) IS NULL"),
    ("str_is_not_null", lambda: col("s").str.len().is_not_null(), "length(s) IS NOT NULL"),
    ("date_is_null", lambda: col("ts").dt.year().is_null(), "year(ts) IS NULL"),
    (
        "date_trunc_is_not_null",
        lambda: col("ts").dt.truncate("month").is_not_null(),
        "date_trunc('month', ts) IS NOT NULL",
    ),
    ("nan_is_null", lambda: col("f").is_nan().is_null(), "isnan(f) IS NULL"),
    (
        "negation_is_null",
        lambda: (~(col("i") > lit(0))).is_null(),
        "(NOT (i > 0)) IS NULL",
    ),
    ("arith_is_null", lambda: (col("i") + col("j")).is_null(), "(i + j) IS NULL"),
    (
        "arith_is_not_null",
        lambda: (col("i") * col("j")).is_not_null(),
        "(i * j) IS NOT NULL",
    ),
    (
        "comparison_is_null",
        lambda: (col("i") < col("j")).is_null(),
        "(i < j) IS NULL",
    ),
    (
        "comparison_is_not_null",
        lambda: (col("i") >= col("j")).is_not_null(),
        "(i >= j) IS NOT NULL",
    ),
    (
        "chained_is_null",
        lambda: col("s").str.upper().str.trim().is_null(),
        "trim(upper(s)) IS NULL",
    ),
]


@pytest.mark.parametrize(
    ("expr", "sql"),
    [(e, s) for _, e, s in _STRICTNESS_CASES],
    ids=[n for n, _, _ in _STRICTNESS_CASES],
)
def test_null_strictness_matches_duckdb(duck, t, expr, sql):
    out = bt.from_arrow(t).select(r=expr()).collect()
    assert_same(out, duck.sql(f"SELECT {sql} AS r FROM t"))


# --- three-valued logic -----------------------------------------------------

_THREE_VALUED_CASES = [
    (
        "tautology",
        lambda: col("i").is_null() | col("i").is_not_null(),
        "i IS NULL OR i IS NOT NULL",
    ),
    (
        "contradiction",
        lambda: col("i").is_null() & col("i").is_not_null(),
        "i IS NULL AND i IS NOT NULL",
    ),
    ("null_of_null_check", lambda: col("i").is_null().is_null(), "(i IS NULL) IS NULL"),
    (
        "coalesce_truncated",
        lambda: bt.coalesce(col("i"), lit(0), col("j")),
        "coalesce(i, 0, j)",
    ),
    (
        "coalesce_not_null",
        lambda: bt.coalesce(col("i"), lit(0)).is_not_null(),
        "coalesce(i, 0) IS NOT NULL",
    ),
    (
        "greatest_is_null",
        lambda: bt.greatest(col("i"), col("j")).is_null(),
        "greatest(i, j) IS NULL",
    ),
    (
        "greatest_is_not_null",
        lambda: bt.greatest(col("i"), col("j")).is_not_null(),
        "greatest(i, j) IS NOT NULL",
    ),
    ("least_is_null", lambda: bt.least(col("i"), col("j")).is_null(), "least(i, j) IS NULL"),
    (
        "least_is_not_null",
        lambda: bt.least(col("i"), col("j")).is_not_null(),
        "least(i, j) IS NOT NULL",
    ),
]


@pytest.mark.parametrize(
    ("expr", "sql"),
    [(e, s) for _, e, s in _THREE_VALUED_CASES],
    ids=[n for n, _, _ in _THREE_VALUED_CASES],
)
def test_three_valued_logic_matches_duckdb(duck, t, expr, sql):
    out = bt.from_arrow(t).select(r=expr()).collect()
    assert_same(out, duck.sql(f"SELECT {sql} AS r FROM t"))


def test_case_null_test_matches_duckdb(duck, t):
    expr = bt.when(col("i") > lit(0)).then(col("i")).otherwise(col("j")).is_null()
    out = bt.from_arrow(t).select(r=expr).collect()
    assert_same(
        out,
        duck.sql("SELECT (CASE WHEN i > 0 THEN i ELSE j END) IS NULL AS r FROM t"),
    )


# --- abs / sign intervals ---------------------------------------------------

_ABS_OPS = [("<", "lt"), ("<=", "le"), (">", "gt"), (">=", "ge"), ("=", "eq"), ("<>", "ne")]


@pytest.mark.parametrize(("sym", "name"), _ABS_OPS, ids=[n for _, n in _ABS_OPS])
def test_abs_interval_matches_duckdb_on_floats(duck, edge, sym, name):
    expr = {
        "lt": col("f").abs() < lit(2),
        "le": col("f").abs() <= lit(2),
        "gt": col("f").abs() > lit(2),
        "ge": col("f").abs() >= lit(2),
        "eq": col("f").abs() == lit(2),
        "ne": col("f").abs() != lit(2),
    }[name]
    out = bt.from_arrow(edge).select(r=expr).collect()
    assert_same(out, duck.sql(f"SELECT abs(f) {sym} 2 AS r FROM t"))


@pytest.mark.parametrize(("sym", "name"), _ABS_OPS, ids=[n for _, n in _ABS_OPS])
def test_abs_interval_matches_duckdb_on_integers(duck, t, sym, name):
    expr = {
        "lt": col("i").abs() < lit(5),
        "le": col("i").abs() <= lit(5),
        "gt": col("i").abs() > lit(5),
        "ge": col("i").abs() >= lit(5),
        "eq": col("i").abs() == lit(5),
        "ne": col("i").abs() != lit(5),
    }[name]
    out = bt.from_arrow(t).select(r=expr).collect()
    assert_same(out, duck.sql(f"SELECT abs(i) {sym} 5 AS r FROM t"))


def test_sign_comparison_on_a_float_is_left_alone(duck, edge):
    # The rules decline a float argument; the engine must still agree with the oracle on
    # the untouched shape, which is what proves declining was the right call.
    out = bt.from_arrow(edge).select(r=col("f").sign() == lit(1)).collect()
    assert_same(out, duck.sql("SELECT sign(f) = 1 AS r FROM t"))


_SIGN_INT_CASES = [
    ("eq_one", col("i").sign() == lit(1), "sign(i) = 1"),
    ("eq_minus_one", col("i").sign() == lit(-1), "sign(i) = -1"),
    ("gt_zero", col("i").sign() > lit(0), "sign(i) > 0"),
    ("lt_zero", col("i").sign() < lit(0), "sign(i) < 0"),
    ("ge_one", col("i").sign() >= lit(1), "sign(i) >= 1"),
    ("le_minus_one", col("i").sign() <= lit(-1), "sign(i) <= -1"),
    ("eq_zero", col("i").sign() == lit(0), "sign(i) = 0"),
    ("ne_zero", col("i").sign() != lit(0), "sign(i) <> 0"),
    ("ge_zero", col("i").sign() >= lit(0), "sign(i) >= 0"),
    ("le_zero", col("i").sign() <= lit(0), "sign(i) <= 0"),
    ("gt_minus_one", col("i").sign() > lit(-1), "sign(i) > -1"),
    ("lt_one", col("i").sign() < lit(1), "sign(i) < 1"),
]


@pytest.mark.parametrize(
    ("expr", "sql"),
    [(e, s) for _, e, s in _SIGN_INT_CASES],
    ids=[n for n, _, _ in _SIGN_INT_CASES],
)
def test_sign_comparison_matches_duckdb_on_integer_edges(duck, edge, expr, sql):
    out = bt.from_arrow(edge).select(r=expr).collect()
    assert_same(out, duck.sql(f"SELECT {sql} AS r FROM t"))


# --- floor / ceil / bit_count / bucket intervals ----------------------------

_ROUNDING_SQL = {
    "lt": "< 1", "le": "<= 1", "gt": "> 1", "ge": ">= 1", "eq": "= 1", "ne": "<> 1",
}  # fmt: skip


@pytest.mark.parametrize("name", list(_ROUNDING_SQL))
def test_floor_interval_matches_duckdb(duck, edge, name):
    expr = {
        "lt": col("f").floor() < lit(1),
        "le": col("f").floor() <= lit(1),
        "gt": col("f").floor() > lit(1),
        "ge": col("f").floor() >= lit(1),
        "eq": col("f").floor() == lit(1),
        "ne": col("f").floor() != lit(1),
    }[name]
    out = bt.from_arrow(edge).select(r=expr).collect()
    assert_same(out, duck.sql(f"SELECT floor(f) {_ROUNDING_SQL[name]} AS r FROM t"))


@pytest.mark.parametrize("name", list(_ROUNDING_SQL))
def test_ceil_interval_matches_duckdb(duck, edge, name):
    expr = {
        "lt": col("f").ceil() < lit(1),
        "le": col("f").ceil() <= lit(1),
        "gt": col("f").ceil() > lit(1),
        "ge": col("f").ceil() >= lit(1),
        "eq": col("f").ceil() == lit(1),
        "ne": col("f").ceil() != lit(1),
    }[name]
    out = bt.from_arrow(edge).select(r=expr).collect()
    assert_same(out, duck.sql(f"SELECT ceil(f) {_ROUNDING_SQL[name]} AS r FROM t"))


def test_bit_count_zero_test_matches_duckdb(duck, edge):
    out = (
        bt.from_arrow(edge)
        .select(a=col("i").bit_count() == lit(0), b=col("i").bit_count() > lit(0))
        .collect()
    )
    assert_same(out, duck.sql("SELECT bit_count(i) = 0 AS a, bit_count(i) > 0 AS b FROM t"))


@pytest.mark.parametrize("name", list(_ROUNDING_SQL))
def test_floor_div_bucket_matches_duckdb(duck, t, name):
    bucket = col("i") // lit(3)
    expr = {
        "lt": bucket < lit(1),
        "le": bucket <= lit(1),
        "gt": bucket > lit(1),
        "ge": bucket >= lit(1),
        "eq": bucket == lit(1),
        "ne": bucket != lit(1),
    }[name]
    out = bt.from_arrow(t).select(r=expr).collect()
    # DuckDB's `//` is floored division, matching the engine's `floor_div`.
    assert_same(out, duck.sql(f"SELECT (i // 3) {_ROUNDING_SQL[name]} AS r FROM t"))


# --- the rewrites survive a filter and an empty input -----------------------


def test_interval_rewrite_inside_a_filter_matches_duckdb(duck, t):
    out = bt.from_arrow(t).filter(col("i").abs() < lit(6)).select(i=col("i")).collect()
    assert_same(out, duck.sql("SELECT i FROM t WHERE abs(i) < 6"))


def test_null_strictness_inside_a_filter_matches_duckdb(duck, t):
    out = bt.from_arrow(t).filter(col("s").str.upper().is_not_null()).select(s=col("s")).collect()
    assert_same(out, duck.sql("SELECT s FROM t WHERE upper(s) IS NOT NULL"))


def test_empty_input(duck, empty):
    out = (
        bt.from_arrow(empty)
        .select(
            a=col("f").abs() < lit(2),
            b=col("i").sign() == lit(1),
            c=col("f").floor() == lit(1),
            d=(col("i") // lit(3)) == lit(1),
            e=col("s").str.upper().is_null(),
        )
        .collect()
    )
    assert_same(
        out,
        duck.sql(
            "SELECT abs(f) < 2 AS a, sign(i) = 1 AS b, floor(f) = 1 AS c, "
            "(i // 3) = 1 AS d, upper(s) IS NULL AS e FROM t"
        ),
    )
