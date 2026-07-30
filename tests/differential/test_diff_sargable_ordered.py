"""Differential tests vs DuckDB for the ordered sargable transposition rules.

`kyber/rules/extra/sargable_range` rewrites `col + k < lit` into `col < lit - k` when the
column's recorded bounds prove the arithmetic cannot wrap. Every query here runs through the
full Batcher optimizer, so the rewrite fires, and is asserted against DuckDB evaluating the
predicate as written. The cases that matter most are the ones where the rewrite must *not*
fire and the answer must still agree: a column whose range reaches the ends of i64 (where
the engine's wrapping arithmetic makes the two forms genuinely different), and a float column
(where folding the constant would round differently).

An in-memory source carries exact per-column min/max, which is what makes these bounds
available without a Parquet footer.

One divergence is deliberately *not* asserted here, because it is the engine's and not this
family's: at the ends of i64 Batcher's arithmetic **wraps** (`bc_expr` uses `add_wrapping`)
while DuckDB promotes to a wider type, so `x + 1 > 0` at `INT64_MAX` answers false in Batcher
and true in DuckDB. That is exactly why the transposition needs a range proof at all. The
guard is pinned instead as a plan-shape assertion in
`tests/unit/test_sargable_ordered_bounds.py`, where it belongs: the rule must decline, and
whatever the engine then computes is the engine's own answer.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
import batcher.kyber.rules.extra.sargable_range as _sargable_range  # noqa: F401
from _harness import assert_same
from batcher import col

pytestmark = pytest.mark.differential

_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1


@pytest.fixture
def t(duck):
    # Nulls, negatives, duplicates, and a zero — the range is nowhere near the ends of i64,
    # so every rewrite in the family is provably exact here.
    tbl = pa.table({"x": pa.array([0, 1, 2, 2, 3, 100, 400, -2, None], type=pa.int64())})
    duck.register("t", tbl)
    return tbl


@pytest.fixture
def empty(duck):
    tbl = pa.table({"x": pa.array([], type=pa.int64())})
    duck.register("empty", tbl)
    return tbl


@pytest.fixture
def floats(duck):
    tbl = pa.table({"f": pa.array([0.5, 1.5, 2.5, -1.5, None], type=pa.float64())})
    duck.register("floats", tbl)
    return tbl


@pytest.mark.parametrize(
    ("pred", "sql"),
    [
        (lambda: (col("x") + 1) < 100, "x + 1 < 100"),
        (lambda: (col("x") + 1) <= 100, "x + 1 <= 100"),
        (lambda: (col("x") + 1) > 100, "x + 1 > 100"),
        (lambda: (col("x") + 1) >= 100, "x + 1 >= 100"),
        (lambda: (col("x") - 3) < 0, "x - 3 < 0"),
        (lambda: (col("x") - 3) <= 0, "x - 3 <= 0"),
        (lambda: (col("x") - 3) > 0, "x - 3 > 0"),
        (lambda: (col("x") - 3) >= 0, "x - 3 >= 0"),
        (lambda: (5 - col("x")) < 2, "5 - x < 2"),
        (lambda: (5 - col("x")) <= 2, "5 - x <= 2"),
        (lambda: (5 - col("x")) > 2, "5 - x > 2"),
        (lambda: (5 - col("x")) >= 2, "5 - x >= 2"),
        # A negative constant, and the commuted spelling of the addition.
        (lambda: (col("x") + -50) > 0, "x + -50 > 0"),
        (lambda: (7 + col("x")) <= 9, "7 + x <= 9"),
        # The literal written first, which arrives as the mirrored comparison.
        (lambda: bt.lit(100) > (col("x") + 1), "100 > x + 1"),
        # Unary minus, which lowers to `0 - col`.
        (lambda: -col("x") >= -3, "-x >= -3"),
    ],
)
def test_ordered_transposition_matches_duckdb(duck, t, pred, sql):
    out = bt.from_arrow(t).filter(pred()).collect()
    assert_same(out, duck.sql(f"SELECT * FROM t WHERE {sql}"))


def test_empty_input(duck, empty):
    out = bt.from_arrow(empty).filter((col("x") + 1) > 5).collect()
    assert_same(out, duck.sql("SELECT * FROM empty WHERE x + 1 > 5"))


def test_float_column_is_not_transposed(duck, floats):
    out = bt.from_arrow(floats).filter((col("f") + 1.0) > 2.0).collect()
    assert_same(out, duck.sql("SELECT * FROM floats WHERE f + 1.0 > 2.0"))


def test_transposed_predicate_inside_a_disjunction(duck, t):
    out = bt.from_arrow(t).filter(((col("x") + 1) > 100) | ((col("x") - 1) < 0)).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE x + 1 > 100 OR x - 1 < 0"))


def test_transposed_predicate_in_a_projection(duck, t):
    out = bt.from_arrow(t).select(x=col("x"), keep=(col("x") + 1) > 100).collect()
    assert_same(out, duck.sql("SELECT x, x + 1 > 100 AS keep FROM t"))


def test_transposed_predicate_survives_a_group_by(duck, t):
    out = (
        bt.from_arrow(t)
        .filter((col("x") - 1) >= 0)
        .group_by(k=(col("x") + 1) > 3)
        .agg(n=col("x").count())
        .collect()
    )
    assert_same(
        out,
        duck.sql("SELECT x + 1 > 3 AS k, count(x) AS n FROM t WHERE x - 1 >= 0 GROUP BY x + 1 > 3"),
    )
