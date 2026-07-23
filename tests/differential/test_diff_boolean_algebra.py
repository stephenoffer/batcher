"""Boolean/CASE/COALESCE/NULL simplifications must match DuckDB after optimization.

Every rule in `kyber.rules.boolean_algebra` is result-preserving only if the engine's
three-valued logic and total-order comparisons agree with DuckDB. These run each
rewritten shape through the FULL optimizer (via `.collect()`) and assert equality vs
DuckDB — over null rows and over empty input, the two places a boolean rewrite is most
likely to diverge.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
import batcher.kyber.rules.extra.boolean_algebra
from _harness import assert_same
from batcher import col
from batcher.plan.expr_ir import Col, InList
from batcher.plan.expr_ir.constructors import coalesce


@pytest.fixture
def t(duck):
    tbl = pa.table({"a": [1, 2, 3, None], "x": [10, 20, 30, 40]})
    duck.register("t", tbl)
    return tbl


@pytest.fixture
def empty(duck):
    tbl = pa.table({"a": pa.array([], type=pa.int64()), "x": pa.array([], type=pa.int64())})
    duck.register("t", tbl)
    return tbl


# --- annihilators -----------------------------------------------------------


def test_and_false_is_empty(duck, t):
    out = bt.from_arrow(t).filter((col("a") > 1) & bt.lit(False)).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE (a > 1) AND FALSE"))


def test_or_true_is_everything(duck, t):
    out = bt.from_arrow(t).filter((col("a") > 1) | bt.lit(True)).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE (a > 1) OR TRUE"))


def test_and_false_empty_input(duck, empty):
    out = bt.from_arrow(empty).filter((col("a") > 1) & bt.lit(False)).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE (a > 1) AND FALSE"))


# --- idempotence ------------------------------------------------------------


def test_and_idempotent(duck, t):
    out = bt.from_arrow(t).filter((col("a") > 1) & (col("a") > 1)).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE (a > 1) AND (a > 1)"))


def test_or_idempotent(duck, t):
    out = bt.from_arrow(t).filter((col("a") > 1) | (col("a") > 1)).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE (a > 1) OR (a > 1)"))


# --- absorption -------------------------------------------------------------


def test_and_absorption(duck, t):
    x, y = col("a") > 1, col("x") < 25
    out = bt.from_arrow(t).filter(x & (x | y)).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE (a > 1) AND ((a > 1) OR (x < 25))"))


def test_or_absorption(duck, t):
    x, y = col("a") > 1, col("x") < 25
    out = bt.from_arrow(t).filter(x | (x & y)).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE (a > 1) OR ((a > 1) AND (x < 25))"))


# --- complementation --------------------------------------------------------


def test_complement_and_is_empty(duck, t):
    out = bt.from_arrow(t).filter(col("a").is_null() & ~col("a").is_null()).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE a IS NULL AND NOT (a IS NULL)"))


def test_complement_or_is_everything(duck, t):
    out = bt.from_arrow(t).filter(col("a").is_not_null() | ~col("a").is_not_null()).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE a IS NOT NULL OR NOT (a IS NOT NULL)"))


# --- NOT over comparison ----------------------------------------------------


def test_fold_not_lt(duck, t):
    # The null-a row is excluded by `NOT (a < 2)` either way (comparison is null).
    out = bt.from_arrow(t).filter(~(col("a") < 2)).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE NOT (a < 2)"))


def test_fold_not_eq(duck, t):
    out = bt.from_arrow(t).filter(~(col("a") == 2)).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE NOT (a = 2)"))


# --- boolean equality against a literal -------------------------------------


def test_bool_eq_true(duck, t):
    out = bt.from_arrow(t).filter((col("a") > 1) == True).collect()  # noqa: E712
    assert_same(out, duck.sql("SELECT * FROM t WHERE (a > 1) = TRUE"))


def test_bool_eq_false(duck, t):
    out = bt.from_arrow(t).filter((col("a") > 1) == False).collect()  # noqa: E712
    assert_same(out, duck.sql("SELECT * FROM t WHERE (a > 1) = FALSE"))


# --- IN-list cleanup --------------------------------------------------------


def test_single_in_list(duck, t):
    out = bt.from_arrow(t).filter(InList(Col("a"), (2,))).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE a IN (2)"))


def test_dedup_in_list(duck, t):
    out = bt.from_arrow(t).filter(InList(Col("a"), (2, 2, 3, 3))).collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE a IN (2, 2, 3, 3)"))


# --- COALESCE flattening ----------------------------------------------------


def test_coalesce_nested(duck, t):
    out = bt.from_arrow(t).select(r=coalesce(col("a"), coalesce(col("x"), bt.lit(0)))).collect()
    assert_same(out, duck.sql("SELECT COALESCE(a, COALESCE(x, 0)) AS r FROM t"))


def test_coalesce_truncate_after_literal(duck, t):
    out = bt.from_arrow(t).select(r=coalesce(col("a"), bt.lit(99), col("x"))).collect()
    assert_same(out, duck.sql("SELECT COALESCE(a, 99, x) AS r FROM t"))


def test_coalesce_empty_input(duck, empty):
    out = bt.from_arrow(empty).select(r=coalesce(col("a"), coalesce(col("x"), bt.lit(0)))).collect()
    assert_same(out, duck.sql("SELECT COALESCE(a, COALESCE(x, 0)) AS r FROM t"))
