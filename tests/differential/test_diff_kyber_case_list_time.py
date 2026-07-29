"""The `conditional_algebra`, `collections_algebra` and `temporal_algebra` rewrites must
match DuckDB after the full optimizer runs.

Each case is written in its *original* spelling; Batcher optimizes it into the pushed,
merged, direct or interval form and the oracle evaluates it as written. The fixtures carry
a NULL in every column, an empty list, and instants on both sides of the Unix epoch — the
rows each family's correctness argument turns on.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
import batcher.kyber.rules.collections_algebra
import batcher.kyber.rules.conditional_algebra
import batcher.kyber.rules.temporal_algebra
from _harness import assert_same
from batcher import col, lit
from batcher.plan.expr_ir.func_nodes import (
    DateFunc,
    DateOffset,
    ListContains,
    ListFunc,
    ListPosition,
)


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "a": pa.array([1, 2, 3, None, 0], type=pa.int64()),
            "s": pa.array(["x", "y", None, "z", ""], type=pa.string()),
            "l": pa.array([[1, 2, 3], [2], [], None, [3, 2, 1]], type=pa.list_(pa.int64())),
            "ts": pa.array(
                [
                    dt.datetime(2020, 1, 1),
                    dt.datetime(1969, 12, 31, 23, 59, 59),
                    dt.datetime(1970, 1, 1),
                    None,
                    dt.datetime(2024, 2, 29, 12, 30),
                ],
                type=pa.timestamp("us"),
            ),
        }
    )
    duck.register("t", tbl)
    return tbl


@pytest.fixture
def empty(duck):
    tbl = pa.table(
        {
            "a": pa.array([], type=pa.int64()),
            "s": pa.array([], type=pa.string()),
            "l": pa.array([], type=pa.list_(pa.int64())),
            "ts": pa.array([], type=pa.timestamp("us")),
        }
    )
    duck.register("t", tbl)
    return tbl


_CASES = [
    (
        "push_upper",
        lambda: bt.when(col("a") > lit(1)).then(lit("ab")).otherwise(lit("cd")).str.upper(),
        "upper(CASE WHEN a > 1 THEN 'ab' ELSE 'cd' END)",
    ),
    (
        "push_abs",
        lambda: bt.when(col("a") > lit(1)).then(lit(-2)).otherwise(lit(3)).abs(),
        "abs(CASE WHEN a > 1 THEN -2 ELSE 3 END)",
    ),
    (
        "push_year",
        lambda: bt.when(col("a") > lit(1)).then(col("ts")).otherwise(col("ts")).dt.year(),
        "year(CASE WHEN a > 1 THEN ts ELSE ts END)",
    ),
    (
        "push_and",
        lambda: (
            bt.when(col("a") > lit(1)).then(lit(True)).otherwise(lit(False)) & (col("a") < lit(3))
        ),
        "(CASE WHEN a > 1 THEN true ELSE false END) AND (a < 3)",
    ),
    (
        "push_bit_xor",
        lambda: bt.when(col("a") > lit(1)).then(lit(1)).otherwise(lit(2)) ^ col("a"),
        "xor(CASE WHEN a > 1 THEN 1 ELSE 2 END, a)",
    ),
    (
        "merge_equal_branches",
        lambda: (
            bt.when(col("a") == lit(1))
            .then(lit("x"))
            .when(col("a") == lit(2))
            .then(lit("x"))
            .otherwise(lit("y"))
        ),
        "CASE WHEN a = 1 THEN 'x' WHEN a = 2 THEN 'x' ELSE 'y' END",
    ),
    (
        "non_adjacent_branches",
        lambda: (
            bt.when(col("a") == lit(1))
            .then(lit("x"))
            .when(col("a") == lit(2))
            .then(lit("z"))
            .when(col("a") == lit(3))
            .then(lit("x"))
            .otherwise(lit("y"))
        ),
        "CASE WHEN a = 1 THEN 'x' WHEN a = 2 THEN 'z' WHEN a = 3 THEN 'x' ELSE 'y' END",
    ),
    (
        "nested_settled_condition",
        lambda: (
            bt.when(col("a") > lit(1))
            .then(lit(1))
            .otherwise(bt.when(col("a") > lit(1)).then(lit(2)).otherwise(lit(3)))
        ),
        "CASE WHEN a > 1 THEN 1 ELSE (CASE WHEN a > 1 THEN 2 ELSE 3 END) END",
    ),
    (
        "contains_through_sort",
        lambda: ListContains(ListFunc("sort", col("l")), 2),
        "list_contains(list_sort(l), 2)",
    ),
    (
        "contains_through_reverse",
        lambda: ListContains(ListFunc("reverse", col("l")), 2),
        "list_contains(list_reverse(l), 2)",
    ),
    (
        "contains_through_unique",
        lambda: ListContains(ListFunc("unique", col("l")), 2),
        "list_contains(list_distinct(l), 2)",
    ),
    (
        "sort_of_reverse",
        lambda: ListFunc("sort", ListFunc("reverse", col("l"))),
        "list_sort(list_reverse(l))",
    ),
    (
        "len_of_transform",
        lambda: ListFunc("len", col("l").list.transform(bt.element() * lit(2))),
        "len(list_transform(l, x -> x * 2))",
    ),
    (
        "identity_transform",
        lambda: col("l").list.transform(bt.element()),
        "list_transform(l, x -> x)",
    ),
    (
        "epoch_ge",
        lambda: DateFunc("epoch", col("ts")) >= lit(0),
        "epoch(ts) >= 0",
    ),
    (
        "epoch_eq",
        lambda: DateFunc("epoch", col("ts")) == lit(0),
        "epoch(ts) = 0",
    ),
    (
        "epoch_lt",
        lambda: DateFunc("epoch", col("ts")) < lit(0),
        "epoch(ts) < 0",
    ),
    (
        "epoch_ne",
        lambda: DateFunc("epoch", col("ts")) != lit(0),
        "epoch(ts) <> 0",
    ),
    (
        "offset_days",
        lambda: DateOffset(col("ts"), 0, 1, 0) < lit(dt.datetime(2020, 1, 5)),
        "(ts + INTERVAL 1 DAY) < TIMESTAMP '2020-01-05 00:00:00'",
    ),
    (
        "offset_micros",
        lambda: DateOffset(col("ts"), 0, 0, 1_500_000) >= lit(dt.datetime(1970, 1, 1)),
        "(ts + INTERVAL 1500000 MICROSECOND) >= TIMESTAMP '1970-01-01 00:00:00'",
    ),
]


@pytest.mark.parametrize(
    ("expr", "sql"), [(e, s) for _, e, s in _CASES], ids=[n for n, _, _ in _CASES]
)
def test_rewrite_matches_duckdb(duck, t, expr, sql):
    out = bt.from_arrow(t).select(r=expr()).collect()
    assert_same(out, duck.sql(f"SELECT {sql} AS r FROM t"))


@pytest.mark.parametrize(
    ("expr", "sql"), [(e, s) for _, e, s in _CASES], ids=[n for n, _, _ in _CASES]
)
def test_rewrite_matches_duckdb_on_empty_input(duck, empty, expr, sql):
    out = bt.from_arrow(empty).select(r=expr()).collect()
    assert_same(out, duck.sql(f"SELECT {sql} AS r FROM t"))


def test_epoch_interval_inside_a_filter_matches_duckdb(duck, t):
    out = (
        bt.from_arrow(t)
        .filter(DateFunc("epoch", col("ts")) >= lit(0))
        .select(ts=col("ts"))
        .collect()
    )
    assert_same(out, duck.sql("SELECT ts FROM t WHERE epoch(ts) >= 0"))


def test_list_position_is_left_untouched_and_still_matches(duck, t):
    # No rule may turn this into `list_contains`: the two disagree on an empty list.
    out = bt.from_arrow(t).select(r=ListPosition(col("l"), 2) > lit(0)).collect()
    assert_same(out, duck.sql("SELECT list_position(l, 2) > 0 AS r FROM t"))
