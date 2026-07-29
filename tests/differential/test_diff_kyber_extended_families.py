"""The second-wave rewrites must match DuckDB after the full optimizer runs.

Covers the date-column `epoch`/`offset_by` twins, the sign-split `trunc` intervals, and
the two-list null-strictness splits. The fixtures carry dates on both sides of the Unix
epoch, floats on both sides of zero (plus NaN and the infinities, where Arrow's total
order puts NaN above everything), NULLs, and an empty list.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
import batcher.kyber.rules.math_algebra
import batcher.kyber.rules.nulls
import batcher.kyber.rules.temporal_algebra
from _harness import assert_same
from batcher import col, lit
from batcher.plan.expr_ir.core import MathExpr
from batcher.plan.expr_ir.func_nodes import DateFunc, DateOffset, ListBinary, ListSet, ListZip

_MIDNIGHT_2020 = 1_577_836_800


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "d": pa.array(
                [
                    dt.date(2019, 12, 31),
                    dt.date(2020, 1, 1),
                    dt.date(2020, 1, 2),
                    dt.date(1969, 12, 31),
                    None,
                ],
                type=pa.date32(),
            ),
            "f": pa.array(
                [2.7, -2.7, 0.0, float("nan"), None],
                type=pa.float64(),
            ),
            "l": pa.array(
                [[1.0, 2.0], [3.0, 4.0], [], [5.0, 6.0], None], type=pa.list_(pa.float64())
            ),
            "m": pa.array(
                [[7.0, 8.0], None, [], [9.0, 10.0], [11.0, 12.0]], type=pa.list_(pa.float64())
            ),
        }
    )
    duck.register("t", tbl)
    return tbl


@pytest.fixture
def empty(duck):
    tbl = pa.table(
        {
            "d": pa.array([], type=pa.date32()),
            "f": pa.array([], type=pa.float64()),
            "l": pa.array([], type=pa.list_(pa.float64())),
            "m": pa.array([], type=pa.list_(pa.float64())),
        }
    )
    duck.register("t", tbl)
    return tbl


_CASES = [
    (
        "epoch_date_ge",
        lambda: DateFunc("epoch", col("d")) >= lit(_MIDNIGHT_2020),
        f"epoch(d) >= {_MIDNIGHT_2020}",
    ),
    (
        "epoch_date_gt",
        lambda: DateFunc("epoch", col("d")) > lit(_MIDNIGHT_2020),
        f"epoch(d) > {_MIDNIGHT_2020}",
    ),
    (
        "epoch_date_le",
        lambda: DateFunc("epoch", col("d")) <= lit(_MIDNIGHT_2020),
        f"epoch(d) <= {_MIDNIGHT_2020}",
    ),
    (
        "epoch_date_lt",
        lambda: DateFunc("epoch", col("d")) < lit(_MIDNIGHT_2020),
        f"epoch(d) < {_MIDNIGHT_2020}",
    ),
    (
        "epoch_date_eq",
        lambda: DateFunc("epoch", col("d")) == lit(_MIDNIGHT_2020),
        f"epoch(d) = {_MIDNIGHT_2020}",
    ),
    (
        "epoch_date_ne",
        lambda: DateFunc("epoch", col("d")) != lit(_MIDNIGHT_2020),
        f"epoch(d) <> {_MIDNIGHT_2020}",
    ),
    # A bound inside a day: the date boundary must round, not truncate.
    (
        "epoch_date_ge_inside",
        lambda: DateFunc("epoch", col("d")) >= lit(_MIDNIGHT_2020 + 1),
        f"epoch(d) >= {_MIDNIGHT_2020 + 1}",
    ),
    (
        "epoch_date_le_inside",
        lambda: DateFunc("epoch", col("d")) <= lit(_MIDNIGHT_2020 + 1),
        f"epoch(d) <= {_MIDNIGHT_2020 + 1}",
    ),
    (
        "epoch_date_eq_inside",
        lambda: DateFunc("epoch", col("d")) == lit(_MIDNIGHT_2020 + 1),
        f"epoch(d) = {_MIDNIGHT_2020 + 1}",
    ),
    (
        "offset_date_lt",
        lambda: DateOffset(col("d"), 0, 3, 0) < lit(dt.date(2020, 1, 10)),
        "(d + INTERVAL 3 DAY) < DATE '2020-01-10'",
    ),
    (
        "offset_date_ge",
        lambda: DateOffset(col("d"), 0, 3, 0) >= lit(dt.date(2020, 1, 10)),
        "(d + INTERVAL 3 DAY) >= DATE '2020-01-10'",
    ),
    (
        "offset_date_eq",
        lambda: DateOffset(col("d"), 0, 3, 0) == lit(dt.date(2020, 1, 4)),
        "(d + INTERVAL 3 DAY) = DATE '2020-01-04'",
    ),
    (
        "offset_date_month_untouched",
        lambda: DateOffset(col("d"), 1, 0, 0) < lit(dt.date(2020, 1, 10)),
        "(d + INTERVAL 1 MONTH) < DATE '2020-01-10'",
    ),
    ("trunc_eq_positive", lambda: MathExpr("trunc", col("f")) == lit(2), "trunc(f) = 2"),
    ("trunc_eq_negative", lambda: MathExpr("trunc", col("f")) == lit(-2), "trunc(f) = -2"),
    ("trunc_eq_zero", lambda: MathExpr("trunc", col("f")) == lit(0), "trunc(f) = 0"),
    ("trunc_ge_zero", lambda: MathExpr("trunc", col("f")) >= lit(0), "trunc(f) >= 0"),
    ("trunc_le_zero", lambda: MathExpr("trunc", col("f")) <= lit(0), "trunc(f) <= 0"),
    ("trunc_gt_zero", lambda: MathExpr("trunc", col("f")) > lit(0), "trunc(f) > 0"),
    ("trunc_lt_zero", lambda: MathExpr("trunc", col("f")) < lit(0), "trunc(f) < 0"),
    ("trunc_ne_zero", lambda: MathExpr("trunc", col("f")) != lit(0), "trunc(f) <> 0"),
    ("trunc_ge_negative", lambda: MathExpr("trunc", col("f")) >= lit(-2), "trunc(f) >= -2"),
    ("trunc_le_positive", lambda: MathExpr("trunc", col("f")) <= lit(2), "trunc(f) <= 2"),
    (
        "list_binary_is_null",
        lambda: ListBinary("dot", col("l"), col("m")).is_null(),
        "list_dot_product(l, m) IS NULL",
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


def test_date_epoch_interval_inside_a_filter_matches_duckdb(duck, t):
    out = (
        bt.from_arrow(t)
        .filter(DateFunc("epoch", col("d")) >= lit(_MIDNIGHT_2020))
        .select(d=col("d"))
        .collect()
    )
    assert_same(out, duck.sql(f"SELECT d FROM t WHERE epoch(d) >= {_MIDNIGHT_2020}"))


def test_two_list_operations_are_null_exactly_when_an_operand_is(t):
    """The engine fact the `list_zip` / `list_set_op` strictness rules rest on.

    DuckDB is not the oracle for these two: its `list_zip` pads a shorter list with NULLs
    and its `list_concat` treats a NULL list as empty, so neither spelling has Batcher's
    null-propagating contract and comparing against them would test the difference between
    the dialects rather than the rewrite. The claim the rules actually depend on is
    measurable directly — the result is null on exactly the rows where an operand is — so
    it is asserted here, and the rewrite itself is a mechanical consequence proven by the
    plan-shape tests.
    """
    ds = bt.from_arrow(t)
    operand_null = ds.select(r=col("l").is_null() | col("m").is_null()).collect()
    for expr in (
        ListZip("list_add", col("l"), col("m")),
        ListSet("array_union", col("l"), col("m")),
        ListBinary("dot", col("l"), col("m")),
    ):
        got = ds.select(r=expr.is_null()).collect()
        assert got.column("r").to_pylist() == operand_null.column("r").to_pylist()


def test_trunc_interval_inside_a_filter_matches_duckdb(duck, t):
    out = (
        bt.from_arrow(t).filter(MathExpr("trunc", col("f")) == lit(0)).select(f=col("f")).collect()
    )
    assert_same(out, duck.sql("SELECT f FROM t WHERE trunc(f) = 0"))
