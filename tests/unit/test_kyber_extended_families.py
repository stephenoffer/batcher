"""Plan-shape tests for the second wave of rule families.

Covers the date-column twins of the `epoch` and `offset_by` rewrites, the `trunc`
intervals (whose bounds split at zero rather than shifting by a constant), and the
null-strictness families added for the two-list operations and the windowing constructors.
Each group asserts the documented shape and the decline its correctness argument requires.
"""

from __future__ import annotations

import datetime as dt

import pytest

import batcher as bt
from batcher import col, lit
from batcher.kyber.optimizer import optimize_logical
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.plan.expr_ir import Binary, Expr
from batcher.plan.expr_ir.core import MathExpr
from batcher.plan.expr_ir.func_nodes import (
    DateFunc,
    DateOffset,
    ListBinary,
    ListSet,
    ListSlice,
    ListZip,
    WindowBuckets,
    WindowStart,
)

_MIDNIGHT_2020 = 1_577_836_800  # 2020-01-01T00:00:00Z
_D2020 = dt.date(2020, 1, 1)


def _ds():
    """Date `d`, Timestamp `t`, Float64 `f`, two List<Float64> columns."""
    return bt.from_pydict(
        {
            "d": [dt.date(2020, 1, 1), dt.date(2021, 2, 3)],
            "t": [dt.datetime(2020, 1, 1), dt.datetime(2021, 2, 3)],
            "f": [2.7, -2.7],
            "l": [[1.0, 2.0], [3.0]],
            "m": [[4.0, 5.0], [6.0]],
        }
    )


def _proj(expr: Expr):
    return _ds().select(r=expr)._plan


def _rule(name: str):
    for r in DEFAULT_REGISTRY.rules():
        if r.name == name:
            return r
    raise AssertionError(f"rule {name!r} is not registered")


def _fire(name: str, expr: Expr) -> dict:
    node = _proj(expr)
    out = _rule(name).apply(node, None)
    assert out.to_ir() != node.to_ir(), f"{name} did not fire"
    return out.items[0].expr.to_ir()


def _noop(name: str, expr: Expr) -> None:
    node = _proj(expr)
    assert _rule(name).apply(node, None).to_ir() == node.to_ir()


# --- epoch over a Date column -----------------------------------------------


@pytest.mark.parametrize(
    ("op", "seconds", "want_op", "want_day"),
    [
        # A bound landing exactly on midnight keeps the day it names.
        ("ge", _MIDNIGHT_2020, "ge", _D2020),
        ("le", _MIDNIGHT_2020, "le", _D2020),
        ("gt", _MIDNIGHT_2020, "ge", dt.date(2020, 1, 2)),
        ("lt", _MIDNIGHT_2020, "le", dt.date(2019, 12, 31)),
        # A bound inside a day rounds to the boundary that keeps the same set of dates.
        ("ge", _MIDNIGHT_2020 + 1, "ge", dt.date(2020, 1, 2)),
        ("le", _MIDNIGHT_2020 + 1, "le", _D2020),
        ("gt", _MIDNIGHT_2020 + 1, "ge", dt.date(2020, 1, 2)),
        ("lt", _MIDNIGHT_2020 + 1, "le", _D2020),
    ],
)
def test_epoch_over_a_date_becomes_a_date_comparison(op, seconds, want_op, want_day):
    expr = Binary(op, DateFunc("epoch", col("d")), lit(seconds))
    got = _fire(f"epoch_{op}_to_date_range", expr)
    assert got == Binary(want_op, col("d"), lit(want_day)).to_ir()


@pytest.mark.parametrize("op", ["eq", "ne"])
def test_epoch_equality_over_a_date_folds_only_at_a_midnight(op):
    exact = Binary(op, DateFunc("epoch", col("d")), lit(_MIDNIGHT_2020))
    assert _fire(f"epoch_{op}_to_date_range", exact) == Binary(op, col("d"), lit(_D2020)).to_ir()
    inside = Binary(op, DateFunc("epoch", col("d")), lit(_MIDNIGHT_2020 + 1))
    _noop(f"epoch_{op}_to_date_range", inside)


def test_epoch_date_rule_declines_a_timestamp_column():
    _noop("epoch_ge_to_date_range", DateFunc("epoch", col("t")) >= lit(_MIDNIGHT_2020))


def test_epoch_timestamp_rule_declines_a_date_column():
    _noop("epoch_ge_to_instant_range", DateFunc("epoch", col("d")) >= lit(_MIDNIGHT_2020))


# --- offset_by over a Date column -------------------------------------------


@pytest.mark.parametrize("op", ["lt", "le", "gt", "ge", "eq", "ne"])
def test_date_offset_moves_onto_the_literal(op):
    expr = Binary(op, DateOffset(col("d"), 0, 3, 0), lit(dt.date(2020, 1, 10)))
    want = Binary(op, col("d"), lit(dt.date(2020, 1, 7)))
    assert _fire(f"offset_by_{op}_to_shifted_date", expr) == want.to_ir()


def test_date_offset_declines_a_microsecond_component():
    _noop(
        "offset_by_lt_to_shifted_date",
        DateOffset(col("d"), 0, 0, 5) < lit(dt.date(2020, 1, 10)),
    )


def test_date_offset_declines_a_calendar_month():
    _noop(
        "offset_by_lt_to_shifted_date",
        DateOffset(col("d"), 1, 0, 0) < lit(dt.date(2020, 1, 10)),
    )


def test_date_offset_declines_a_timestamp_column():
    _noop(
        "offset_by_lt_to_shifted_date",
        DateOffset(col("t"), 0, 3, 0) < lit(dt.date(2020, 1, 10)),
    )


# --- trunc intervals --------------------------------------------------------


@pytest.mark.parametrize(
    ("op", "k", "want"),
    [
        # Above zero `trunc` behaves like `floor`.
        ("eq", 2, (col("f") >= lit(2)) & (col("f") < lit(3))),
        ("ge", 2, col("f") >= lit(2)),
        ("le", 2, col("f") < lit(3)),
        # Below zero it behaves like `ceil`.
        ("eq", -2, (col("f") > lit(-3)) & (col("f") <= lit(-2))),
        ("ge", -2, col("f") > lit(-3)),
        ("le", -2, col("f") <= lit(-2)),
        # At zero the bucket straddles it: (-1, 1).
        ("eq", 0, (col("f") > lit(-1)) & (col("f") < lit(1))),
        ("ge", 0, col("f") > lit(-1)),
        ("le", 0, col("f") < lit(1)),
        ("gt", 0, col("f") >= lit(1)),
        ("lt", 0, col("f") <= lit(-1)),
    ],
)
def test_trunc_comparison_becomes_the_sign_split_interval(op, k, want):
    expr = Binary(op, MathExpr("trunc", col("f")), lit(k))
    assert _fire(f"trunc_{op}_to_range", expr) == want.to_ir()


def test_trunc_inequality_is_the_complement_of_the_bucket():
    expr = MathExpr("trunc", col("f")) != lit(0)
    got = _fire("trunc_ne_to_range", expr)
    assert got == ((col("f") <= lit(-1)) | (col("f") >= lit(1))).to_ir()


def test_trunc_declines_a_fractional_bound():
    _noop("trunc_eq_to_range", MathExpr("trunc", col("f")) == lit(2.5))


# --- null strictness over the two-list operations ---------------------------


@pytest.mark.parametrize(
    ("family", "expr"),
    [
        ("list_zip", ListZip("list_add", col("l"), col("m"))),
        ("list_set_op", ListSet("array_union", col("l"), col("m"))),
        ("list_binary", ListBinary("dot", col("l"), col("m"))),
    ],
)
def test_null_test_splits_over_a_two_list_operation(family, expr):
    assert (
        _fire(f"is_null_through_{family}", expr.is_null())
        == (col("l").is_null() | col("m").is_null()).to_ir()
    )
    assert (
        _fire(f"is_not_null_through_{family}", expr.is_not_null())
        == (col("l").is_not_null() & col("m").is_not_null()).to_ir()
    )


@pytest.mark.parametrize(
    ("family", "expr"),
    [
        ("list_slice", ListSlice(col("l"), 0, 1)),
        ("window_start", WindowStart(col("t"), 3_600_000_000)),
        ("window_buckets", WindowBuckets(col("t"), 3_600_000_000, 1_800_000_000)),
    ],
)
def test_null_test_moves_through_a_unary_list_or_window_construct(family, expr):
    assert _fire(f"is_null_through_{family}", expr.is_null()) == expr.input.is_null().to_ir()
    assert (
        _fire(f"is_not_null_through_{family}", expr.is_not_null())
        == expr.input.is_not_null().to_ir()
    )


def test_simhash_is_not_a_strictness_family():
    # `list_simhash` answers null for an *empty* list, which is not a null input.
    names = {r.name for r in DEFAULT_REGISTRY.rules()}
    assert "is_null_through_list_simhash" not in names


# --- end to end -------------------------------------------------------------


@pytest.mark.parametrize(
    "expr",
    [
        DateFunc("epoch", col("d")) >= lit(_MIDNIGHT_2020),
        DateOffset(col("d"), 0, 3, 0) < lit(dt.date(2020, 1, 10)),
        MathExpr("trunc", col("f")) == lit(0),
        ListZip("list_add", col("l"), col("m")).is_null(),
    ],
)
def test_optimizer_is_idempotent(expr):
    once = optimize_logical(_proj(expr))
    assert optimize_logical(once).to_ir() == once.to_ir()
