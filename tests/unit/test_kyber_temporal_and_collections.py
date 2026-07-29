"""Plan-shape tests for the `temporal_algebra` and `collections_algebra` families.

The temporal rules must produce the documented instant interval and must decline the two
shapes their correctness argument excludes: a calendar-month offset (a month is not a
fixed duration) and a non-`Timestamp` argument. The list rules must reach the direct
membership test, must drop a reordering a membership test cannot see, and must decline the
folds whose answer Python cannot reproduce exactly.
"""

from __future__ import annotations

import datetime as dt

import pytest

import batcher as bt
from batcher import array, col, lit
from batcher.kyber.optimizer import optimize_logical
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.plan.expr_ir import Binary, Expr
from batcher.plan.expr_ir.func_nodes import (
    DateFunc,
    DateOffset,
    ListContains,
    ListFilter,
    ListFunc,
    ListPosition,
    ListTransform,
)

_EPOCH_2020 = 1_577_836_800  # 2020-01-01T00:00:00Z
_T2020 = dt.datetime(2020, 1, 1)
_T2020_PLUS_1S = dt.datetime(2020, 1, 1, 0, 0, 1)


def _ds():
    """Timestamp `t`, Date `d`, List `l`, Int64 `i`."""
    return bt.from_pydict(
        {
            "t": [dt.datetime(2020, 1, 1), dt.datetime(2021, 2, 3)],
            "d": [dt.date(2020, 1, 1), dt.date(2021, 2, 3)],
            "l": [[1, 2, 3], [4]],
            "i": [1, 2],
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


# --- epoch ------------------------------------------------------------------

_EPOCH_EXPECTED = {
    "lt": col("t") < lit(_T2020),
    "le": col("t") < lit(_T2020_PLUS_1S),
    "gt": col("t") >= lit(_T2020_PLUS_1S),
    "ge": col("t") >= lit(_T2020),
    "eq": (col("t") >= lit(_T2020)) & (col("t") < lit(_T2020_PLUS_1S)),
    "ne": (col("t") < lit(_T2020)) | (col("t") >= lit(_T2020_PLUS_1S)),
}


@pytest.mark.parametrize("op", list(_EPOCH_EXPECTED))
def test_epoch_comparison_becomes_an_instant_interval(op):
    expr = Binary(op, DateFunc("epoch", col("t")), lit(_EPOCH_2020))
    assert _fire(f"epoch_{op}_to_instant_range", expr) == _EPOCH_EXPECTED[op].to_ir()


def test_epoch_accepts_the_literal_on_the_left():
    expr = Binary("le", lit(_EPOCH_2020), DateFunc("epoch", col("t")))
    assert _fire("epoch_ge_to_instant_range", expr) == _EPOCH_EXPECTED["ge"].to_ir()


def test_epoch_declines_a_date_argument():
    _noop("epoch_ge_to_instant_range", DateFunc("epoch", col("d")) >= lit(_EPOCH_2020))


def test_epoch_declines_a_second_count_outside_the_representable_range():
    _noop("epoch_ge_to_instant_range", DateFunc("epoch", col("t")) >= lit(10**18))


def test_epoch_declines_a_non_literal_bound():
    _noop("epoch_ge_to_instant_range", DateFunc("epoch", col("t")) >= col("i"))


# --- offset_by --------------------------------------------------------------


@pytest.mark.parametrize("op", ["lt", "le", "gt", "ge", "eq", "ne"])
def test_offset_moves_onto_the_literal(op):
    expr = Binary(op, DateOffset(col("t"), 0, 1, 0), lit(dt.datetime(2020, 1, 5)))
    want = Binary(op, col("t"), lit(dt.datetime(2020, 1, 4)))
    assert _fire(f"offset_by_{op}_to_shifted_instant", expr) == want.to_ir()


def test_offset_with_microseconds_shifts_the_literal_exactly():
    expr = DateOffset(col("t"), 0, 0, 1_500_000) < lit(dt.datetime(2020, 1, 5))
    want = col("t") < lit(dt.datetime(2020, 1, 4, 23, 59, 58, 500_000))
    assert _fire("offset_by_lt_to_shifted_instant", expr) == want.to_ir()


def test_offset_declines_a_calendar_month():
    _noop(
        "offset_by_lt_to_shifted_instant",
        DateOffset(col("t"), 1, 0, 0) < lit(dt.datetime(2020, 1, 5)),
    )


def test_offset_declines_a_date_argument():
    _noop(
        "offset_by_lt_to_shifted_instant",
        DateOffset(col("d"), 0, 1, 0) < lit(dt.datetime(2020, 1, 5)),
    )


# --- list membership (deliberately unrewritten) ------------------------------


def test_list_position_is_never_rewritten_to_a_membership_test():
    # `list_position` answers NULL for an empty list while `list_contains` answers false,
    # so the two disagree on every empty-list row and no rule may equate them.
    names = {r.name for r in DEFAULT_REGISTRY.rules()}
    assert not any(name.startswith("list_position_") for name in names)


# --- list ordering ----------------------------------------------------------


@pytest.mark.parametrize("fn", ["sort", "reverse", "unique"])
def test_membership_sees_through_a_reordering(fn):
    expr = ListContains(ListFunc(fn, col("l")), 2)
    assert _fire(f"list_contains_through_list_{fn}", expr) == ListContains(col("l"), 2).to_ir()


def test_sort_absorbs_a_reversal():
    expr = ListFunc("sort", ListFunc("reverse", col("l")))
    assert _fire("collapse_list_sort_of_list_reverse", expr) == ListFunc("sort", col("l")).to_ir()


def test_length_sees_through_a_transform():
    expr = ListFunc("len", ListTransform(col("l"), bt.element() * lit(2)))
    assert _fire("list_len_through_list_transform", expr) == ListFunc("len", col("l")).to_ir()


def test_length_does_not_see_through_a_filter():
    # A filter can drop elements, so the length is not preserved.
    _noop(
        "list_len_through_list_transform",
        ListFunc("len", ListFilter(col("l"), bt.element() > lit(1))),
    )


def test_length_does_not_see_through_dedup():
    names = {r.name for r in DEFAULT_REGISTRY.rules()}
    assert "list_len_through_list_unique" not in names


# --- list folds -------------------------------------------------------------


@pytest.mark.parametrize(
    ("fn", "elements", "want"),
    [
        ("len", [1, 2, 3], 3),
        ("n_unique", [1, 1, 2], 2),
        ("min", [3, 1, 2], 1),
        ("max", [3, 1, 2], 3),
        ("sum", [1, 2, 3], 6),
        ("product", [2, 3], 6),
    ],
)
def test_reduction_over_a_literal_array_folds(fn, elements, want):
    expr = ListFunc(fn, array(*[lit(v) for v in elements]))
    assert _fire(f"fold_list_{fn}_of_literal_array", expr) == lit(want).to_ir()


def test_sum_of_float_literals_is_left_to_the_engine():
    _noop("fold_list_sum_of_literal_array", ListFunc("sum", array(lit(0.1), lit(0.2))))


def test_reduction_over_a_non_literal_array_is_left_alone():
    _noop("fold_list_min_of_literal_array", ListFunc("min", array(col("i"), lit(2))))


@pytest.mark.parametrize(
    ("fn", "elements", "want"),
    [
        ("sort", [3, 1, 2], [1, 2, 3]),
        ("reverse", [1, 2, 3], [3, 2, 1]),
        ("unique", [1, 1, 2], [1, 2]),
    ],
)
def test_reordering_of_a_literal_array_folds(fn, elements, want):
    expr = ListFunc(fn, array(*[lit(v) for v in elements]))
    assert _fire(f"fold_list_{fn}_of_literal_array", expr) == array(*[lit(v) for v in want]).to_ir()


def test_reordering_declines_a_non_literal_array():
    _noop("fold_list_sort_of_literal_array", ListFunc("sort", array(col("i"), lit(1))))


def test_membership_over_a_literal_array_folds():
    assert (
        _fire("fold_list_contains_of_literal_array", ListContains(array(lit(1), lit(2)), 5))
        == lit(False).to_ir()
    )
    assert (
        _fire("fold_list_position_of_literal_array", ListPosition(array(lit(1), lit(2)), 2))
        == lit(2).to_ir()
    )


def test_identity_transform_and_filter_drop_out():
    assert (
        _fire("drop_identity_list_transform", ListTransform(col("l"), bt.element()))
        == col("l").to_ir()
    )
    assert _fire("drop_identity_list_filter", ListFilter(col("l"), lit(True))) == col("l").to_ir()


def test_non_identity_transform_is_left_alone():
    _noop("drop_identity_list_transform", ListTransform(col("l"), bt.element() * lit(2)))


# --- end to end -------------------------------------------------------------


@pytest.mark.parametrize(
    "expr",
    [
        DateFunc("epoch", col("t")) >= lit(_EPOCH_2020),
        DateOffset(col("t"), 0, 1, 0) < lit(dt.datetime(2020, 1, 5)),
        ListPosition(col("l"), 2) > lit(0),
        ListFunc("sort", array(lit(3), lit(1))),
        ListFunc("len", ListTransform(col("l"), bt.element() * lit(2))),
    ],
)
def test_optimizer_is_idempotent(expr):
    once = optimize_logical(_proj(expr))
    assert optimize_logical(once).to_ir() == once.to_ir()
