"""Plan-shape tests for the `kyber.rules.predicate_algebra.bounds` family.

Each rule must union the disjunction into the single documented comparison, and must
decline the neighbouring shape whose union is not a single interval: two *disjoint*
ranges, two bounds on different columns, a comparison against a non-literal, and a pair of
literals whose types are not comparable.
"""

from __future__ import annotations

import datetime as dt

import pytest

import batcher as bt
from batcher import col, lit
from batcher.kyber.optimizer import optimize_logical
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.plan.expr_ir import Expr, InList


def _ds():
    return bt.from_pydict(
        {"x": [1, 2, 3], "y": [4, 5, 6], "s": ["a", "b", "c"], "b": [True, False, True]}
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


# --- widening same-direction bounds -----------------------------------------


@pytest.mark.parametrize(
    ("left", "right", "want"),
    [
        (col("x") < lit(3), col("x") < lit(7), col("x") < lit(7)),
        (col("x") < lit(7), col("x") < lit(3), col("x") < lit(7)),
        (col("x") <= lit(3), col("x") < lit(7), col("x") < lit(7)),
        (col("x") < lit(3), col("x") <= lit(3), col("x") <= lit(3)),
    ],
)
def test_upper_bound_disjunction_keeps_the_weaker(left, right, want):
    assert _fire("widen_upper_bound_disjunction", left | right) == want.to_ir()


@pytest.mark.parametrize(
    ("left", "right", "want"),
    [
        (col("x") > lit(7), col("x") >= lit(3), col("x") >= lit(3)),
        (col("x") >= lit(3), col("x") > lit(7), col("x") >= lit(3)),
        (col("x") > lit(3), col("x") >= lit(3), col("x") >= lit(3)),
    ],
)
def test_lower_bound_disjunction_keeps_the_weaker(left, right, want):
    assert _fire("widen_lower_bound_disjunction", left | right) == want.to_ir()


def test_bound_disjunction_declines_opposite_directions():
    _noop("widen_upper_bound_disjunction", (col("x") < lit(3)) | (col("x") > lit(7)))


def test_bound_disjunction_declines_different_columns():
    _noop("widen_upper_bound_disjunction", (col("x") < lit(3)) | (col("y") < lit(7)))


def test_bound_disjunction_declines_a_non_literal_bound():
    _noop("widen_upper_bound_disjunction", (col("x") < lit(3)) | (col("x") < col("y")))


def test_bound_disjunction_declines_incomparable_literals():
    _noop("widen_upper_bound_disjunction", (col("x") < lit(3)) | (col("s") < lit("a")))


def test_bound_disjunction_declines_booleans():
    # Python orders True as 1; a boolean column must not be widened by an integer rule.
    _noop("widen_upper_bound_disjunction", (col("b") < lit(True)) | (col("b") < lit(False)))


# --- closing a bound with an equality ---------------------------------------


@pytest.mark.parametrize(
    ("bound", "want"),
    [(col("x") > lit(3), col("x") >= lit(3)), (col("x") >= lit(3), col("x") >= lit(3))],
)
def test_equality_closes_a_lower_bound(bound, want):
    assert _fire("close_lower_bound_with_equality", (col("x") == lit(3)) | bound) == want.to_ir()


@pytest.mark.parametrize(
    ("bound", "want"),
    [(col("x") < lit(3), col("x") <= lit(3)), (col("x") <= lit(3), col("x") <= lit(3))],
)
def test_equality_closes_an_upper_bound(bound, want):
    assert _fire("close_upper_bound_with_equality", (col("x") == lit(3)) | bound) == want.to_ir()


def test_equality_at_a_different_literal_does_not_close_the_bound():
    _noop("close_lower_bound_with_equality", (col("x") == lit(5)) | (col("x") > lit(3)))


# --- an equality absorbing an implied bound ---------------------------------


@pytest.mark.parametrize(
    "bound",
    [
        col("x") > lit(1),
        col("x") >= lit(5),
        col("x") < lit(9),
        col("x") <= lit(5),
        col("x") != lit(2),
    ],
)
def test_equality_absorbs_a_bound_it_satisfies(bound):
    got = _fire("equality_absorbs_implied_bound", (col("x") == lit(5)) & bound)
    assert got == (col("x") == lit(5)).to_ir()


def test_equality_does_not_absorb_a_bound_it_fails():
    _noop("equality_absorbs_implied_bound", (col("x") == lit(5)) & (col("x") > lit(9)))


# --- merging IN lists -------------------------------------------------------


def test_in_list_absorbs_an_equality():
    got = _fire("merge_in_list_disjunction", InList(col("x"), (1, 2)) | (col("x") == lit(3)))
    assert got == InList(col("x"), (1, 2, 3)).to_ir()


def test_two_in_lists_merge():
    got = _fire("merge_in_list_disjunction", InList(col("x"), (1, 2)) | InList(col("x"), (2, 3)))
    assert got == InList(col("x"), (1, 2, 3)).to_ir()


def test_in_list_merge_declines_different_columns():
    _noop("merge_in_list_disjunction", InList(col("x"), (1, 2)) | InList(col("y"), (3,)))


def test_two_bare_equalities_are_left_to_the_existing_rule():
    _noop("merge_in_list_disjunction", (col("x") == lit(1)) | (col("x") == lit(2)))


# --- unioning ranges --------------------------------------------------------


def test_overlapping_ranges_union():
    left = (col("x") >= lit(1)) & (col("x") <= lit(5))
    right = (col("x") >= lit(4)) & (col("x") <= lit(9))
    got = _fire("union_overlapping_range_disjunction", left | right)
    assert got == ((col("x") >= lit(1)) & (col("x") <= lit(9))).to_ir()


def test_touching_ranges_union():
    left = (col("x") >= lit(1)) & (col("x") <= lit(4))
    right = (col("x") >= lit(4)) & (col("x") <= lit(9))
    got = _fire("union_overlapping_range_disjunction", left | right)
    assert got == ((col("x") >= lit(1)) & (col("x") <= lit(9))).to_ir()


def test_nested_range_union_keeps_the_outer_range():
    left = (col("x") >= lit(1)) & (col("x") <= lit(9))
    right = (col("x") >= lit(4)) & (col("x") <= lit(5))
    got = _fire("union_overlapping_range_disjunction", left | right)
    assert got == ((col("x") >= lit(1)) & (col("x") <= lit(9))).to_ir()


def test_disjoint_ranges_do_not_union():
    left = (col("x") >= lit(1)) & (col("x") <= lit(2))
    right = (col("x") >= lit(7)) & (col("x") <= lit(9))
    _noop("union_overlapping_range_disjunction", left | right)


def test_ranges_on_different_columns_do_not_union():
    left = (col("x") >= lit(1)) & (col("x") <= lit(5))
    right = (col("y") >= lit(4)) & (col("y") <= lit(9))
    _noop("union_overlapping_range_disjunction", left | right)


def test_range_union_works_on_dates():
    left = (col("x") >= lit(dt.date(2020, 1, 1))) & (col("x") <= lit(dt.date(2020, 6, 1)))
    right = (col("x") >= lit(dt.date(2020, 5, 1))) & (col("x") <= lit(dt.date(2020, 9, 1)))
    got = _fire("union_overlapping_range_disjunction", left | right)
    want = (col("x") >= lit(dt.date(2020, 1, 1))) & (col("x") <= lit(dt.date(2020, 9, 1)))
    assert got == want.to_ir()


# --- end to end -------------------------------------------------------------


@pytest.mark.parametrize(
    "expr",
    [
        (col("x") < lit(3)) | (col("x") < lit(7)),
        (col("x") == lit(3)) | (col("x") > lit(3)),
        (col("x") == lit(5)) & (col("x") > lit(1)),
        ((col("x") >= lit(1)) & (col("x") <= lit(5)))
        | ((col("x") >= lit(4)) & (col("x") <= lit(9))),
    ],
)
def test_optimizer_is_idempotent(expr):
    once = optimize_logical(_proj(expr))
    assert optimize_logical(once).to_ir() == once.to_ir()


# --- collapse_degenerate_range_to_equality ----------------------------------
#
# The one rule here that reads a *conjunction* of two ordered bounds rather than a
# disjunction. It exists because an equality is a different kind of predicate to the skipping
# machinery, not because it is shorter: a bloom filter refutes an equality and says nothing
# about a range, and equality selectivity comes off the most-common-values sketch instead of
# being interpolated from the quantile grid.

_DEGENERATE = "collapse_degenerate_range_to_equality"


def test_zero_width_range_becomes_an_equality():
    got = _fire(_DEGENERATE, (col("x") >= lit(3)) & (col("x") <= lit(3)))
    assert got == (col("x") == lit(3)).to_ir()


def test_zero_width_range_in_either_order():
    got = _fire(_DEGENERATE, (col("x") <= lit(3)) & (col("x") >= lit(3)))
    assert got == (col("x") == lit(3)).to_ir()


def test_zero_width_range_over_a_string():
    got = _fire(_DEGENERATE, (col("s") >= lit("b")) & (col("s") <= lit("b")))
    assert got == (col("s") == lit("b")).to_ir()


def test_zero_width_range_over_a_date():
    day = dt.date(2020, 5, 1)
    got = _fire(_DEGENERATE, (col("x") >= lit(day)) & (col("x") <= lit(day)))
    assert got == (col("x") == lit(day)).to_ir()


def test_zero_width_range_with_the_literal_written_first():
    # `3 <= x AND 3 >= x` is the same pair mirrored; `_comparison` normalizes both.
    got = _fire(_DEGENERATE, (lit(3) <= col("x")) & (lit(3) >= col("x")))
    assert got == (col("x") == lit(3)).to_ir()


@pytest.mark.parametrize(
    "expr",
    [
        # A real interval, not a point.
        (col("x") >= lit(3)) & (col("x") <= lit(4)),
        # Strict bounds at one literal are a *contradiction*, which is
        # `filter_range_contradiction`'s job — this rule must not turn one into an equality.
        (col("x") > lit(3)) & (col("x") < lit(3)),
        (col("x") > lit(3)) & (col("x") <= lit(3)),
        (col("x") >= lit(3)) & (col("x") < lit(3)),
        # Same direction twice is `tighten_comparison_bounds`'s job.
        (col("x") >= lit(3)) & (col("x") >= lit(3)),
        # Different columns share no point.
        (col("x") >= lit(3)) & (col("y") <= lit(3)),
        # Incomparable literals decline rather than guessing the engine's coercion.
        (col("x") >= lit(3)) & (col("x") <= lit("3")),
        # A disjunction is a different rule's shape entirely.
        (col("x") >= lit(3)) | (col("x") <= lit(3)),
    ],
)
def test_degenerate_range_declines(expr):
    node = _proj(expr)
    assert _rule(_DEGENERATE).apply(node, None).to_ir() == node.to_ir()


def test_degenerate_range_collapse_is_idempotent():
    once = optimize_logical(_proj((col("x") >= lit(3)) & (col("x") <= lit(3))))
    assert optimize_logical(once).to_ir() == once.to_ir()
