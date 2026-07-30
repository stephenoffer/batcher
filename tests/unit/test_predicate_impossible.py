"""Plan-shape and does-not-fire tests for the two unsatisfiable-predicate rules.

`filter_arithmetic_contradiction` empties a filter whose conjunct no integer can satisfy, from
the range or bit lattice its own arithmetic produces; `filter_function_range_contradiction` does
the same from the image of a function. Both are claims about *every* value the left side can
take, so the tests that carry the weight are the satisfiable neighbours — one value off each
boundary — and the guards: the integer-type guard, the ASCII restriction on the case-folding
refutation, and `<>`, which a value outside the range makes true on every row rather than false.
"""

from __future__ import annotations

import datetime as dt

import pytest

import batcher as bt
from batcher import col
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.extra.predicate_impossible import (
    filter_arithmetic_contradiction,
    filter_function_range_contradiction,
)
from batcher.plan.expr_ir import Binary, Lit


def _plan(pred):
    ds = bt.from_pydict(
        {
            "i": [1, 2, 3],
            "j": [7, 8, 9],
            "f": [1.0, 2.0, 3.0],
            "s": ["ab", "cd", "ef"],
            "lst": [[1, 2], [3], []],
            "ts": [dt.datetime(2024, 1, 1), dt.datetime(2024, 6, 15), dt.datetime(2024, 12, 31)],
        }
    )
    return ds.filter(pred)._plan


def _empties(pred) -> bool:
    """Whether the rule reduces the filter to constant FALSE."""
    out = filter_arithmetic_contradiction(_plan(pred), None)
    return out is not None and out.predicate.to_ir() == Lit(False).to_ir()


def test_both_rules_are_registered():
    names = {r.name for r in DEFAULT_REGISTRY.rules()}
    assert {"filter_arithmetic_contradiction", "filter_function_range_contradiction"} <= names


# --- remainder magnitude: |x % k| <= |k| - 1 ---------------------------------


@pytest.mark.parametrize(
    "pred",
    [
        col("i") % 10 == 15,  # above the reachable maximum
        col("i") % 10 == -15,  # below the reachable minimum
        col("i") % 10 == 10,  # exactly |k| is already unreachable
        col("i") % 10 >= 10,
        col("i") % 10 > 9,
        col("i") % 10 < -9,
        col("i") % 10 <= -10,
        col("i") % -10 == 12,  # a negative divisor has the same magnitude bound
    ],
)
def test_unsatisfiable_remainder_comparison_empties_the_filter(pred):
    assert _empties(pred)


@pytest.mark.parametrize(
    "pred",
    [
        col("i") % 10 == 9,  # the reachable maximum
        col("i") % 10 == -9,  # the reachable minimum
        col("i") % 10 == 0,
        col("i") % 10 > 8,
        col("i") % 10 >= 9,
        col("i") % 10 < -8,
        col("i") % 10 <= -9,
        col("i") % 10 != 15,  # never-equal is satisfied by every row, not refuted
    ],
)
def test_satisfiable_remainder_comparison_is_left_alone(pred):
    assert filter_arithmetic_contradiction(_plan(pred), None) is None


def test_zero_divisor_is_not_refuted():
    # `x % 0` is NULL on every row rather than a value, so the range argument does not apply.
    # NULL is already not TRUE, so nothing is lost by leaving it.
    assert filter_arithmetic_contradiction(_plan(col("i") % 0 == 5), None) is None


# --- multiplication parity ---------------------------------------------------


@pytest.mark.parametrize(
    "pred",
    [
        col("i") * 2 == 7,  # an even coefficient can never produce an odd product
        col("i") * 4 == 6,  # 4 | product, and 4 does not divide 6
        col("i") * 8 == 12,
        col("i") * -2 == 7,  # the sign does not change the parity argument
        col("i") * 0 == 5,  # the reachable set is {0}
    ],
)
def test_unsatisfiable_product_empties_the_filter(pred):
    assert _empties(pred)


@pytest.mark.parametrize(
    "pred",
    [
        col("i") * 2 == 8,  # 8 is even: two solutions exist under wrapping arithmetic
        col("i") * 3 == 7,  # an odd coefficient is a bijection — always solvable
        col("i") * 4 == 8,
        col("i") * 0 == 0,  # satisfied by every row
        col("i") * 2 != 7,  # `<>` is satisfied by every row, not refuted
        col("i") * 2 < 7,  # wrapping breaks the ordered argument, so it is not made
    ],
)
def test_satisfiable_product_is_left_alone(pred):
    assert filter_arithmetic_contradiction(_plan(pred), None) is None


# --- bit masks ---------------------------------------------------------------


@pytest.mark.parametrize(
    "pred",
    [
        col("i").bitwise_and(12) == 3,  # bits 0 and 1 are outside the mask
        col("i").bitwise_and(12) == 13,
        col("i").bitwise_or(12) == 3,  # `|` forces bits 2 and 3, which 3 lacks
        col("i").bitwise_or(12) == 8,
        col("i").bitwise_and(-4) == 3,  # a negative mask reads as its 64-bit pattern
    ],
)
def test_unsatisfiable_bit_mask_empties_the_filter(pred):
    assert _empties(pred)


@pytest.mark.parametrize(
    "pred",
    [
        col("i").bitwise_and(12) == 4,
        col("i").bitwise_and(12) == 12,
        col("i").bitwise_and(12) == 0,
        col("i").bitwise_or(12) == 13,
        col("i").bitwise_or(12) == 12,
        col("i").bitwise_and(12) != 3,  # `<>` is not refuted
        col("i").bitwise_and(12) < 3,  # only equality is analyzed
    ],
)
def test_satisfiable_bit_mask_is_left_alone(pred):
    assert filter_arithmetic_contradiction(_plan(pred), None) is None


# --- the guards --------------------------------------------------------------


def test_float_operand_is_left_alone():
    # `%` on a float has the same magnitude property, but `*` has no parity one and a bit
    # operation is not this shape at all — so the whole rule gates on an integer operand.
    assert filter_arithmetic_contradiction(_plan(col("f") % 10 == 15), None) is None


def test_non_constant_divisor_is_left_alone():
    assert filter_arithmetic_contradiction(_plan(col("i") % col("j") == 15), None) is None


def test_two_column_product_is_left_alone():
    assert filter_arithmetic_contradiction(_plan(col("i") * col("j") == 7), None) is None


def test_already_constant_predicate_is_left_alone():
    assert filter_arithmetic_contradiction(_plan(Lit(False)), None) is None


def test_literal_written_first_is_normalized():
    # `15 = i % 10` is the same claim written the other way round.
    assert _empties(Binary("eq", Lit(15), col("i") % 10))
    # ...and so is the mirrored ordered comparison: `10 <= i % 10` is `i % 10 >= 10`.
    assert _empties(Binary("le", Lit(10), col("i") % 10))


def test_a_refutable_conjunct_empties_the_whole_conjunction():
    # A Filter keeps a row only where every conjunct is TRUE, so one never-TRUE conjunct is
    # enough — whatever the siblings say.
    assert _empties((col("i") > 0) & (col("i") % 10 == 15))
    assert _empties((col("i") % 10 == 15) & (col("j") < 100))


def test_a_refutable_disjunct_does_not_empty_the_filter():
    # Inside an OR the conjunct is not controlling: the other side can still be TRUE.
    assert filter_arithmetic_contradiction(_plan((col("i") > 0) | (col("i") % 10 == 15)), None) is (
        None
    )


def test_rule_is_idempotent():
    once = filter_arithmetic_contradiction(_plan(col("i") % 10 == 15), None)
    assert filter_arithmetic_contradiction(once, None) is None


# --- through the whole optimizer ---------------------------------------------


@pytest.mark.parametrize(
    "pred",
    [
        col("i") % 10 == 15,
        col("i") * 2 == 7,
        col("i").bitwise_and(12) == 3,
        col("i").bitwise_or(12) == 3,
    ],
)
def test_refutation_becomes_the_canonical_empty_relation(pred):
    # The payoff is not the constant FALSE but what the empty-relation rules then do with it:
    # the filter disappears and the scan is capped at zero rows.
    from batcher.kyber.optimizer import optimize_logical
    from batcher.plan.logical import Limit, Scan
    from batcher.plan.visitor import walk

    ds = bt.from_pydict({"i": [1, 2, 3, 15, 20]})
    out = optimize_logical(ds.filter(pred)._plan)
    assert [type(n) for n in walk(out)] == [Limit, Scan]
    assert out.n == 0


# --- the image of a function -------------------------------------------------
#
# `filter_function_range_contradiction` refutes from what the function can return at all, which
# is a property of the function rather than of the data — so unlike its arithmetic sibling it
# resolves no schema. The ranges were verified against the engine, not assumed, and the two
# weekday conventions living side by side (`dayofweek` is Sunday-0, `isodow` is Monday-1) are
# exactly where a guess would have made it wrong at one end.


def _image_empties(pred) -> bool:
    out = filter_function_range_contradiction(_plan(pred), None)
    return out is not None and out.predicate.to_ir() == Lit(False).to_ir()


@pytest.mark.parametrize(
    "pred",
    [
        col("ts").dt.month() == 13,
        col("ts").dt.month() == 0,
        col("ts").dt.month() > 12,
        col("ts").dt.quarter() == 5,
        col("ts").dt.day() == 32,
        col("ts").dt.day() == 0,
        col("ts").dt.hour() >= 24,
        col("ts").dt.minute() > 59,
        col("ts").dt.second() == 60,
        col("ts").dt.week() == 54,
        col("ts").dt.dayofweek() == 7,  # Sunday = 0, so 7 is out of range
        col("ts").dt.isodow() == 0,  # Monday = 1, so 0 is out of range
        col("ts").dt.dayofyear() == 367,
        col("s").str.len_chars() < 0,
        col("s").str.len_chars() == -1,
        col("i").abs() == -5,
        col("i").abs() < 0,
        col("f").abs() <= -0.5,
        col("i").sign() == 5,
        col("i").sign() == -2,
        col("s").str.to_uppercase() == "ab",  # an uppercased string has no ASCII lowercase
        col("s").str.to_uppercase() == "Ab",  # one offending character is enough
        col("s").str.to_lowercase() == "AB",
        # Every counting function: a length or a match count is never negative.
        col("s").str.len_bytes() < 0,
        col("s").str.count_matches("a") == -1,
        col("lst").list.len() < 0,
        col("lst").list.len() == -1,
        # The two float functions whose *lower* bound is safe against NaN.
        col("f").sqrt() < 0,
        col("f").exp() < 0,
    ],
)
def test_value_outside_the_image_empties_the_filter(pred):
    assert _image_empties(pred)


@pytest.mark.parametrize(
    "pred",
    [
        col("ts").dt.month() == 12,  # the reachable maximum
        col("ts").dt.month() == 1,  # the reachable minimum
        col("ts").dt.month() > 11,
        col("ts").dt.hour() == 23,
        col("ts").dt.hour() >= 23,
        col("ts").dt.second() == 59,
        col("ts").dt.week() == 53,
        col("ts").dt.dayofweek() == 0,
        col("ts").dt.dayofweek() == 6,
        col("ts").dt.isodow() == 7,
        col("ts").dt.dayofyear() == 366,
        col("ts").dt.year() == -5,  # a year has no useful bound, so nothing is claimed
        col("s").str.len_chars() == 0,
        col("i").abs() == 0,
        col("f").abs() == -0.0,  # -0.0 equals 0.0, so this is satisfiable
        col("i").sign() == -1,
        col("s").str.to_uppercase() == "AB",
        col("s").str.to_uppercase() == "A1_",  # no letters to offend
        col("s").str.to_lowercase() == "ab",
        col("ts").dt.month() != 13,  # `<>` is true on every row, not refuted
        col("lst").list.len() == 0,
        col("lst").list.len() >= 0,
        col("s").str.len_bytes() == 0,
        col("s").str.count_matches("a") == 0,
        # `sqrt(-0.0)` is `-0.0`, which equals `0.0`, so the inclusive bound is reachable.
        col("f").sqrt() == 0,
        # `exp` reaches zero by underflow, so the bound is inclusive rather than exclusive.
        col("f").exp() == 0,
        # No float function carries an *upper* bound, because `sin(NaN)` would break one.
        col("f").sqrt() > 1e300,
        col("f").exp() > 1e300,
    ],
)
def test_value_inside_the_image_is_left_alone(pred):
    assert filter_function_range_contradiction(_plan(pred), None) is None


def test_non_ascii_case_literal_is_left_alone():
    # The general Unicode case mappings are locale-sensitive at the edges, so the rule claims
    # nothing about a non-ASCII letter rather than guessing.
    assert filter_function_range_contradiction(_plan(col("s").str.to_uppercase() == "é"), None) is (
        None
    )


def test_image_refutation_ignores_a_comparison_between_two_calls():
    pred = col("ts").dt.month() == col("ts").dt.day()
    assert filter_function_range_contradiction(_plan(pred), None) is None


def test_image_literal_written_first_is_normalized():
    assert _image_empties(Binary("eq", Lit(13), col("ts").dt.month()))
    assert _image_empties(Binary("le", Lit(24), col("ts").dt.hour()))


def test_image_refutation_is_idempotent():
    once = filter_function_range_contradiction(_plan(col("ts").dt.month() == 13), None)
    assert filter_function_range_contradiction(once, None) is None


def test_image_refutation_empties_a_larger_conjunction():
    assert _image_empties((col("i") > 0) & (col("ts").dt.month() == 13))


def test_image_refutation_does_not_fire_inside_a_disjunction():
    pred = (col("i") > 0) | (col("ts").dt.month() == 13)
    assert filter_function_range_contradiction(_plan(pred), None) is None


def test_no_float_function_carries_an_upper_image_bound():
    """A float upper bound would be unsound, and this states why mechanically.

    `sin(x) <= 1` is true mathematically and false in the engine: `sin(NaN)` is NaN, which the
    total order places *above* every finite value, so `sin(x) > 1` is TRUE on a NaN row rather
    than impossible. Refuting it would drop that row. A *lower* bound has no such hazard, since
    a NaN is never below anything — so every float entry must be lower-only, and this fails if
    one ever grows an upper bound without that argument being revisited.
    """
    from batcher.kyber.rules.extra.predicate_impossible import _IMAGE
    from batcher.plan.expr_ir.core import MathExpr

    bounded_above = {
        fn
        for (node_type, fn), (_lo, hi) in _IMAGE.items()
        if node_type is MathExpr and hi is not None
    }
    # `sign` is the one exception, and it is one because it never returns a NaN at all: the
    # engine answers 0.0 for a NaN input, which the family's tests pin directly.
    assert bounded_above == {"sign"}
