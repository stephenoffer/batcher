"""Selectivity mathematics: the identities a probability must satisfy.

These are not tuning preferences. Each is an identity that a *fraction of rows kept* has
to obey, and each was violated:

* `P(x < v) + P(x >= v) = 1` — the boundary's point mass has to land on exactly one side.
  `lt` was estimated as `le`, so `x < 5` and `x <= 5` were the same number.
* `sel(p) + sel(NOT p) = 1 - f_null(p)` — SQL keeps only TRUE rows, so a NULL operand is
  dropped by `p` *and* by `NOT p`. Both complements assumed 2-valued logic.
* `x IN (v1, ..., vk)` is a union of **mutually exclusive** equalities, so it is their
  sum over distinct literals — duplicated literals must not inflate it.
* a quantile grid's endpoints carry probability `probs[0]` / `probs[-1]`, not 0 / 1.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.config import active_config
from batcher.kyber.stats.selectivity import predicate_selectivity as sel
from batcher.plan.expr_ir import InList

pytestmark = pytest.mark.unit

_CFG = active_config().optimizer.cardinality
_NDV = {"x": 10.0}
# A uniform grid over [0, 10]: F(5) = 0.5, and each of the 10 distinct values holds 1/10.
_QUANTILES = {"x": {"probs": [0.0, 1.0], "values": [0.0, 10.0]}}


def _s(expr, *, nulls=None):
    return sel(expr, _NDV, _CFG, _QUANTILES, None, None, nulls)


# --- range boundaries: the point mass belongs to exactly one side -----------------


def test_strict_and_non_strict_differ_by_the_point_mass():
    below = _s(bt.col("x") < 5)
    at_or_below = _s(bt.col("x") <= 5)
    assert at_or_below - below == pytest.approx(1.0 / _NDV["x"])


def test_lt_and_ge_partition_the_rows():
    assert _s(bt.col("x") < 5) + _s(bt.col("x") >= 5) == pytest.approx(1.0)


def test_le_and_gt_partition_the_rows():
    assert _s(bt.col("x") <= 5) + _s(bt.col("x") > 5) == pytest.approx(1.0)


def test_flipped_operand_order_is_equivalent():
    # `5 > x` is `x < 5`. Built from the raw IR node: Python's reflected comparison would
    # hand back the very same `Binary("lt", Col, Lit)`, so the test would assert nothing.
    from batcher.plan.expr_ir import Binary, Col, Lit

    flipped = Binary("gt", Lit(5), Col("x"))
    assert _s(flipped) == pytest.approx(_s(bt.col("x") < 5))


# --- quantile grid endpoints ------------------------------------------------------


def test_grid_endpoints_carry_their_boundary_probability():
    from batcher.kyber.stats.selectivity import _fraction_below

    # An equi-depth grid inset from the true extremes: values[0] is the 10th percentile.
    probs, values = [0.1, 0.5, 0.9], [10.0, 20.0, 30.0]
    assert _fraction_below(10.0, probs, values) == pytest.approx(0.1)  # not 0.0
    assert _fraction_below(30.0, probs, values) == pytest.approx(0.9)  # not 1.0
    assert _fraction_below(9.0, probs, values) == pytest.approx(0.0)  # strictly below
    assert _fraction_below(31.0, probs, values) == pytest.approx(1.0)  # strictly above
    assert _fraction_below(20.0, probs, values) == pytest.approx(0.5)


# --- three-valued logic -----------------------------------------------------------

_NULLS = {"x": 0.30}


def test_a_predicate_and_its_negation_lose_the_null_rows():
    p = bt.col("x") == 5
    assert _s(p, nulls=_NULLS) + _s(~p, nulls=_NULLS) == pytest.approx(1.0 - _NULLS["x"])


def test_inequality_drops_nulls():
    # `x != 5` is TRUE only where x is non-null and unequal.
    assert _s(bt.col("x") != 5, nulls=_NULLS) == pytest.approx(1.0 - 0.30 - 0.1)


def test_is_null_uses_the_measured_null_fraction():
    assert _s(bt.col("x").is_null(), nulls=_NULLS) == pytest.approx(0.30)
    assert _s(bt.col("x").is_not_null(), nulls=_NULLS) == pytest.approx(0.70)


def test_is_null_is_two_valued_so_its_negation_is_exact():
    # `IS NULL` never evaluates to NULL, so the complement is exact (no null mass lost).
    p = bt.col("x").is_null()
    assert _s(p, nulls=_NULLS) + _s(~p, nulls=_NULLS) == pytest.approx(1.0)


def test_an_unmeasured_null_count_subtracts_no_mass():
    """Unknown null mass is 0, not the `null_selectivity` prior.

    An unmeasured column is usually null-free, and guessing a 5% null mass would shrink
    every negation's estimate — under-budgeting memory — on no evidence at all. So the
    complement stays exactly what it was before 3-valued logic was modelled.
    """
    assert _s(bt.col("x") != 5) == pytest.approx(1.0 - 0.1)
    assert _s(~(bt.col("x") == 5)) == pytest.approx(1.0 - 0.1)


def test_is_null_keeps_its_own_prior_when_unmeasured():
    # `IS NULL` has nothing else to say cold, so it still reports the configured prior.
    assert _s(bt.col("x").is_null()) == pytest.approx(_CFG.null_selectivity)
    assert _s(bt.col("x").is_not_null()) == pytest.approx(1.0 - _CFG.null_selectivity)


# --- IN lists ---------------------------------------------------------------------


def test_in_list_is_the_sum_of_mutually_exclusive_equalities():
    # `x = 1` and `x = 2` cannot both hold, so the union is the sum: 3/10.
    assert _s(InList(bt.col("x"), (1, 2, 3))) == pytest.approx(0.3)


def test_in_list_deduplicates_its_literals():
    assert _s(InList(bt.col("x"), (1, 1, 2))) == pytest.approx(0.2)


def test_empty_in_list_matches_nothing():
    assert _s(InList(bt.col("x"), ())) == pytest.approx(0.0)


def test_in_list_is_clamped_to_one():
    assert _s(InList(bt.col("x"), tuple(range(50)))) == pytest.approx(1.0)


def test_in_list_uses_measured_frequencies_for_skewed_values():
    mcv = {"x": {"1": 0.5}}  # value 1 holds half the rows
    got = sel(InList(bt.col("x"), (1, 2)), _NDV, _CFG, None, mcv)
    assert got == pytest.approx(0.5 + 0.1)  # measured skew + uniform for the other


# --- most-common-value lookup is type-tolerant ------------------------------------


def test_mcv_lookup_matches_across_numeric_spellings():
    # The table is keyed by `str(measured_value)`; an int column stores "5" while a `5.0`
    # literal renders as "5.0". The lookup must not miss on exactly the skewed values.
    int_table = {"x": {"5": 0.42}}
    float_table = {"x": {"5.0": 0.42}}
    assert sel(bt.col("x") == 5.0, _NDV, _CFG, None, int_table) == pytest.approx(0.42)
    assert sel(bt.col("x") == 5, _NDV, _CFG, None, float_table) == pytest.approx(0.42)


def test_bool_literal_does_not_match_an_integer_mcv_entry():
    # `True` is an `int` subclass; it must not collide with the value 1.
    assert sel(bt.col("x") == True, _NDV, _CFG, None, {"x": {"1": 0.9}}) != 0.9  # noqa: E712
