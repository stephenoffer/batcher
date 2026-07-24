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
    from batcher.kyber.stats.selectivity.scalars import _fraction_below

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
    # The listed value takes its measured 0.5; value 2 is *not* a most-common value, so it
    # draws from the residual — the 0.5 of mass the table leaves, shared by the other 9 of
    # the 10 distinct values — rather than the whole column's uniform 1/10.
    assert got == pytest.approx(0.5 + 0.5 / 9)


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


# --- same-column ranges are one interval, not two independent conjuncts -----------
#
# `x BETWEEN lo AND hi` desugars to `x >= lo AND x <= hi`. The two bounds carve a single
# interval out of one distribution, so their joint mass is the CDF difference
# `F(hi) - F(lo)` — not the exponential-backoff of two "loosely dependent" conjuncts,
# which roughly doubles the estimate. A bounded range is the most common selective filter
# in analytics, so this identity gates real join-order and build-side decisions.


def test_closed_interval_is_the_cdf_difference():
    # `2 <= x <= 8` keeps values 2..8: F(8) - F(2 strict below) = 0.8 - 0.2 = 0.6.
    got = _s((bt.col("x") >= 2) & (bt.col("x") <= 8))
    assert got == pytest.approx(_s(bt.col("x") <= 8) - _s(bt.col("x") < 2))


def test_open_interval_excludes_both_boundary_masses():
    # `2 < x < 8`: F(8 strict) - F(2) = (0.8 - 0.1) - 0.2 = 0.5.
    got = _s((bt.col("x") > 2) & (bt.col("x") < 8))
    assert got == pytest.approx(_s(bt.col("x") < 8) - _s(bt.col("x") <= 2))


def test_interval_is_tighter_than_backoff_of_its_bounds():
    # The whole point: backoff over-estimates a bounded range. The interval must be smaller.
    interval = _s((bt.col("x") >= 2) & (bt.col("x") <= 8))
    lo, hi = sorted((_s(bt.col("x") >= 2), _s(bt.col("x") <= 8)))
    backoff = lo * hi**0.5
    assert interval < backoff


def test_redundant_same_side_bounds_take_the_tightest():
    # `x > 2 AND x > 5` is just `x > 5`; the looser bound adds nothing.
    assert _s((bt.col("x") > 2) & (bt.col("x") > 5)) == pytest.approx(_s(bt.col("x") > 5))


def test_flipped_interval_bounds_are_equivalent():
    from batcher.plan.expr_ir import Binary, Col, Lit

    flipped = Binary("le", Lit(2), Col("x")) & Binary("ge", Lit(8), Col("x"))  # 2<=x AND 8>=x
    assert _s(flipped) == pytest.approx(_s((bt.col("x") >= 2) & (bt.col("x") <= 8)))


def test_empty_interval_is_non_negative():
    # `x >= 8 AND x <= 2` is unsatisfiable; the estimate must clamp to 0, never go negative.
    assert _s((bt.col("x") >= 8) & (bt.col("x") <= 2)) == pytest.approx(0.0)


def test_interval_with_extra_conjunct_still_combines():
    # `2 <= x <= 8 AND (x = 5)` — the interval collapses, the equality stays its own term.
    both = _s((bt.col("x") >= 2) & (bt.col("x") <= 8) & (bt.col("x") == 5))
    interval = _s((bt.col("x") >= 2) & (bt.col("x") <= 8))
    assert 0.0 < both <= interval


def test_interval_without_stats_falls_back_to_backoff():
    # No quantiles and no bounds → the interval is uncomputable, so behaviour is unchanged:
    # each bound is the flat `range_selectivity`, combined by exponential backoff (s * s^0.5).
    plain = sel((bt.col("y") >= 2) & (bt.col("y") <= 8), {}, _CFG, None, None, None)
    s = _CFG.range_selectivity
    assert plain == pytest.approx(s * s**0.5)


# --- column = column is the Selinger containment case, not the flat default -------


def test_column_equality_uses_one_over_max_ndv():
    # `a = b` matches ~1/max(d_a, d_b) of rows under uniformity/containment.
    ndv = {"a": 100.0, "b": 10.0}
    assert sel(bt.col("a") == bt.col("b"), ndv, _CFG) == pytest.approx(1.0 / 100.0)


def test_column_equality_over_estimates_when_one_ndv_missing():
    # Only `a` known: max >= d_a, so 1/d_a is the safe over-estimate (never OOM the build).
    assert sel(bt.col("a") == bt.col("b"), {"a": 100.0}, _CFG) == pytest.approx(1.0 / 100.0)


def test_column_equality_falls_back_without_any_ndv():
    assert sel(bt.col("a") == bt.col("b"), {}, _CFG) == pytest.approx(_CFG.eq_selectivity)


# --- LIKE selectivity is read from where the wildcards fall -----------------------


def _like(colname, pat, fn="like"):
    from batcher.plan.expr_ir import Col, StrFunc

    return StrFunc(fn=fn, input=Col(colname), pattern=pat)


def test_like_no_wildcard_is_equality():
    # `col LIKE 'FOO'` is exact equality → 1/ndv, not the blunt substring prior.
    assert sel(_like("x", "FOO"), _NDV, _CFG) == pytest.approx(1.0 / _NDV["x"])


def test_like_no_wildcard_uses_measured_skew():
    assert sel(_like("x", "FOO"), _NDV, _CFG, None, {"x": {"FOO": 0.6}}) == pytest.approx(0.6)


def test_like_anchored_prefix_is_prefix_selectivity():
    assert sel(_like("x", "AIR%"), _NDV, _CFG) == pytest.approx(_CFG.prefix_selectivity)


def test_like_anchored_suffix_is_prefix_selectivity():
    assert sel(_like("x", "%ing"), _NDV, _CFG) == pytest.approx(_CFG.prefix_selectivity)


def test_like_unanchored_substring_stays_substring_selectivity():
    assert sel(_like("x", "%foo%"), _NDV, _CFG) == pytest.approx(_CFG.substring_selectivity)


def test_like_interior_wildcard_is_substring():
    assert sel(_like("x", "a%b"), _NDV, _CFG) == pytest.approx(_CFG.substring_selectivity)


def test_like_underscore_wildcard_is_substring():
    # `_` matches a single char; an anchored `A_C%` isn't a plain prefix, so stay blunt.
    assert sel(_like("x", "A_C%"), _NDV, _CFG) == pytest.approx(_CFG.substring_selectivity)


def test_ilike_is_analyzed_like_like():
    assert sel(_like("x", "AIR%", "ilike"), _NDV, _CFG) == pytest.approx(_CFG.prefix_selectivity)


# --- date-part predicates over a bounded uniform domain ---------------------------


def _dpart(fn, colname="d"):
    from batcher.plan.expr_ir import Col, DateFunc

    return DateFunc(fn=fn, input=Col(colname))


def test_date_part_equality_is_one_over_period():
    assert sel(_dpart("month") == 6, {}, _CFG) == pytest.approx(1.0 / 12.0)
    assert sel(_dpart("quarter") == 3, {}, _CFG) == pytest.approx(1.0 / 4.0)
    assert sel(_dpart("day_of_week") == 0, {}, _CFG) == pytest.approx(1.0 / 7.0)


def test_date_part_range_uses_the_discrete_domain():
    # month(d) <= 6 keeps 6/12; the strict `< 6` drops the boundary month (5/12).
    assert sel(_dpart("month") <= 6, {}, _CFG) == pytest.approx(6.0 / 12.0)
    assert sel(_dpart("month") < 6, {}, _CFG) == pytest.approx(5.0 / 12.0)
    assert sel(_dpart("hour", "ts") < 9, {}, _CFG) == pytest.approx(9.0 / 24.0)


def test_date_part_lt_and_ge_partition_the_domain():
    below = sel(_dpart("month") < 6, {}, _CFG)
    at_or_above = sel(_dpart("month") >= 6, {}, _CFG)
    assert below + at_or_above == pytest.approx(1.0)


def test_date_part_in_list_is_k_over_period():
    from batcher.plan.expr_ir import InList

    assert sel(InList(_dpart("month"), (6, 7, 8)), {}, _CFG) == pytest.approx(3.0 / 12.0)


def test_date_part_string_field_equality_still_uses_period():
    # monthname → a string, so no range, but equality is still 1 of 12.
    assert sel(_dpart("monthname") == "June", {}, _CFG) == pytest.approx(1.0 / 12.0)


def test_unbounded_date_part_is_left_to_the_default():
    # year has no bounded domain here (the sargable rewrite handles it); stays the default.
    assert sel(_dpart("year") == 1995, {}, _CFG) == pytest.approx(_CFG.eq_selectivity)


# --- a literal outside the column's bounds matches nothing -------------------------
#
# Bounds are a valid (loose) superset of the actual values, so a literal beyond them
# provably cannot appear. The range path already estimates `col > max` at 0; equality must
# agree, or a stale/partition-mismatched `col = v` keeps 1/ndv of the table it can't match.

_BOUNDS = {"x": (0.0, 10.0)}


def test_equality_outside_bounds_is_zero():
    assert sel(bt.col("x") == 999, _NDV, _CFG, None, None, _BOUNDS) == pytest.approx(0.0)
    assert sel(bt.col("x") == -5, _NDV, _CFG, None, None, _BOUNDS) == pytest.approx(0.0)


def test_equality_inside_bounds_is_unaffected():
    assert sel(bt.col("x") == 5, _NDV, _CFG, None, None, _BOUNDS) == pytest.approx(1.0 / 10.0)


def test_inequality_outside_bounds_keeps_everything():
    # `x != 999` is TRUE for every (non-null) row when 999 can't occur.
    assert sel(bt.col("x") != 999, _NDV, _CFG, None, None, _BOUNDS) == pytest.approx(1.0)


def test_equality_without_bounds_is_unchanged():
    assert sel(bt.col("x") == 999, _NDV, _CFG) == pytest.approx(1.0 / _NDV["x"])


def test_in_list_drops_out_of_bounds_values():
    # `x IN (5, 999)` over x ∈ [0, 10]: only 5 can occur, so 1/ndv, not 2/ndv.
    got = sel(InList(bt.col("x"), (5, 999)), _NDV, _CFG, None, None, _BOUNDS)
    assert got == pytest.approx(1.0 / _NDV["x"])


def test_in_list_all_out_of_bounds_is_zero():
    got = sel(InList(bt.col("x"), (999, 1000)), _NDV, _CFG, None, None, _BOUNDS)
    assert got == pytest.approx(0.0)


# --- a bare boolean column predicate keeps its TRUE rows ---------------------------


def test_bare_boolean_column_uses_measured_true_frequency():
    assert sel(bt.col("flag"), {}, _CFG, None, {"flag": {"True": 0.05}}) == pytest.approx(0.05)


def test_bare_boolean_column_defaults_when_unmeasured():
    assert sel(bt.col("flag"), {}, _CFG) == pytest.approx(_CFG.default_filter_selectivity)


def test_negated_boolean_column_complements_the_true_fraction():
    assert sel(~bt.col("flag"), {}, _CFG, None, {"flag": {"True": 0.05}}) == pytest.approx(0.95)


# --- NOT IN / NOT LIKE drop the null rows (3-valued complement) --------------------


def test_not_in_subtracts_the_null_mass():
    # `col NOT IN (...)` is NULL where col is NULL, so those rows are dropped like `NOT p`.
    got = sel(~InList(bt.col("x"), (1, 2, 3)), {"x": 10.0}, _CFG, None, None, None, {"x": 0.3})
    assert got == pytest.approx((1.0 - 0.3) - 0.3)


def test_not_like_subtracts_the_null_mass():
    got = sel(~bt.col("x").str.contains("a"), {}, _CFG, None, None, None, {"x": 0.3})
    assert got == pytest.approx((1.0 - 0.3) - _CFG.substring_selectivity)


def test_not_in_without_null_info_is_unchanged():
    got = sel(~InList(bt.col("x"), (1, 2, 3)), {"x": 10.0}, _CFG)
    assert got == pytest.approx(1.0 - 0.3)


# --- OR of same-column equalities sums (disjoint values), like an IN list ----------


def test_or_of_same_column_equalities_sums():
    # `x = 1 OR x = 2` selects disjoint values, so the union is the sum 2/10 — the same as
    # `x IN (1, 2)`, not the inclusion-exclusion `a + b - a*b` which undercounts.
    assert _s((bt.col("x") == 1) | (bt.col("x") == 2)) == pytest.approx(0.2)
    assert _s((bt.col("x") == 1) | (bt.col("x") == 2) | (bt.col("x") == 3)) == pytest.approx(0.3)


def test_or_of_same_column_equalities_matches_in_list():
    ored = _s((bt.col("x") == 1) | (bt.col("x") == 2))
    inlist = _s(InList(bt.col("x"), (1, 2)))
    assert ored == pytest.approx(inlist)


def test_or_across_columns_is_the_independent_union():
    # Different columns → assume independence: 1 - (1-a)(1-b), unchanged from before.
    a, b = _s(bt.col("x") == 1), _s(bt.col("y") == 2)
    assert _s((bt.col("x") == 1) | (bt.col("y") == 2)) == pytest.approx(a + b - a * b)


# --- COALESCE(x, c) = v (the fill_null cleaning shape) is not the blunt 0.5 --------


def _coalesce_eq(fill, v):
    from batcher.plan.expr_ir import Binary, Coalesce, Col, Lit

    return Binary("eq", Coalesce(inputs=[Col("x"), Lit(fill)]), Lit(v))


def test_coalesce_equality_when_value_differs_from_fill():
    # coalesce(x, 0) = 5 matches only where x = 5 (non-null) → 1/ndv.
    assert sel(_coalesce_eq(0, 5), _NDV, _CFG) == pytest.approx(1.0 / _NDV["x"])


def test_coalesce_equality_when_value_equals_fill_adds_null_mass():
    # coalesce(x, 0) = 0 also matches the null rows (which the fill maps to 0).
    got = sel(_coalesce_eq(0, 0), _NDV, _CFG, None, None, None, {"x": 0.2})
    assert got == pytest.approx(1.0 / _NDV["x"] + 0.2)


def test_coalesce_equality_is_far_below_the_default():
    # The whole point: it no longer hits default_filter_selectivity (0.5).
    assert sel(_coalesce_eq(0, 5), _NDV, _CFG) < _CFG.default_filter_selectivity


# --- COALESCE as a predicate (null-safe equality desugars to it) -------------------


def test_coalesce_with_false_fill_is_exactly_its_argument():
    from batcher.plan.expr_ir import Coalesce, Lit

    # `coalesce(p, FALSE)` is TRUE exactly where `p` is TRUE — the selectivity is identical.
    p = bt.col("x") == 5
    guarded = Coalesce(inputs=[p, Lit(False)])
    assert _s(guarded) == pytest.approx(_s(p))


def test_null_safe_equality_is_not_a_coin_flip():
    # `eq_missing` desugars to `coalesce(x = y, FALSE) OR (x IS NULL AND y IS NULL)`. It used
    # to collapse to ~0.5 (the blunt default for the COALESCE); it should track the equality.
    got = sel(bt.col("x").eq_missing(bt.col("y")), {"x": 10.0, "y": 10.0}, _CFG)
    assert got < 0.2  # far below the 0.5 default
    assert got >= 1.0 / 10.0  # at least the col=col equality mass


def test_coalesce_with_non_false_fill_is_bounded_above():
    from batcher.plan.expr_ir import Coalesce, Lit

    # A TRUE fill can only add back the rows where `p` is NULL, so it stays within
    # sel(p) + null_mass(p) and never exceeds 1.
    p = bt.col("x") == 5
    guarded = Coalesce(inputs=[p, Lit(True)])
    got = sel(guarded, _NDV, _CFG, None, None, None, {"x": 0.3})
    assert _s(p) <= got <= 1.0


# --- CASE as a predicate, and constant predicates ---------------------------------


def test_constant_predicates_keep_all_or_nothing():
    from batcher.plan.expr_ir import Lit

    assert _s(Lit(True)) == pytest.approx(1.0)
    assert _s(Lit(False)) == pytest.approx(0.0)


def test_boolean_case_collapses_to_its_condition():
    from batcher.plan.expr_ir import Case, Lit

    cond = bt.col("x") == 5
    assert _s(Case(branches=[(cond, Lit(True))], otherwise=Lit(False))) == pytest.approx(_s(cond))


def test_inverted_boolean_case_is_the_complement():
    from batcher.plan.expr_ir import Case, Lit

    cond = bt.col("x") == 5
    got = _s(Case(branches=[(cond, Lit(False))], otherwise=Lit(True)))
    assert got == pytest.approx(1.0 - _s(cond))


def test_case_branches_partition_and_stay_a_probability():
    from batcher.plan.expr_ir import Case, Lit

    # Two branches plus an otherwise — the result must stay in [0, 1].
    got = _s(
        Case(
            branches=[(bt.col("x") == 1, Lit(True)), (bt.col("x") == 2, Lit(False))],
            otherwise=Lit(True),
        )
    )
    assert 0.0 <= got <= 1.0
