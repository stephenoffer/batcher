"""Cold-start cardinality: the estimates a plan gets before anything has been measured.

Kyber's join ordering is only as good as its first-run estimates, and three of them used
to be blind guesses:

* a range predicate against a column whose exact `min`/`max` were already known fell back
  to the Selinger 1/3 constant;
* a string-pattern (`LIKE '%x%'`) or `IN` predicate fell back to the 0.5 "unknown filter"
  constant;
* a join above a join lost its key `ndv` (it was dropped unconditionally), so
  `_estimate_join` fell back to `max(|L|, |R|)`.

Together those chose TPC-H Q9's plan: join `lineitem` to `partsupp` and `orders` first —
two multi-gigabyte intermediates — and only then apply the 5%-selective `part` filter.
These tests pin the estimates, not the plan, so they stay meaningful as rules change.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

from batcher.config import CardinalityConfig
from batcher.kyber.stats.columns import join_columns
from batcher.kyber.stats.selectivity import predicate_selectivity
from batcher.plan.expr_ir import Binary, Col, InList, Lit, StrFunc
from batcher.plan.logical import Join, Scan
from batcher.plan.logical.join import JoinOutputCol
from batcher.plan.schema import SchemaRef
from batcher.plan.stats import ColumnStat, Provenance, RelStats

pytestmark = pytest.mark.unit

CFG = CardinalityConfig()


def _bounds(lo, hi):
    return {"d": (lo, hi)}


# --- range selectivity from exact min/max ------------------------------------


def test_date_range_interpolates_between_exact_bounds():
    """`d < 1995-01-01` over a 1992..1999 span keeps ~3/7 of the rows, not 1/3."""
    span = _bounds(dt.date(1992, 1, 1), dt.date(1999, 1, 1))
    sel = predicate_selectivity(
        Binary("lt", Col("d"), Lit(dt.date(1995, 1, 1))), {}, CFG, bounds=span
    )
    assert sel == pytest.approx(3 / 7, abs=0.01)


def test_range_beyond_the_bounds_is_saturated_not_guessed():
    span = _bounds(dt.date(1992, 1, 1), dt.date(1999, 1, 1))
    below = predicate_selectivity(
        Binary("lt", Col("d"), Lit(dt.date(1990, 1, 1))), {}, CFG, bounds=span
    )
    above = predicate_selectivity(
        Binary("lt", Col("d"), Lit(dt.date(2001, 1, 1))), {}, CFG, bounds=span
    )
    assert below == 0.0
    assert above == 1.0


def test_numeric_range_interpolates_between_exact_bounds():
    """An *integer* column's range is counted, not interpolated as a continuum.

    `d <= 25` over integer bounds `[0, 100]` keeps 26 of the 101 values the range contains, not
    the 25/100 a continuous reading gives. The two barely differ over a span this wide; they
    differ a great deal at the ends and on a narrow range, which is why the count is what
    `_fraction_below_bounds` uses for a discrete column — the same form
    `_date_part_range_selectivity` already used for a bounded field like `month`.
    """
    sel = predicate_selectivity(Binary("le", Col("d"), Lit(25)), {}, CFG, bounds=_bounds(0, 100))
    assert sel == pytest.approx(26 / 101, abs=1e-9)


def test_a_float_column_keeps_the_continuous_reading():
    """A continuous column must NOT be counted: there is no "next" value to divide by.

    Deciding discreteness from whether the bounds *happen* to be whole numbers would catch this
    one — `0.0` and `100.0` both are — and then read a `Float64` column as holding 101 values.
    The type is what decides, so this stays at the interpolated 0.25.
    """
    sel = predicate_selectivity(
        Binary("le", Col("d"), Lit(25.0)), {}, CFG, bounds=_bounds(0.0, 100.0)
    )
    assert sel == pytest.approx(0.25, abs=1e-9)


def test_a_predicate_at_the_minimum_no_longer_estimates_zero_rows():
    """`d <= min` matches every row holding the minimum, so it cannot be zero.

    The continuous form answered a flat 0 at the lower bound, unable to tell "below the minimum"
    from "equal to it" — and a zero-row estimate is the worst kind to be wrong by, because
    build-side choice, join order, broadcast sizing and the adaptive gate all read it as "this
    subtree is empty". Over integer bounds `[1, 4]` the answer is one value's worth.
    """
    sel = predicate_selectivity(Binary("le", Col("d"), Lit(1)), {}, CFG, bounds=_bounds(1, 4))
    assert sel == pytest.approx(0.25, abs=1e-9)
    # ...and strictly below the minimum really is zero.
    below = predicate_selectivity(Binary("lt", Col("d"), Lit(1)), {}, CFG, bounds=_bounds(1, 4))
    assert below == 0.0


def test_a_comparison_selectivity_stays_within_zero_and_one():
    """`F` and `eq` come from different estimators, so their difference is not bounded.

    A skewed column whose measured mass at `x` exceeds the interpolated `F(x)` made `lt` come
    back negative and `ge` exceed one, and a negative selectivity propagates as a negative row
    estimate. Both are clamped now, and the mass is also used as a floor on `F` for the two
    comparisons that read it directly, since a CDF is never below the point mass at that value.
    """
    from batcher.kyber.stats.selectivity.leaves import _from_cdf

    for op in ("le", "lt", "gt", "ge"):
        assert 0.0 <= _from_cdf(op, 0.1, 0.4) <= 1.0
        assert 0.0 <= _from_cdf(op, 0.9, 0.05) <= 1.0
    # The floor: `P(v <= x)` is at least `P(v = x)`.
    assert _from_cdf("le", 0.1, 0.4) == pytest.approx(0.4)
    # ...but `lt` is not floored, or it would claim nothing lies below an interior `x`.
    assert _from_cdf("lt", 0.5, 0.25) == pytest.approx(0.25)


def test_range_without_bounds_still_falls_back_to_the_constant():
    sel = predicate_selectivity(Binary("lt", Col("d"), Lit(5)), {}, CFG)
    assert sel == pytest.approx(CFG.range_selectivity)


def test_quantiles_win_over_bounds_when_both_are_known():
    """A learned histogram is sharper than the uniformity assumption; it takes priority."""
    quantiles = {"d": {"probs": [0.0, 0.5, 1.0], "values": [0.0, 90.0, 100.0]}}
    sel = predicate_selectivity(
        Binary("le", Col("d"), Lit(90)), {}, CFG, quantiles=quantiles, bounds=_bounds(0, 100)
    )
    assert sel == pytest.approx(0.5, abs=1e-9)


def test_string_bounds_do_not_interpolate():
    """A non-ordinal literal must not be forced onto the numeric grid."""
    sel = predicate_selectivity(Binary("lt", Col("d"), Lit("m")), {}, CFG, bounds={"d": ("a", "z")})
    assert sel == pytest.approx(CFG.range_selectivity)


# --- string-pattern and IN-list selectivity ----------------------------------


def test_substring_pattern_is_selective_not_a_coin_flip():
    sel = predicate_selectivity(StrFunc("contains", Col("d"), pattern="green"), {}, CFG)
    assert sel == pytest.approx(CFG.substring_selectivity)
    assert sel < CFG.default_filter_selectivity


def test_prefix_pattern_uses_the_prefix_constant():
    sel = predicate_selectivity(StrFunc("starts_with", Col("d"), pattern="ab"), {}, CFG)
    assert sel == pytest.approx(CFG.prefix_selectivity)


def test_in_list_is_k_over_ndv():
    sel = predicate_selectivity(InList(Col("d"), [1, 2, 3]), {"d": 60.0}, CFG)
    assert sel == pytest.approx(3 / 60)


def test_in_list_saturates_at_one():
    sel = predicate_selectivity(InList(Col("d"), list(range(20))), {"d": 4.0}, CFG)
    assert sel == 1.0


def test_in_list_without_ndv_uses_the_equality_constant():
    sel = predicate_selectivity(InList(Col("d"), [1, 2]), {}, CFG)
    assert sel == pytest.approx(2 * CFG.eq_selectivity)


# --- ndv propagation through a join ------------------------------------------


def _join_of() -> Join:
    left = Scan(0, SchemaRef(pa.schema([("lk", pa.int64())])))
    right = Scan(1, SchemaRef(pa.schema([("rk", pa.int64())])))
    return Join(
        left,
        right,
        left_keys=("lk",),
        right_keys=("rk",),
        join_type="inner",
        output=(
            JoinOutputCol(side="left", name="lk", alias="lk"),
            JoinOutputCol(side="right", name="rk", alias="rk"),
        ),
    )


def _stats(ndv: float, rows: float) -> RelStats:
    name = "lk" if rows > 100 else "rk"
    return RelStats(rows, Provenance.EXACT, {name: ColumnStat(ndv=ndv, min=1, max=int(ndv))})


def test_join_carries_ndv_forward_capped_by_output_rows():
    """A join invents no values, so `ndv_out <= min(ndv_in, rows_out)` — and that bound
    is what a join *above* this one needs to estimate its own cardinality.

    For an equi-**key** the bound is tighter still: a value survives only if it appears on
    *both* sides, so the surviving distinct count is at most `min(d_L, d_R)` — 10,000 here,
    not the left side's own 200,000. Both key columns therefore carry the same bound, which
    is what an equality between them means."""
    node = _join_of()
    cols = join_columns(node, _stats(200_000, 6_000_000), _stats(10_000, 10), out_rows=320_000)
    assert cols["lk"].ndv == pytest.approx(10_000)  # containment: min(d_L, d_R)
    assert cols["rk"].ndv == pytest.approx(10_000)


def test_join_ndv_is_capped_by_a_small_output():
    node = _join_of()
    cols = join_columns(node, _stats(200_000, 6_000_000), _stats(10_000, 10), out_rows=500)
    assert cols["lk"].ndv == pytest.approx(500)  # cannot exceed the rows that survive


def test_join_ndv_is_dropped_when_output_rows_are_unknown():
    node = _join_of()
    cols = join_columns(node, _stats(200_000, 6_000_000), _stats(10_000, 10))
    assert cols["lk"].ndv is None


def test_join_never_reports_exact_provenance():
    """A join may drop the extremes, so bounds survive but `EXACT` must not."""
    node = _join_of()
    cols = join_columns(node, _stats(200_000, 6_000_000), _stats(10_000, 10), out_rows=320_000)
    assert cols["lk"].provenance is not Provenance.EXACT
    assert cols["lk"].min == 1  # bounds still carried through
