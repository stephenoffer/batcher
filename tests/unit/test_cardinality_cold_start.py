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
    sel = predicate_selectivity(Binary("le", Col("d"), Lit(25)), {}, CFG, bounds=_bounds(0, 100))
    assert sel == pytest.approx(0.25, abs=1e-9)


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
