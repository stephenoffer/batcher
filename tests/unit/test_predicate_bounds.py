"""Conjuncts on one column are priced against what the others already prove.

`d_year IN (2000..2003) AND d_year >= 2000 AND d_year <= 2003` keeps the rows the `IN` keeps.
Multiplying three selectivities priced the two implied ranges against the table's whole
`[1900, 2100]` span, a further 4x cut, and TPC-DS q23's date side read 41 dates for 1,461.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.api.source_stats import build_estimator
from batcher.kyber.stats.predicate_bounds import bounded_by_predicate, without_implied_ranges
from batcher.plan.expr_ir import Binary, Col, InList, Lit
from batcher.plan.stats import ColumnStat

pytestmark = pytest.mark.unit


def _calendar() -> bt.Dataset:
    return bt.from_pydict({"year": [1900 + i // 10 for i in range(2010)]})


def _rows(ds: bt.Dataset) -> float:
    return build_estimator(ds._sources, None).estimate(ds._plan).rows


def test_implied_ranges_do_not_cut_the_estimate_again():
    base = _calendar()
    listed = base.filter(bt.col("year").is_in([2000, 2001, 2002, 2003]))
    stacked = base.filter(
        bt.col("year").is_in([2000, 2001, 2002, 2003])
        & (bt.col("year") >= 2000)
        & (bt.col("year") <= 2003)
    )
    assert _rows(stacked) == pytest.approx(_rows(listed))
    assert 20 <= _rows(stacked) <= 60  # 40 rows actually survive


def test_a_range_the_point_set_does_not_imply_is_kept():
    conjuncts = [InList(Col("y"), (2000, 2003)), Binary("ge", Col("y"), Lit(2002))]
    assert len(without_implied_ranges(conjuncts)) == 2
    implied = [InList(Col("y"), (2000, 2003)), Binary("le", Col("y"), Lit(2003))]
    assert len(without_implied_ranges(implied)) == 1


def test_a_filtered_columns_bounds_tighten_to_its_predicate():
    stat = ColumnStat(min=1900, max=2100)
    predicate = Binary("and", Binary("ge", Col("y"), Lit(1998)), Binary("lt", Col("y"), Lit(2001)))
    assert (
        bounded_by_predicate(stat, predicate).min,
        bounded_by_predicate(stat, predicate).max,
    ) == (
        1998,
        2001,
    )
    listed = bounded_by_predicate(stat, InList(Col("y"), (1950, 2010)))
    assert (listed.min, listed.max) == (1950, 2010)


def test_a_literal_of_another_type_never_becomes_a_bound():
    stat = ColumnStat(min=1.5, max=9.5)
    assert bounded_by_predicate(stat, Binary("ge", Col("y"), Lit(3))) is stat


def test_mixed_literal_types_on_one_column_are_not_compared() -> None:
    conjuncts = [
        Binary("eq", Col("x"), Lit(1)),
        Binary("eq", Col("x"), Lit("1")),
        Binary("ge", Col("x"), Lit(0)),
    ]
    # No common order between 1 and "1": nothing may be pruned, and nothing may raise.
    assert without_implied_ranges(conjuncts) == conjuncts
