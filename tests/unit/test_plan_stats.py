"""Neutral stats algebra: `Provenance`, `weakest`, `ColumnStat`, `RelStats`.

These pin the firewall the metadata-first layer rests on: trust only ever
composes downward (`weakest`), and `EXACT` is the sole provenance that may
answer a query.
"""

from __future__ import annotations

import itertools

import pytest

from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance, RelStats, weakest


def test_provenance_order_exact_is_strongest():
    assert min(Provenance) is Provenance.EXACT
    assert max(Provenance) is Provenance.DEFAULT
    assert Provenance.EXACT.is_exact
    assert not Provenance.SKETCH.is_exact
    assert not Provenance.LEARNED.is_exact
    assert not Provenance.DEFAULT.is_exact


@pytest.mark.parametrize("a,b", list(itertools.product(Provenance, repeat=2)))
def test_weakest_is_max_and_never_upgrades(a: Provenance, b: Provenance):
    result = weakest(a, b)
    # The combiner can never produce a stronger (smaller) provenance than either input.
    assert result.value >= a.value
    assert result.value >= b.value
    assert result is (a if a.value >= b.value else b)


def test_weakest_empty_is_default():
    assert weakest() is Provenance.DEFAULT


def test_weakest_single():
    assert weakest(Provenance.SKETCH) is Provenance.SKETCH


def test_column_stat_downgrade_weakens_but_keeps_values():
    c = ColumnStat(min=1, max=9, null_count=0, ndv=9, provenance=Provenance.EXACT)
    d = c.downgrade(Provenance.DEFAULT)
    assert (d.min, d.max, d.null_count, d.ndv) == (1, 9, 0, 9)
    assert d.provenance is Provenance.DEFAULT
    # Downgrading to a stronger floor than current is a no-op on provenance.
    assert c.downgrade(Provenance.EXACT).provenance is Provenance.EXACT


def test_relstats_column_default_when_absent():
    rs = RelStats(rows=10, provenance=Provenance.EXACT)
    assert rs.rows_exact
    empty = rs.column("missing")
    assert empty.min is None and empty.provenance is Provenance.DEFAULT


def test_source_statistics_to_relstats_exact():
    ss = SourceStatistics(
        row_count=100,
        columns={"x": ColumnStat(min=0, max=99, provenance=Provenance.EXACT)},
        sorted_by=("x",),
    )
    rs = ss.to_relstats(default_rows=1e12)
    assert rs.rows == 100 and rs.rows_exact
    assert rs.sorted_by == ("x",)
    assert rs.column("x").max == 99


def test_source_statistics_estimate_is_not_exact():
    ss = SourceStatistics(row_count=100, exact_rows=False)
    rs = ss.to_relstats(default_rows=1e12)
    assert rs.rows == 100
    assert not rs.rows_exact
    assert rs.provenance is Provenance.SKETCH


def test_source_statistics_unknown_rows_uses_default():
    ss = SourceStatistics()
    rs = ss.to_relstats(default_rows=1234.0)
    assert rs.rows == 1234.0 and not rs.rows_exact
    assert SourceStatistics(row_count=0).is_empty()
