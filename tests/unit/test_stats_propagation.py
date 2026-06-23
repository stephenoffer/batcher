"""Column-statistics propagation through `StatsEstimator`.

Pins both the propagation rules (which operators carry min/max/null/ndv, and
how) and the provenance firewall the metadata-answer layer rests on: a `Filter`
never yields an EXACT row count, min/max carried through a row-shrinking operator
is never EXACT, and a global aggregate derives an answer only from EXACT inputs.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col, count, lit
from batcher.kyber.stats import StatsEstimator
from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance


def _exact_source(rows: int, **cols: ColumnStat) -> SourceStatistics:
    return SourceStatistics(row_count=rows, columns=cols, exact_rows=True)


def _ds():
    return bt.from_arrow(pa.table({"x": list(range(100)), "y": list(range(100))}))


def _est(ds, src_stats):
    return StatsEstimator(ds._sources, source_stats=[src_stats])


def test_scan_seeds_exact_column_stats():
    ds = _ds()
    src = _exact_source(100, x=ColumnStat(min=0, max=99, null_count=0, provenance=Provenance.EXACT))
    rs = _est(ds, src).estimate(ds._plan)
    assert rs.rows == 100 and rs.rows_exact
    assert rs.column("x").max == 99 and rs.column("x").provenance is Provenance.EXACT


def test_project_col_passthrough_and_literal_exact():
    ds = _ds().select(a=col("x"), b=lit(7))
    src = _exact_source(100, x=ColumnStat(min=0, max=99, provenance=Provenance.EXACT))
    rs = _est(ds, src).estimate(ds._plan)
    assert rs.column("a").max == 99 and rs.column("a").provenance is Provenance.EXACT
    b = rs.column("b")
    assert b.min == 7 and b.max == 7 and b.ndv == 1 and b.provenance is Provenance.EXACT


def test_filter_never_exact_rows_and_bounds_downgraded():
    ds = _ds().filter(col("x") > lit(50))
    src = _exact_source(100, x=ColumnStat(min=0, max=99, null_count=0, provenance=Provenance.EXACT))
    rs = _est(ds, src).estimate(ds._plan)
    assert not rs.rows_exact  # the firewall: filtered count is never exact
    # min/max survive as bounds but are no longer trusted as exact extremes.
    assert rs.column("x").max == 99
    assert rs.column("x").provenance is not Provenance.EXACT


def test_sort_preserves_exact_value_set_and_records_order():
    ds = _ds().sort("x")
    src = _exact_source(100, x=ColumnStat(min=0, max=99, null_count=0, provenance=Provenance.EXACT))
    rs = _est(ds, src).estimate(ds._plan)
    assert rs.rows == 100 and rs.rows_exact
    assert rs.column("x").provenance is Provenance.EXACT  # reordering preserves values
    assert rs.sorted_by == ("x",)


def test_limit_caps_rows_exact_when_child_exact():
    ds = _ds().limit(10)
    src = _exact_source(100, x=ColumnStat(min=0, max=99, provenance=Provenance.EXACT))
    rs = _est(ds, src).estimate(ds._plan)
    assert rs.rows == 10 and rs.rows_exact
    # but the value bounds are no longer exact extremes (a prefix may drop them).
    assert rs.column("x").provenance is not Provenance.EXACT


def test_union_all_sums_rows_and_merges_bounds():
    a = bt.from_arrow(pa.table({"x": [1, 2, 3]}))
    b = bt.from_arrow(pa.table({"x": [4, 5]}))
    ds = a.union(b)
    sa = _exact_source(3, x=ColumnStat(min=1, max=3, null_count=0, provenance=Provenance.EXACT))
    sb = _exact_source(2, x=ColumnStat(min=4, max=5, null_count=0, provenance=Provenance.EXACT))
    rs = StatsEstimator(ds._sources, source_stats=[sa, sb]).estimate(ds._plan)
    assert rs.rows == 5 and rs.rows_exact
    assert rs.column("x").min == 1 and rs.column("x").max == 5
    assert rs.column("x").provenance is Provenance.EXACT


def test_global_aggregate_derivations_exact():
    ds = _ds().agg(c=count(), mx=col("x").max(), mn=col("x").min())
    src = _exact_source(100, x=ColumnStat(min=0, max=99, null_count=0, provenance=Provenance.EXACT))
    rs = _est(ds, src).estimate(ds._plan)
    assert rs.rows == 1 and rs.rows_exact
    assert rs.column("c").min == 100  # count(*) = exact rows
    assert rs.column("mx").min == 99 and rs.column("mn").min == 0


def test_count_col_uses_null_count():
    ds = _ds().agg(c=col("x").count())
    src = _exact_source(
        100, x=ColumnStat(min=0, max=99, null_count=10, provenance=Provenance.EXACT)
    )
    rs = _est(ds, src).estimate(ds._plan)
    assert rs.column("c").min == 90  # rows - null_count


def test_count_distinct_only_from_exact_ndv():
    ds = _ds().agg(n=col("x").n_unique())
    # SKETCH ndv must NOT be answerable (HLL is approximate).
    sketch = _exact_source(100, x=ColumnStat(ndv=42, provenance=Provenance.SKETCH))
    rs_sketch = _est(ds, sketch).estimate(ds._plan)
    assert "n" not in rs_sketch.columns

    exact = _exact_source(100, x=ColumnStat(ndv=42, provenance=Provenance.EXACT))
    rs_exact = _est(ds, exact).estimate(ds._plan)
    assert rs_exact.column("n").min == 42


def test_learned_ndv_never_taints_exact_footer_column():
    # Regression: a learned (approximate) ndv must not be merged into an EXACT
    # footer column, or count_distinct would wrongly answer from an HLL estimate.
    ds = _ds().agg(n=col("x").n_unique())
    src = _exact_source(100, x=ColumnStat(min=0, max=99, null_count=0, provenance=Provenance.EXACT))
    learned = {"__column_ndv__": {"x": 42.0}}  # a stale/approximate distinct count
    rs = StatsEstimator(ds._sources, learned, source_stats=[src]).estimate(ds._plan)
    # The exact footer column keeps min/max EXACT and gains NO exact ndv, so the
    # count_distinct output is not derivable from metadata (must execute).
    assert "n" not in rs.columns


def test_estimate_unaffected_when_no_source_stats():
    # Back-compat: with no source_stats the estimator falls back to row_count().
    ds = _ds()
    rs = StatsEstimator(ds._sources).estimate(ds._plan)
    assert rs.rows == 100 and rs.rows_exact
    assert rs.columns == {}
