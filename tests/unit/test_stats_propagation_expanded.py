"""Unit tests for the expanded column-stat propagation and scalar derivations.

These pin the *provenance firewall* around each new shortcut: a derivation is
produced only from `Provenance.EXACT` inputs, a row-shrinking/duplicating operator
downgrades away from EXACT (so it can never wrongly answer an exact terminal), and
a value SQL leaves NULL (all-null/empty group) is not derived. Correctness vs
DuckDB lives in `tests/differential/test_diff_metadata_expanded.py`; this file
proves the shortcut *fires* on EXACT input and *falls back* otherwise.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col, count, lit
from batcher.kyber.stats import StatsEstimator
from batcher.kyber.stats import aggregate_columns as agg_cols
from batcher.kyber.stats import columns as col_prop
from batcher.plan.expr_ir import Col
from batcher.plan.logical import Filter
from batcher.plan.logical.aggregate import SortKeySpec
from batcher.plan.logical.window import Window, WindowFuncSpec
from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance, RelStats


def _ds():
    return bt.from_arrow(pa.table({"x": list(range(6)), "y": list(range(6))}))


def _bool_ds():
    return bt.from_arrow(pa.table({"flag": [True, False, True]}))


def _exact_source(rows: int, **cols: ColumnStat) -> SourceStatistics:
    return SourceStatistics(row_count=rows, columns=cols, exact_rows=True)


def _est(ds, src):
    return StatsEstimator(ds._sources, source_stats=[src])


# --- bool_and / bool_or ---------------------------------------------------
def test_bool_and_or_derive_from_exact_minmax():
    ds = _bool_ds().agg(a=col("flag").bool_and(), o=col("flag").bool_or())
    src = _exact_source(
        3, flag=ColumnStat(min=False, max=True, null_count=0, provenance=Provenance.EXACT)
    )
    rs = _est(ds, src).estimate(ds._plan)
    assert rs.column("a").min is False  # bool_and == (min is True)
    assert rs.column("o").min is True  # bool_or == (max is True)
    assert rs.column("a").provenance is Provenance.EXACT


def test_bool_and_all_true_is_true():
    ds = _bool_ds().agg(a=col("flag").bool_and())
    src = _exact_source(
        3, flag=ColumnStat(min=True, max=True, null_count=0, provenance=Provenance.EXACT)
    )
    assert _est(ds, src).estimate(ds._plan).column("a").min is True


def test_bool_and_not_derived_from_sketch_minmax():
    # A non-EXACT boolean bound must never answer an exact bool_and.
    ds = _bool_ds().agg(a=col("flag").bool_and())
    src = _exact_source(3, flag=ColumnStat(min=False, max=True, provenance=Provenance.SKETCH))
    assert "a" not in _est(ds, src).estimate(ds._plan).columns


def test_bool_and_all_null_group_not_derived():
    ds = _bool_ds().agg(a=col("flag").bool_and())
    src = _exact_source(
        3, flag=ColumnStat(min=True, max=True, null_count=3, provenance=Provenance.EXACT)
    )
    assert "a" not in _est(ds, src).estimate(ds._plan).columns


def test_bool_agg_rejects_non_boolean_column():
    # An integer column's min/max must not be read as a boolean aggregate.
    stat = ColumnStat(min=0, max=5, null_count=0, provenance=Provenance.EXACT)
    child = RelStats(6, Provenance.EXACT, {"x": stat})
    assert agg_cols._derive_scalar_aggregate("bool_and", Col("x"), child) is None


# --- count(col) of a non-null column --------------------------------------
def test_count_non_null_column_is_rows():
    ds = _ds().agg(c=col("x").count())
    src = _exact_source(6, x=ColumnStat(min=0, max=5, null_count=0, provenance=Provenance.EXACT))
    assert _est(ds, src).estimate(ds._plan).column("c").min == 6


# --- sum with the empty-group guard ---------------------------------------
def test_sum_empty_relation_not_derived():
    child = RelStats(
        0, Provenance.EXACT, {"v": ColumnStat(total_sum=0.0, provenance=Provenance.EXACT)}
    )
    assert agg_cols._derive_scalar_aggregate("sum", Col("v"), child) is None


# --- grouped-aggregate group-key propagation ------------------------------
def test_group_key_minmax_exact_ndv_not_claimed():
    ds = _ds().group_by("x").agg(c=count())
    src = _exact_source(
        6, x=ColumnStat(min=0, max=5, null_count=0, ndv=6, provenance=Provenance.EXACT)
    )
    rs = _est(ds, src).estimate(ds._plan)
    key = rs.column("x")
    assert (key.min, key.max) == (0, 5)
    assert key.provenance is Provenance.EXACT  # grouping preserves extremes
    assert key.ndv is None  # group count is only an estimate → not claimed EXACT


def test_group_key_count_distinct_not_answerable_from_group_count():
    # Guard: a grouped key's ndv must NOT let count_distinct answer from the (estimated)
    # number of groups.
    ds = _ds().group_by("x").agg(c=count()).agg(n=col("x").n_unique())
    src = _exact_source(
        6, x=ColumnStat(min=0, max=5, null_count=0, ndv=6, provenance=Provenance.EXACT)
    )
    rs = _est(ds, src).estimate(ds._plan)
    assert "n" not in rs.columns  # falls back to execution


# --- window preserves input column stats ----------------------------------
def test_window_preserves_input_exact_stats():
    ds = _ds()
    src = _exact_source(6, x=ColumnStat(min=0, max=5, null_count=0, provenance=Provenance.EXACT))
    win = Window(
        ds._plan,
        partition_keys=(),
        order_keys=(SortKeySpec(col("x")),),
        functions=(WindowFuncSpec("row_number", None, "rn"),),
    )
    rs = _est(ds, src).estimate(win)
    assert rs.rows == 6 and rs.rows_exact
    assert rs.column("x").provenance is Provenance.EXACT and rs.column("x").max == 5


# --- identity cast carries stats; a value-changing cast drops them --------
def test_identity_cast_carries_stats():
    ds = _ds().select(w=col("x").cast("int64"))
    src = _exact_source(6, x=ColumnStat(min=0, max=5, null_count=0, provenance=Provenance.EXACT))
    rs = _est(ds, src).estimate(ds._plan)
    assert rs.column("w").max == 5 and rs.column("w").provenance is Provenance.EXACT


def test_value_changing_cast_drops_stats():
    ds = _ds().select(w=col("x").cast("float64"))  # int64 -> float64 is not identity
    src = _exact_source(6, x=ColumnStat(min=0, max=5, null_count=0, provenance=Provenance.EXACT))
    assert "w" not in _est(ds, src).estimate(ds._plan).columns


def test_try_cast_never_carries_stats():
    ds = _ds().select(w=col("x").try_cast("int64"))
    src = _exact_source(6, x=ColumnStat(min=0, max=5, null_count=0, provenance=Provenance.EXACT))
    assert "w" not in _est(ds, src).estimate(ds._plan).columns


# --- constant-boolean filters ---------------------------------------------
def test_filter_true_preserves_child_exactly():
    ds = _ds()
    src = _exact_source(6, x=ColumnStat(min=0, max=5, null_count=0, provenance=Provenance.EXACT))
    rs = _est(ds, src).estimate(Filter(ds._plan, lit(True)))
    assert rs.rows == 6 and rs.rows_exact
    assert rs.column("x").provenance is Provenance.EXACT  # nothing dropped


def test_filter_false_is_exact_empty():
    ds = _ds()
    src = _exact_source(6, x=ColumnStat(min=0, max=5, provenance=Provenance.EXACT))
    rs = _est(ds, src).estimate(Filter(ds._plan, lit(False)))
    assert rs.rows == 0 and rs.rows_exact


def test_general_filter_still_downgrades():
    # Regression: a non-constant predicate must NOT be treated as trivially true.
    ds = _ds()
    src = _exact_source(6, x=ColumnStat(min=0, max=5, null_count=0, provenance=Provenance.EXACT))
    rs = _est(ds, src).estimate(Filter(ds._plan, col("x") > lit(2)))
    assert not rs.rows_exact
    assert rs.column("x").provenance is not Provenance.EXACT


# --- join: preserved column values are downgraded bounds, never EXACT -----
def test_join_columns_downgrade_from_exact():
    left = RelStats(
        5, Provenance.EXACT, {"a": ColumnStat(min=0, max=9, provenance=Provenance.EXACT)}
    )
    right = RelStats(
        3, Provenance.EXACT, {"b": ColumnStat(min=1, max=4, provenance=Provenance.EXACT)}
    )
    from batcher.plan.logical.join import Join, JoinOutputCol

    ds = bt.from_arrow(pa.table({"a": [0, 1], "b2": [1, 2]}))
    # Build a Join node just to exercise join_columns' output mapping.
    node = Join(
        left=ds._plan,
        right=ds._plan,
        left_keys=("a",),
        right_keys=("a",),
        join_type="inner",
        output=(
            JoinOutputCol("left", "a", "a"),
            JoinOutputCol("right", "b", "b"),
        ),
    )
    cols = col_prop.join_columns(node, left, right)
    assert cols["a"].max == 9 and cols["a"].provenance is not Provenance.EXACT
    assert cols["b"].min == 1 and cols["b"].provenance is not Provenance.EXACT
    assert cols["a"].null_count is None and cols["a"].ndv is None
