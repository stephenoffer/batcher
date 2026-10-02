"""A filter on one ascending column narrows the columns that ascend with it.

A date dimension stores `d_date_sk`, `d_week_seq` and `d_year` in one order, so `d_year = X`
keeps one contiguous run of rows: 52 weeks, not the several hundred that modelling the kept
rows as a random sample predicts. In TPC-DS q72 that difference was a sevenfold fan-out the
estimator could not see (2.5M rows estimated, 16.4M actual), and a join order built on it.
"""

from __future__ import annotations

import random

import pytest

import batcher as bt
from batcher.api.source_stats import build_estimator, collect_source_stats

pytestmark = pytest.mark.unit

_DAYS = 20 * 364  # twenty 52-week years, so a year is exactly 52 weeks of 7 days


def _date_dim(shuffled: bool = False) -> bt.Dataset:
    rows = [
        {"sk": 1_000 + i, "week": i // 7, "year": 2000 + i // 364, "tag": i % 3}
        for i in range(_DAYS)
    ]
    if shuffled:
        random.Random(7).shuffle(rows)
    return bt.from_pydict({k: [r[k] for r in rows] for k in rows[0]})


def _week_ndv_after_year_filter(ds: bt.Dataset) -> float:
    filtered = ds.filter(bt.col("year") == 2005)
    est = build_estimator(filtered._sources, None)
    return est.estimate(filtered._plan).columns["week"].ndv


def test_the_source_reports_which_columns_ascend() -> None:
    (stats,) = collect_source_stats(_date_dim()._sources, None)
    assert set(stats.ascending) == {"sk", "week", "year"}, "`tag` cycles, so it must not ascend"
    (narrowed,) = collect_source_stats(
        _date_dim()._sources, None, need_columns={"sk", "week", "year", "tag"}
    )
    assert set(narrowed.ascending) == {"sk", "week", "year"}


def test_a_year_keeps_its_weeks_not_a_random_sample_of_them() -> None:
    ndv = _week_ndv_after_year_filter(_date_dim())
    assert 40 <= ndv <= 65, f"one year is 52 weeks, estimated {ndv}"


def test_control_the_same_rows_out_of_order_keep_the_random_sample_model() -> None:
    """Shuffled, the columns ascend in no order, so nothing licenses the narrowing."""
    (stats,) = collect_source_stats(_date_dim(shuffled=True)._sources, None)
    assert stats.ascending == ()
    assert _week_ndv_after_year_filter(_date_dim(shuffled=True)) > 200


def test_a_range_the_kept_run_satisfies_is_priced_as_keeping_it() -> None:
    """The year's surrogate keys all lie in its run, so a range covering them keeps every row.

    Without the narrowing, `sk` still spans all twenty years above the year filter, and a
    range covering one year's keys reads as keeping a twentieth of it.
    """
    ds = _date_dim()
    year = ds.filter(bt.col("year") == 2005)
    lo, hi = 1_000 + 5 * 364, 1_000 + 6 * 364 - 1
    covered = year.filter((bt.col("sk") >= lo) & (bt.col("sk") <= hi))
    est = build_estimator(covered._sources, None)
    kept = est.estimate(covered._plan).rows / est.estimate(year._plan).rows
    assert kept > 0.8, f"a range holding every kept key was priced at {kept:.2f} of the rows"


def test_the_narrowing_never_moves_the_sound_bounds() -> None:
    """`min`/`max` stay the source's bounds: rules may read them as proofs, and a run
    position inferred under a uniformity assumption is not one."""
    filtered = _date_dim().filter(bt.col("year") == 2005)
    stat = build_estimator(filtered._sources, None).estimate(filtered._plan).columns["sk"]
    assert (stat.min, stat.max) == (1_000, 1_000 + _DAYS - 1)


def _year_ndv_after_joining(dim: bt.Dataset, fact_keys: list[int]) -> float:
    facts = bt.from_pydict({"f_sk": fact_keys, "v": list(range(len(fact_keys)))})
    joined = facts.join(dim, left_on="f_sk", right_on="sk")
    est = build_estimator(joined._sources, None)
    return est.estimate(joined._plan).columns["year"].ndv


def test_an_inner_join_confines_the_dimension_to_the_facts_key_range() -> None:
    """Facts dated in 2005-2006 join to two of the dimension's twenty years, not all twenty.

    TPC-DS `store_sales ⋈ date_dim` is the shape: its fact keys span five of `d_year`'s 201
    values, and a `GROUP BY customer, d_year` above the join read the unnarrowed 201.
    """
    lo = 1_000 + 5 * 364
    keys = [lo + (i * 7) % (2 * 364) for i in range(5_000)]
    ndv = _year_ndv_after_joining(_date_dim(), keys)
    assert 1.5 <= ndv <= 3.5, f"the facts span two years, estimated {ndv}"
    # Control: facts spanning the whole dimension confine nothing.
    wide = [1_000 + (i * 7) % _DAYS for i in range(5_000)]
    assert _year_ndv_after_joining(_date_dim(), wide) > 15


def test_control_an_unordered_dimension_is_not_narrowed_by_a_join() -> None:
    lo = 1_000 + 5 * 364
    keys = [lo + (i * 7) % (2 * 364) for i in range(5_000)]
    assert _year_ndv_after_joining(_date_dim(shuffled=True), keys) > 15


def test_a_join_to_a_filtered_dimension_counts_only_the_facts_inside_its_run() -> None:
    """Facts spread over five years, joined to the dimension cut to one of them, keep a year.

    The fact key has fewer distinct values than the filtered dimension (weekly against daily),
    so containment assumed every fact key finds a partner. They only do inside the kept run,
    which the filter records on the key's quantile grid; TPC-DS q22 was estimated 5x high.
    """
    dim = _date_dim()
    one_year = dim.filter(bt.col("year") == 2005)
    # Every week of 2003-2007: 260 fact keys against the one year's 364 dimension keys.
    weekly = [1_000 + 3 * 364 + 7 * w for w in range(5 * 52)]
    facts = bt.from_pydict({"f_sk": weekly * 5, "v": list(range(len(weekly) * 5))})
    from batcher import core

    # Measure both keys' distinct counts through a *different* query: running the one under
    # test would record its own cardinality, and the estimate would replay the measurement.
    facts.join(dim, left_on="f_sk", right_on="sk").collect()
    joined = facts.join(one_year, left_on="f_sk", right_on="sk")
    est = build_estimator(joined._sources, core.default_hub()).estimate(joined._plan).rows
    actual = 52 * 5  # one year of weeks, each fact week five times
    assert actual / 3 <= est <= actual * 3, (est, actual)
