"""An integral column's `[min, max]` bounds its distinct count, and the estimator uses it.

Neither an in-memory Arrow table nor a Parquet footer records a distinct count, so a
surrogate-key column used to reach the join estimator with `ndv=None` — and the fallback
there is `max(|L|, |R|)`, which prices every join as a key lookup. A column bounded `[1,
18000]` cannot hold more than 18,000 distinct values whatever its row count, and that one
subtraction is what makes a many-to-many join visible to the cost model.
"""

from __future__ import annotations

import datetime

import batcher as bt
from batcher.api.source_stats import collect_source_stats
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.stats.columns import integral_range_ndv
from batcher.plan.stats import ColumnStat, Provenance


def _estimator(ds):
    """The estimator the optimizer builds — statistics come from the bound sources."""
    return CardinalityEstimator(
        ds._sources, {}, source_stats=collect_source_stats(ds._sources, None)
    )


def _stat(lo, hi):
    return ColumnStat(min=lo, max=hi, provenance=Provenance.EXACT)


def test_an_integer_range_narrower_than_the_relation_bounds_the_distinct_count():
    assert integral_range_ndv(_stat(1, 18_000), rows=1_441_548.0) == 18_000.0


def test_a_dense_surrogate_key_is_recognized_as_unique():
    # span == rows is the shape of every dimension table's primary key.
    assert integral_range_ndv(_stat(1, 18_000), rows=18_000.0) == 18_000.0


def test_a_range_wider_than_the_relation_claims_nothing():
    # `min(ndv, rows)` is already the better statement, and "one distinct value per row"
    # would be a guess rather than a bound.
    assert integral_range_ndv(_stat(1, 1_000_000), rows=1_000.0) is None


def test_a_measured_distinct_count_is_never_overridden():
    measured = ColumnStat(min=1, max=100, ndv=7.0, ndv_provenance=Provenance.SKETCH)
    assert integral_range_ndv(measured, rows=1_000.0) is None


def test_a_float_range_says_nothing_about_how_many_values_occur():
    assert integral_range_ndv(_stat(0.0, 1.0), rows=1_000.0) is None


def test_a_boolean_column_is_not_treated_as_an_integer():
    assert integral_range_ndv(_stat(False, True), rows=1_000.0) is None


def test_a_date_range_bounds_the_distinct_count_by_its_days():
    lo, hi = datetime.date(2024, 1, 1), datetime.date(2024, 1, 10)
    assert integral_range_ndv(_stat(lo, hi), rows=5_000.0) == 10.0


def test_a_timestamp_range_is_left_alone():
    # Sub-day precision this bound cannot see; bounding at the day count would be wrong.
    lo = datetime.datetime(2024, 1, 1, 0, 0)
    hi = datetime.datetime(2024, 1, 2, 0, 0)
    assert integral_range_ndv(_stat(lo, hi), rows=5_000.0) is None


def test_the_bound_reaches_the_scan_estimate_and_can_never_answer_count_distinct():
    ds = bt.from_pydict({"k": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]})
    est = _estimator(ds)
    stat = est.estimate(ds._plan).columns["k"]
    assert stat.ndv == 3.0  # [1, 3] over 10 rows
    assert not stat.ndv_is_exact


def test_a_many_to_many_join_is_no_longer_priced_as_a_key_lookup():
    # Both sides hold 3 distinct keys over many rows: the join fans out, and with no ndv
    # the estimate was `max(|L|, |R|)` — smaller than either side's contribution.
    left = bt.from_pydict({"k": [1, 2, 3] * 40})
    right = bt.from_pydict({"k": [1, 2, 3] * 30, "v": list(range(90))})
    joined = left.join(right, on="k")
    est = _estimator(joined)
    assert est.estimate(joined._plan).rows > max(120, 90)
