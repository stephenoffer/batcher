"""A filter on a column constrains that column's own value domain.

`filter_columns` shrinks every column's distinct count with Cardenas' formula, which models
the survivors as a **random subset** of the rows. That is the right model for a column the
predicate says nothing about, and the wrong one for the column it filters on: `WHERE k < 100`
over a key with 1,000 values leaves exactly 100 of them, not the 878 a random 10% sample of
the rows would be expected to touch.

The shape is `WHERE d >= '2024-01-01' GROUP BY d` — a date-restricted rollup, and any
`GROUP BY` or `DISTINCT` over a filtered key. The 8.8x error reached the group count, the
hash-aggregate's memory envelope, and every join above it.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.config import active_config
from batcher.kyber.stats import StatsEstimator

pytestmark = pytest.mark.unit

_CFG = active_config().optimizer.cardinality
_ROWS = 5000
_KEYS = 1000


def _estimate(dataset) -> float:
    stats = [s.statistics() for s in dataset._sources]
    return (
        StatsEstimator(dataset._sources, {}, _CFG, source_stats=stats).estimate(dataset._plan).rows
    )


@pytest.fixture
def keyed():
    """5,000 rows over 1,000 key values, plus a payload column the filter says nothing about."""
    return bt.from_pydict(
        {"k": [i % _KEYS for i in range(_ROWS)], "v": [float(i) for i in range(_ROWS)]}
    )


@pytest.mark.parametrize(
    ("predicate", "surviving_keys"),
    [
        (bt.col("k") < 100, 100),
        (bt.col("k") < 500, 500),
        (bt.col("k") == 7, 1),
        (bt.col("k").is_in([1, 2, 3]), 3),
        ((bt.col("k") >= 200) & (bt.col("k") < 300), 100),
    ],
)
def test_a_group_by_on_a_filtered_key_counts_the_keys_the_filter_admits(
    keyed, predicate, surviving_keys
):
    grouped = keyed.filter(predicate).group_by("k").agg(s=bt.col("v").sum())
    estimated, executed = _estimate(grouped), grouped.count()
    assert executed == surviving_keys, "fixture drifted"
    assert estimated == pytest.approx(executed, rel=0.15)


def test_distinct_over_a_filtered_key_agrees_with_the_group_by(keyed):
    filtered = keyed.filter(bt.col("k") < 100)
    assert _estimate(filtered.select("k").distinct()) == pytest.approx(
        _estimate(filtered.group_by("k").agg(s=bt.col("v").sum()))
    )


def test_a_filter_on_another_column_keeps_the_random_subset_model(keyed):
    """The rule applies to the column the predicate names, not to every column.

    `v` is unconstrained by a predicate on `k`, so its surviving distinct count is still the
    random-subset estimate — which is *above* the number of admitted key values.
    """
    filtered = keyed.filter(bt.col("k") < 100)
    stats = [s.statistics() for s in filtered._sources]
    columns = (
        StatsEstimator(filtered._sources, {}, _CFG, source_stats=stats)
        .estimate(filtered._plan)
        .columns
    )
    assert columns["k"].ndv == pytest.approx(100.0, rel=0.15)
    assert columns["v"].ndv is None or columns["v"].ndv > columns["k"].ndv


def test_a_null_test_does_not_shrink_the_value_domain():
    """`IS NOT NULL` selects on nullity; dropping the nulls removes no distinct value."""
    rows = 3000
    values = [None if i % 3 == 0 else i % 300 for i in range(rows)]
    dataset = bt.from_pydict({"k": values})
    filtered = dataset.filter(bt.col("k").is_not_null())
    stats = [s.statistics() for s in filtered._sources]
    columns = (
        StatsEstimator(
            filtered._sources, {"__column_ndv__": {"k": 300.0}}, _CFG, source_stats=stats
        )
        .estimate(filtered._plan)
        .columns
    )
    assert columns["k"].ndv == pytest.approx(300.0, rel=0.05)


def test_the_tightening_only_ever_sharpens(keyed):
    """It is a cap: a predicate that admits the whole domain changes nothing."""
    wide = keyed.filter(bt.col("k") < 10_000)
    stats = [s.statistics() for s in wide._sources]
    columns = (
        StatsEstimator(wide._sources, {}, _CFG, source_stats=stats).estimate(wide._plan).columns
    )
    assert columns["k"].ndv == pytest.approx(_KEYS, rel=0.05)
