"""A range on a *derived* date field is still one interval, not two independent bounds.

`_range_column` wants a bare `Col`, so `month(d)` and `hour(ts)` never grouped: `WHERE MONTH(d)
BETWEEN 3 AND 5` combined by exponential backoff as though its two bounds were independent
predicates. Each bound alone really does keep most or half of the year — `MONTH(d) >= 3` keeps
ten twelfths — and neither mentions the width between them, which is the whole answer.

Measured over 6,000 rows spanning 400 days: 2,282 estimated against 1,296 actual, against the
1,500 the three-of-twelve domain implies.

Business-hours (`HOUR(ts) BETWEEN 9 AND 17`), seasonal and weekday windows are all this shape,
and they are exactly the temporal filters the year/decade sargable rewrite deliberately leaves
alone — so this is the only place they get sharpened.
"""

from __future__ import annotations

import datetime

import pytest

import batcher as bt
from batcher.config import active_config
from batcher.kyber.stats import StatsEstimator
from batcher.kyber.stats.selectivity import predicate_selectivity as sel

pytestmark = pytest.mark.unit

_CFG = active_config().optimizer.cardinality


def _s(expr):
    return sel(expr, {}, _CFG, None, None, None)


@pytest.mark.parametrize(
    ("predicate", "expected"),
    [
        ((bt.col("d").dt.month() >= 3) & (bt.col("d").dt.month() <= 5), 3 / 12),
        ((bt.col("d").dt.month() >= 1) & (bt.col("d").dt.month() <= 12), 1.0),
        ((bt.col("d").dt.month() > 3) & (bt.col("d").dt.month() < 5), 1 / 12),
        ((bt.col("t").dt.hour() >= 9) & (bt.col("t").dt.hour() <= 17), 9 / 24),
        ((bt.col("d").dt.day_of_week() >= 5) & (bt.col("d").dt.day_of_week() <= 6), 2 / 7),
    ],
    ids=["spring", "whole-year", "strict", "business-hours", "weekend"],
)
def test_a_derived_field_range_is_one_interval(predicate, expected):
    assert _s(predicate) == pytest.approx(expected, rel=1e-9)


def test_a_contradictory_interval_keeps_nothing():
    assert _s((bt.col("d").dt.month() >= 9) & (bt.col("d").dt.month() <= 3)) == pytest.approx(0.0)


def test_it_is_far_below_the_independent_combination_it_replaced():
    """Each bound alone keeps most of the domain; backoff therefore lands much too high."""
    lower = _s(bt.col("d").dt.month() >= 3)
    upper = _s(bt.col("d").dt.month() <= 5)
    combined = _s((bt.col("d").dt.month() >= 3) & (bt.col("d").dt.month() <= 5))
    assert lower > 0.8 and upper > 0.4
    assert combined < lower * upper  # even the pure product over-states it


def test_two_different_fields_do_not_merge():
    """`MONTH(d) >= 3 AND HOUR(t) <= 5` are two predicates, not one interval."""
    month = _s(bt.col("d").dt.month() >= 3)
    hour = _s(bt.col("t").dt.hour() <= 5)
    combined = _s((bt.col("d").dt.month() >= 3) & (bt.col("t").dt.hour() <= 5))
    # Exponential backoff over the two, most selective first — the ordinary cross-field path.
    first, second = sorted((month, hour))
    assert combined == pytest.approx(first * second**0.5, rel=1e-9)


def test_the_estimate_tracks_the_executed_row_count():
    day0 = datetime.date(2022, 1, 1)
    rows = 3650  # ten years, so every month is evenly represented
    frame = bt.from_pydict({"d": [day0 + datetime.timedelta(days=i) for i in range(rows)]})
    filtered = frame.filter((bt.col("d").dt.month() >= 3) & (bt.col("d").dt.month() <= 5))
    stats = [s.statistics() for s in filtered._sources]
    estimated = (
        StatsEstimator(filtered._sources, {}, _CFG, source_stats=stats)
        .estimate(filtered._plan)
        .rows
    )
    executed = filtered.count()
    assert estimated == pytest.approx(executed, rel=0.1), (
        f"estimated {estimated:.1f} against {executed} executed rows"
    )
