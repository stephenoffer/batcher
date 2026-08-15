"""A string column has an order, and its measured values place a literal on it.

`_ordinal` refuses a string deliberately: a string column's footer bounds may be
byte-truncated, so an ordinal built from them is not sound for *pruning*, where a wrong
answer drops rows. The consequence was that `s >= 'blue'` had no CDF at all and took
Selinger's 1/3 constant — 2,667 rows estimated against 6,420 actual — and `NOT (s < 'zzz')`,
which matches nothing because every value is below `'zzz'`, also estimated 2,667 against 0.

The most-common-value table is a measured sample of the distribution on the order that does
exist, and nothing here is used for pruning. That also restores an estimate the optimizer had
been destroying: `starts_with(s, p)` is rewritten into the sargable range
`s >= p AND s < p⁺`, which bypassed the measured-match rule in `patterns` and landed on the
same constant — so a text predicate that estimated exactly *before* optimization estimated at
a third of the table after it.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.config import active_config
from batcher.kyber.optimizer.facade import optimize_logical
from batcher.kyber.stats import StatsEstimator
from batcher.kyber.stats.selectivity import predicate_selectivity as sel

pytestmark = pytest.mark.unit

_CFG = active_config().optimizer.cardinality
# Five values, evenly split, all of the mass — the shape of a status/country/category column.
_COLOURS = ("amber", "blue", "green", "red", "violet")
_MCV = {"s": dict.fromkeys(_COLOURS, 0.2)}
_NDV = {"s": 5.0}


def _s(expr):
    return sel(expr, _NDV, _CFG, None, _MCV)


@pytest.mark.parametrize(
    ("predicate", "expected"),
    [
        (bt.col("s") < "blue", 0.2),  # amber
        (bt.col("s") <= "blue", 0.4),  # amber, blue
        (bt.col("s") >= "blue", 0.8),  # all but amber
        (bt.col("s") > "blue", 0.6),  # green, red, violet
        (bt.col("s") < "zzz", 1.0),  # every value is below it
        (bt.col("s") >= "zzz", 0.0),  # none is at or above it
        (bt.col("s") < "aaa", 0.0),  # none is below it
    ],
)
def test_an_ordering_comparison_reads_the_measured_values(predicate, expected):
    assert _s(predicate) == pytest.approx(expected)


def test_a_certainty_and_its_negation_still_partition():
    assert _s(bt.col("s") < "zzz") + _s(~(bt.col("s") < "zzz")) == pytest.approx(1.0)


def test_a_prefix_range_keeps_the_interval_it_names():
    """`LIKE 'red%'` is rewritten to `s >= 'red' AND s < 'ree'` — one value's worth."""
    prefix = (bt.col("s") >= "red") & (bt.col("s") < "ree")
    assert _s(prefix) == pytest.approx(0.2)


def test_a_boundary_at_an_absent_value_carries_no_mass():
    """The CDF boundary and the equality floor are different questions.

    `residual_eq_frequency` answers the rarest listed frequency for an unlisted value, and is
    right to as an *equality* estimate: it must not claim a predicate matches exactly nothing.
    Borrowing that floor as a CDF boundary subtracts a whole value's worth from an interval
    that does not contain the value, which collapsed the prefix range above from 1,636 rows
    to 56.
    """
    from batcher.kyber.stats.selectivity.scalars import _point_mass

    assert _point_mass("s", "red", _NDV, _MCV) == pytest.approx(0.2)
    assert _point_mass("s", "ree", _NDV, _MCV) == pytest.approx(0.0)


def test_a_numeric_column_is_not_compared_lexicographically():
    """The table is keyed by `str(value)`, where `"10" < "9"` — so only strings qualify."""
    ndv = {"n": 100.0}
    mcv = {"n": {"9": 0.5, "10": 0.5}}
    # Falls back to the ordinal path (bounds/quantiles/constant), never to key comparison.
    assert sel(bt.col("n") < 10, ndv, _CFG, None, mcv) == pytest.approx(_CFG.range_selectivity)


def test_the_estimate_tracks_the_executed_row_count():
    rows = 5000
    values = [_COLOURS[i % 5] for i in range(rows)]
    frame = bt.from_pydict({"s": values})
    learned = {
        "__column_mcv__": {"s": {c: values.count(c) / rows for c in _COLOURS}},
        "__column_ndv__": {"s": 5.0},
    }
    for predicate, expected_rows in [
        (bt.col("s") >= "blue", 4000),
        (bt.col("s") < "zzz", 5000),
        (bt.col("s").str.starts_with("red"), 1000),
    ]:
        dataset = frame.filter(predicate)
        stats = [s.statistics() for s in dataset._sources]
        plan = optimize_logical(dataset._plan, sources=dataset._sources, source_stats=stats)
        estimated = (
            StatsEstimator(dataset._sources, learned, _CFG, source_stats=stats).estimate(plan).rows
        )
        executed = dataset.count()
        assert executed == expected_rows, "fixture drifted"
        assert estimated == pytest.approx(executed, rel=0.1), (
            f"{predicate!r}: estimated {estimated:.1f} against {executed}"
        )


# --- a string literal outside the measured bounds ---------------------------------


_BOUNDS = {"s": ("amber", "violet")}


@pytest.mark.parametrize("absent", ["zzz", "aaa", "Aardvark"])
def test_a_string_outside_the_bounds_matches_nothing(absent):
    """`_outside_bounds` declined every string, so `s = 'zzz'` kept a fifth of the table.

    Sound despite the truncation caveat that keeps strings off the ordinal axis elsewhere: a
    Parquet writer truncates a `min` downwards and a `max` upwards, so the stored pair is a
    *superset* of the real range. Python compares `str` by code point and the engine by UTF-8
    bytes, which is the same order.
    """
    assert sel(bt.col("s") == absent, _NDV, _CFG, None, _MCV, _BOUNDS) == pytest.approx(0.0)


def test_an_in_list_drops_only_the_out_of_range_literals():
    listed = sel(bt.col("s").is_in(["zzz", "red"]), _NDV, _CFG, None, _MCV, _BOUNDS)
    assert listed == pytest.approx(0.2)


def test_a_string_inside_the_bounds_is_unaffected():
    assert sel(bt.col("s") == "red", _NDV, _CFG, None, _MCV, _BOUNDS) == pytest.approx(0.2)


def test_the_complement_keeps_everything():
    assert sel(bt.col("s") != "zzz", _NDV, _CFG, None, _MCV, _BOUNDS) == pytest.approx(1.0)


def test_a_parquet_source_carries_the_string_bounds_this_relies_on(tmp_path):
    """The fix is only reachable because real sources record string min/max."""
    target = str(tmp_path / "t.parquet")
    bt.from_pydict({"s": [_COLOURS[i % 5] for i in range(500)]}).write.parquet(target)
    dataset = bt.read.parquet(target)
    stats = dataset._sources[0].statistics()
    assert stats.columns["s"].min == "amber"
    assert stats.columns["s"].max == "violet"
    absent = dataset.filter(bt.col("s") == "zzz")
    estimated = (
        StatsEstimator(
            absent._sources, {}, _CFG, source_stats=[s.statistics() for s in absent._sources]
        )
        .estimate(absent._plan)
        .rows
    )
    assert estimated == pytest.approx(0.0)
    assert absent.count() == 0
