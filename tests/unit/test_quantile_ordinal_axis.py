"""A learned quantile grid and a predicate literal must sit on the same number line.

Core measures a quantile grid from raw Arrow values, so a `date32` column's grid counts
epoch **days** and a `timestamp[us]` column's counts epoch **microseconds**. Kyber consults
that grid with a Python `date`/`datetime` off the predicate. The two used to be placed
differently — `date.toordinal()` counts from year 1, a 719,163-day offset, and
`datetime.timestamp()` reads a naive value in the *local* zone and returns seconds — so every
temporal literal landed far outside its own column's grid.

That is not a merely-imprecise estimate. `o_orderdate BETWEEN '1995-01-01' AND '1996-12-31'`
over TPC-H `orders` came back as **0 rows** instead of 455,112, which made every join with
`orders` look free: TPC-H Q8's plan flipped from joining the 1,327-row filtered `part` to
`lineitem` first into carrying a 1.8M-row intermediate through four joins, and the query went
from 26 ms to 835 ms. It only bit from a query's *second* execution, because the first has no
grid to read.

These tests pin the axis agreement itself at both ends — what Core records and what Kyber
reads — rather than any one plan.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

from batcher.config import CardinalityConfig
from batcher.core.stats import column_statistics
from batcher.kyber.stats.selectivity import predicate_selectivity
from batcher.plan.expr_ir import Binary, Col, Lit
from batcher.plan.stats import (
    AXIS_DATE,
    AXIS_DATETIME,
    AXIS_NUMERIC,
    arrow_ordinal_axis,
    ordinal_with_axis,
)

pytestmark = pytest.mark.unit

_CFG = CardinalityConfig()


def _between(column: str, lo: object, hi: object) -> Binary:
    return Binary(
        "and",
        Binary("ge", Col(column), Lit(lo)),
        Binary("le", Col(column), Lit(hi)),
    )


def _dates(n: int, start: dt.date) -> pa.Table:
    return pa.table({"d": pa.array([start + dt.timedelta(days=i) for i in range(n)], pa.date32())})


def test_date_literal_and_arrow_date_share_the_epoch_day_axis() -> None:
    axis, position = ordinal_with_axis(dt.date(1995, 1, 1))
    assert axis == AXIS_DATE
    assert position == 9131.0  # days since 1970-01-01, which is what `date32` stores
    assert arrow_ordinal_axis(pa.date32()) == (AXIS_DATE, 1.0)
    assert arrow_ordinal_axis(pa.date64()) == (AXIS_DATE, 86_400_000.0)


def test_naive_datetime_is_read_as_utc_like_arrow_reads_a_zoneless_timestamp() -> None:
    axis, position = ordinal_with_axis(dt.datetime(1995, 1, 1))
    assert axis == AXIS_DATETIME
    # Arrow stores a zoneless timestamp as a UTC epoch offset; a local-zone reading would
    # shift the value by the host's UTC offset and silently move a date-boundary predicate.
    assert position == 788_918_400.0
    for unit, divisor in (("s", 1.0), ("ms", 1e3), ("us", 1e6), ("ns", 1e9)):
        assert arrow_ordinal_axis(pa.timestamp(unit)) == (AXIS_DATETIME, divisor)
        assert arrow_ordinal_axis(pa.timestamp(unit, tz="UTC")) == (AXIS_DATETIME, divisor)


def test_unordered_types_have_no_axis() -> None:
    assert ordinal_with_axis(True) is None
    assert ordinal_with_axis("1995-01-01") is None
    assert arrow_ordinal_axis(pa.bool_()) is None
    assert arrow_ordinal_axis(pa.string()) is None
    assert arrow_ordinal_axis(pa.list_(pa.int64())) is None


def test_measured_date_grid_is_recorded_on_the_axis_its_literal_uses() -> None:
    table = _dates(400, dt.date(1995, 1, 1))
    _ndv, quantiles, _bytes = column_statistics(table.to_batches(), ["d"])
    grid = quantiles["d"]
    assert grid["axis"] == AXIS_DATE
    lo, hi = ordinal_with_axis(dt.date(1995, 1, 1))[1], ordinal_with_axis(dt.date(1996, 2, 4))[1]
    assert grid["values"][0] == pytest.approx(lo)
    assert grid["values"][-1] == pytest.approx(hi)


def test_measured_timestamp_grid_is_recorded_in_epoch_seconds() -> None:
    stamps = [dt.datetime(1995, 1, 1) + dt.timedelta(hours=i) for i in range(200)]
    table = pa.table({"t": pa.array(stamps, pa.timestamp("us"))})
    _ndv, quantiles, _bytes = column_statistics(table.to_batches(), ["t"])
    grid = quantiles["t"]
    assert grid["axis"] == AXIS_DATETIME
    assert grid["values"][0] == pytest.approx(ordinal_with_axis(stamps[0])[1])
    assert grid["values"][-1] == pytest.approx(ordinal_with_axis(stamps[-1])[1])


def test_a_date_range_covering_half_the_grid_estimates_about_half() -> None:
    """The regression: this used to estimate 0.0 once a grid had been measured."""
    table = _dates(1000, dt.date(1992, 1, 1))
    _ndv, quantiles, _bytes = column_statistics(table.to_batches(), ["d"])
    covered = _between("d", dt.date(1992, 1, 1), dt.date(1993, 5, 15))  # 500 of 1000 days
    assert predicate_selectivity(covered, {}, _CFG, quantiles) == pytest.approx(0.5, abs=0.05)


def test_a_date_range_outside_the_grid_still_estimates_nothing() -> None:
    """The axis fix must not blunt the estimator: a genuinely disjoint range is still empty."""
    table = _dates(1000, dt.date(1992, 1, 1))
    _ndv, quantiles, _bytes = column_statistics(table.to_batches(), ["d"])
    disjoint = _between("d", dt.date(2020, 1, 1), dt.date(2020, 12, 31))
    assert predicate_selectivity(disjoint, {}, _CFG, quantiles) == pytest.approx(0.0, abs=1e-6)


def test_a_grid_on_a_foreign_axis_is_declined_rather_than_misread() -> None:
    """A grid whose axis does not match the literal's must not be interpolated against.

    A hub persisted before grids carried an axis holds raw storage units, which read as
    `AXIS_NUMERIC`. Reading a `date` literal against one is exactly the bug; declining falls
    back to the column's bounds (or the Selinger constant), which is merely imprecise.
    """
    legacy = {"d": {"probs": [0.0, 0.5, 1.0], "values": [8035.0, 9200.0, 10440.0]}}
    assert legacy["d"].get("axis", AXIS_NUMERIC) == AXIS_NUMERIC
    covered = _between("d", dt.date(1992, 1, 1), dt.date(1998, 12, 31))
    # Declined -> the Selinger range constant, not the 0.0 a mis-axed read would give.
    assert predicate_selectivity(covered, {}, _CFG, legacy) > 0.1


def test_numeric_columns_are_unaffected() -> None:
    table = pa.table({"n": pa.array(list(range(1000)), pa.int64())})
    _ndv, quantiles, _bytes = column_statistics(table.to_batches(), ["n"])
    assert quantiles["n"]["axis"] == AXIS_NUMERIC
    assert quantiles["n"]["values"][0] == pytest.approx(0.0)
    assert quantiles["n"]["values"][-1] == pytest.approx(999.0)
    half = _between("n", 0, 499)
    assert predicate_selectivity(half, {}, _CFG, quantiles) == pytest.approx(0.5, abs=0.05)
