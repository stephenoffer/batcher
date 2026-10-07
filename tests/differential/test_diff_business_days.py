"""The business-day calendar, held to numpy's ``busday_offset``/``busday_count``/``is_busday``.

DuckDB has no business-day functions, so numpy -- whose semantics Polars and pandas both
follow -- is the oracle. One engine node answers all three operations, so they share one
definition of a business day; the cases below exercise every one of them over the same
calendar so a disagreement between them cannot hide.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential

_START = dt.date(2023, 12, 18)
_DAYS = [_START + dt.timedelta(days=i) for i in range(0, 40, 3)]
_HOLIDAYS = [dt.date(2023, 12, 25), dt.date(2024, 1, 1), dt.date(2023, 12, 30)]  # one a Saturday
_COUNTS = [0, 1, -1, 4, -7, 10, 23, -15, 2, 5, -3, 1, 9, 0]

_CALENDARS = [
    ({}, {}),
    ({"holidays": _HOLIDAYS}, {"holidays": _HOLIDAYS}),
    ({"weekmask": "1111001"}, {"weekmask": "1111001"}),
    (
        {"holidays": _HOLIDAYS, "weekmask": [True, True, True, True, False, False, True]},
        {"holidays": _HOLIDAYS, "weekmask": "1111001"},
    ),
]


def _np(fn, *args, **kw):
    out = fn(*args, **kw)
    return [None if v is None else v for v in np.asarray(out).tolist()]


@pytest.mark.parametrize(("ours", "theirs"), _CALENDARS)
def test_is_business_day_matches_numpy(ours, theirs):
    ds = bt.from_pydict({"d": [*_DAYS, None]})
    if not ours:
        ours = {"holidays": [], "weekmask": "1111100"}
    got = ds.select(r=col("d").dt.is_business_day(**ours)).to_pydict()["r"]
    assert got == [*_np(np.is_busday, _DAYS, **theirs), None]


@pytest.mark.parametrize(("ours", "theirs"), _CALENDARS)
@pytest.mark.parametrize("roll", ["forward", "backward"])
def test_add_business_days_matches_numpy(ours, theirs, roll):
    ds = bt.from_pydict({"d": [*_DAYS, None], "n": [*_COUNTS, 3]})
    got = ds.select(r=col("d").dt.add_business_days(col("n"), roll=roll, **ours)).to_pydict()
    expected = [
        d.astype(dt.date)
        for d in np.busday_offset(_DAYS, _COUNTS, roll=roll, **theirs).astype("datetime64[D]")
    ]
    assert got["r"] == [*expected, None]


@pytest.mark.parametrize(("ours", "theirs"), _CALENDARS)
def test_business_day_count_matches_numpy(ours, theirs):
    ends = [d + dt.timedelta(days=k) for d, k in zip(_DAYS, _COUNTS, strict=True)]
    ds = bt.from_pydict({"a": [*_DAYS, None], "b": [*ends, _START]})
    got = ds.select(r=bt.business_day_count("a", "b", **ours)).to_pydict()["r"]
    assert got == [*_np(np.busday_count, _DAYS, ends, **theirs), None]


def test_a_weekend_start_raises_unless_rolled():
    saturday = bt.from_pydict({"d": [dt.date(2024, 1, 6)]})
    with pytest.raises(Exception, match="not a business day"):
        saturday.select(r=col("d").dt.add_business_days(1)).collect()
    # numpy rolls a Saturday forward to Monday 2024-01-08 before moving zero days.
    out = saturday.select(r=col("d").dt.add_business_days(0, roll="forward")).to_pydict()
    assert out == {"r": [dt.date(2024, 1, 8)]}


def test_a_timestamp_keeps_its_clock_and_counts_by_date():
    ds = bt.from_pydict(
        {"t": [dt.datetime(2024, 1, 5, 17, 30)], "u": [dt.datetime(2024, 1, 8, 9, 0)]}
    )
    out = ds.select(
        r=col("t").dt.add_business_days(1), n=bt.business_day_count(col("t"), col("u"))
    ).to_pydict()
    assert out == {"r": [dt.datetime(2024, 1, 8, 17, 30)], "n": [1]}


def test_the_default_is_business_day_keeps_its_historical_ir():
    """No calendar arguments -> the weekday composition every tier already compiles."""
    assert col("d").dt.is_business_day().to_ir() == (col("d").dt.weekday() <= 5).to_ir()


def test_malformed_calendars_are_refused_at_plan_time():
    with pytest.raises(bt.PlanError, match="weekmask"):
        col("d").dt.add_business_days(1, weekmask="11111")
    with pytest.raises(bt.PlanError, match="no weekday"):
        col("d").dt.is_business_day(weekmask="0000000")
    with pytest.raises(bt.PlanError, match="holiday"):
        col("d").dt.add_business_days(1, holidays=["not-a-date"])
    with pytest.raises(bt.PlanError, match="roll"):
        col("d").dt.add_business_days(1, roll="nearest")
