"""`time_series_split(..., gap=n)` leaves a buffer between train and validation (AP-410)."""

from __future__ import annotations

import datetime as dt

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.splitting import time_series_split

pytestmark = pytest.mark.unit


def _bounds(train: bt.Dataset, validate: bt.Dataset, column: str):
    lo = train.agg(m=bt.col(column).max()).to_pydict()["m"][0]
    hi = validate.agg(m=bt.col(column).min()).to_pydict()["m"][0]
    return lo, hi


def test_gap_zero_reproduces_the_adjacent_split():
    ds = bt.from_pydict({"t": list(range(100))})
    plain = [(a.count(), b.count()) for a, b in time_series_split(ds, "t", 4)]
    zero = [(a.count(), b.count()) for a, b in time_series_split(ds, "t", 4, gap=0)]
    assert plain == zero == [(20, 20), (40, 20), (60, 20), (80, 19)]


@pytest.mark.parametrize("expanding", [True, False])
def test_an_integer_gap_separates_train_from_validation(expanding):
    ds = bt.from_pydict({"t": list(range(100))})
    for train, validate in time_series_split(ds, "t", 4, gap=7, expanding=expanding):
        lo, hi = _bounds(train, validate, "t")
        assert lo < hi - 7


def test_a_rolling_window_keeps_its_width_under_a_gap():
    ds = bt.from_pydict({"t": list(range(100))})
    plain = [a.count() for a, _ in time_series_split(ds, "t", 4, expanding=False)]
    gapped = [a.count() for a, _ in time_series_split(ds, "t", 4, expanding=False, gap=5)]
    assert plain == [20, 20, 20, 20]
    assert gapped == [15, 20, 20, 20]


def test_a_timestamp_column_splits_and_takes_a_timedelta_gap():
    start = dt.datetime(2024, 1, 1)
    ds = bt.from_pydict({"t": [start + dt.timedelta(days=i) for i in range(100)]})
    gap = dt.timedelta(days=3)
    splits = ds.ml.time_series_split("t", 4, gap=gap)
    assert [(a.count(), b.count()) for a, b in splits] == [(17, 20), (37, 20), (57, 20), (77, 19)]
    for train, validate in splits:
        lo, hi = _bounds(train, validate, "t")
        assert lo < hi - gap


def test_a_date_column_takes_a_timedelta_gap():
    start = dt.date(2024, 1, 1)
    ds = bt.from_pydict({"d": [start + dt.timedelta(days=i) for i in range(50)]})
    for train, validate in time_series_split(ds, "d", 2, gap=dt.timedelta(days=2)):
        lo, hi = _bounds(train, validate, "d")
        assert lo < hi - dt.timedelta(days=2)


@pytest.mark.parametrize(
    ("values", "gap", "match"),
    [
        (list(range(10)), -1, "must not be negative"),
        (list(range(10)), dt.timedelta(days=1), "must be a number"),
        ([dt.datetime(2024, 1, i + 1) for i in range(10)], 1, "must be a datetime.timedelta"),
        (list(range(10)), "1", "number or a datetime.timedelta"),
    ],
)
def test_a_gap_of_the_wrong_kind_or_sign_is_refused(values, gap, match):
    with pytest.raises(PlanError, match=match):
        time_series_split(bt.from_pydict({"t": values}), "t", 2, gap=gap)
