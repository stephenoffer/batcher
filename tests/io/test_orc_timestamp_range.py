"""ORC refuses a timestamp it would silently rewrite, rather than writing a wrong one.

ORC stores timestamps as nanoseconds and nothing else, so `pyarrow.orc` converts every
other precision on the way in. A value outside the nanosecond epoch range does not fail
that conversion, it **wraps**: writing ``2300-01-01`` and reading it back returned
``1715-06-13 00:25:26.290448384``, with no error from pyarrow, from ORC, or from Batcher,
and a perfectly valid file on disk.

That is the worst shape a defect can take here — a write that reports success and stores
different data — and the inputs are ordinary. ``9999-12-31`` is a standard far-future
sentinel and pre-1677 dates are routine in historical data. Parquet, Arrow and Avro all
round-trip these exactly, so nothing about using ORC would lead a user to suspect it.

The check is deliberately narrow: only a timestamp column whose unit is not already
nanoseconds, and only its min and max, so an in-range write pays one vectorized kernel per
timestamp column and nothing per row.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import SchemaError

pytestmark = pytest.mark.unit


def _write(tmp_path, array):
    target = str(tmp_path / "out")
    bt.from_arrow(pa.table({"ts": array})).write.orc(target)
    return target


def test_an_in_range_timestamp_round_trips_exactly(tmp_path):
    values = [dt.datetime(2024, 1, 1), dt.datetime(1970, 1, 1), None]
    target = _write(tmp_path, pa.array(values, type=pa.timestamp("us")))
    assert bt.read.orc(target).collect().to_pydict()["ts"] == values


@pytest.mark.parametrize(
    "moment",
    [
        pytest.param(dt.datetime(2300, 1, 1), id="far_future"),
        pytest.param(dt.datetime(9999, 12, 31), id="sentinel_9999"),
        pytest.param(dt.datetime(1600, 1, 1), id="pre_1677_historical"),
    ],
)
def test_a_timestamp_orc_would_wrap_is_refused(tmp_path, moment):
    array = pa.array([moment, dt.datetime(2024, 1, 1)], type=pa.timestamp("us"))
    with pytest.raises(SchemaError, match="ORC keeps timestamps in nanoseconds"):
        _write(tmp_path, array)


def test_the_refusal_names_the_column_and_a_format_that_can_hold_it(tmp_path):
    array = pa.array([dt.datetime(2300, 1, 1)], type=pa.timestamp("us"))
    with pytest.raises(SchemaError) as excinfo:
        _write(tmp_path, array)
    message = str(excinfo.value)
    assert "'ts'" in message
    assert "Parquet" in message


def test_an_all_null_timestamp_column_has_no_bound_to_exceed(tmp_path):
    target = _write(tmp_path, pa.array([None, None], type=pa.timestamp("us")))
    assert bt.read.orc(target).collect().to_pydict()["ts"] == [None, None]


def test_a_nanosecond_column_is_not_checked_because_it_is_what_orc_stores(tmp_path):
    # Already in ORC's own unit, so nothing is converted and nothing can wrap.
    values = [dt.datetime(2024, 1, 1), None]
    target = _write(tmp_path, pa.array(values, type=pa.timestamp("ns")))
    assert bt.read.orc(target).collect().to_pydict()["ts"] == values


def test_a_timezone_aware_column_is_checked_in_its_own_zone(tmp_path):
    array = pa.array([dt.datetime(2300, 1, 1)], type=pa.timestamp("us", tz="UTC"))
    with pytest.raises(SchemaError, match="ORC keeps timestamps in nanoseconds"):
        _write(tmp_path, array)


def test_the_streaming_write_path_is_guarded_too(tmp_path):
    # `write.orc` over a streamed plan appends batch by batch through a different method
    # than the whole-table write, and an unguarded path there would corrupt just as well.
    array = pa.array([dt.datetime(2300, 1, 1)] * 3, type=pa.timestamp("us"))
    source = bt.from_arrow(pa.table({"ts": array})).filter(bt.col("ts").is_not_null())
    with pytest.raises(SchemaError, match="ORC keeps timestamps in nanoseconds"):
        source.write.orc(str(tmp_path / "streamed"))
