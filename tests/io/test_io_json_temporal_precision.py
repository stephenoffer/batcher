"""A date or timestamp written to JSON keeps its value exactly.

JSON has no temporal type, so the value must be a number or a string. It was a number,
and the number was wrong. `ndjson_vectorized` declined temporal columns, so every one of
them fell through to the pandas encoder -- which reads a timestamp column's raw integers
as *nanoseconds* whatever the column's unit actually is. Only `timestamp[ns]` survived:

    stored                     correct epoch value        written
    timestamp[s]               1709210096                 1709
    timestamp[ms]              1709210096123              1709210
    timestamp[us]              1709210096123456           1709210096

`timestamp[us]` is what the FFI boundary normalizes to, so the common case was out by a
factor of a million, and silently: the output is a plausible-looking number that decodes
to the wrong instant rather than anything that raises.

The writer now renders ISO-8601 through Arrow's cast, exact at the column's own
resolution -- the spelling `msgpack` already used and that DuckDB and Spark produce. The
*type* still does not survive a round trip (Arrow's JSON reader infers a bare date as
`timestamp[s]` and leaves a sub-second instant a string); that is a reader limit and is
recorded in `test_format_fidelity_matrix.LOSSY`. This module pins the values.
"""

from __future__ import annotations

import datetime as dt
import json

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.io

#: An instant with a non-zero microsecond field, so truncation is visible.
_INSTANT = dt.datetime(2024, 2, 29, 12, 34, 56, 123456)

#: Each timestamp unit against the ISO-8601 text it must produce at that resolution.
_BY_UNIT = {
    "s": "2024-02-29 12:34:56",
    "ms": "2024-02-29 12:34:56.123",
    "us": "2024-02-29 12:34:56.123456",
    "ns": "2024-02-29 12:34:56.123456000",
}


def _written(tbl: pa.Table, tmp_path) -> list[dict]:
    path = tmp_path / "out.json"
    bt.from_arrow(tbl).write.json(str(path))
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.mark.parametrize("unit", sorted(_BY_UNIT))
def test_every_timestamp_unit_is_written_exactly(unit, tmp_path):
    """The bug: every unit but `ns` was divided by a million on the way out."""
    tbl = pa.table({"v": pa.array([_INSTANT, None], pa.timestamp(unit))})
    assert _written(tbl, tmp_path) == [{"v": _BY_UNIT[unit]}, {"v": None}]


def test_a_date_is_written_as_an_iso_date(tmp_path):
    """Not epoch milliseconds, and not a timestamp -- a date is a date."""
    tbl = pa.table(
        {"v": pa.array([dt.date(2024, 2, 29), None, dt.date(1969, 12, 31)], pa.date32())}
    )
    assert _written(tbl, tmp_path) == [{"v": "2024-02-29"}, {"v": None}, {"v": "1969-12-31"}]


def test_a_timezone_aware_instant_keeps_its_offset(tmp_path):
    """A tz-aware column renders the zone, so the instant is unambiguous."""
    tbl = pa.table({"v": pa.array([_INSTANT.replace(tzinfo=dt.UTC)], pa.timestamp("us", tz="UTC"))})
    assert _written(tbl, tmp_path) == [{"v": "2024-02-29 12:34:56.123456Z"}]


def test_a_timestamp_nested_in_a_struct_is_written_exactly(tmp_path):
    """The struct renderer recurses through `_values`, so it inherits the same fix."""
    field = pa.struct([("when", pa.timestamp("us")), ("n", pa.int64())])
    tbl = pa.table({"v": pa.array([{"when": _INSTANT, "n": 1}, None], field)})
    assert _written(tbl, tmp_path) == [
        {"v": {"when": "2024-02-29 12:34:56.123456", "n": 1}},
        {"v": None},
    ]


def test_temporal_alongside_the_other_renderable_types(tmp_path):
    """A mixed table still escapes strings and renders floats exactly."""
    tbl = pa.table(
        {
            "i": pa.array([1], pa.int64()),
            "f": pa.array([1.5], pa.float64()),
            "s": pa.array(['a"b'], pa.string()),
            "d": pa.array([dt.date(2024, 2, 29)], pa.date32()),
            "ts": pa.array([_INSTANT], pa.timestamp("us")),
        }
    )
    assert _written(tbl, tmp_path) == [
        {"i": 1, "f": 1.5, "s": 'a"b', "d": "2024-02-29", "ts": "2024-02-29 12:34:56.123456"}
    ]


def test_the_microseconds_survive_a_write_and_read(tmp_path):
    """End to end: the text read back still carries the sub-second field."""
    tbl = pa.table({"ts": pa.array([_INSTANT], pa.timestamp("us"))})
    path = tmp_path / "rt.json"
    bt.from_arrow(tbl).write.json(str(path))
    back = bt.read.json(str(path)).to_pydict()["ts"]
    # The reader leaves a sub-second instant as text (see the module docstring); what
    # matters here is that the microseconds are still in it rather than truncated away.
    assert back == ["2024-02-29 12:34:56.123456"]
