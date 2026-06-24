"""Coverage for `dt.convert_timezone` — DST-aware tz conversion (chrono-tz).

DuckDB's `convert_timezone` needs the ICU extension (not loaded here), so the oracle
is Python's `zoneinfo` (the same IANA database the engine uses).
"""

from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential


def _convert(naive: datetime.datetime, from_tz: str, to_tz: str) -> datetime.datetime:
    """Reference: read `naive` as wall-clock in from_tz → wall-clock in to_tz."""
    aware = naive.replace(tzinfo=ZoneInfo(from_tz))
    return aware.astimezone(ZoneInfo(to_tz)).replace(tzinfo=None)


def test_convert_timezone_dst_aware():
    # July → EDT (UTC-4); January → EST (UTC-5).
    instants = [
        datetime.datetime(2024, 7, 1, 12, 0, 0),
        datetime.datetime(2024, 1, 1, 12, 0, 0),
        datetime.datetime(2024, 3, 15, 0, 30, 0),
    ]
    ds = bt.from_arrow(pa.table({"ts": instants}))
    out = (
        ds.select(ny=col("ts").dt.convert_timezone("UTC", "America/New_York"))
        .collect()
        .to_pydict()["ny"]
    )
    expected = [_convert(t, "UTC", "America/New_York") for t in instants]
    assert out == expected


def test_convert_timezone_roundtrip_and_nulls():
    instants = [datetime.datetime(2024, 6, 1, 9, 0, 0), None]
    ds = bt.from_arrow(pa.table({"ts": instants}))
    out = (
        ds.select(
            rt=col("ts")
            .dt.convert_timezone("UTC", "Asia/Tokyo")
            .dt.convert_timezone("Asia/Tokyo", "UTC")
        )
        .collect()
        .to_pydict()["rt"]
    )
    assert out[0] == instants[0]  # round-trip is identity
    assert out[1] is None  # null propagates


def test_convert_timezone_unknown_zone_errors():
    ds = bt.from_arrow(pa.table({"ts": [datetime.datetime(2024, 1, 1)]}))
    with pytest.raises(Exception, match="timezone"):
        ds.select(x=col("ts").dt.convert_timezone("UTC", "Not/AZone")).collect()
