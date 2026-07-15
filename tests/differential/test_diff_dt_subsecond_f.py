"""`strftime`/`strptime` fractional-second (`%f`) parity with DuckDB.

DuckDB (like Python/C ``strftime``) renders and parses ``%f`` as **microseconds**
right-padded to 6 digits (``.5`` ↔ ``500000``). chrono — the engine's formatter —
instead treats ``%f`` as 9-digit *nanoseconds* on format (``.123456`` → ``.123456000``)
and as a raw nanosecond integer on parse (``.123456`` → 123 µs, ``.5`` → 0 µs), a
silently ~1000×-wrong result. These pin the corrected microsecond semantics.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa

import batcher as bt
from batcher import col


def test_strftime_subsecond_f_is_microseconds(duck):
    """``strftime(ts, '…%S.%f')`` renders 6-digit microseconds, not 9-digit nanos."""
    from conftest import assert_same

    ts = [
        dt.datetime(2024, 2, 15, 13, 45, 30, 123456),
        dt.datetime(1969, 6, 15, 13, 45, 30, 500000),
        dt.datetime(2020, 12, 31, 23, 59, 59, 0),
        None,
    ]
    t = pa.table({"ts": pa.array(ts, pa.timestamp("us"))})
    duck.register("sf", t)
    out = bt.from_arrow(t).select(
        r=col("ts").dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    ).collect()
    assert_same(out, duck.sql("SELECT strftime(ts, '%Y-%m-%d %H:%M:%S.%f') r FROM sf"))


def test_strptime_subsecond_f_scales_as_microseconds(duck):
    """``strptime(s, '…%S.%f')`` scales the fraction to microseconds (``.5`` → 500000)."""
    from conftest import assert_same

    s = [
        "2024-02-15 13:45:30.123456",
        "2024-02-15 13:45:30.5",
        "2024-02-15 13:45:30.123",
        "bad",
        None,
    ]
    t = pa.table({"s": pa.array(s, pa.string())})
    duck.register("sp", t)
    out = bt.from_arrow(t).select(
        d=col("s").str.to_datetime("%Y-%m-%d %H:%M:%S.%f")
    ).collect()
    assert_same(out, duck.sql("SELECT try_strptime(s, '%Y-%m-%d %H:%M:%S.%f') d FROM sp"))
