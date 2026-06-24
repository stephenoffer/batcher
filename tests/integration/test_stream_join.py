"""Workstream H — watermark-bounded stream-stream interval join.

`left.join_stream(right, on=..., left_time=, right_time=, within=)` joins two streams
on keys *and* an event-time interval, buffering both sides and evicting state past the
watermark so memory stays bounded (Spark stream-stream join).
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt

_BASE = dt.datetime(2024, 1, 1, 0, 0, 0)
_L = pa.schema([("k", pa.string()), ("lts", pa.timestamp("us")), ("lv", pa.int64())])
_R = pa.schema([("k", pa.string()), ("rts", pa.timestamp("us")), ("rv", pa.int64())])


def _lb(rows):
    return pa.RecordBatch.from_pydict(
        {
            "k": [k for k, _, _ in rows],
            "lts": [_BASE + dt.timedelta(minutes=m) for _, m, _ in rows],
            "lv": [v for _, _, v in rows],
        },
        schema=_L,
    )


def _rb(rows):
    return pa.RecordBatch.from_pydict(
        {
            "k": [k for k, _, _ in rows],
            "rts": [_BASE + dt.timedelta(minutes=m) for _, m, _ in rows],
            "rv": [v for _, _, v in rows],
        },
        schema=_R,
    )


@pytest.mark.integration
def test_interval_join_matches_within_window():
    def left():
        yield _lb([("a", 0, 1), ("b", 1, 2)])
        yield _lb([("a", 10, 3)])

    def right():
        yield _rb([("a", 2, 10)])  # matches a@0 (|0-2|=2 ≤ 5)
        yield _rb([("a", 8, 20), ("b", 30, 30)])  # a@8 matches a@10 (|10-8|=2); b@30 vs b@1 → no

    ls = bt.from_batches(left, _L, bounded=False)
    rs = bt.from_batches(right, _R, bounded=False)
    out = pa.Table.from_batches(
        list(
            ls.join_stream(
                rs, on="k", left_time="lts", right_time="rts", within="5m"
            ).iter_batches()
        )
    )
    pairs = sorted(zip(out.column("lv").to_pylist(), out.column("rv").to_pylist(), strict=True))
    # a@0↔a@2 (1,10); a@10↔a@8 (3,20). b never matches within 5m.
    assert pairs == [(1, 10), (3, 20)]


@pytest.mark.integration
def test_interval_join_bounded_sources():
    # Over bounded sources it is a plain inner join + the interval filter.
    left = bt.from_arrow(pa.table({"k": ["a", "a"], "lts": [_BASE, _BASE], "lv": [1, 2]}))
    right = bt.from_arrow(
        pa.table({"k": ["a"], "rts": [_BASE + dt.timedelta(minutes=3)], "rv": [9]}, schema=_R)
    )
    out = left.join_stream(right, on="k", left_time="lts", right_time="rts", within="5m").collect()
    assert sorted(zip(out.column("lv").to_pylist(), out.column("rv").to_pylist(), strict=True)) == [
        (1, 9),
        (2, 9),
    ]
