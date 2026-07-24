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
def test_interval_join_same_named_time_columns():
    # When both streams name their event-time column the same, the join output suffixes
    # the right one (`ts` / `ts_right`). The interval filter must difference the two
    # distinct output columns, not read the left column twice — which would make every
    # diff 0 and pass every pair through the window regardless of its real time distance.
    same_l = pa.schema([("k", pa.string()), ("ts", pa.timestamp("us")), ("lv", pa.int64())])
    same_r = pa.schema([("k", pa.string()), ("ts", pa.timestamp("us")), ("rv", pa.int64())])

    def left():
        yield pa.RecordBatch.from_pydict({"k": ["a"], "ts": [_BASE], "lv": [1]}, schema=same_l)

    def right():
        # a@2min is within 5m of a@0; a@100min is far outside and must NOT match.
        yield pa.RecordBatch.from_pydict(
            {
                "k": ["a", "a"],
                "ts": [_BASE + dt.timedelta(minutes=2), _BASE + dt.timedelta(minutes=100)],
                "rv": [10, 11],
            },
            schema=same_r,
        )

    ls = bt.from_batches(left, same_l, bounded=False)
    rs = bt.from_batches(right, same_r, bounded=False)
    joined = ls.join_stream(rs, on="k", left_time="ts", right_time="ts", within="5m")
    out = pa.Table.from_batches(list(joined.iter_batches()))
    pairs = sorted(zip(out.column("lv").to_pylist(), out.column("rv").to_pylist(), strict=True))
    assert pairs == [(1, 10)]


@pytest.mark.integration
def test_stream_join_three_way_raises_clear_error():
    # The buffered symmetric hash join drives exactly two sides. Chaining a third stream
    # (`a.join_stream(b).join_stream(c)`) must fail with a message that names the
    # two-stream limit, not fall through to the generic "must materialize" error.
    from batcher._internal.errors import PlanError

    def gen(name, tcol):
        sch = pa.schema([("k", pa.string()), (tcol, pa.timestamp("us")), (name, pa.int64())])

        def factory():
            yield pa.RecordBatch.from_pydict({"k": ["a"], tcol: [_BASE], name: [1]}, schema=sch)

        return bt.from_batches(factory, sch, bounded=False)

    a = gen("av", "ats")
    b = gen("bv", "bts")
    c = gen("cv", "cts")
    j = a.join_stream(b, on="k", left_time="ats", right_time="bts", within="5m")
    three = j.join_stream(c, on="k", left_time="ats", right_time="cts", within="5m")
    with pytest.raises(PlanError, match=r"join_stream.*exactly two streams"):
        list(three.iter_batches())


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
