"""Workstream H — watermark-bounded streaming deduplication.

`ds.drop_duplicates_within_watermark(subset, event_time, lateness)` keeps the first
row per key within the event-time watermark, evicting keys the watermark has passed
so seen-key state stays bounded — Spark's ``dropDuplicatesWithinWatermark``.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt

_BASE = dt.datetime(2024, 1, 1, 0, 0, 0)
_SCHEMA = pa.schema([("k", pa.string()), ("ts", pa.timestamp("us")), ("v", pa.int64())])


def _rb(rows):
    ts = [_BASE + dt.timedelta(minutes=m) for _, m, _ in rows]
    return pa.RecordBatch.from_pydict(
        {"k": [k for k, _, _ in rows], "ts": ts, "v": [v for _, _, v in rows]}, schema=_SCHEMA
    )


@pytest.mark.integration
def test_dedup_keeps_first_per_key():
    def batches():
        yield _rb([("a", 0, 1), ("b", 1, 2), ("a", 2, 3)])  # 'a' dup within batch
        yield _rb([("b", 5, 4), ("c", 6, 5)])  # 'b' dup across batches

    ds = bt.from_batches(batches, _SCHEMA, bounded=False).drop_duplicates_within_watermark(
        ["k"], event_time="ts", lateness="1h"
    )
    out = pa.Table.from_batches(list(ds.iter_batches()))
    got = dict(zip(out.column("k").to_pylist(), out.column("v").to_pylist(), strict=True))
    # First occurrence of each key kept: a→1, b→2, c→5 (the dup a@3 and b@4 dropped).
    assert got == {"a": 1, "b": 2, "c": 5}


@pytest.mark.integration
def test_dedup_evicts_past_watermark():
    # With a tiny lateness, a key reappearing long after the watermark passed it is
    # treated as new (state was evicted) — bounded memory, Spark semantics.
    def batches():
        yield _rb([("a", 0, 1)])  # watermark advances to 0 - 0 = 0
        yield _rb([("a", 120, 2)])  # advances to 120 - lateness; key 'a'@0 evicted
        yield _rb([("a", 121, 3)])  # 'a' still within watermark of the 120 event → dropped

    ds = bt.from_batches(batches, _SCHEMA, bounded=False).drop_duplicates_within_watermark(
        ["k"], event_time="ts", lateness="1m"
    )
    out = pa.Table.from_batches(list(ds.iter_batches()))
    # a@0 emitted; a@120 dropped (a@0 still in state when checked); a@121 emitted
    # (a@0 evicted after batch 2 advanced the watermark). Spark semantics.
    assert sorted(out.column("v").to_pylist()) == [1, 3]


@pytest.mark.integration
def test_dedup_bounded_source_is_exact_distinct():
    # Over a bounded source it is exact deduplication (no watermark needed).
    tbl = pa.table({"k": ["a", "a", "b"], "ts": [_BASE] * 3, "v": [1, 2, 3]})
    out = (
        bt.from_arrow(tbl)
        .drop_duplicates_within_watermark(["k"], event_time="ts", lateness="1h")
        .collect()
    )
    assert sorted(out.column("k").to_pylist()) == ["a", "b"]


@pytest.mark.integration
def test_dedup_state_cap_fails_loudly():
    # A tiny `streaming_state_max_bytes` with a long lateness (watermark never evicts
    # keys) makes the seen-key state exceed the cap, so dedup raises a clear
    # ResourceError instead of growing unbounded.
    import dataclasses

    from batcher._internal.errors import ResourceError
    from batcher.config import active_config, config_context

    def batches():
        yield _rb([("a", 0, 1), ("b", 1, 2)])
        yield _rb([("c", 2, 3), ("d", 3, 4)])

    ds = bt.from_batches(batches, _SCHEMA, bounded=False).drop_duplicates_within_watermark(
        ["k"], event_time="ts", lateness="1h"
    )
    cfg = active_config()
    tiny = cfg.replace(memory=dataclasses.replace(cfg.memory, streaming_state_max_bytes=1))
    with config_context(tiny), pytest.raises(ResourceError, match="streaming state"):
        list(ds.iter_batches())
