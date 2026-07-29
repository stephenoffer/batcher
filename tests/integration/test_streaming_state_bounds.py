"""Retained streaming state: fragmentation, targeted eviction, durability, stop latency.

Each contract here concerns state or cadence rather than results, so each is asserted
against the structure the operator actually retains or the syscalls it actually makes.
"""

from __future__ import annotations

import datetime as dt
import os

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.api.terminal.stream.watermark import _MAX_STATE_CHUNKS, _compact
from batcher.io.formats.streaming.checkpoint.state_store import StateStore
from batcher.io.formats.streaming.dev import RateSource


# --------------------------------------------------------------------------
# Retained state must not fragment without bound.
# --------------------------------------------------------------------------
def test_state_compaction_triggers_only_past_the_chunk_ceiling():
    """State grows by `concat_tables` (one chunk per micro-batch) and shrinks by `filter`,
    which preserves the chunk structure — so an hour at a 100ms trigger carried a
    36,000-chunk table into every anti-join. The byte cap never saw it: the bytes were fine.
    """
    one = pa.table({"a": [1]})
    small = pa.concat_tables([one] * 4)
    assert _compact(small) is small  # under the ceiling: no copy

    big = pa.concat_tables([one] * (_MAX_STATE_CHUNKS + 5))
    compacted = _compact(big)
    assert compacted.column(0).num_chunks == 1
    assert compacted.to_pydict() == big.to_pydict()


def test_state_compaction_is_a_no_op_for_absent_or_column_less_state():
    assert _compact(None) is None
    empty = pa.table({})
    assert _compact(empty) is empty


def test_a_long_running_dedup_keeps_its_seen_table_compact():
    """End-to-end: many micro-batches through the watermark dedup must not leave the
    retained seen-key table with one chunk per batch."""
    base = dt.datetime(2024, 1, 1)
    schema = pa.schema([("ts", pa.timestamp("us")), ("k", pa.int64())])

    def batches():
        for i in range(_MAX_STATE_CHUNKS + 20):
            yield pa.RecordBatch.from_pylist(
                [{"ts": base + dt.timedelta(seconds=i), "k": i}], schema=schema
            )

    ds = bt.from_batches(batches, schema, bounded=False)
    deduped = ds.drop_duplicates_within_watermark(["k"], event_time="ts", lateness="1h")
    assert sum(b.num_rows for b in deduped.iter_batches()) == _MAX_STATE_CHUNKS + 20


# --------------------------------------------------------------------------
# The stream-stream join evicts the side that needs it, not both every time.
# --------------------------------------------------------------------------
def test_the_interval_join_still_matches_after_targeted_eviction():
    """Halving the eviction work must not change which pairs survive the window."""
    base = dt.datetime(2024, 1, 1)
    schema = pa.schema([("ts", pa.timestamp("us")), ("k", pa.int64()), ("v", pa.int64())])

    def side(offset_seconds: int, tag: int):
        def gen():
            for i in range(6):
                yield pa.RecordBatch.from_pylist(
                    [
                        {
                            "ts": base + dt.timedelta(seconds=i * 10 + offset_seconds),
                            "k": i,
                            "v": tag * 100 + i,
                        }
                    ],
                    schema=schema,
                )

        return gen

    left = bt.from_batches(side(0, 1), schema, bounded=False)
    right = bt.from_batches(side(2, 2), schema, bounded=False)
    joined = left.join_stream(
        right, on="k", left_time="ts", right_time="ts", within="30s", lateness="1m"
    )
    rows = sum(b.num_rows for b in joined.iter_batches())
    assert rows == 6  # every pair is 2s apart, well inside the 30s interval


# --------------------------------------------------------------------------
# The state snapshot must be durable before its commit is recorded.
# --------------------------------------------------------------------------
def test_a_state_snapshot_is_fsynced_before_it_is_renamed(tmp_path, monkeypatch):
    """The rename is atomic, not durable. The engine snapshots state and *then* records the
    commit, so without the syncs a crash could leave the commit on disk and the state not —
    and recovery resumes past consumed data with an empty running aggregate."""
    order: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace

    def spy_fsync(fd):
        order.append("fsync")
        return real_fsync(fd)

    def spy_replace(src, dst):
        order.append("replace")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", spy_fsync)
    monkeypatch.setattr(os, "replace", spy_replace)

    store = StateStore(str(tmp_path / "state"))
    store.snapshot(7, pa.record_batch({"g": [1, 2], "n": [10, 20]}))

    renamed_before_durable = "the file was renamed before it was durable"
    assert order.index("fsync") < order.index("replace"), renamed_before_durable
    assert order[-1] == "fsync", "the directory entry for the rename was never synced"


def test_a_snapshot_round_trips_and_prunes(tmp_path):
    store = StateStore(str(tmp_path / "state"))
    state = pa.record_batch({"g": [1, 2], "n": [10, 20]})
    store.snapshot(3, state)
    store.snapshot(4, state)
    assert store.restore(4).to_pydict() == state.to_pydict()
    store.prune(4)
    assert store.restore(3) is None
    assert store.restore(4) is not None
    # No temp files survive a completed snapshot.
    assert not [n for n in os.listdir(str(tmp_path / "state")) if n.endswith(".tmp")]


# --------------------------------------------------------------------------
# The rate source: stop latency and an unrepresentable rate.
# --------------------------------------------------------------------------
def test_a_paced_rate_source_observes_a_stop_without_waiting_out_its_tick():
    """A paced source sleeps a full second between batches, so `stop()` waited out the
    sleep before the driver thread could be joined."""
    source = RateSource(rows_per_second=2)
    fired = {"stop": False}
    source.set_stop_signal(lambda: fired["stop"])

    stream = source.iter_batches()
    first = next(stream)
    assert first.num_rows == 2
    fired["stop"] = True
    assert list(stream) == []  # the nap is cut short and the loop ends


def test_an_unattached_rate_source_keeps_its_old_behaviour():
    source = RateSource(rows_per_second=3, num_rows=6, pace=False)
    assert sum(b.num_rows for b in source.iter_batches()) == 6


@pytest.mark.parametrize("rate", [0, -1, 1_000_001])
def test_an_unrepresentable_rate_is_refused(rate):
    """Above one row per microsecond the spacing floors to zero and every row gets the
    epoch — a windowed aggregate then puts the whole run in one bucket."""
    with pytest.raises(PlanError):
        RateSource(rows_per_second=rate)


def test_the_fastest_representable_rate_still_spaces_its_rows():
    source = RateSource(rows_per_second=1_000_000, num_rows=3, pace=False)
    stamps = next(iter(source.iter_batches())).column("timestamp").to_pylist()
    assert len(set(stamps)) == 3
