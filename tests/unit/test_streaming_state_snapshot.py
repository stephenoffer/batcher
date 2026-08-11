"""A snapshot with no rows is still a snapshot.

The windowed fold deliberately snapshots a *zero-column* batch when its watermark has
advanced past every open window — the whole payload rides in the schema metadata, so
there is no sidecar file to keep consistent. That is the ordinary state of a windowed
query between windows, and it is the exact state the end-of-stream flush leaves behind.

`StateStore.restore` dropped it: `Table.to_batches()` returns an empty list for a table
with no rows, so the store answered `None`, the engine skipped `restore_state` entirely,
and the watermark rewound to whatever the next batch happened to carry. Rows the old
watermark had correctly ruled late were then re-admitted and folded into windows that had
already been emitted.
"""

from __future__ import annotations

import datetime

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.core.streaming import _window_key, _WindowedAggFold
from batcher.io.formats.streaming.checkpoint.state_store import StateStore

pytestmark = pytest.mark.unit

_SCHEMA = pa.schema([("ts", pa.timestamp("us")), ("v", pa.int64())])
_BASE = datetime.datetime(2024, 1, 1)
_WATERMARK_META = b"12345"


def _at(minutes: int) -> datetime.datetime:
    return _BASE + datetime.timedelta(minutes=minutes)


@pytest.fixture
def store(tmp_path) -> StateStore:
    return StateStore(str(tmp_path / "state"))


def test_a_zero_column_snapshot_round_trips_with_its_metadata(store):
    meta = {b"__wm": _WATERMARK_META}
    store.snapshot(7, pa.RecordBatch.from_pylist([], schema=pa.schema([], metadata=meta)))

    restored = store.restore(7)
    assert restored is not None, "a watermark-only snapshot was dropped on restore"
    assert restored.schema.metadata == meta


def test_a_typed_but_empty_snapshot_round_trips_too(store):
    meta = {b"__wm": _WATERMARK_META}
    schema = pa.schema([("k", pa.int64())], metadata=meta)
    store.snapshot(8, pa.RecordBatch.from_pylist([], schema=schema))

    restored = store.restore(8)
    assert restored is not None
    assert restored.num_columns == 1 and restored.num_rows == 0
    assert restored.schema.metadata == meta


def test_a_snapshot_with_rows_is_unchanged(store):
    store.snapshot(9, pa.record_batch({"k": [1, 2]}))
    assert store.restore(9).num_rows == 2


def test_an_absent_snapshot_is_still_none(store):
    assert store.restore(99) is None


def test_prune_removes_orphaned_temp_files(store, tmp_path):
    """A crash between the write and the rename leaves one behind, and nothing else ever
    removed it — so the directory this method bounds grew one orphan per crash."""
    directory = tmp_path / "state"
    (directory / "batch-00000005.arrow.tmp").write_bytes(b"partial")
    store.snapshot(6, pa.record_batch({"k": [1]}))

    store.prune(6)
    assert not (directory / "batch-00000005.arrow.tmp").exists()
    assert store.restore(6) is not None


def _windowed_fold():
    ds = bt.from_pydict({"ts": [_at(0)], "v": [1]}).with_watermark("ts", "5 minutes")
    agg = ds.group_by(w=bt.window(col("ts"), "10 minutes")).agg(total=col("v").sum())._plan
    return agg, _WindowedAggFold(agg, _window_key(agg))


def test_the_end_of_stream_flush_leaves_exactly_the_shape_that_was_being_dropped(store):
    """Not a synthetic case: `finalize` empties the fold, so the *last* snapshot of every
    windowed query is the zero-column one."""
    agg, fold = _windowed_fold()
    fold.push(pa.record_batch({"ts": [_at(0)], "v": [1]}, schema=_SCHEMA))
    fold.flush()

    state = fold.state()
    assert state.num_columns == 0, "the flush left something other than the metadata-only state"
    assert fold._wm is not None

    store.snapshot(3, state)
    restored = store.restore(3)
    assert restored is not None

    resumed = _WindowedAggFold(agg, _window_key(agg))
    resumed.restore(restored)
    assert resumed._wm == fold._wm, "the watermark rewound across the restart"
