"""Micro-batch epoch integrity: one sink write per epoch, and no id collisions on restart.

Both contracts here failed *silently* and only on a plan shape or a lifecycle step that the
existing suite did not combine. They are pinned against fakes rather than a live sink so the
sink's exactly-once check — which is what actually swallowed the rows — is visible.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.core.streaming_query import StreamingQueryEngine
from batcher.core.streaming_runner import LocalRunner
from batcher.io.formats.streaming.checkpoint.store import CheckpointStore
from batcher.plan.streaming import Trigger


class _IdempotentSink:
    """A sink keyed on `batch_id`, the way `FileStreamSink` and `DeltaStreamSink` are.

    A second write under an id it has already seen is *dropped*, exactly as a `part-batch`
    file that already exists is skipped and a recorded `(app_id, batch_id)` Delta transaction
    commits nothing.
    """

    def __init__(self) -> None:
        self.written: dict[int, pa.Table] = {}
        self.dropped: list[int] = []

    def open(self) -> None:
        pass

    def write_batch(self, batch_id: int, table: pa.Table) -> str | None:
        if batch_id in self.written:
            self.dropped.append(batch_id)
            return f"already:{batch_id}"
        self.written[batch_id] = table
        return f"ok:{batch_id}"

    def close(self) -> None:
        pass

    def rows(self) -> list[int]:
        out: list[int] = []
        for table in self.written.values():
            out.extend(table.column("a").to_pylist())
        return sorted(out)


class _SplittingProcessor:
    """A processor whose epoch output is several record batches — a plan wider than a
    morsel, or a union, produces exactly this."""

    def process(self, batch: pa.RecordBatch) -> list[pa.RecordBatch]:
        values = batch.column("a").to_pylist()
        return [pa.record_batch({"a": [v]}) for v in values]


class _FiniteSource:
    bounded = True

    def __init__(self, batches: list[pa.RecordBatch]) -> None:
        self._batches = batches

    def schema(self) -> pa.Schema:
        return self._batches[0].schema

    def row_count(self) -> int | None:
        return sum(b.num_rows for b in self._batches)

    def read(self, projection=None):
        return list(self._batches)

    def iter_batches(self, projection=None):
        yield from self._batches


def test_an_epoch_whose_output_is_several_batches_is_written_once_and_whole():
    """Each output batch used to be a separate `write_batch` under the *same* `batch_id`, so
    the sink's own exactly-once check dropped everything after the first."""
    sink = _IdempotentSink()
    source = _FiniteSource([pa.record_batch({"a": [1, 2, 3]})])
    runner = LocalRunner(source, _SplittingProcessor(), sink)

    staged = runner.stage(0)
    consumed, emitted = runner.publish(0, staged)

    assert (consumed, emitted) == (3, 3)
    assert sink.dropped == []
    assert sink.rows() == [1, 2, 3]


def test_an_epoch_with_no_output_writes_nothing():
    class _Dropping:
        def process(self, batch):
            return [pa.record_batch({"a": pa.array([], type=pa.int64())})]

    sink = _IdempotentSink()
    source = _FiniteSource([pa.record_batch({"a": [1, 2]})])
    runner = LocalRunner(source, _Dropping(), sink)
    assert runner.publish(0, runner.stage(0)) == (2, 0)
    assert sink.written == {}


# --------------------------------------------------------------------------
# The end-of-stream flush must claim the batch id it wrote under.
# --------------------------------------------------------------------------
class _FlushingProcessor:
    """Emits nothing per micro-batch and one batch at end-of-stream — the shape of a
    windowed aggregate whose windows are all still open when the query stops."""

    def __init__(self) -> None:
        self.seen: list[int] = []

    def process(self, batch: pa.RecordBatch) -> list[pa.RecordBatch]:
        self.seen.extend(batch.column("a").to_pylist())
        return []

    def finalize(self) -> list[pa.RecordBatch]:
        return [pa.record_batch({"a": list(self.seen)})] if self.seen else []

    def snapshot_state(self) -> pa.RecordBatch | None:
        return pa.record_batch({"a": list(self.seen)})

    def restore_state(self, state: pa.RecordBatch) -> None:
        self.seen = list(state.column("a").to_pylist())


class _Checkpointable(_FiniteSource):
    """A finite source with a resume position, so the engine exercises the checkpoint path."""

    def __init__(self, batches):
        super().__init__(batches)
        self._at = 0

    def iter_batches(self, projection=None):
        for i, b in enumerate(self._batches):
            if i < self._at:
                continue
            self._at = i + 1
            yield b

    def snapshot_position(self) -> dict:
        return {"at": self._at}

    def seek(self, position: dict) -> None:
        self._at = int(position.get("at", 0))


def _run(sink, source, processor, checkpoint):
    """Start a query and wait for it to drain. `checkpoint` may be a store or a location."""
    if isinstance(checkpoint, str):
        checkpoint = CheckpointStore(checkpoint)
    engine = StreamingQueryEngine(
        name="flush-test",
        source=source,
        sink=sink,
        processor=processor,
        trigger=Trigger.available_now(),
        output_mode="append",
        checkpoint=checkpoint,
    )
    engine.start()
    assert engine.await_termination(30) is True
    return engine


def test_the_final_flush_claims_its_batch_id_so_a_restart_is_not_swallowed(tmp_path):
    """The flush wrote under the id the next micro-batch would take and recorded no commit,
    so recovery resumed at that same id — and the restarted query's very first epoch hit the
    sink's idempotency check against the *previous run's flush*, dropping the whole epoch."""
    sink = _IdempotentSink()
    location = str(tmp_path / "ckpt")
    first = _run(
        sink,
        _Checkpointable([pa.record_batch({"a": [1, 2]})]),
        _FlushingProcessor(),
        CheckpointStore(location),
    )
    flushed_at = max(sink.written)
    assert sink.rows() == [1, 2]

    # The flush's batch id is committed, so recovery starts strictly after it. A new store
    # over the same directory is what a restart opens — the query's own store is closed with
    # its loop, so nothing outlives the query holding its connections.
    reopened = CheckpointStore(location)
    assert reopened.commits.is_committed(flushed_at)
    assert first._batches == flushed_at + 1
    reopened.close()

    # The restarted query replays the same topic and resumes after the checkpointed batch.
    replayed = _Checkpointable([pa.record_batch({"a": [1, 2]}), pa.record_batch({"a": [3, 4]})])
    second = _run(sink, replayed, _FlushingProcessor(), CheckpointStore(location))
    assert sink.dropped == []  # nothing collided with the previous run's flush
    assert {3, 4} <= set(sink.rows())  # the epoch after the restart really landed
    assert second._batches > flushed_at + 1


def test_a_query_with_nothing_to_flush_records_only_the_drain(tmp_path):
    """One epoch of rows, then the drain's terminal checkpoint — and no flush batch, because
    a stateless processor has nothing left open when the source is spent."""

    class _Plain:
        def process(self, batch):
            return [batch]

    sink = _IdempotentSink()
    location = str(tmp_path / "ckpt")
    engine = _run(sink, _Checkpointable([pa.record_batch({"a": [7]})]), _Plain(), location)
    assert sink.rows() == [7]
    assert engine._batches == 2  # batch 0 carried the rows, batch 1 is the drain marker

    store = CheckpointStore(location)
    # The drain marker records where the source finally stood, so recovery resumes *after*
    # the window rather than replaying it.
    assert store.offsets.position_at(1) == {0: {"at": 1}}
    assert store.commits.is_committed(1)
    store.close()


# --------------------------------------------------------------------------
# Checkpoint log pruning is amortized but still bounded.
# --------------------------------------------------------------------------
def test_log_pruning_is_amortized_and_still_bounds_the_logs(tmp_path):
    """Pruning per commit cost two extra `DELETE`s and two fsyncs an epoch to remove a single
    row. The property that matters is that the logs stay bounded, not that they hold one row."""
    from batcher.io.formats.streaming.checkpoint.store import _PRUNE_EVERY

    store = CheckpointStore(str(tmp_path / "ckpt"))
    for batch_id in range(_PRUNE_EVERY * 3):
        store.record_offsets(batch_id, {0: {"at": batch_id}})
        store.commit(batch_id)
        store.prune_logs(batch_id)

    rows = store.offsets._conn.execute("SELECT COUNT(*) FROM offsets").fetchone()[0]
    assert 0 < rows <= _PRUNE_EVERY + 1  # bounded by the stride, not by the query's lifetime
    # The last committed batch's position is what recovery reads, and it survived every sweep.
    last = _PRUNE_EVERY * 3 - 1
    assert store.offsets.position_at(last) == {0: {"at": last}}
    assert store.commits.last_committed() == last


@pytest.mark.parametrize("log", ["offsets", "commits"])
def test_the_checkpoint_logs_run_in_write_ahead_mode(tmp_path, log):
    store = CheckpointStore(str(tmp_path / "ckpt"))
    conn = getattr(store, log)._conn
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
