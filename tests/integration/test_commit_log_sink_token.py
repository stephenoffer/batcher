"""The commit log must record *what* the sink wrote, not only that a batch committed.

Every `StreamSink` returns a token — a `part-batch` path, a Delta `(app_id, batch_id)`
marker, a `foreach_batch:{id}` receipt — and the commit log has always had a `sink_token`
column. Nothing carried the value between them, so the column was NULL for every row ever
written, and there was no reader for it either: a write-only column is not a record.
"""

from __future__ import annotations

import pyarrow as pa

from batcher.core.streaming_query import StreamingQueryEngine
from batcher.core.streaming_runner import LocalRunner
from batcher.io.formats.streaming.checkpoint.store import CheckpointStore
from batcher.plan.streaming import Trigger


class _TokenSink:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def open(self) -> None:
        pass

    def write_batch(self, batch_id: int, table: pa.Table) -> str | None:
        self.calls.append(batch_id)
        return f"part-batch{batch_id:05d}.parquet"

    def close(self) -> None:
        pass


class _Source:
    bounded = True

    def __init__(self, batches):
        self._batches = batches
        self._at = 0

    def schema(self):
        return self._batches[0].schema

    def row_count(self):
        return sum(b.num_rows for b in self._batches)

    def read(self, projection=None):
        return list(self._batches)

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


class _Passthrough:
    def process(self, batch):
        return [batch]


def _run(location: str, batches, sink):
    engine = StreamingQueryEngine(
        name="token-test",
        source=_Source(batches),
        sink=sink,
        processor=_Passthrough(),
        trigger=Trigger.available_now(),
        output_mode="append",
        checkpoint=CheckpointStore(location),
    )
    engine.start()
    assert engine.await_termination(30) is True
    return engine


def test_the_commit_log_records_the_token_the_sink_returned(tmp_path):
    location = str(tmp_path / "ckpt")
    sink = _TokenSink()
    _run(location, [pa.record_batch({"a": [1, 2]}), pa.record_batch({"a": [3]})], sink)

    store = CheckpointStore(location)
    try:
        for batch_id in sink.calls:
            assert store.commits.is_committed(batch_id)
            assert store.commits.sink_token(batch_id) == f"part-batch{batch_id:05d}.parquet"
    finally:
        store.close()


def test_a_batch_that_wrote_nothing_records_no_token(tmp_path):
    """The drain marker and an epoch the processor emptied both commit without a write."""

    class _Dropping:
        def process(self, batch):
            return []

    location = str(tmp_path / "ckpt")
    sink = _TokenSink()
    engine = StreamingQueryEngine(
        name="token-empty",
        source=_Source([pa.record_batch({"a": [1]})]),
        sink=sink,
        processor=_Dropping(),
        trigger=Trigger.available_now(),
        output_mode="append",
        checkpoint=CheckpointStore(location),
    )
    engine.start()
    assert engine.await_termination(30) is True

    store = CheckpointStore(location)
    try:
        assert sink.calls == []
        last = store.commits.last_committed()
        assert last is not None
        assert store.commits.sink_token(last) is None
    finally:
        store.close()


def test_the_reader_answers_none_for_an_uncommitted_batch(tmp_path):
    store = CheckpointStore(str(tmp_path / "ckpt"))
    try:
        store.commit(7, "tok")
        assert store.commits.sink_token(7) == "tok"
        assert store.commits.sink_token(8) is None
    finally:
        store.close()


def test_the_runner_reports_the_token_of_the_epoch_it_just_published():
    sink = _TokenSink()
    source = _Source([pa.record_batch({"a": [1, 2]})])
    runner = LocalRunner(source, _Passthrough(), sink)
    assert runner.last_sink_token() is None  # nothing published yet
    runner.publish(4, runner.stage(4))
    assert runner.last_sink_token() == "part-batch00004.parquet"


def test_an_epoch_with_no_output_clears_the_previous_token():
    """Otherwise a batch that wrote nothing would be recorded under the *previous* batch's
    token, which is worse than recording none at all."""

    class _EmptyAfterFirst:
        def __init__(self):
            self.seen = 0

        def process(self, batch):
            self.seen += 1
            return [batch] if self.seen == 1 else []

    sink = _TokenSink()
    source = _Source([pa.record_batch({"a": [1]}), pa.record_batch({"a": [2]})])
    runner = LocalRunner(source, _EmptyAfterFirst(), sink)
    runner.publish(0, runner.stage(0))
    assert runner.last_sink_token() == "part-batch00000.parquet"
    runner.publish(1, runner.stage(1))
    assert runner.last_sink_token() is None
