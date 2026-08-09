"""Streaming sinks: idempotency tokens, memory-store lifecycle, and Delta race safety.

These tests exercise `batcher.io.formats.streaming.sinks` directly at the `StreamSink`
boundary, the way the engine's micro-batch runner drives it (`open` -> many
`write_batch` -> `close`). None of them need a live cluster.

The Delta conflict-handling test uses a *fake* ``SINKS.get("delta")`` double so it runs
with no ``deltalake`` extra installed: it reproduces the check-then-commit race (two
writers sharing an ``app_id`` both pass the pre-check, the log rejects the loser at
commit time) and pins that the loser treats the rejection as already-committed rather
than crashing the query.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher._internal.errors import CommitError
from batcher.io.formats.streaming import sinks as sinks_mod
from batcher.io.formats.streaming.sinks import (
    ForeachBatchStreamSink,
    ForeachStreamSink,
    MemoryStreamSink,
    memory_table,
)


# --------------------------------------------------------------------------
# foreach / foreach_batch: a batch-id-bearing token, not None, so the commit
# log records that the micro-batch was delivered.
# --------------------------------------------------------------------------
def test_foreach_batch_returns_a_batch_id_token():
    seen: list[tuple[int, int]] = []
    sink = ForeachBatchStreamSink(lambda table, bid: seen.append((bid, table.num_rows)))
    sink.open()
    token = sink.write_batch(7, pa.table({"x": pa.array([1, 2, 3], pa.int64())}))
    sink.close()

    assert token == "foreach_batch:7"
    assert seen == [(7, 3)]


def test_foreach_returns_a_batch_id_token():
    rows: list[dict] = []
    sink = ForeachStreamSink(rows.append)
    sink.open()
    token = sink.write_batch(4, pa.table({"x": pa.array([10, 20], pa.int64())}))
    sink.close()

    assert token == "foreach:4"
    assert rows == [{"x": 10}, {"x": 20}]


# --------------------------------------------------------------------------
# memory sink: read-after-stop is its whole purpose, so close() frees nothing.
# --------------------------------------------------------------------------
def test_memory_sink_accumulates_and_is_read_after_close():
    name = "test_mem_accumulate"
    sinks_mod._MEMORY.pop(name, None)  # start clean regardless of prior runs
    sink = MemoryStreamSink(name)
    sink.open()
    sink.write_batch(0, pa.table({"x": pa.array([1], pa.int64())}))
    sink.write_batch(1, pa.table({"x": pa.array([2], pa.int64())}))
    sink.close()

    # read-after-stop is the memory sink's whole purpose: close() frees nothing.
    assert memory_table(name).column("x").to_pylist() == [1, 2]
    sinks_mod._MEMORY.pop(name, None)  # cleanup


# --------------------------------------------------------------------------
# Delta sink: the check-then-commit race, reproduced with a fake sink double.
# --------------------------------------------------------------------------
class _FakeWritten:
    def __init__(self, rows: int) -> None:
        self.rows = rows
        self.path = "part-0.parquet"


class _FakeDeltaSink:
    """A ``delta`` sink double whose commit conflicts, driven by class-level state.

    ``committed`` models the shared Delta log: ``is_committed`` reads it, and
    ``commit`` either records the transaction or — when ``conflict_on_commit`` is set —
    raises ``CommitError`` the way the real optimistic-concurrency check does when a
    peer won the race. Setting ``peer_wins`` makes the log show the transaction as
    already present after that rejection (the benign case); leaving it clear models a
    genuine commit failure (the transaction never lands).
    """

    committed = False
    conflict_on_commit = False
    peer_wins = False
    commit_calls = 0

    def __init__(self, *, app_id=None, txn_version=None, **_):
        self.app_id = app_id
        self.txn_version = txn_version

    def is_committed(self, _uri):
        return type(self).committed

    def write(self, _table, _uri):
        return _FakeWritten(rows=5)

    def commit(self, _manifest, _uri):
        cls = type(self)
        cls.commit_calls += 1
        if cls.conflict_on_commit:
            if cls.peer_wins:
                cls.committed = True  # the peer's transaction is now in the log
            raise CommitError("Delta commit conflicted with a concurrent writer")
        cls.committed = True


@pytest.fixture
def fake_delta(monkeypatch):
    _FakeDeltaSink.committed = False
    _FakeDeltaSink.conflict_on_commit = False
    _FakeDeltaSink.peer_wins = False
    _FakeDeltaSink.commit_calls = 0
    monkeypatch.setattr(
        "batcher.io.formats.SINKS.get",
        lambda name: _FakeDeltaSink if name == "delta" else None,
    )
    return _FakeDeltaSink


def _delta_sink(uri="mem://t"):
    return sinks_mod.DeltaStreamSink(uri, query_name="q")


def test_delta_happy_path_commits_once(fake_delta):
    sink = _delta_sink()
    sink.open()
    token = sink.write_batch(0, pa.table({"id": pa.array([1], pa.int64())}))
    assert token == "delta:0:5"
    assert fake_delta.commit_calls == 1
    assert fake_delta.committed is True


def test_delta_precheck_skips_a_replayed_batch(fake_delta):
    fake_delta.committed = True  # the transaction is already in the log
    sink = _delta_sink()
    sink.open()
    token = sink.write_batch(2, pa.table({"id": pa.array([2], pa.int64())}))
    assert token == "delta:2:already-committed"
    assert fake_delta.commit_calls == 0  # nothing committed, no duplicate


def test_delta_commit_conflict_won_by_peer_is_treated_as_committed(fake_delta):
    """The race this fix pins.

    Two writers share the ``app_id`` and both pass the pre-check. The loser's commit is
    rejected by the log, but the batch *is* durably committed by the peer, so the loser
    must not crash the query — it re-reads the log, sees the transaction, and reports it
    as already-committed.
    """
    fake_delta.conflict_on_commit = True
    fake_delta.peer_wins = True
    sink = _delta_sink()
    sink.open()
    token = sink.write_batch(3, pa.table({"id": pa.array([3], pa.int64())}))
    assert token == "delta:3:already-committed"
    assert fake_delta.commit_calls == 1


def test_delta_genuine_commit_failure_still_raises(fake_delta):
    """A conflict the log does *not* resolve to a committed transaction is a real error.

    If the transaction is still absent after the rejection, swallowing it would drop the
    micro-batch silently. The sink must re-raise.
    """
    fake_delta.conflict_on_commit = True
    fake_delta.peer_wins = False
    sink = _delta_sink()
    sink.open()
    with pytest.raises(CommitError):
        sink.write_batch(4, pa.table({"id": pa.array([4], pa.int64())}))


# --- console options and the ForeachWriter shape ------------------------------------


def test_the_console_sink_truncates_wide_strings_by_default(capsys):
    """Spark truncates by default, and a Kafka stream's `value` column is exactly the
    shape that fills a terminal without it."""
    from batcher.io.formats.streaming.sinks import ConsoleStreamSink

    ConsoleStreamSink().write_batch(0, pa.table({"v": ["x" * 50]}))
    printed = capsys.readouterr().out
    assert "x" * 20 in printed
    assert "x" * 21 not in printed


def test_the_console_sink_can_be_told_not_to_truncate(capsys):
    from batcher.io.formats.streaming.sinks import ConsoleStreamSink

    ConsoleStreamSink(truncate=False).write_batch(0, pa.table({"v": ["x" * 50]}))
    assert "x" * 50 in capsys.readouterr().out


def test_the_console_sink_takes_an_explicit_truncation_width(capsys):
    from batcher.io.formats.streaming.sinks import ConsoleStreamSink

    ConsoleStreamSink(truncate=5).write_batch(0, pa.table({"v": ["abcdefghij"]}))
    out = capsys.readouterr().out
    assert "abcde" in out and "abcdef" not in out


def test_truncation_leaves_non_string_columns_alone(capsys):
    from batcher.io.formats.streaming.sinks import ConsoleStreamSink

    ConsoleStreamSink().write_batch(0, pa.table({"n": [1234567890123456789]}))
    assert "1234567890123456789" in capsys.readouterr().out


def test_a_foreach_writer_gets_open_process_close_in_order():
    from batcher.io.formats.streaming.sinks import ForeachStreamSink, ForeachWriter

    class _Writer(ForeachWriter):
        def __init__(self):
            self.log = []

        def open(self, partition_id, epoch_id):
            self.log.append(("open", partition_id, epoch_id))
            return True

        def process(self, row):
            self.log.append(("process", row["n"]))

        def close(self, error):
            self.log.append(("close", error))

    writer = _Writer()
    ForeachStreamSink(writer).write_batch(7, pa.table({"n": [1, 2]}))
    assert writer.log == [("open", 0, 7), ("process", 1), ("process", 2), ("close", None)]


def test_a_foreach_writer_that_declines_an_epoch_processes_nothing():
    from batcher.io.formats.streaming.sinks import ForeachStreamSink, ForeachWriter

    class _Writer(ForeachWriter):
        def __init__(self):
            self.rows = []

        def open(self, partition_id, epoch_id):
            return False

        def process(self, row):
            self.rows.append(row)

    writer = _Writer()
    token = ForeachStreamSink(writer).write_batch(0, pa.table({"n": [1]}))
    assert writer.rows == []
    assert token.endswith("skipped")


def test_a_failing_epoch_still_closes_the_writer_with_the_error():
    """The whole reason the shape exists: the connection is released either way."""
    from batcher.io.formats.streaming.sinks import ForeachStreamSink, ForeachWriter

    class _Writer(ForeachWriter):
        def __init__(self):
            self.closed = []

        def process(self, row):
            raise RuntimeError("destination refused")

        def close(self, error):
            self.closed.append(error)

    writer = _Writer()
    with pytest.raises(RuntimeError, match="destination refused"):
        ForeachStreamSink(writer).write_batch(0, pa.table({"n": [1]}))
    assert isinstance(writer.closed[0], RuntimeError)


def test_a_plain_function_still_works_unchanged():
    from batcher.io.formats.streaming.sinks import ForeachStreamSink

    seen = []
    ForeachStreamSink(seen.append).write_batch(0, pa.table({"n": [1, 2]}))
    assert [r["n"] for r in seen] == [1, 2]


def test_an_object_with_only_process_is_accepted():
    """Spark's `ForeachWriter` requires all three; a writer with just `process` is the
    minimal useful shape and must not need the base class."""
    from batcher.io.formats.streaming.sinks import ForeachStreamSink

    class _Minimal:
        def __init__(self):
            self.rows = []

        def process(self, row):
            self.rows.append(row)

    writer = _Minimal()
    ForeachStreamSink(writer).write_batch(0, pa.table({"n": [1]}))
    assert len(writer.rows) == 1
