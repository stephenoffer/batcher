"""Auto Loader state handling and streaming-sink memory behavior.

The contracts here are the ones that only bite at scale or over time: per-trigger
filesystem chatter, a completion list that goes quadratic on a backlog, a guard whose cost
grows with what it guards, and a row-wise sink that materializes a whole micro-batch as
Python objects before the first row is written.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from batcher._internal.errors import IOError as BatcherIOError
from batcher.io.formats.streaming.autoloader import IncrementalFileSource
from batcher.io.formats.streaming.seen_store import SeenStore
from batcher.io.formats.streaming.sinks import (
    _MEMORY,
    ForeachStreamSink,
    MemoryStreamSink,
    memory_table,
)


def _write(dir_path, name: str, rows: int = 2) -> str:
    path = str(dir_path / name)
    pq.write_table(pa.table({"a": list(range(rows))}), path)
    return path


def _source(tmp_path, **kwargs) -> IncrementalFileSource:
    return IncrementalFileSource(
        str(tmp_path / "data"),
        "parquet",
        state_dir=str(tmp_path / "state"),
        **kwargs,
    )


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    return d


# --------------------------------------------------------------------------
# The seen-store connection is opened once, not per trigger.
# --------------------------------------------------------------------------
def test_the_seen_store_is_opened_once_for_the_life_of_the_source(tmp_path, data_dir, monkeypatch):
    """`discover()` and `confirm()` each opened a fresh SQLite connection, preceded by a
    `mkdirs`. A streaming query calls both once per trigger, so a 200ms cadence paid four
    filesystem round-trips a second before listing a single file."""
    opened: list[str] = []
    real_init = SeenStore.__init__

    def counting_init(self, path):
        opened.append(path)
        real_init(self, path)

    monkeypatch.setattr(SeenStore, "__init__", counting_init)

    src = _source(tmp_path)
    for i in range(5):
        _write(data_dir, f"f{i}.parquet")
        src.discover()
        src.complete(src._pending)
        src.confirm()
    assert len(opened) == 1
    src.close()
    src.close()  # idempotent


def test_the_store_is_reopened_after_close(tmp_path, data_dir):
    src = _source(tmp_path)
    _write(data_dir, "a.parquet")
    assert src.discover() == [str(data_dir / "a.parquet")]
    src.complete(src._pending)
    src.confirm()
    src.close()
    # A fresh source over the same state directory still sees the file as processed.
    reopened = _source(tmp_path)
    assert reopened.discover() == []
    reopened.close()


def test_seen_store_survives_a_reopen_under_write_ahead_logging(tmp_path):
    path = str(tmp_path / "seen.sqlite")
    store = SeenStore(path)
    mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
    store.mark_many([("a", 1, 1.0), ("b", 2, 2.0)])
    store.close()
    reopened = SeenStore(path)
    assert reopened.unseen(["a", "b", "c"]) == ["c"]
    reopened.close()


# --------------------------------------------------------------------------
# `complete()` must not be quadratic in the backlog.
# --------------------------------------------------------------------------
def test_complete_dedups_without_scanning_what_it_has_already_collected(tmp_path):
    src = _source(tmp_path)
    files = [f"f{i}.parquet" for i in range(2000)]
    src.complete(files)
    src.complete(files)  # a re-reported file is still recorded once
    assert src._completed == files
    assert src._completed_set == set(files)
    src.close()


def test_confirm_clears_both_the_list_and_its_membership_set(tmp_path, data_dir):
    src = _source(tmp_path)
    _write(data_dir, "a.parquet")
    new = src.discover()
    src.complete(new)
    src.confirm()
    assert src._completed == [] and src._completed_set == set()
    # Re-completing the same path after a confirm records it again (it is a fresh epoch's
    # claim), rather than being swallowed by a stale membership set.
    src.complete(new)
    assert src._completed == new
    src.close()


def test_seek_rebuilds_the_membership_set_so_a_recovered_file_confirms_once(tmp_path, data_dir):
    path = _write(data_dir, "a.parquet")
    src = _source(tmp_path)
    src.seek({"pending": [path]})
    assert src._completed_set == set()  # confirmed and cleared
    src.close()
    reopened = _source(tmp_path)
    assert reopened.discover() == []  # the recovered file is durably seen
    reopened.close()


# --------------------------------------------------------------------------
# Schema inference on an empty directory.
# --------------------------------------------------------------------------
def test_schema_on_an_empty_directory_names_the_path_not_an_index_error(tmp_path, data_dir):
    src = _source(tmp_path)
    with pytest.raises(BatcherIOError, match=r"no \.parquet files yet"):
        src.schema()
    src.close()


# --------------------------------------------------------------------------
# Sinks.
# --------------------------------------------------------------------------
def test_foreach_converts_one_record_batch_at_a_time():
    """`Table.to_pylist()` builds a dict per row of the *entire* micro-batch before the
    first `fn(row)` runs, on top of the Arrow table it came from."""
    seen: list[int] = []

    table = pa.Table.from_batches([pa.record_batch({"a": [i, i + 1]}) for i in range(0, 10, 2)])

    def fn(row):
        seen.append(row["a"])

    assert ForeachStreamSink(fn).write_batch(3, table) == "foreach:3"
    assert seen == list(range(10))
    assert table.num_rows == 10  # five chunks of two, converted chunk by chunk


def test_foreach_handles_an_empty_micro_batch():
    calls: list[object] = []
    empty = pa.table({"a": pa.array([], type=pa.int64())})
    assert ForeachStreamSink(calls.append).write_batch(0, empty) == "foreach:0"
    assert calls == []


def test_the_memory_sink_guard_tracks_a_running_total_not_a_full_rescan():
    """Summing `nbytes` over every retained table on every micro-batch made the guard
    against unbounded growth itself quadratic in the number of micro-batches."""
    sink = MemoryStreamSink("running-total")
    sink.open()
    batch = pa.table({"a": list(range(100))})
    for i in range(4):
        sink.write_batch(i, batch)
    assert sink._held == batch.nbytes * 4
    assert memory_table("running-total").num_rows == 400
    _MEMORY.pop("running-total", None)


def test_complete_mode_replaces_and_does_not_accumulate():
    sink = MemoryStreamSink("complete-mode", output_mode="complete")
    sink.open()
    for i in range(3):
        sink.write_batch(i, pa.table({"a": [i]}))
    assert memory_table("complete-mode").to_pydict() == {"a": [2]}
    # The retained size is the *current* table's, not zero and not a running total: complete
    # mode replaces what it holds each micro-batch, and the size guard has to measure what is
    # actually held or a high-cardinality running result grows past the budget unnoticed.
    assert sink._held == pa.table({"a": [2]}).nbytes
    _MEMORY.pop("complete-mode", None)


def test_reopening_a_memory_sink_resets_its_running_total():
    sink = MemoryStreamSink("reopened")
    sink.open()
    sink.write_batch(0, pa.table({"a": list(range(10))}))
    assert sink._held > 0
    sink.open()
    assert sink._held == 0
    assert memory_table("reopened").num_rows == 0
    _MEMORY.pop("reopened", None)
