"""A shuffle file is published whole by a clean close, or not at all.

Shuffle map and reduce tasks are speculated (`gather_with_backups`, one backup by default),
so two copies of a task write the same deterministic path, and the barrier hands the first
finisher's paths to the reducers while the slower copy is still running. Written in place,
the slower copy re-opened the path with `O_TRUNC` under a reader already given it: the
skewed-sort integration tests failed on a loaded 64-core gate with `ArrowInvalid: Tried
reading schema message, was null or length 0`, a reducer reading a zero-byte bucket. A file
caught half rewritten would have been worse, a stream that simply ends early, read as a
complete, smaller bucket.

These pin the three properties that close it: a second writer to a published path does not
disturb it until it publishes, a write that fails part-way publishes nothing, and an aborted
or failed writer leaves no partial file behind.
"""

from __future__ import annotations

import os

import pyarrow as pa
import pytest

from batcher.dist.shuffle_io import IpcWriter, read_ipc, write_ipc, write_ipc_round_robin

pytestmark = pytest.mark.unit


def _batch(lo: int, n: int) -> pa.RecordBatch:
    return pa.record_batch({"v": pa.array(range(lo, lo + n), type=pa.int64())})


def _rows(path: str) -> int:
    return sum(b.num_rows for b in read_ipc(path))


def test_a_second_writer_does_not_disturb_the_published_file_until_it_publishes(tmp_path):
    """The speculative backup's position: the same path, written while it is being read."""
    path = str(tmp_path / "m0_r0.arrow")
    write_ipc([_batch(0, 1000)], path)

    late = IpcWriter(path)
    late.write(_batch(0, 400))  # opened and part-written: in place, this truncated `path`
    assert _rows(path) == 1000
    with open(path, "rb") as reader_held:  # a reducer that opened the first copy
        late.write(_batch(400, 600))
        late.close()
        held = pa.ipc.open_stream(reader_held).read_all()
    assert held.num_rows == 1000
    assert _rows(path) == 1000


def test_a_write_that_fails_part_way_publishes_nothing(tmp_path):
    path = str(tmp_path / "bucket.arrow")
    with pytest.raises(RuntimeError, match="worker lost"), IpcWriter(path) as writer:
        writer.write(_batch(0, 10))
        raise RuntimeError("worker lost")
    assert not os.path.exists(path)
    assert os.listdir(tmp_path) == [], "the partial file was left behind"


def test_a_failed_rewrite_leaves_the_published_file_intact(tmp_path):
    """A cancelled backup must not replace a complete bucket with the rows it got to."""
    path = str(tmp_path / "bucket.arrow")
    write_ipc([_batch(0, 1000)], path)
    with pytest.raises(KeyboardInterrupt), IpcWriter(path) as writer:
        writer.write(_batch(0, 10))
        raise KeyboardInterrupt  # what a best-effort `ray.cancel` raises in the loser
    assert _rows(path) == 1000
    assert sorted(os.listdir(tmp_path)) == ["bucket.arrow"]


def test_abort_reports_no_file_and_close_after_it_is_a_no_op(tmp_path):
    path = str(tmp_path / "bucket.arrow")
    writer = IpcWriter(path)
    writer.write(_batch(0, 5))
    writer.abort()
    assert writer.close() is None
    assert not writer.is_open
    assert os.listdir(tmp_path) == []


def test_a_clean_close_publishes_at_the_named_path(tmp_path):
    """Positive control: everything above is about what does *not* appear."""
    path = str(tmp_path / "bucket.arrow")
    with IpcWriter(path) as writer:
        writer.write(_batch(0, 7))
    assert writer.close() == path
    assert _rows(path) == 7
    assert os.listdir(tmp_path) == ["bucket.arrow"]


def test_round_robin_publishes_every_partition_or_none(tmp_path):
    paths = [str(tmp_path / f"part-{i}.arrow") for i in range(3)]
    write_ipc_round_robin(iter([_batch(0, 4), _batch(4, 4)]), _batch(0, 0).schema, paths)
    assert [_rows(p) for p in paths] == [4, 4, 0]

    def failing():
        yield _batch(0, 4)
        raise OSError("source went away")

    fresh = [str(tmp_path / f"fresh-{i}.arrow") for i in range(3)]
    with pytest.raises(OSError, match="went away"):
        write_ipc_round_robin(failing(), _batch(0, 0).schema, fresh)
    assert not any(os.path.exists(p) for p in fresh)
    assert sorted(os.listdir(tmp_path)) == sorted(os.path.basename(p) for p in paths)
