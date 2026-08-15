"""The shard corpus at scale, and after a crash.

Two properties this format has to hold that a small-corpus test cannot see. At scale the
manifest must not grow with the shard count — a petabyte corpus in 256 MB shards is four
million shards, and a manifest naming each one is parsed on every rank at startup. And a
write that dies must leave something: the corpus is hours of work, and losing it because the
manifest was only published at the end is not a failure mode a long job can carry.
"""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import FormatError
from batcher.io.formats.ml.shards import (
    ShardIndex,
    ShardReader,
    read_shard_index,
    shard_name,
    write_shards,
)


def _batches(rows: int, per_batch: int = 10, fail_at: int | None = None):
    for start in range(0, rows, per_batch):
        if fail_at is not None and start == fail_at:
            raise RuntimeError("the writer's process died")
        yield pa.record_batch({"x": np.arange(start, start + per_batch, dtype=np.int64)})


def test_the_manifest_does_not_grow_with_the_shard_count(tmp_path):
    # The manifest used to name every shard: ~43 bytes each, so 172 MB of JSON and four
    # million Python strings resident on every rank for a petabyte corpus.
    small = tmp_path / "small"
    large = tmp_path / "large"
    write_shards(_batches(50), str(small), rows_per_shard=10)  # 5 shards
    write_shards(_batches(5000), str(large), rows_per_shard=10)  # 500 shards
    small_bytes = (small / "index.json").stat().st_size
    large_bytes = (large / "index.json").stat().st_size
    # A hundred times the shards may cost the extra digits in two integers, and nothing
    # else. Naming each shard cost ~43 bytes apiece, which is 172 MB at four million.
    assert large_bytes - small_bytes < 32, (
        f"100x the shards grew the manifest by {large_bytes - small_bytes} bytes"
    )
    assert large_bytes < 1024
    index = read_shard_index(str(large))
    assert index.uniform
    assert index.explicit_paths == () and index.explicit_rows == ()
    assert index.shard_count == 500


def test_locating_a_row_touches_nothing_per_shard(tmp_path):
    write_shards(_batches(5000), str(tmp_path), rows_per_shard=10)
    index = read_shard_index(str(tmp_path))
    assert index.locate(0) == (0, 0)
    assert index.locate(4999) == (499, 9)
    assert index.locate(1234) == (123, 4)
    with pytest.raises(IndexError):
        index.locate(5000)
    # The compatibility views still answer, and agree with the O(1) accessors.
    assert index.shard_rows == tuple([10] * 500)
    assert index.starts[:3] == (0, 10, 20)
    assert index.shard_paths[7] == index.shard_path(7)


def test_shard_names_sort_lexicographically_into_numeric_order():
    # At five digits they stopped agreeing at 100,000 — `shard-100000` sorts before
    # `shard-99999` — so the documented `bt.read.arrow(dir + "/*.arrow")` fallback, which
    # orders by name, silently returned the rows in the wrong order on exactly the corpora
    # too large for anyone to notice by looking.
    names = [shard_name(i) for i in (0, 9, 99999, 100000, 100001, 9999999)]
    assert sorted(names) == names


def test_a_crashed_write_leaves_a_readable_corpus(tmp_path):
    with pytest.raises(RuntimeError, match="died"):
        write_shards(_batches(1000, fail_at=600), str(tmp_path), rows_per_shard=10)
    partial = read_shard_index(str(tmp_path))
    assert partial.total_rows > 0, "the manifest was never published, so nothing is readable"
    assert partial.total_rows % 10 == 0, "a partial shard must not be described as complete"
    rows = ShardReader(str(tmp_path), cache_size=64).take(list(range(partial.total_rows)))
    # Order-sensitive: a partial corpus must still be the *prefix* of the intended one.
    assert rows.column("x").to_pylist() == list(range(partial.total_rows))


def test_a_crashed_write_resumes_without_rewriting_what_landed(tmp_path):
    with pytest.raises(RuntimeError):
        write_shards(_batches(1000, fail_at=600), str(tmp_path), rows_per_shard=10)
    before = read_shard_index(str(tmp_path)).total_rows
    final = write_shards(_batches(1000), str(tmp_path), rows_per_shard=10, resume=True)
    assert final.total_rows == 1000
    assert before < 1000, "the crash test is not exercising a partial write"
    rows = ShardReader(str(tmp_path), cache_size=200).take(list(range(1000)))
    assert rows.column("x").to_pylist() == list(range(1000))


def test_resuming_a_finished_corpus_is_a_no_op(tmp_path):
    write_shards(_batches(100), str(tmp_path), rows_per_shard=10)
    again = write_shards(_batches(100), str(tmp_path), rows_per_shard=10, resume=True)
    assert (again.total_rows, again.shard_count) == (100, 10)
    assert again.schema is not None, "a no-op resume must not forget the corpus schema"


def test_resume_refuses_a_different_shard_width(tmp_path):
    # Two widths in one corpus makes a global row index mean two different things.
    write_shards(_batches(100), str(tmp_path), rows_per_shard=10)
    with pytest.raises(FormatError, match="rows_per_shard"):
        write_shards(_batches(200), str(tmp_path), rows_per_shard=25, resume=True)


def test_resume_on_a_directory_that_was_never_written_is_a_fresh_write(tmp_path):
    index = write_shards(_batches(50), str(tmp_path), rows_per_shard=10, resume=True)
    assert index.total_rows == 50


def test_an_index_that_names_its_shards_is_still_read(tmp_path):
    # The representation older corpora use, and what a hand-assembled manifest may look
    # like. It keeps the per-shard tuples and the bisect rather than the arithmetic.
    write_shards(_batches(50), str(tmp_path), rows_per_shard=10)
    document = json.loads((tmp_path / "index.json").read_text())
    document.pop("uniform")
    document.pop("shard_count")
    document["shards"] = [{"path": shard_name(i), "rows": 10} for i in range(5)]
    (tmp_path / "index.json").write_text(json.dumps(document))

    index = read_shard_index(str(tmp_path))
    assert not index.uniform
    assert index.shard_count == 5
    assert index.locate(34) == (3, 4)
    assert ShardReader(str(tmp_path)).take([49, 0]).column("x").to_pylist() == [49, 0]


def test_a_ragged_index_locates_rows_across_uneven_shards(tmp_path):
    # A hand-built manifest may describe shards of differing widths; the arithmetic path
    # would place every row wrong, so this must take the bisect.
    index = ShardIndex(
        rows_per_shard=10,
        total_rows=25,
        shard_count=3,
        explicit_paths=("a", "b", "c"),
        explicit_rows=(5, 15, 5),
    )
    assert [index.locate(i) for i in (0, 4, 5, 19, 20, 24)] == [
        (0, 0),
        (0, 4),
        (1, 0),
        (1, 14),
        (2, 0),
        (2, 4),
    ]


class _FlakyFilesystem:
    """A filesystem whose first `n` opens fail, then behave. Wrapping rather than patching
    because the real filesystem objects use ``__slots__`` and reject attribute assignment."""

    def __init__(self, real, error: Exception, failures: int) -> None:
        self._real = real
        self._error = error
        self._left = failures
        self.opens = 0

    def open(self, path, *args, **kwargs):
        self.opens += 1
        if self._left > 0:
            self._left -= 1
            raise self._error
        return self._real.open(path, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_a_shard_read_survives_a_transient_failure(tmp_path, monkeypatch):
    # A training run reads its corpus for hours or days against object storage, so it will
    # meet a 503 or a dropped connection. Without a retry that blip ends the job.
    write_shards(_batches(50), str(tmp_path), rows_per_shard=10)
    monkeypatch.setattr("batcher.io.base._transient.time.sleep", lambda _s: None)
    reader = ShardReader(str(tmp_path), cache_size=8)
    flaky = _FlakyFilesystem(reader._fs, ConnectionError("503 SlowDown"), failures=1)
    reader._fs = flaky
    assert reader.take([0, 1, 2]).column("x").to_pylist() == [0, 1, 2]
    assert flaky.opens == 2, "the failed read should have been retried exactly once"


def test_a_permanent_read_failure_is_not_retried(tmp_path, monkeypatch):
    # Retrying a fact that will not change only spends the budget before failing anyway.
    write_shards(_batches(50), str(tmp_path), rows_per_shard=10)
    monkeypatch.setattr("batcher.io.base._transient.time.sleep", lambda _s: None)
    reader = ShardReader(str(tmp_path), cache_size=8)
    gone = _FlakyFilesystem(reader._fs, FileNotFoundError("NoSuchKey"), failures=99)
    reader._fs = gone
    with pytest.raises(FileNotFoundError):
        reader.take([0])
    assert gone.opens == 1


def test_the_public_writer_carries_resume_through(tmp_path):
    ds = bt.from_pydict({"x": list(range(60))})
    ds.ml.write_shards(str(tmp_path), rows_per_shard=10)
    index = ds.ml.write_shards(str(tmp_path), rows_per_shard=10, resume=True)
    assert index.total_rows == 60


class _CountingFilesystem:
    """Wraps a filesystem to count concurrent `atomic_writer` scopes and their peak."""

    def __init__(self, real, fail_on: str | None = None) -> None:
        import threading

        self._real = real
        self._fail_on = fail_on
        self._lock = threading.Lock()
        self.live = 0
        self.peak = 0
        self.written: list[str] = []

    def atomic_writer(self, path):
        import contextlib

        @contextlib.contextmanager
        def _scope():
            import time

            with self._lock:
                self.live += 1
                self.peak = max(self.peak, self.live)
            try:
                if self._fail_on and path.endswith(self._fail_on):
                    raise OSError("disk on fire")
                with self._real.atomic_writer(path) as handle:
                    yield handle
                time.sleep(0.005)  # stand in for an object-store round trip
                with self._lock:
                    self.written.append(path)
            finally:
                with self._lock:
                    self.live -= 1

        return _scope()

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.fixture
def counting_fs(monkeypatch):
    """Install a counting filesystem into the shard writer and hand it back."""
    import batcher.io.formats.ml.shards.writer as writer_module

    real = writer_module.resolve_filesystem
    holder: dict[str, _CountingFilesystem] = {}

    def _install(fail_on: str | None = None):
        def _resolve(path, **kwargs):
            if "fs" not in holder:
                holder["fs"] = _CountingFilesystem(real(path, **kwargs), fail_on)
            return holder["fs"]

        monkeypatch.setattr(writer_module, "resolve_filesystem", _resolve)
        return holder

    return _install


@pytest.mark.parametrize("concurrency", [1, 2, 8])
def test_concurrent_writing_produces_an_identical_corpus(tmp_path, concurrency):
    # Shards are published out of order but must land under the right names: shard k holds
    # rows [k*rows_per_shard, ...), or every global row index means something else.
    out = tmp_path / str(concurrency)
    index = write_shards(_batches(1000), str(out), rows_per_shard=10, write_concurrency=concurrency)
    assert (index.total_rows, index.shard_count) == (1000, 100)
    rows = ShardReader(str(out), cache_size=200).take(list(range(1000)))
    assert rows.column("x").to_pylist() == list(range(1000))


def test_writing_holds_only_the_shards_in_flight(tmp_path, counting_fs):
    # Without backpressure the packer runs ahead of the writers and the whole corpus queues
    # up in memory — the regression the streaming rewrite removed, reintroduced by a pool.
    holder = counting_fs()
    write_shards(_batches(2000), str(tmp_path), rows_per_shard=10, write_concurrency=3)
    fs = holder["fs"]
    assert fs.peak <= 4, f"{fs.peak} shards were in flight for a concurrency of 3"
    assert fs.peak > 1, "the writes did not overlap at all"


def test_a_failure_under_concurrency_leaves_a_contiguous_manifest(tmp_path, counting_fs):
    # Shards complete out of order, so a manifest that counted *completions* rather than a
    # contiguous prefix would describe a shard that is not there. Failing shard 5 must leave
    # a corpus of exactly shards 0-4, whatever landed after it.
    holder = counting_fs(fail_on="00000005.arrow")
    with pytest.raises(OSError, match="disk on fire"):
        write_shards(_batches(1000), str(tmp_path), rows_per_shard=10, write_concurrency=8)
    index = read_shard_index(str(tmp_path))
    assert index.shard_count == 5, f"manifest describes {index.shard_count} shards, not 5"
    assert index.total_rows == 50
    assert len(holder["fs"].written) > 5, "no later shards landed; the test proves nothing"
    rows = ShardReader(str(tmp_path), cache_size=16).take(list(range(50)))
    assert rows.column("x").to_pylist() == list(range(50))
