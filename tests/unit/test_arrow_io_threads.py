"""pyarrow's global IO pool is the ceiling on Batcher's remote reads, and it defaults to 8.

`io._concurrent.read_each_file` fans 64 blob or footer reads across a `ThreadPoolExecutor`,
and every one of them then queues inside `S3FileSystem.open_input_file` behind pyarrow's own
IO threads. Eight of them is 2.4x slower on a 1,000-object S3 read than sixty-four, and the
engine's own concurrency knob cannot see it — raising `BATCHER_FOOTER_CONCURRENCY` from 64 to
1024 moved that read by under 5 %, because the requests were never the bound.

These tests pin the three properties that make lifting it safe rather than merely faster:
it happens only when the engine actually reaches a remote store, it only ever raises, and an
explicit `ARROW_IO_THREADS` wins.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.io import filesystem as fsmod

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _restore_io_threads():
    """The pool is process-global, so every test here puts it back."""
    before = pa.io_thread_count()
    yield
    pa.set_io_thread_count(before)


def test_a_remote_resolution_lifts_the_pool_to_the_engines_fan_out_width():
    pa.set_io_thread_count(8)
    fsmod._raise_arrow_io_threads()
    assert pa.io_thread_count() == fsmod._ARROW_IO_THREADS


def test_a_local_path_leaves_the_pool_alone():
    """A local read is a syscall on page cache; there is no round trip to overlap, and the
    pool is shared with everything else in the interpreter."""
    pa.set_io_thread_count(8)
    fsmod.resolve_filesystem("/tmp/some/local/path.parquet")
    assert pa.io_thread_count() == 8


def test_it_only_ever_raises():
    """A caller who has already asked for more keeps it — this lifts a bad default, it does
    not impose a policy."""
    pa.set_io_thread_count(fsmod._ARROW_IO_THREADS * 2)
    fsmod._raise_arrow_io_threads()
    assert pa.io_thread_count() == fsmod._ARROW_IO_THREADS * 2


def test_an_explicit_arrow_io_threads_wins(monkeypatch):
    """`ARROW_IO_THREADS` is pyarrow's own way to set the pool, so a deployment that has
    tuned it has already said what it wants."""
    monkeypatch.setenv("ARROW_IO_THREADS", "4")
    pa.set_io_thread_count(8)
    fsmod._raise_arrow_io_threads()
    assert pa.io_thread_count() == 8
