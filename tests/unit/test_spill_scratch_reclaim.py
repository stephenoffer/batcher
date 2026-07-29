"""Spill scratch is reclaimed on the failure paths, including the remote tier.

The local tier is swept by the caller's `rmtree`, so a leak there is invisible in a test
directory that gets deleted anyway. The remote tier has no such backstop: an object left
behind by a failed query stays in the bucket, accumulating and billable, with nothing
recording that it existed. These pin the two exits that reach it.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.carbonite.spill.store import TieredSpillStore


def _batch(n: int = 8) -> pa.RecordBatch:
    return pa.record_batch({"a": list(range(n))})


def test_cleanup_aborts_a_writer_that_was_never_closed(tmp_path):
    """A partition phase abandoned by an exception leaves an open writer. It has no closed
    path, so it is in neither tier's list and `cleanup` could not see it at all."""
    store = TieredSpillStore(str(tmp_path))
    writer = store.writer("bucket_0")
    writer.write(_batch())
    partial = writer.path
    assert partial is not None
    assert store._local_pending > 0

    store.cleanup()

    import os

    assert not os.path.exists(partial)
    assert store._local_pending == 0  # the byte charge went back
    assert store._open_writers == set()


def test_cleanup_is_idempotent_and_safe_after_a_normal_close(tmp_path):
    store = TieredSpillStore(str(tmp_path))
    writer = store.writer("bucket_0")
    writer.write(_batch())
    handle = writer.close()
    assert handle is not None
    assert store._open_writers == set()

    store.cleanup()
    store.cleanup()
    assert store._local_used == 0


def test_an_empty_bucket_leaves_nothing_registered(tmp_path):
    store = TieredSpillStore(str(tmp_path))
    writer = store.writer("empty")
    assert writer.close() is None
    assert store._open_writers == set()


def test_a_failing_close_releases_its_byte_charge(tmp_path):
    """`close()` was the one exit that could not reach `abort()`, so a finalize that fails —
    the flush that finally hits a full disk, a remote upload that is refused — left the
    bucket's bytes charged against the live local budget for the rest of the process, and
    every later bucket then read the local tier as full."""
    store = TieredSpillStore(str(tmp_path))
    writer = store.writer("doomed")
    writer.write(_batch())
    partial = writer.path
    charged = store._local_pending
    assert charged > 0

    def boom():
        raise OSError("no space left on device")

    writer._writer.close = boom
    with pytest.raises(OSError, match="no space"):
        writer.close()

    import os

    assert store._local_pending == 0
    assert not os.path.exists(partial)
    assert store._open_writers == set()


def test_abort_and_close_both_deregister_exactly_once(tmp_path):
    store = TieredSpillStore(str(tmp_path))
    aborted = store.writer("a")
    aborted.write(_batch())
    aborted.abort()
    aborted.abort()  # idempotent
    assert store._open_writers == set()
    assert store._local_pending == 0
