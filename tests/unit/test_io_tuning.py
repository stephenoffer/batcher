"""I/O hardware-utilization tuning: filesystem reuse + read concurrency.

These are result-invariant knobs that cut the many-small-files object-store tax — a
memoized filesystem (no per-split credential re-walk) and a lifted pyarrow IO thread pool.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.io import filesystem as fsmod

pytestmark = pytest.mark.unit


def test_resolve_filesystem_is_memoized_per_authority(monkeypatch):
    # Two object keys in the same bucket resolve to the *same* filesystem façade, and the
    # expensive `from_uri` (credential walk + connection pool) runs once — not per split.
    calls = {"n": 0}

    class _FS:
        type_name = "s3"

    class _FakeFS:
        @staticmethod
        def from_uri(p):
            calls["n"] += 1
            return _FS(), p.split("://", 1)[1].rstrip("/")

    monkeypatch.setattr(fsmod.pafs, "FileSystem", _FakeFS)
    fsmod._resolve_uri_fs.cache_clear()
    a = fsmod.resolve_filesystem("s3://bucket/a.parquet")
    b = fsmod.resolve_filesystem("s3://bucket/b.parquet")
    assert a is b  # same cached façade for two keys in one bucket
    assert calls["n"] == 1  # the credential walk ran once, not per object path
    fsmod._resolve_uri_fs.cache_clear()


def test_ensure_io_threads_lifts_the_default_cap():
    from batcher.io.filesystem import ensure_io_threads

    before = pa.io_thread_count()
    try:
        pa.set_io_thread_count(4)  # simulate pyarrow's low default
        ensure_io_threads.cache_clear()
        ensure_io_threads()
        assert pa.io_thread_count() >= 8  # a wide S3 read is no longer throttled to 8
    finally:
        pa.set_io_thread_count(max(1, before))
        ensure_io_threads.cache_clear()


def test_cap_arrow_cpu_threads_lowers_to_usable_cores(monkeypatch):
    from batcher._internal import hardware
    from batcher.io.filesystem import cap_arrow_cpu_threads

    before = pa.cpu_count()
    try:
        # Container throttled to 4 usable cores on a wider host: pyarrow's compute/decode
        # pool must not stay at the host count and over-subscribe onto cores it can't run on.
        monkeypatch.setattr(hardware, "available_cpu_count", lambda: 4)
        pa.set_cpu_count(64)  # simulate pyarrow's host-sized default
        cap_arrow_cpu_threads()
        assert pa.cpu_count() == 4
        # Never *raises* above a caller's lower explicit choice.
        pa.set_cpu_count(2)
        cap_arrow_cpu_threads()
        assert pa.cpu_count() == 2
    finally:
        pa.set_cpu_count(max(1, before))
