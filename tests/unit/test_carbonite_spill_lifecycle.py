"""The spill bucket's three exits: close, abort, and the context manager that picks one.

Each test here pins a way the old single-exit writer leaked. The expensive one is
`abort`: a writer abandoned mid-stream left its bytes charged against the store's live
local budget forever, so every later bucket in the process read the local tier as full
and overflowed to object storage — a permanent throughput cliff caused by one unrelated
error, invisible to every correctness test.
"""

from __future__ import annotations

import os

import pyarrow as pa
import pytest

from batcher._internal.errors import PlanError
from batcher.carbonite.spill import SpillTier, TieredSpillStore
from batcher.carbonite.spill import disk as spill_disk


def _batch(n: int = 8) -> pa.RecordBatch:
    return pa.record_batch({"v": pa.array(list(range(n)), type=pa.int64())})


def test_close_is_idempotent(tmp_path) -> None:
    """A second close returns the same handle and charges the store once."""
    store = TieredSpillStore(str(tmp_path / "s"), compression=None)
    w = store.writer("b0")
    w.write(_batch())
    first = w.close()
    second = w.close()

    assert first is not None
    assert first == second
    assert store.stats()["local_buckets"] == 1
    assert store.local_bytes == first.nbytes


def test_abort_releases_the_pending_byte_charge(tmp_path) -> None:
    """The leak: an abandoned writer used to strand its bytes in the live local budget."""
    store = TieredSpillStore(str(tmp_path / "s"), compression=None)
    w = store.writer("b0")
    w.write(_batch(1000))
    assert store.stats()["local_pending_bytes"] > 0

    w.abort()

    assert store.stats()["local_pending_bytes"] == 0
    assert store.local_bytes == 0
    assert not os.path.exists(str(tmp_path / "s" / "b0.arrow"))


def test_abort_is_idempotent_and_never_raises(tmp_path) -> None:
    store = TieredSpillStore(str(tmp_path / "s"), compression=None)
    w = store.writer("b0")
    w.write(_batch())
    w.abort()
    w.abort()  # must not raise, must not double-release
    assert store.stats()["local_pending_bytes"] == 0


def test_context_manager_aborts_on_an_exception(tmp_path) -> None:
    store = TieredSpillStore(str(tmp_path / "s"), compression=None)
    with pytest.raises(RuntimeError), store.writer("b0") as w:
        w.write(_batch(500))
        raise RuntimeError("operator blew up mid-partition")

    assert store.stats()["local_pending_bytes"] == 0
    assert store.bucket_count == 0


def test_spill_aborts_when_the_batch_source_raises(tmp_path) -> None:
    """`spill` takes an iterable, so a generator that fails part-way must not strand bytes."""

    def batches():
        yield _batch(500)
        raise ValueError("upstream failed")

    store = TieredSpillStore(str(tmp_path / "s"), compression=None)
    with pytest.raises(ValueError):
        store.spill(batches(), "b0")

    assert store.stats()["local_pending_bytes"] == 0
    assert store.bucket_count == 0


def test_handle_carries_row_count_and_compression_ratio(tmp_path) -> None:
    store = TieredSpillStore(str(tmp_path / "s"), compression=None)
    handle = store.spill([_batch(10), _batch(10)], "b0")

    assert handle is not None
    assert handle.num_rows == 20
    assert handle.compression_ratio > 0
    assert handle.is_remote is False


def test_release_frees_one_bucket_incrementally(tmp_path) -> None:
    """A reduce that consumes buckets one at a time can give each one's disk back."""
    store = TieredSpillStore(str(tmp_path / "s"), compression=None)
    a = store.spill([_batch(10)], "b0")
    b = store.spill([_batch(10)], "b1")
    assert store.bucket_count == 2

    store.release(a)

    assert store.bucket_count == 1
    assert store.local_bytes == b.nbytes
    assert not os.path.exists(a.path)
    store.release(a)  # already gone — a no-op, not an error
    assert store.bucket_count == 1


def test_store_is_a_context_manager_that_cleans_up(tmp_path) -> None:
    with TieredSpillStore(str(tmp_path / "s"), compression=None) as store:
        handle = store.spill([_batch(10)], "b0")
        assert os.path.exists(handle.path)
    assert not os.path.exists(handle.path)
    assert store.total_bytes == 0


@pytest.mark.parametrize("name", ["../escape", "a/b", "", ".hidden", "with space"])
def test_unsafe_bucket_names_are_rejected(tmp_path, name) -> None:
    """A bucket name becomes a path component; it must not be able to leave the dir."""
    store = TieredSpillStore(str(tmp_path / "s"))
    with pytest.raises(PlanError, match="safe path component"):
        store.writer(name)


def test_remote_buckets_are_cleaned_up_too(tmp_path) -> None:
    """Remote scratch is scratch: leaving it behind bills the operator forever."""
    fsspec = pytest.importorskip("fsspec", reason="fsspec (cloud extra) not installed")
    uri = "memory://batcher-spill-cleanup-test"
    store = TieredSpillStore(
        str(tmp_path / "s"), remote_uri=uri, local_budget_bytes=0, compression=None
    )
    handle = store.spill([_batch(10)], "b0")
    assert handle is not None and handle.tier is SpillTier.REMOTE
    fs, _, paths = fsspec.get_fs_token_paths(handle.path)
    assert fs.exists(paths[0])
    assert store.remote_bytes > 0
    assert store.overflowed == 1

    store.cleanup()

    assert not fs.exists(paths[0])
    assert store.remote_bytes == 0
    assert store.bucket_count == 0


def test_free_disk_reading_is_ttl_cached(tmp_path, monkeypatch) -> None:
    """A 4,096-way spill asked the kernel 4,096 times for a number that cannot move."""
    spill_disk.reset_disk_sampling()
    calls = {"n": 0}

    def counted(_path):
        calls["n"] += 1
        return 1 << 40

    monkeypatch.setattr(spill_disk, "read_free_disk_bytes", counted)
    for _ in range(50):
        spill_disk.free_disk_bytes(str(tmp_path))

    assert calls["n"] == 1
    spill_disk.reset_disk_sampling()
    spill_disk.free_disk_bytes(str(tmp_path))
    assert calls["n"] == 2


def test_classifying_the_volume_does_not_route_around_the_cache(tmp_path, monkeypatch) -> None:
    """`disk_pressure` reads free space through the TTL cache and then took `total` twice
    with raw `shutil.disk_usage` calls — once directly, once inside `disk_floor_bytes`.

    The store consults the pressure once per bucket open, so a 4,096-way partitioned spill
    made 8,192 uncached `statvfs` calls: more than the storm the cache was added to stop,
    for a number that moves even less than free space does.
    """
    import shutil

    spill_disk.reset_disk_sampling()
    calls = {"n": 0}

    class _Usage:
        total = 100 << 30
        used = 0
        free = 80 << 30

    def counted(_path):
        calls["n"] += 1
        return _Usage()

    monkeypatch.setattr(shutil, "disk_usage", counted)
    for _ in range(50):
        spill_disk.disk_pressure(str(tmp_path))

    # One reading of free, one of capacity, for the whole window.
    assert calls["n"] == 2


def test_disk_floor_shrinks_on_a_small_volume(tmp_path, monkeypatch) -> None:
    """A fixed 256 MiB reserve is a quarter of a 1 GiB container scratch mount."""
    import shutil

    class _Usage:
        total = 1 << 30  # 1 GiB
        used = 0
        free = 1 << 30

    monkeypatch.setattr(shutil, "disk_usage", lambda _p: _Usage())
    floor = spill_disk.disk_floor_bytes(str(tmp_path))
    assert floor < spill_disk.DISK_FLOOR_BYTES
    assert floor == int((1 << 30) * 0.05)


def test_quota_exhaustion_is_classified_as_out_of_space() -> None:
    """`EDQUOT` has the same cause and the same fix as `ENOSPC`."""
    import errno

    assert spill_disk.is_out_of_space(OSError(errno.ENOSPC, "no space"))
    if hasattr(errno, "EDQUOT"):
        assert spill_disk.is_out_of_space(OSError(errno.EDQUOT, "over quota"))
    assert not spill_disk.is_out_of_space(OSError(errno.EACCES, "denied"))


def test_an_unknown_compression_codec_is_recorded_not_silent(monkeypatch) -> None:
    """A typo used to become a silently uncompressed bucket on a per-byte-priced tier."""
    seen = []
    monkeypatch.setattr(
        "batcher.carbonite.spill.disk.note_suppressed",
        lambda *a, **k: seen.append(a),
    )
    assert spill_disk.ipc_options("zstandard") is None
    assert seen, "a codec that does not exist must be reported, not silently dropped"


def test_a_truncated_bucket_is_refused_rather_than_read_short(tmp_path) -> None:
    """A spilled bucket cut at a message boundary must fail, not shorten the answer.

    An Arrow IPC *stream* truncated at a message boundary -- the last complete batch present,
    the end-of-stream marker gone -- is byte-for-byte a shorter valid stream, so
    `pa.ipc.open_stream` returns the batches it finds and reports success. The reducer then
    computes a correct answer over the wrong rows, with nothing recording that it happened.

    The boundary is constructed exactly, from a reference bucket of three batches minus its
    8-byte end-of-stream marker, rather than by cutting an arbitrary fraction: a cut that
    lands *inside* a message arrow catches on its own, and only the boundary case needs the
    row count `SpillHandle` has always carried for it.
    """
    from batcher._internal.errors import ResourceError

    eos_bytes = 8  # continuation marker + zero length

    store = TieredSpillStore(str(tmp_path / "s"), compression=None)

    # A reference bucket of three batches: its length is the boundary.
    ref = store.writer("ref")
    for _ in range(3):
        ref.write(_batch(1000))
    ref_handle = ref.close()
    boundary = os.path.getsize(ref_handle.path) - eos_bytes

    # The real bucket: five batches, cut back to the three-batch boundary.
    w = store.writer("b0")
    for _ in range(5):
        w.write(_batch(1000))
    handle = w.close()
    assert handle.num_rows == 5000
    with open(handle.path, "r+b") as fh:
        fh.truncate(boundary)

    with pytest.raises(ResourceError) as exc:
        list(store.read_stream(handle))
    assert "3000" in str(exc.value) and "5000" in str(exc.value)


def test_an_intact_bucket_reads_back_without_complaint(tmp_path) -> None:
    """The truncation check must not fire on a healthy bucket, or it would be a way to fail
    every spilling query rather than a guard on a rare one."""
    store = TieredSpillStore(str(tmp_path / "s"), compression=None)
    w = store.writer("b0")
    for _ in range(4):
        w.write(_batch(500))
    handle = w.close()

    assert sum(b.num_rows for b in store.read_stream(handle)) == 2000
    assert sum(b.num_rows for b in store.read(handle)) == 2000
