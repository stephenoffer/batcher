"""Carbonite tiered spill store: local NVMe tier, object-storage overflow.

Pins the streaming write/read (a partition never co-resides whole in memory), the
tiering decision (stay local until the budget is exhausted, then overflow), and the
round-trip (spill → load is byte-identical Arrow). The remote tier is exercised over
an in-memory fsspec filesystem when fsspec is available.
"""

from __future__ import annotations

import shutil

import pyarrow as pa
import pytest

from batcher.carbonite.spill import SpillTier, TieredSpillStore

pytestmark = pytest.mark.unit


def _batch(n, base=0):
    return pa.record_batch({"k": list(range(base, base + n)), "v": [base] * n})


def test_local_spill_roundtrip(tmp_path):
    store = TieredSpillStore(str(tmp_path / "spill"))
    handle = store.spill([_batch(100), _batch(100, 100)])
    assert handle.tier is SpillTier.LOCAL
    assert store.local_bytes == handle.nbytes > 0

    loaded = store.read(handle)
    assert sum(b.num_rows for b in loaded) == 200
    assert loaded[0].column("k").to_pylist() == list(range(100))


def test_streaming_writer_does_not_buffer_whole_partition(tmp_path):
    # The writer streams batch-by-batch; closing yields one handle covering them all.
    store = TieredSpillStore(str(tmp_path / "spill"))
    w = store.writer("bucket_0")
    for i in range(10):
        w.write(_batch(50, i * 50))
    handle = w.close()
    assert handle is not None
    assert sum(b.num_rows for b in store.read(handle)) == 500


def test_no_remote_stays_local_even_over_budget(tmp_path):
    # With a budget but no remote URI, everything still lands locally (no overflow
    # target) — the store never drops data.
    store = TieredSpillStore(str(tmp_path / "spill"), local_budget_bytes=1)
    handle = store.spill([_batch(50)])
    assert handle.tier is SpillTier.LOCAL


def test_local_budget_is_clamped_to_free_disk(tmp_path):
    # An unrealistically large configured budget is clamped to a safe fraction of the
    # scratch volume's measured free space, so the local tier can't fill the filesystem
    # before overflow triggers.
    free = shutil.disk_usage(str(tmp_path)).free
    store = TieredSpillStore(str(tmp_path / "spill"), local_budget_bytes=free * 1000)
    assert store._local_budget is not None
    assert store._local_budget <= free
    assert store._local_budget < free * 1000


def test_clamp_to_free_disk_stats_nearest_existing_ancestor(tmp_path):
    # A spill dir that does not exist yet (created lazily on first spill) must still be
    # clamped — by stat'ing its nearest existing ancestor (the same filesystem), not by
    # silently dropping the disk-aware bound and keeping the too-large configured budget.
    from batcher.carbonite.spill import _clamp_to_free_disk

    free = shutil.disk_usage(str(tmp_path)).free
    missing = str(tmp_path / "does" / "not" / "exist" / "yet")
    clamped = _clamp_to_free_disk(missing, free * 1000)
    assert clamped is not None
    assert clamped <= free  # bounded by the ancestor's real free space, not the raw budget


def test_local_budget_derived_from_free_disk_when_unset(tmp_path):
    # With no configured budget the store still bounds the local tier to measured free
    # disk rather than leaving it unbounded.
    store = TieredSpillStore(str(tmp_path / "spill"))
    assert store._local_budget is not None and store._local_budget > 0


def test_remote_tier_always_compresses_even_when_local_is_raw():
    # C13: the remote tier is slow object storage priced by bytes transferred, so it
    # compresses even when the local NVMe tier is left uncompressed.
    from batcher.carbonite.spill import _ipc_options, _remote_ipc_options

    if _ipc_options("lz4") is None:
        pytest.skip("lz4 codec not built into this pyarrow")
    assert _ipc_options(None) is None  # local honors "no compression"
    assert _remote_ipc_options(None) is not None  # remote upgrades to LZ4
    assert _remote_ipc_options("auto") is not None
    assert _remote_ipc_options("zstd") is not None  # an explicit codec is honored


def test_empty_partition_is_tolerated(tmp_path):
    # C18: an empty/all-empty bucket is intrinsic to a shuffle, not an error — it
    # opens no file and returns no handle.
    store = TieredSpillStore(str(tmp_path / "spill"))
    assert store.spill([]) is None
    assert store.spill([_batch(0)]) is None
    assert store.local_bytes == 0


def test_overflow_to_object_storage(tmp_path):
    pytest.importorskip("fsspec", reason="fsspec (cloud extra) not installed")
    # budget 0 → the local tier is already "full", so the first bucket overflows.
    store = TieredSpillStore(
        str(tmp_path / "spill"),
        remote_uri="memory://batcher-spill-test-a",
        local_budget_bytes=0,
        compression=None,
    )
    handle = store.spill([_batch(100)])
    assert handle.tier is SpillTier.REMOTE
    assert store.local_bytes == 0  # nothing landed locally
    assert sum(b.num_rows for b in store.read(handle)) == 100


def test_local_fills_then_later_buckets_overflow(tmp_path):
    pytest.importorskip("fsspec", reason="fsspec (cloud extra) not installed")
    # A positive budget: the first bucket lands local; once cumulative local bytes
    # reach the budget, subsequent buckets overflow to object storage.
    store = TieredSpillStore(
        str(tmp_path / "spill"),
        remote_uri="memory://batcher-spill-test-b",
        local_budget_bytes=1,
        compression=None,
    )
    first = store.spill([_batch(100)], name="b0")
    second = store.spill([_batch(100)], name="b1")
    assert first.tier is SpillTier.LOCAL
    assert second.tier is SpillTier.REMOTE
    assert store.local_bytes == first.nbytes


def test_concurrent_open_writers_overflow_on_live_budget(tmp_path):
    pytest.importorskip("fsspec", reason="fsspec (cloud extra) not installed")
    # The partition phase holds one writer per bucket open at once and interleaves writes
    # across them (a hash partition scatters every input batch into all buckets). The tier
    # is decided on a bucket's first write from the local budget — but `_local_used` only
    # grows when a bucket *closes*, so historically none of the still-open siblings ever
    # observed the others' growth: with a tiny budget and a configured remote tier, all
    # buckets stayed LOCAL and the local disk filled far past its budget, defeating the
    # "overflow to object storage before the disk fills" guarantee. Live in-flight
    # accounting fixes it — once the streamed bytes cross the budget, later-opened buckets
    # overflow. (The first bucket to receive data correctly stays local: the budget is not
    # yet touched when it opens.)
    store = TieredSpillStore(
        str(tmp_path / "spill"),
        remote_uri="memory://batcher-spill-test-concurrent",
        local_budget_bytes=1,  # any real write immediately exhausts the local tier
        compression=None,
    )
    writers = {i: store.writer(f"bucket_{i}") for i in range(4)}
    for _ in range(3):  # interleave writes across all open buckets
        for w in writers.values():
            w.write(_batch(200))
    handles = {i: w.close() for i, w in writers.items()}

    tiers = [handles[i].tier for i in range(4)]
    assert tiers[0] is SpillTier.LOCAL  # first bucket to be written lands local
    assert all(t is SpillTier.REMOTE for t in tiers[1:])  # the rest overflow off the budget
    # Live accounting balances exactly back out once every bucket closes.
    assert store._local_pending == 0
    # No data is lost or misrouted across the overflow: every bucket round-trips its rows.
    for i in range(4):
        assert sum(b.num_rows for b in store.read(handles[i])) == 600


def test_cleanup_removes_local(tmp_path):
    store = TieredSpillStore(str(tmp_path / "spill"))
    store.spill([_batch(10)])
    store.cleanup()
    assert store.local_bytes == 0


def test_lost_local_file_raises_retryable_resource_error(tmp_path):
    # Phase 3c: an ephemeral/spot node's local NVMe can be reclaimed mid-query,
    # vanishing the spilled partition. Reading it must surface a clear, retryable
    # ResourceError (the distributed recovery path recomputes on it), not a cryptic
    # OSError that crashes the query.
    import os

    from batcher._internal.errors import ResourceError

    store = TieredSpillStore(str(tmp_path / "spill"))
    handle = store.spill([_batch(50)])
    os.remove(handle.path)  # simulate the disk being reclaimed

    with pytest.raises(ResourceError, match="reclaimed"):
        store.read(handle)
    with pytest.raises(ResourceError, match="reclaimed"):
        list(store.read_stream(handle))


def test_spill_handle_reports_uncompressed_logical_size(tmp_path):
    """`logical_nbytes` is the uncompressed in-memory size the reducer must budget against.

    The grace-recursion decision (`dist/spill.py::_reduce_agg_bucket`) reads a bucket back
    into RAM, which decompresses it — so it must budget against the *uncompressed* size, not
    the on-disk compressed `nbytes` (which for a compressible bucket can be far smaller and
    would let an over-large bucket skip re-spill recursion and OOM `combine_finalize`).
    """
    store = TieredSpillStore(str(tmp_path / "spill"), compression="zstd")
    # A single repeated value over many rows: hugely compressible, so on-disk << resident.
    batch = pa.record_batch({"k": pa.array([7] * 20_000), "v": pa.array([1] * 20_000)})
    handle = store.spill([batch])
    # The logical size is exactly the batch's in-memory footprint...
    assert handle.logical_nbytes == batch.nbytes
    # ...and never smaller than the compressed on-disk size (here strictly larger).
    assert handle.logical_nbytes >= handle.nbytes


def test_spill_handle_logical_size_accumulates_across_batches(tmp_path):
    """A streamed multi-batch bucket's `logical_nbytes` sums every batch's uncompressed size."""
    store = TieredSpillStore(str(tmp_path / "spill"))
    b1, b2 = _batch(100), _batch(100, 100)
    handle = store.spill([b1, b2])
    assert handle.logical_nbytes == b1.nbytes + b2.nbytes
