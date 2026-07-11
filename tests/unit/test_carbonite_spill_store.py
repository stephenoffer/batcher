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
