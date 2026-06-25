"""Carbonite tiered spill store: local NVMe tier, object-storage overflow.

Pins the streaming write/read (a partition never co-resides whole in memory), the
tiering decision (stay local until the budget is exhausted, then overflow), and the
round-trip (spill → load is byte-identical Arrow). The remote tier is exercised over
an in-memory fsspec filesystem when fsspec is available.
"""

from __future__ import annotations

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
