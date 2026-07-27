"""The disk side of the envelope: the pressure ladder, and reading a bucket back.

Carbonite governs memory with a four-level ladder every component reads, and governed disk
with a single boolean consulted in one place. That asymmetry matters more than it looks:
an out-of-memory operator can spill, and an out-of-space write cannot be retried or
degraded at all — it fails the query.
"""

from __future__ import annotations

import shutil

import pyarrow as pa
import pytest

from batcher.carbonite.memory.pool import process_pool, reset_process_pool
from batcher.carbonite.spill import DiskPressure, SpillTier, TieredSpillStore
from batcher.carbonite.spill import disk as spill_disk

pytestmark = pytest.mark.unit


def _batch(n: int = 8) -> pa.RecordBatch:
    return pa.record_batch({"v": pa.array(list(range(n)), type=pa.int64())})


class _Usage:
    """A `shutil.disk_usage` result with a chosen total and free."""

    def __init__(self, total: int, free: int) -> None:
        self.total = total
        self.free = free
        self.used = total - free


@pytest.fixture(autouse=True)
def _fresh_disk_sampling():
    spill_disk.reset_disk_sampling()
    yield
    spill_disk.reset_disk_sampling()


def _volume(monkeypatch, total: int, free: int) -> None:
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: _Usage(total, free))
    spill_disk.reset_disk_sampling()


def test_the_ladder_is_ordered_and_monotone_in_free_space(monkeypatch, tmp_path) -> None:
    total = 100 << 30  # 100 GiB, so the fixed reserve (256 MiB) is the binding floor
    seen = []
    for free_gib in (80, 40, 20, 10, 1):
        _volume(monkeypatch, total, free_gib << 30)
        seen.append(spill_disk.disk_pressure(str(tmp_path)))
    assert seen == sorted(seen), "a fuller volume must never classify as roomier"
    assert seen[0] is DiskPressure.NORMAL
    assert DiskPressure.FULL > DiskPressure.ELEVATED > DiskPressure.NORMAL


def test_elevated_fires_while_there_is_still_room_to_react(monkeypatch, tmp_path) -> None:
    """The point of a middle rung is being reached before the reserve floor."""
    total = 100 << 30
    _volume(monkeypatch, total, 20 << 30)  # 20% free: under a quarter, far above the floor
    level = spill_disk.disk_pressure(str(tmp_path))
    assert level is DiskPressure.ELEVATED
    assert spill_disk.disk_floor_bytes(str(tmp_path)) < (20 << 30)


def test_an_unstatable_volume_reads_as_normal(monkeypatch, tmp_path) -> None:
    """A measurement that could not be taken is not evidence of a problem."""

    def _boom(_p):
        raise OSError("no such filesystem")

    monkeypatch.setattr(shutil, "disk_usage", _boom)
    spill_disk.reset_disk_sampling()
    assert spill_disk.disk_pressure(str(tmp_path)) is DiskPressure.NORMAL


def test_an_elevated_volume_routes_new_buckets_off_the_local_tier(monkeypatch, tmp_path) -> None:
    """Waiting for FULL means the first bucket to notice already has nowhere to go."""
    pytest.importorskip("fsspec", reason="fsspec (cloud extra) not installed")
    store = TieredSpillStore(
        str(tmp_path / "s"),
        remote_uri="memory://batcher-disk-ladder-test",
        local_budget_bytes=1 << 40,  # the budget alone would keep everything local
        compression=None,
    )
    _volume(monkeypatch, 100 << 30, 10 << 30)  # 10% free — ELEVATED, not yet FULL
    assert store.disk_pressure() is DiskPressure.ELEVATED

    handle = store.spill([_batch()], "b0")
    assert handle is not None and handle.tier is SpillTier.REMOTE
    assert store.stats()["disk_pressure"] == "ELEVATED"
    store.cleanup()


def test_an_ample_volume_keeps_the_fast_local_tier(monkeypatch, tmp_path) -> None:
    store = TieredSpillStore(
        str(tmp_path / "s"),
        remote_uri="memory://batcher-disk-ladder-ample",
        local_budget_bytes=1 << 40,
        compression=None,
    )
    _volume(monkeypatch, 100 << 30, 90 << 30)
    handle = store.spill([_batch()], "b0")
    assert handle is not None and handle.tier is SpillTier.LOCAL
    store.cleanup()


def test_reading_a_bucket_back_is_accounted_against_the_pool(tmp_path) -> None:
    """Reading is the one step of spilling that can undo it; nothing checked the budget."""
    reset_process_pool()
    try:
        pool = process_pool(1 << 30)
        store = TieredSpillStore(str(tmp_path / "s"), compression=None)
        handle = store.spill([_batch(1000)], "b0")
        assert handle is not None and handle.logical_nbytes > 0

        with store.read_reserved(handle) as batches:
            first = next(iter(batches))
            assert first.num_rows > 0
            assert pool.used >= handle.logical_nbytes, "the read is not accounted"
        assert pool.used == 0, "the accounting is released when the read ends"
        store.cleanup()
    finally:
        reset_process_pool()


def test_reading_reserves_the_uncompressed_size_not_the_file_size(tmp_path) -> None:
    """Budgeting against the on-disk size under-reserves by exactly the compression ratio."""
    reset_process_pool()
    try:
        pool = process_pool(1 << 30)
        # Highly compressible: a constant column under zstd.
        rows = pa.record_batch({"v": pa.array([7] * 50_000, type=pa.int64())})
        store = TieredSpillStore(str(tmp_path / "s"), compression="zstd")
        handle = store.spill([rows], "b0")
        assert handle is not None
        if handle.compression_ratio <= 1.5:
            pytest.skip("this pyarrow build did not compress the bucket")

        with store.read_reserved(handle):
            assert pool.used == handle.logical_nbytes
            assert pool.used > handle.nbytes, "reserved only the compressed size"
        store.cleanup()
    finally:
        reset_process_pool()


def test_reading_without_a_pool_still_works(tmp_path) -> None:
    reset_process_pool()
    try:
        store = TieredSpillStore(str(tmp_path / "s"), compression=None)
        handle = store.spill([_batch(10)], "b0")
        with store.read_reserved(handle) as batches:
            assert sum(b.num_rows for b in batches) == 10
        store.cleanup()
    finally:
        reset_process_pool()
