"""The memory ledger and the over-release counter: figures the pressure `max` collapses.

The pressure monitor classifies the maximum of both pools and the process footprint,
which is right for a decision and useless for a diagnosis: it cannot say whether a full
box is full of reserved operator state or of memory no pool was asked for (pyarrow buffers,
UDF tensors). The ledger reports those separately (F081). The over-release counter makes a
release that outran its reservation visible instead of silently clamped (F083).
"""

from __future__ import annotations

import dataclasses

import pytest

from batcher.carbonite.manager import ResourceManager
from batcher.carbonite.memory import ledger
from batcher.carbonite.memory.pool import BufferPool, _FallbackPool, reset_process_pool
from batcher.config import Config

pytestmark = pytest.mark.unit

_MIB = 1 << 20


@pytest.fixture(autouse=True)
def _clean_pool():
    reset_process_pool()
    yield
    reset_process_pool()


class _Pool:
    def __init__(self, used: int) -> None:
        self.used = used


def _patch(monkeypatch, *, control, engine, rss, cgroup=None) -> None:
    monkeypatch.setattr(
        ledger, "current_process_pool", lambda: None if control is None else _Pool(control)
    )
    monkeypatch.setattr(
        ledger,
        "engine_pool_stats",
        lambda: None if engine is None else {"used_bytes": engine},
    )
    monkeypatch.setattr(ledger.probe, "process_rss_bytes", lambda: rss)
    monkeypatch.setattr(ledger.probe, "cgroup_current_bytes", lambda: cgroup)


def test_accounted_is_the_max_of_the_two_reservations_never_the_sum(monkeypatch) -> None:
    """Both pools describe the same running work, so summing them charges it twice."""
    _patch(monkeypatch, control=300 * _MIB, engine=500 * _MIB, rss=2048 * _MIB, cgroup=4096)
    out = ledger.memory_ledger()
    assert out["control_plane_reserved_bytes"] == 300 * _MIB
    assert out["engine_reserved_bytes"] == 500 * _MIB
    assert out["accounted_bytes"] == 500 * _MIB
    assert out["resident_bytes"] == 2048 * _MIB
    assert out["cgroup_unreclaimable_bytes"] == 4096
    # The bytes no pool was asked for: the foreign allocations the max would hide.
    assert out["unaccounted_bytes"] == (2048 - 500) * _MIB


def test_reserved_ahead_of_allocation_floors_unaccounted_at_zero(monkeypatch) -> None:
    """Reserve-before-allocate means reservations can exceed RSS; that is not negative."""
    _patch(monkeypatch, control=None, engine=900 * _MIB, rss=100 * _MIB)
    out = ledger.memory_ledger()
    assert out["control_plane_reserved_bytes"] is None
    assert out["unaccounted_bytes"] == 0


def test_an_unreadable_resident_set_is_none_not_zero(monkeypatch) -> None:
    """Zero would claim no foreign memory; unknown must read as unknown."""
    _patch(monkeypatch, control=None, engine=None, rss=None)
    out = ledger.memory_ledger()
    assert out["engine_reserved_bytes"] is None
    assert out["accounted_bytes"] == 0
    assert out["resident_bytes"] is None
    assert out["unaccounted_bytes"] is None


def test_the_manager_snapshot_carries_the_ledger() -> None:
    cfg = Config()
    cfg = dataclasses.replace(
        cfg, memory=dataclasses.replace(cfg.memory, max_memory_bytes=512 * _MIB)
    )
    stats = ResourceManager(cfg).stats()
    assert set(stats["memory_ledger"]) == {
        "control_plane_reserved_bytes",
        "engine_reserved_bytes",
        "accounted_bytes",
        "pyarrow_allocated_bytes",
        "resident_bytes",
        "cgroup_unreclaimable_bytes",
        "unaccounted_bytes",
    }


def test_python_side_arrow_buffers_are_measured_not_inferred() -> None:
    """pyarrow's own pool is the one foreign allocator the ledger can read exactly (F062)."""
    import pyarrow as pa

    before = ledger.memory_ledger()["pyarrow_allocated_bytes"]
    held = pa.array(range(250_000), pa.int64())  # 2 MB built on the Python side
    after = ledger.memory_ledger()["pyarrow_allocated_bytes"]
    assert after - before >= held.nbytes


def test_the_fallback_pool_counts_an_over_release() -> None:
    """The pure-Python mirror keeps the Rust pool's semantics, counter included."""
    pool = _FallbackPool(1000)
    assert pool.try_reserve(100)
    pool.release(100)
    assert pool.over_released == 0
    pool.release(40)
    assert pool.used == 0
    assert pool.over_released == 40


def test_buffer_pool_stats_report_over_release() -> None:
    """A balanced reserve block releases exactly what it held: the counter stays at 0."""
    pool = BufferPool(1 << 20)
    with pool.reserve(4096) as ok:
        assert ok
    assert pool.stats()["over_released_bytes"] == 0
