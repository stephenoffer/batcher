"""Reclaimable page cache is not memory pressure.

`memory.current` counts anonymous memory *plus* every clean file page the kernel is caching,
and the kernel drops that cache long before it OOM-kills anything. Reading it as pressure made
a box that had merely *read files* look like a box about to die: measured on a 30 GiB host,
22.6 GiB "current" of which 15.3 GiB was cache, so `PressureMonitor` reported ELEVATED from a
cold start — and ELEVATED halves every morsel (`_MORSEL_PRESSURE_FACTORS`), for the whole run.

That is the shape worth pinning: not a crash, but a silent, permanent throttle that no
correctness test could see. The guard itself is untouched — everything it exists to catch
(the Flight shuffle store, off-pool pyarrow buffers) is anonymous, and stays counted.
"""

from __future__ import annotations

import pytest

from batcher.carbonite.memory import probe
from batcher.carbonite.memory.pool import process_pool
from batcher.carbonite.memory.pressure import PressureLevel, PressureMonitor
from batcher.config import Config, MemoryConfig, config_context

pytestmark = pytest.mark.unit

_GIB = 1 << 30


def test_the_page_cache_is_subtracted_from_the_cgroup_reading(monkeypatch):
    """The regression: 22.6 GiB charged, 15.3 GiB of it cache — only the rest is real."""
    monkeypatch.setattr(probe, "_cgroup_total_bytes", lambda: int(22.6 * _GIB))
    monkeypatch.setattr(probe, "_cgroup_file_cache_bytes", lambda: int(15.3 * _GIB))
    assert probe.cgroup_current_bytes() == pytest.approx(7.3 * _GIB, rel=1e-6)


def test_anonymous_memory_is_still_counted_in_full(monkeypatch):
    """The guard is not weakened: an off-pool anonymous buffer is exactly what it must see."""
    monkeypatch.setattr(probe, "_cgroup_total_bytes", lambda: 8 * _GIB)
    monkeypatch.setattr(probe, "_cgroup_file_cache_bytes", lambda: 0)
    assert probe.cgroup_current_bytes() == 8 * _GIB


def _cache_heavy_host(monkeypatch) -> None:
    """The measured reading after loading TPC-H sf1 on a 30 GiB box: 24.3 GiB charged to the
    cgroup, of which 15.3 GiB is page cache. Only 9 GiB is anonymous — but the raw 0.81
    fraction sails past the 0.765 ELEVATED line, and every morsel is halved from here on."""
    monkeypatch.setattr(probe, "_cgroup_total_bytes", lambda: int(24.3 * _GIB))
    monkeypatch.setattr(probe, "_cgroup_file_cache_bytes", lambda: int(15.3 * _GIB))
    monkeypatch.setattr(probe, "total_memory_bytes", lambda: 30 * _GIB)


def test_a_cache_heavy_host_does_not_throttle_an_idle_engine(monkeypatch):
    """The consequence that cost real throughput: pressure must read NORMAL, not ELEVATED."""
    _cache_heavy_host(monkeypatch)
    cfg = Config()
    with config_context(cfg):
        process_pool(24 * _GIB)  # a pool exists and holds nothing
        assert PressureMonitor(cfg).classify() is PressureLevel.NORMAL


def test_reading_files_does_not_halve_the_morsel(monkeypatch):
    """The damage itself. ELEVATED scales the morsel by 0.5, so a cache-warm box silently ran
    the whole engine at half batch size — a throughput regression no correctness test can see.
    `None` is the contract for "keep the configured target"."""
    from batcher.carbonite import ResourceManager

    _cache_heavy_host(monkeypatch)
    cfg = Config()
    with config_context(cfg):
        process_pool(24 * _GIB)
        assert ResourceManager(cfg).recommend_morsel_target() is None


def test_a_genuinely_full_host_still_reports_pressure(monkeypatch):
    """The other half: anonymous memory near the ceiling must still escalate."""
    monkeypatch.setattr(probe, "_cgroup_total_bytes", lambda: 29 * _GIB)
    monkeypatch.setattr(probe, "_cgroup_file_cache_bytes", lambda: 0)
    monkeypatch.setattr(probe, "total_memory_bytes", lambda: 30 * _GIB)
    cfg = Config()
    with config_context(cfg):
        process_pool(24 * _GIB)  # the pool is empty; the anonymous footprint is what bites
        assert PressureMonitor(cfg).classify() >= PressureLevel.CRITICAL


def test_no_cgroup_leaves_the_reading_absent(monkeypatch):
    """Bare metal with no cgroup: the caller falls back to RSS, not to a bogus zero."""
    monkeypatch.setattr(probe, "_cgroup_total_bytes", lambda: None)
    assert probe.cgroup_current_bytes() is None


def test_the_engine_envelope_beats_a_quiet_host(monkeypatch):
    """A tiny envelope the engine has filled is pressure even when the host is idle."""
    monkeypatch.setattr(probe, "_cgroup_total_bytes", lambda: 1 * _GIB)
    monkeypatch.setattr(probe, "_cgroup_file_cache_bytes", lambda: 0)
    monkeypatch.setattr(probe, "total_memory_bytes", lambda: 30 * _GIB)
    cfg = Config().replace(memory=MemoryConfig(max_memory_bytes=1_000))
    with config_context(cfg):
        pool = process_pool(1_000)
        with pool.reserve(950):
            assert PressureMonitor(cfg).classify() >= PressureLevel.CRITICAL
