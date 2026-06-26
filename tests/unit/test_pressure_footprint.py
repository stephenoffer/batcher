"""Phase-3a: pressure sensing must see the process's real footprint.

The buffer pool only tracks its own reservations; the in-memory Flight shuffle store
and off-pool pyarrow buffers are invisible to it. If the monitor classified on the
pool alone it could report NORMAL while the kernel OOM-kills a shuffle-heavy worker.
These assert the off-pool footprint (cgroup current / RSS) drives the level.
"""

from __future__ import annotations

from batcher.carbonite.memory import pressure
from batcher.carbonite.memory.pressure import PressureLevel, PressureMonitor


def test_footprint_drives_pressure_when_pool_is_idle(monkeypatch):
    total = pressure.total_memory_bytes()
    # No cgroup; RSS reports ~96% of the ceiling (off-pool memory the pool can't see).
    monkeypatch.setattr(pressure, "_cgroup_current_bytes", lambda: None)
    monkeypatch.setattr(pressure, "_process_rss_bytes", lambda: int(total * 0.96))
    # No pool initialized, so the pool term contributes nothing.
    monkeypatch.setattr("batcher.carbonite.memory.pool.current_process_pool", lambda: None)

    frac = PressureMonitor._engine_used_fraction()
    assert frac >= 0.95  # the footprint, not a falsely-idle 0.0
    assert PressureMonitor().level() == PressureLevel.CRITICAL


def test_cgroup_current_preferred_over_rss(monkeypatch):
    total = pressure.total_memory_bytes()
    monkeypatch.setattr(pressure, "_cgroup_current_bytes", lambda: int(total * 0.88))
    monkeypatch.setattr(pressure, "_process_rss_bytes", lambda: int(total * 0.10))
    monkeypatch.setattr("batcher.carbonite.memory.pool.current_process_pool", lambda: None)
    frac = PressureMonitor._engine_used_fraction()
    assert 0.85 <= frac <= 0.91  # the cgroup figure, not the lower RSS one


def test_no_footprint_reading_does_not_crash(monkeypatch):
    # Neither cgroup nor RSS available, no pool → falls back, never raises.
    monkeypatch.setattr(pressure, "_cgroup_current_bytes", lambda: None)
    monkeypatch.setattr(pressure, "_process_rss_bytes", lambda: None)
    monkeypatch.setattr("batcher.carbonite.memory.pool.current_process_pool", lambda: None)
    frac = PressureMonitor._engine_used_fraction()
    assert 0.0 <= frac <= 1.0
