"""Carbonite memory governance: buffer pool, pressure sensing, spill decision.

The reserve-before-allocate pool accounts a fixed envelope; the pressure monitor
reads live RAM; the resource manager decides spill by comparing a plan's estimated
peak against the budget. These pin that logic without the compiled engine (the
pool falls back to its pure-Python accounting, the estimator reads a tiny mock).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from batcher.carbonite import ResourceManager
from batcher.carbonite.memory import (
    BufferPool,
    PressureMonitor,
    peak_operator_bytes,
    process_pool,
)
from batcher.config import Config, MemoryConfig, config_context

pytestmark = pytest.mark.unit


def _plan_with_peak(*op_bytes: int):
    """A minimal stand-in for a PhysicalPlan: just `.ops[*].bounds.m_max_bytes`."""
    ops = [SimpleNamespace(bounds=SimpleNamespace(m_max_bytes=b)) for b in op_bytes]
    return SimpleNamespace(ops=ops)


# --- Buffer pool -------------------------------------------------------------


def test_pool_admits_until_full_then_rejects():
    pool = BufferPool(1000)
    assert pool.limit == 1000
    with pool.reserve(400) as granted:
        assert granted is True
        assert pool.used == 400
        assert pool.available == 600
        with pool.reserve(700) as granted_2:
            # Over the limit: rejected, and the pool is left untouched.
            assert granted_2 is False
            assert pool.used == 400
    assert pool.used == 0  # both reservations released on exit


def test_pool_releases_on_exception():
    pool = BufferPool(1000)
    with pytest.raises(ValueError), pool.reserve(500) as granted:
        assert granted is True
        assert pool.used == 500
        raise ValueError("boom")
    assert pool.used == 0  # released despite the exception


def test_process_pool_is_a_singleton_with_reconciled_limit():
    a = process_pool(1 << 30)
    b = process_pool(1 << 20)  # same pool, but the limit is reconciled (C11)
    assert a is b
    assert a.limit == (1 << 20)
    process_pool(1 << 30)  # restore for other tests sharing the process-wide pool


# --- Estimation --------------------------------------------------------------


def test_peak_is_the_dominant_operator():
    # The linear pipeline's footprint is its largest breaker, not the sum.
    assert peak_operator_bytes(_plan_with_peak(100, 5000, 200)) == 5000
    assert peak_operator_bytes(_plan_with_peak()) == 0


# --- Spill decision ----------------------------------------------------------


def test_should_spill_when_estimate_exceeds_cap():
    cfg = Config().replace(memory=MemoryConfig(max_memory_bytes=1000))
    with config_context(cfg):
        rm = ResourceManager()
        # budget = 1000 * hard_limit(0.90) = 900
        assert rm.should_spill(_plan_with_peak(2000)) is True
        assert rm.should_spill(_plan_with_peak(500)) is False


def test_unsized_plan_never_spills():
    cfg = Config().replace(memory=MemoryConfig(max_memory_bytes=1))
    with config_context(cfg):
        # No Kyber estimate (m_max_bytes == 0) → never spill on a guess.
        assert ResourceManager().should_spill(_plan_with_peak(0)) is False


def test_pressure_monitor_reports_sane_memory():
    snap = PressureMonitor().snapshot()
    assert snap.total > 0
    assert 0 <= snap.available <= snap.total
    assert 0.0 <= snap.used_fraction <= 1.0


def test_pressure_level_escalates_instantly_on_a_spike(monkeypatch):
    """Escalation is never smoothed: a rising reading drives the level immediately, so
    protective spill is never delayed by the hysteresis."""
    from batcher.carbonite.memory.pressure import PressureLevel

    mon = PressureMonitor()  # default soft 0.85 / hard 0.90
    readings = iter([0.10, 0.95])
    monkeypatch.setattr(mon, "_engine_used_fraction", lambda: next(readings))
    assert mon.level() == PressureLevel.NORMAL  # 0.10
    assert mon.level() == PressureLevel.CRITICAL  # 0.95 → instant, no smoothing


def test_pressure_level_hysteresis_damps_flapping_near_threshold(monkeypatch):
    """Readings oscillating across the soft line would flap SPILL↔ELEVATED every
    sample; the de-escalation hysteresis holds the level through the brief dips so the
    shuffle's AIMD credit window doesn't oscillate."""
    from batcher.carbonite.memory.pressure import PressureLevel

    mon = PressureMonitor()  # soft 0.85
    readings = iter([0.84, 0.87, 0.84, 0.87, 0.84])  # alternate just under/over soft
    monkeypatch.setattr(mon, "_engine_used_fraction", lambda: next(readings))
    levels = [mon.level() for _ in range(5)]
    # Once SPILL is reached, the 0.84 dips are held at SPILL by the lagging average
    # (ewma stays ≈0.85+) instead of dropping to ELEVATED — no per-sample flapping.
    assert levels[1] == PressureLevel.SPILL  # 0.87
    assert levels[2] == PressureLevel.SPILL  # 0.84 dip, held by ewma
    assert levels[3] == PressureLevel.SPILL  # 0.87
    assert levels[4] == PressureLevel.SPILL  # 0.84 dip, still held
