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
from batcher.carbonite.memory.pressure import PressureLevel
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


def test_pool_tracks_peak_used_high_water():
    pool = BufferPool(1000)
    assert pool.peak_used == 0
    with pool.reserve(400):
        assert pool.peak_used == 400
        with pool.reserve(300):
            assert pool.peak_used == 700  # concurrent high-water
        # Releasing the inner block does not lower the recorded high-water.
        assert pool.used == 400
        assert pool.peak_used == 700
    # A rejected reservation (over limit) never raises the high-water.
    with pool.reserve(900) as granted, pool.reserve(900) as granted_2:
        assert granted is True and granted_2 is False
        assert pool.peak_used == 900
    assert pool.used == 0
    assert pool.peak_used == 900  # survives release — it's a lifetime mark


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


def test_input_that_will_not_fit_routes_out_of_core():
    # The in-memory path resolves every source to Arrow *before* the engine runs, so the
    # input is resident in full however small the result is. That term is invisible to the
    # plan estimate — `m_max_bytes` sizes an operator's working set, so a query whose
    # breakers are all tiny reads as "fits" while the scan feeding them does not. Measured
    # on sf100 TPC-H: a GROUP BY returning four rows materialized ~19 GiB of input.
    cfg = Config().replace(memory=MemoryConfig(max_memory_bytes=1_000))
    with config_context(cfg):
        rm = ResourceManager()
        assert rm.input_exceeds_budget(10_000) is True  # budget = 1000 * 0.90
        assert rm.input_exceeds_budget(100) is False


def test_unsized_input_is_not_read_as_fitting():
    # `0` means the sources could not size themselves. That is an absence of evidence, not
    # evidence of fitting, so it must not be spelled the same way as "small enough" — the
    # caller falls back to its other signals instead.
    cfg = Config().replace(memory=MemoryConfig(max_memory_bytes=1_000))
    with config_context(cfg):
        assert ResourceManager().input_exceeds_budget(0) is False


def test_unsized_plan_does_not_spill_on_a_guess(monkeypatch):
    cfg = Config().replace(memory=MemoryConfig(max_memory_bytes=1))
    with config_context(cfg):
        rm = ResourceManager()
        # No Kyber estimate (m_max_bytes == 0) and no measured pressure → don't spill
        # on a guess. The estimate alone can never *prevent* a spill, only add one.
        monkeypatch.setattr(rm._pressure, "classify", lambda: PressureLevel.NORMAL)
        assert rm.should_spill(_plan_with_peak(0)) is False


def test_unsized_plan_spills_when_memory_pressure_is_measured(monkeypatch):
    # The OOM path this closes: Kyber emits 0 bytes for any operator whose cardinality
    # is unknown, so an un-sized plan used to run fully in memory no matter how much of
    # the box was already gone. The measured footprint — the one signal that cannot be
    # wrong the way an estimate can — now overrules the missing estimate. Spilling is
    # result-invariant, so a false positive costs latency and a false negative costs
    # the process.
    cfg = Config().replace(memory=MemoryConfig(max_memory_bytes=1 << 40))
    with config_context(cfg):
        rm = ResourceManager()
        monkeypatch.setattr(rm._pressure, "classify", lambda: PressureLevel.SPILL)
        assert rm.should_spill(_plan_with_peak(0)) is True
        # And it holds for a plan whose estimate says it comfortably fits.
        assert rm.should_spill(_plan_with_peak(1000)) is True


def test_pressure_monitor_reports_sane_memory():
    snap = PressureMonitor().snapshot()
    assert snap.total > 0
    assert 0 <= snap.available <= snap.total
    assert 0.0 <= snap.used_fraction <= 1.0


def test_available_capped_to_cgroup_headroom(monkeypatch):
    """A cgroup-limited container must not read the host's free RAM as its own headroom —
    on a big host it would over-admit and OOM at the cgroup cap. The available reading is
    clamped to `limit - current`."""
    from batcher.carbonite.memory import pressure

    # Big host (180 GB free) but an 8 GB container already using 3 GB → 5 GB real headroom.
    monkeypatch.setattr(pressure, "_cgroup_limit_bytes", lambda: 8 * 1024**3)
    monkeypatch.setattr(pressure, "_cgroup_current_bytes", lambda: 3 * 1024**3)
    assert pressure._cap_to_cgroup_headroom(180 * 1024**3) == 5 * 1024**3
    # When the host figure is already below the cgroup headroom, it wins (no inflation).
    assert pressure._cap_to_cgroup_headroom(2 * 1024**3) == 2 * 1024**3
    # Over-budget container (current > limit) clamps to 0, never negative.
    monkeypatch.setattr(pressure, "_cgroup_current_bytes", lambda: 9 * 1024**3)
    assert pressure._cap_to_cgroup_headroom(180 * 1024**3) == 0
    # No cgroup cap (bare metal) leaves the host reading untouched.
    monkeypatch.setattr(pressure, "_cgroup_limit_bytes", lambda: None)
    assert pressure._cap_to_cgroup_headroom(180 * 1024**3) == 180 * 1024**3


def test_available_reading_is_shared_within_the_ttl(monkeypatch):
    """The expensive live OS read is sampled once per TTL window and reused, so the
    per-query control-plane cost does not pay it on every decision / back-to-back query.
    """
    from batcher.carbonite.memory import pressure

    pressure.reset_memory_sampling()
    calls = {"n": 0}

    def _counting_read() -> int:
        calls["n"] += 1
        return 8 * 1024**3

    monkeypatch.setattr(pressure.PressureMonitor, "_read_available_bytes", _counting_read)
    mon = pressure.PressureMonitor()
    # Many reads across fresh monitors — all served from the one cached sample.
    for _ in range(50):
        assert mon.available_bytes() > 0
        pressure.PressureMonitor().envelope_bytes()
    assert calls["n"] == 1, "the live OS read must be shared across the TTL window"

    # A reset forces the next sample to re-read (the seam tests use after patching).
    pressure.reset_memory_sampling()
    mon.available_bytes()
    assert calls["n"] == 2
    pressure.reset_memory_sampling()  # don't leak the patched sample into later tests


def test_host_ram_is_memoized(monkeypatch):
    """Host RAM is process-constant, so `total_memory_bytes` must not re-run syscalls
    on every call — only the (cheap, live) container-usage figure stays uncached."""
    from batcher.carbonite.memory import pressure

    pressure.reset_memory_sampling()
    sysconf_calls = {"n": 0}
    real_sysconf = pressure.os.sysconf

    def _counting_sysconf(name):
        sysconf_calls["n"] += 1
        return real_sysconf(name)

    monkeypatch.setattr(pressure.os, "sysconf", _counting_sysconf)
    first = pressure.total_memory_bytes()
    for _ in range(100):
        assert pressure.total_memory_bytes() == first
    # Two sysconf calls for the one first read (SC_PAGE_SIZE + SC_PHYS_PAGES), then memoized.
    assert sysconf_calls["n"] <= 2


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
