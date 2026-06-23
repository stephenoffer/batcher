"""The three memory-limit decisions agree on one budget (C2/C3/C5), and pressure
is classified against the engine's envelope, not the whole machine (C4)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from batcher.carbonite.manager import ResourceManager
from batcher.carbonite.memory.pool import current_process_pool, process_pool
from batcher.carbonite.memory.pressure import PressureLevel, PressureMonitor
from batcher.config import Config, MemoryConfig, config_context

pytestmark = pytest.mark.unit


def _plan_with_peak(peak: int):
    """Minimal PhysicalPlan stand-in: just `.ops[*].bounds.m_max_bytes`."""
    return SimpleNamespace(ops=[SimpleNamespace(bounds=SimpleNamespace(m_max_bytes=peak))])


def test_spill_and_reserve_share_one_budget():
    # C2/C3: should_spill's threshold and reserve's pool cap are the SAME figure —
    # the hard fraction of the once-sampled envelope — so a plan that fails the
    # spill check is exactly the plan the reservation would reject.
    cfg = Config().replace(memory=MemoryConfig(max_memory_bytes=1_000))
    with config_context(cfg):
        rm = ResourceManager()
        with rm.reserve(0):
            pool_limit = current_process_pool().limit
        # hard budget = 1000 * hard_limit(0.90) = 900
        assert pool_limit == 900
        assert rm.should_spill(_plan_with_peak(901)) is True
        assert rm.should_spill(_plan_with_peak(899)) is False


def test_soft_budget_below_hard_budget():
    # C2: soft (admission/throttle) < hard (spill/reserve), both from one envelope.
    cfg = Config().replace(memory=MemoryConfig(max_memory_bytes=1_000))
    with config_context(cfg):
        rm = ResourceManager()
        assert rm._soft_budget() == 850  # 1000 * 0.85
        assert rm._hard_budget() == 900  # 1000 * 0.90
        assert rm._soft_budget() < rm._hard_budget()


def test_reserve_reports_false_when_over_budget():
    # C30: reserve() must report a False when the reservation doesn't fit, so the
    # caller can route to spill instead of racing into an OOM. (The orchestration
    # acts on this False; here we pin the signal itself.)
    cfg = Config().replace(memory=MemoryConfig(max_memory_bytes=1_000))
    with config_context(cfg):
        rm = ResourceManager()  # hard budget = 900
        with rm.reserve(800) as first:
            assert first is True
            with rm.reserve(800) as second:  # 800 + 800 > 900
                assert second is False


def test_pressure_level_tracks_engine_envelope_not_machine():
    # C4: the pressure level reflects how full the engine's buffer pool is (its own
    # envelope), not the whole machine. Reserving past the hard fraction trips
    # CRITICAL even though the machine has plenty of free RAM.
    cfg = Config().replace(memory=MemoryConfig(max_memory_bytes=1_000))
    with config_context(cfg):
        pool = process_pool(1_000)
        with pool.reserve(950):  # 95% of the envelope → past hard_limit 0.90
            assert PressureMonitor(cfg).level() >= PressureLevel.CRITICAL
        # Released → back to NORMAL (the engine holds nothing).
        assert PressureMonitor(cfg).level() == PressureLevel.NORMAL
