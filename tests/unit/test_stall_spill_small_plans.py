"""A kernel stall alone spills only the plans large enough for a spill to relieve it.

`PressureMonitor.stall_floor` raises the level to SPILL while PSI says the cgroup thrashes. On
a node other tenants are exhausting, that holds indefinitely, and taken as an order for every
query it sent a TPC-H sf1 join estimated at 3.5 MB to disk. These pin the narrowing and,
more importantly, everything it must leave alone: a large sized plan, an un-sized plan, and
any plan at all once the engine's *own* byte accounting is in the spill band.
"""

from __future__ import annotations

import pytest

from batcher.carbonite import ResourceManager
from batcher.carbonite.memory.pressure import PressureLevel, PressureMonitor
from batcher.config import Config, MemoryConfig, config_context
from batcher.plan.physical import OpId, PhysicalOp, PhysicalPlan
from batcher.plan.resource import ResourceBounds

pytestmark = pytest.mark.unit

_GIB = 1 << 30
_CFG = Config(memory=MemoryConfig(max_memory_bytes=10 * _GIB, stall_aware_pressure=True))


def _plan(peak_bytes: int) -> PhysicalPlan:
    op = PhysicalOp(
        op_id=OpId(1),
        kind="Aggregate",
        backend="native",
        algorithm="hash",
        bounds=ResourceBounds(m_max_bytes=peak_bytes, c_max_credits=4, n_max_parallelism=4),
        inputs=(),
    )
    return PhysicalPlan(ir={}, output_schema=None, ops=(op,))


@pytest.fixture
def pressure(monkeypatch):
    """Set the PSI `full` share and the byte-accounting used fraction independently."""

    def set_to(*, stall: float, used: float) -> None:
        monkeypatch.setattr(
            "batcher.carbonite.memory.kernel.memory_stall_full", lambda: stall, raising=False
        )
        monkeypatch.setattr(PressureMonitor, "_engine_used_fraction", staticmethod(lambda: used))
        # No OOM history, so the third spill signal cannot be what decides a case here.
        from batcher.carbonite.memory.kernel import KernelMemoryState

        monkeypatch.setattr(
            "batcher.carbonite.memory.kernel.kernel_memory_state",
            lambda: KernelMemoryState(oom_kills=0),
            raising=False,
        )

    return set_to


def _reason(plan: PhysicalPlan) -> str | None:
    with config_context(_CFG):
        return ResourceManager(_CFG).spill_reason(plan)


def test_the_fixture_really_is_stall_only(pressure):
    # The positive control for every case below: the level is SPILL, and only the stall put it
    # there — so a plan staying in memory is the narrowing at work and not a quiet box.
    pressure(stall=0.5, used=0.1)
    monitor = PressureMonitor(_CFG)
    assert monitor.classify() is PressureLevel.SPILL
    assert monitor.accounted_level() is PressureLevel.NORMAL


def test_a_small_sized_plan_stays_in_memory_under_a_stall_alone(pressure):
    pressure(stall=0.5, used=0.1)
    assert _reason(_plan(4 << 20)) is None


def test_a_large_sized_plan_still_spills_under_a_stall(pressure):
    pressure(stall=0.5, used=0.1)
    reason = _reason(_plan(4 * _GIB))
    assert reason is not None
    assert "pressure" in reason


def test_an_unsized_plan_still_spills_under_a_stall(pressure):
    # `0` is Kyber's "could not estimate", which says nothing about the plan being small.
    pressure(stall=0.5, used=0.1)
    assert _reason(_plan(0)) is not None


def test_a_full_envelope_spills_even_a_small_plan(pressure):
    # The byte accounting is the engine's own evidence, and the narrowing never overrides it.
    pressure(stall=0.0, used=0.87)
    assert _reason(_plan(4 << 20)) is not None


def test_a_full_envelope_and_a_stall_together_spill_a_small_plan(pressure):
    pressure(stall=0.5, used=0.87)
    assert _reason(_plan(4 << 20)) is not None


def test_a_quiet_box_spills_nothing(pressure):
    pressure(stall=0.0, used=0.1)
    assert _reason(_plan(4 << 20)) is None
