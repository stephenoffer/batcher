"""Carbonite can see the pool the *engine* reserves against, not only its own.

There are two `MemoryPool`s in a Batcher process and they had never met. Carbonite's
`BufferPool` wraps one the control plane constructs and charges its coarse per-query
reservations to; `execute_plan` charges operator state and the Flight transit buffers to a
separate process-wide pool inside the engine. On a real query the second is by far the
larger footprint, and the control plane could only infer it from process RSS -- which lags
a reservation by however long the operator takes to fill the state it reserved.

`engine_pool_stats` closes that by *reading* the engine's pool. It deliberately does not
merge the two counters: Carbonite reserves a plan's estimated peak and the engine then
reserves the same operator's actual bytes, so one shared counter would double-count every
query and spill it at half its budget.
"""

from __future__ import annotations

import dataclasses

import pytest

from batcher.carbonite.manager import ResourceManager
from batcher.carbonite.memory.pool import (
    engine_pool_stats,
    engine_pool_utilization,
    reset_process_pool,
)
from batcher.carbonite.memory.pressure import PressureLevel, PressureMonitor
from batcher.config import Config

pytestmark = pytest.mark.unit

_MIB = 1 << 20


@pytest.fixture(autouse=True)
def _clean_pool():
    reset_process_pool()
    yield
    reset_process_pool()


def _engine_has_the_reader() -> bool:
    """Whether the built extension exposes the accessor (an older `.so` may not)."""
    from batcher._internal.native import engine_or_none

    mod = engine_or_none()
    return mod is not None and hasattr(mod, "engine_pool_stats")


_needs_reader = pytest.mark.skipif(
    not _engine_has_the_reader(),
    reason="the built extension predates `engine_pool_stats`; rebuild with `just build`",
)


# --- the accessor degrades, always ---------------------------------------------


def test_reading_the_engine_pool_never_raises() -> None:
    """Every caller is a *diagnostic* path, so an absent engine must read as absent.

    Returning `None` rather than zeros is the contract: a dict of zeros asserts that a pool
    exists and is empty, which is a different claim from "no query has run under a budget".
    """
    stats = engine_pool_stats()
    assert stats is None or isinstance(stats, dict)


def test_an_unbounded_engine_pool_has_no_utilization() -> None:
    """A zero limit governs nothing, so it must not read as 100% full.

    The Rust `PoolStats.utilization` reports `1.0` for a zero limit, which is right there
    (a pool that admits nothing is full by definition) and wrong here: pinning the pressure
    level at CRITICAL for a process that opted out of the budget entirely would halve every
    morsel and spill every query.
    """
    stats = engine_pool_stats()
    if stats is not None and stats["limit_bytes"] == 0:
        assert engine_pool_utilization() is None


@_needs_reader
def test_the_engine_pool_reports_the_documented_shape() -> None:
    """`stats()` promises these keys to the profile and the metrics export."""
    import batcher as bt

    cfg = Config()
    cfg = dataclasses.replace(
        cfg, memory=dataclasses.replace(cfg.memory, max_memory_bytes=512 * _MIB)
    )
    with bt.config_context(cfg):
        bt.from_pydict({"g": [i % 8 for i in range(2048)], "x": [1.0] * 2048}).group_by("g").agg(
            s=bt.sum("x")
        ).collect()

    stats = engine_pool_stats()
    assert stats is not None, "a query ran under a budget but the engine pool does not exist"
    assert set(stats) == {
        "limit_bytes",
        "used_bytes",
        "available_bytes",
        "peak_used_bytes",
        "denied",
        "spill_requests",
        "utilization",
        # The data plane's own pressure line and the level it implies. Without them a reader
        # could see `used` and had to assume where the line between "filling its envelope"
        # and "idle while the box is full elsewhere" sat — opposite problems with opposite
        # fixes, and the pool is the only thing that knows which side it is on.
        "soft_limit_bytes",
        "pressure",
    }
    assert stats["limit_bytes"] > 0
    # RAII reservations, so the pool drains between queries; the *peak* is what survives.
    assert stats["used_bytes"] == 0
    assert 0.0 <= stats["utilization"] <= 1.0
    assert 0 < stats["soft_limit_bytes"] <= stats["limit_bytes"]
    # Drained, so the level has to be the quiet one. A pool that reported pressure while
    # holding nothing would make every idle process look like one about to spill.
    assert stats["pressure"] == "NOMINAL"


@_needs_reader
def test_the_manager_reports_both_pools_side_by_side() -> None:
    """The pair is the diagnosis, so one without the other is half an answer."""
    import batcher as bt

    cfg = Config()
    cfg = dataclasses.replace(
        cfg, memory=dataclasses.replace(cfg.memory, max_memory_bytes=512 * _MIB)
    )
    with bt.config_context(cfg):
        bt.from_pydict({"g": [1, 2, 3], "x": [1.0, 2.0, 3.0]}).group_by("g").agg(
            s=bt.sum("x")
        ).collect()
        stats = ResourceManager(cfg).stats()

    assert "engine_pool" in stats, "the engine's own envelope is missing from the snapshot"
    assert stats["engine_pool"]["limit_bytes"] > 0


# --- pressure sees it ----------------------------------------------------------


def test_pressure_reads_the_engine_pool(monkeypatch) -> None:
    """The behaviour this exists for.

    A query holding 95% of the *engine's* envelope classified as NORMAL, because the
    monitor read only the control plane's pool and then waited for RSS to catch up. RSS
    lags the reservation by however long the operator takes to fill the state.
    """
    monkeypatch.setattr(
        "batcher.carbonite.memory.pool.engine_pool_utilization", lambda: 0.95, raising=True
    )
    # Neither the control plane's pool nor the process footprint says anything, so the
    # engine's reading is the only candidate and must carry the classification alone.
    monkeypatch.setattr(
        "batcher.carbonite.memory.probe.cgroup_current_bytes", lambda: 0, raising=True
    )
    monkeypatch.setattr("batcher.carbonite.memory.probe.process_rss_bytes", lambda: 0, raising=True)
    assert PressureMonitor(Config()).classify() >= PressureLevel.SPILL


def test_an_absent_engine_pool_leaves_pressure_where_it_was(monkeypatch) -> None:
    """The negative control: an unbuilt or unbudgeted engine must change nothing."""
    monkeypatch.setattr(
        "batcher.carbonite.memory.pool.engine_pool_utilization", lambda: None, raising=True
    )
    monkeypatch.setattr(
        "batcher.carbonite.memory.probe.cgroup_current_bytes", lambda: 0, raising=True
    )
    monkeypatch.setattr("batcher.carbonite.memory.probe.process_rss_bytes", lambda: 0, raising=True)
    assert PressureMonitor(Config()).classify() is PressureLevel.NORMAL


# --- admission sees it too ------------------------------------------------------


def test_admission_subtracts_the_engine_pool_not_only_its_own(monkeypatch) -> None:
    """Admission budgets against what is *actually* held, across both pools.

    It read only Carbonite's own pool, so on a machine already executing one large query
    it subtracted the estimate and not the bytes the estimate turned into.
    """
    from batcher.carbonite.base import ResourceContext
    from batcher.carbonite.policies import admission as admission_mod

    ctx = ResourceContext(config=Config(), envelope_bytes=100 * _MIB)
    monkeypatch.setattr(
        "batcher.carbonite.memory.pool.engine_pool_stats",
        lambda: {"used_bytes": 90 * _MIB, "limit_bytes": 100 * _MIB},
        raising=True,
    )
    assert admission_mod._bytes_already_held() >= 90 * _MIB, (
        "admission cannot see the memory the engine is actually holding"
    )
    # A plan needing 50 MiB no longer fits an envelope 90 MiB of which is already in use.
    plan = _breaker_plan(50)
    assert not admission_mod.BudgetingAdmission().validate(plan, ctx).feasible


def test_the_two_pools_are_not_double_counted(monkeypatch) -> None:
    """One neighbouring query, counted once: the max of the two readings, not the sum.

    The counters describe the same query from two sides -- an estimate and the bytes it
    turned into -- so adding them refuses queries the box can hold.
    """
    from batcher.carbonite.memory.pool import process_pool
    from batcher.carbonite.policies import admission as admission_mod

    pool = process_pool(100 * _MIB)
    monkeypatch.setattr(
        "batcher.carbonite.memory.pool.engine_pool_stats",
        lambda: {"used_bytes": 40 * _MIB, "limit_bytes": 100 * _MIB},
        raising=True,
    )
    with pool.reserve(30 * _MIB):
        assert admission_mod._bytes_already_held() == 40 * _MIB


def _breaker_plan(peak_mib: int):
    """A one-breaker plan sized `peak_mib` MiB, for the admission checks above."""
    from batcher.plan.physical import PhysicalOp, PhysicalPlan
    from batcher.plan.resource import ResourceBounds

    op = PhysicalOp(
        op_id=0,
        kind="Join",
        backend="native",
        algorithm="",
        bounds=ResourceBounds(m_max_bytes=peak_mib * _MIB, c_max_credits=0, n_max_parallelism=0),
        inputs=(),
    )
    return PhysicalPlan(ir={}, output_schema=None, ops=(op,))
