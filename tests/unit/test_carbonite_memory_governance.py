"""Buffer-pool accounting, pressure classification, and the shape of the learned fit.

Each test pins a way a memory decision could be wrong while every result stayed correct —
the failure mode this subsystem specializes in, since nothing it does can change an answer
and therefore nothing it does gets caught by a differential test.
"""

from __future__ import annotations

import pytest

from batcher.carbonite.memory import probe
from batcher.carbonite.memory.learned import LearnedMemoryModel, _upper_quantile
from batcher.carbonite.memory.pool import (
    BufferPool,
    current_process_pool,
    process_pool,
    reset_process_pool,
)
from batcher.carbonite.memory.pressure import PressureLevel, PressureMonitor
from batcher.config import Config, MemoryConfig, config_context

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh_pool():
    reset_process_pool()
    yield
    reset_process_pool()


# --- the buffer pool ---------------------------------------------------------


def test_a_negative_reservation_cannot_manufacture_headroom() -> None:
    """The pool admits *growth*, so nothing else in it checks the sign of the request."""
    pool = BufferPool(1000)
    with pool.reserve(-500) as granted:
        assert granted is True
        assert pool.used == 0  # NOT -500, which would hand out 1,500 bytes of a 1,000 pool
    assert pool.used == 0
    assert pool.available == 1000


def test_a_zero_reservation_is_free_and_granted() -> None:
    pool = BufferPool(1000)
    with pool.reserve(0) as granted:
        assert granted is True
        assert pool.used == 0


def test_denied_reservations_are_counted() -> None:
    """`peak_used` near the limit only *suggests* the envelope binds; a denial proves it."""
    pool = BufferPool(100)
    with pool.reserve(1000) as granted:
        assert granted is False
    assert pool.denied == 1
    assert pool.stats()["denied"] == 1


def test_the_engines_own_counters_are_preferred_when_it_reports_them() -> None:
    """The control plane sees only its own reservations; the data plane sees the rest."""

    class _EngineBacked:
        limit = 1000
        used = 0
        available = 1000
        peak_used = 900  # the engine reserved for operator state Carbonite never saw
        denied = 3
        spill_requests = 7

        def try_reserve(self, _n):
            return True

        def release(self, _n):
            return None

    pool = BufferPool(1000)
    pool._pool = _EngineBacked()
    assert pool.peak_used == 900
    assert pool.denied == 3
    assert pool.spill_requests == 7
    assert pool.stats()["peak_utilization"] == pytest.approx(0.9)


def test_denials_are_not_double_counted_across_the_boundary() -> None:
    """The engine's counter already includes every denial the control plane caused."""

    class _EngineBacked:
        limit = 10
        used = 10
        available = 0
        peak_used = 10
        denied = 1  # the same refusal the Python side is about to record
        spill_requests = 0

        def try_reserve(self, _n):
            return False

        def release(self, _n):
            return None

    pool = BufferPool(10)
    pool._pool = _EngineBacked()
    with pool.reserve(100) as granted:
        assert granted is False
    assert pool.denied == 1, "summing would report two refusals where one happened"


def test_utilization_and_stats_agree_with_the_accounting() -> None:
    pool = BufferPool(1000)
    assert pool.utilization == 0.0
    with pool.reserve(250):
        assert pool.utilization == 0.25
        stats = pool.stats()
        assert stats["used_bytes"] == 250
        assert stats["available_bytes"] == 750
        assert stats["peak_utilization"] == 0.25
    assert BufferPool(0).utilization == 1.0  # a zero-limit envelope is full by definition


def test_a_deferred_shrink_is_remembered_not_dropped() -> None:
    """The bug: a shrink refused while the pool was busy used to be discarded forever."""
    pool = process_pool(1 << 30)
    with pool.reserve(1024):
        process_pool(1 << 20)  # busy — must not shrink under a running query
        assert pool.limit == 1 << 30
    # Idle again. The next reconcile applies the shrink that was held back, even though
    # this caller asked for the *larger* figure it already has.
    process_pool(1 << 30)
    assert pool.limit == 1 << 20


def test_growth_always_applies_immediately() -> None:
    pool = process_pool(1 << 20)
    with pool.reserve(1024):
        process_pool(1 << 30)  # capacity the autoscaler just added must not wait
        assert pool.limit == 1 << 30


def test_reset_process_pool_drops_the_singleton() -> None:
    process_pool(1 << 20)
    assert current_process_pool() is not None
    reset_process_pool()
    assert current_process_pool() is None


# --- pressure classification -------------------------------------------------


def _monitor(soft: float, hard: float) -> PressureMonitor:
    return PressureMonitor(Config().replace(memory=MemoryConfig(soft_limit=soft, hard_limit=hard)))


def test_the_ladder_survives_an_inverted_soft_hard_config() -> None:
    """`hard < soft` taken literally deletes the ELEVATED and SPILL bands entirely."""
    mon = _monitor(soft=0.85, hard=0.50)
    # `hard` is the cap and wins; `soft` is pulled down to it. There is still a warning
    # band below the cap, which an unordered comparison did not leave: it jumped straight
    # from NORMAL to the cache-clearing, producer-pausing CRITICAL.
    assert mon._classify(0.30) is PressureLevel.NORMAL
    assert mon._classify(0.47) is PressureLevel.ELEVATED
    assert mon._classify(0.60) is PressureLevel.CRITICAL


def test_the_ladder_is_monotone_in_usage() -> None:
    mon = _monitor(soft=0.85, hard=0.90)
    levels = [mon._classify(u / 100) for u in range(0, 101)]
    assert levels == sorted(levels), "a fuller box must never classify as calmer"
    assert levels[0] is PressureLevel.NORMAL
    assert levels[-1] is PressureLevel.CRITICAL


def test_elevated_sits_below_the_soft_limit() -> None:
    mon = _monitor(soft=0.80, hard=0.90)
    assert mon._classify(0.75) is PressureLevel.ELEVATED  # between 0.9 * 0.80 and 0.80
    assert mon._classify(0.70) is PressureLevel.NORMAL


def test_reset_forgets_the_hysteresis_and_flap_history() -> None:
    mon = _monitor(soft=0.85, hard=0.90)
    mon.level()
    mon.level()
    assert mon.flap_rate() is not None
    mon.reset()
    assert mon.flap_rate() is None
    assert mon._ewma is None


def test_headroom_never_goes_negative(monkeypatch) -> None:
    mon = _monitor(soft=0.85, hard=0.90)
    monkeypatch.setattr(PressureMonitor, "_engine_used_fraction", staticmethod(lambda: 0.99))
    assert mon.headroom_bytes() == 0


def test_reset_memory_sampling_clears_the_cgroup_cap(monkeypatch) -> None:
    """The lru_cache on the cap survived the reset, silently overriding every later patch."""
    probe.reset_memory_sampling()
    monkeypatch.setattr(probe, "read_cgroup_bytes", lambda _p: 4096)
    monkeypatch.setattr(probe, "cgroup_v2_dirs", lambda: ["/sys/fs/cgroup"])
    assert probe.cgroup_limit_bytes() == 4096

    probe.reset_memory_sampling()
    monkeypatch.setattr(probe, "read_cgroup_bytes", lambda _p: 8192)
    assert probe.cgroup_limit_bytes() == 8192
    probe.reset_memory_sampling()


# --- the learned fit ---------------------------------------------------------


def test_upper_quantile_matches_the_type_7_definition() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _upper_quantile(values, 0.0) == 1.0
    assert _upper_quantile(values, 1.0) == 5.0
    assert _upper_quantile(values, 0.5) == 3.0
    assert _upper_quantile(values, 0.8) == pytest.approx(4.2)
    assert _upper_quantile([7.0]) == 7.0


def test_the_fit_lands_above_the_median_of_the_samples() -> None:
    """Memory is one-sided: a median is exceeded by half the runs, and those are the OOMs."""
    values = [10.0, 10.0, 10.0, 10.0, 100.0]
    assert _upper_quantile(values) > 10.0  # the median would be exactly 10


def test_widest_row_is_derived_not_supplied() -> None:
    """A test double that forgot the derived field used to disable wide-row credit sizing."""
    model = LearnedMemoryModel(
        _bytes_per_row={"aggregate": 512.0, "scan": 8.0},
        _alpha=0.5,
        _clamp=4.0,
        _row_bytes=64,
        _spill_per_row={},
    )
    assert model.max_bytes_per_row() == 512.0
    assert model.max_bytes_per_row(["Scan"]) == 8.0
    assert LearnedMemoryModel({}, 0.5, 4.0, 64, {}).max_bytes_per_row() is None


def test_the_envelope_reports_the_plans_own_parallelism_and_credits() -> None:
    from batcher.carbonite.base import ResourceContext
    from batcher.carbonite.memory.estimator import OperatorMemoryEstimator, binding_operator
    from batcher.plan.physical import OpId, PhysicalOp, PhysicalPlan
    from batcher.plan.resource import ResourceBounds

    ops = (
        PhysicalOp(
            op_id=OpId(1),
            kind="Scan",
            backend="native",
            algorithm="scan",
            bounds=ResourceBounds(m_max_bytes=10, c_max_credits=2, n_max_parallelism=3),
            inputs=(),
        ),
        PhysicalOp(
            op_id=OpId(2),
            kind="Aggregate",
            backend="native",
            algorithm="hash",
            bounds=ResourceBounds(m_max_bytes=1000, c_max_credits=32, n_max_parallelism=64),
            inputs=(OpId(1),),
        ),
    )
    plan = PhysicalPlan(ir={}, output_schema=None, ops=ops)
    with config_context(Config()):
        env = OperatorMemoryEstimator().envelope(plan, ResourceContext(config=Config()))
    assert env.m_max_bytes == 1000
    assert env.c_max_credits == 32
    assert env.n_max_parallelism == 64
    assert binding_operator(plan) is ops[1]
    assert binding_operator(PhysicalPlan(ir={}, output_schema=None, ops=())) is None


def test_an_infeasible_verdict_names_the_binding_operator() -> None:
    from batcher.carbonite.base import ResourceContext
    from batcher.carbonite.policies.admission import BudgetingAdmission
    from batcher.plan.physical import OpId, PhysicalOp, PhysicalPlan
    from batcher.plan.resource import ResourceBounds

    op = PhysicalOp(
        op_id=OpId(7),
        kind="Aggregate",
        backend="native",
        algorithm="hash",
        bounds=ResourceBounds(m_max_bytes=1 << 60, c_max_credits=0, n_max_parallelism=0),
        inputs=(),
    )
    plan = PhysicalPlan(ir={}, output_schema=None, ops=(op,))
    verdict = BudgetingAdmission(available_bytes=1 << 20).validate(
        plan, ResourceContext(config=Config())
    )
    assert verdict.feasible is False
    assert verdict.binding_op == "Aggregate#7"
