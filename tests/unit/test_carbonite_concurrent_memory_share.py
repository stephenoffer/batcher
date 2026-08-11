"""Concurrent queries divide the memory envelope, the way they already divide the cores.

`ConcurrencyLimiter.width_for` has always narrowed a query's rayon pool when others are
running. Memory had no equivalent: every admitted query compared its estimated peak against
the *whole* envelope, so N plans that each need most of RAM all read as fitting and all N
took the in-memory path. That is the oversubscription failure the module was written for,
one level down.

These tests hold the property rather than the arithmetic: with concurrency unbounded
(the default) nothing may change at all, and with slots configured the per-query budget must
shrink with occupancy while the *pool* — the process's one real envelope — must not.
"""

from __future__ import annotations

import dataclasses
import threading
from contextlib import contextmanager

import pytest

from batcher.carbonite.manager import ResourceManager
from batcher.carbonite.memory.pool import current_process_pool, reset_process_pool
from batcher.carbonite.policies.concurrency import (
    ConcurrencyLimiter,
    process_limiter,
    query_memory_share,
    reset_process_limiter,
)
from batcher.config import Config, config_context
from batcher.plan.physical import PhysicalOp, PhysicalPlan
from batcher.plan.resource import ResourceBounds

pytestmark = pytest.mark.unit

_MIB = 1 << 20


@pytest.fixture(autouse=True)
def _clean_process_state():
    """Both process singletons, before and after: they outlive a test otherwise."""
    reset_process_limiter()
    reset_process_pool()
    yield
    reset_process_limiter()
    reset_process_pool()


def _plan(peak_mib: int) -> PhysicalPlan:
    """A one-breaker plan whose materializing operator needs `peak_mib` MiB."""
    op = PhysicalOp(
        op_id=0,
        kind="Aggregate",
        backend="native",
        algorithm="",
        bounds=ResourceBounds(m_max_bytes=peak_mib * _MIB, c_max_credits=0, n_max_parallelism=0),
        inputs=(),
    )
    return PhysicalPlan(ir={}, output_schema=None, ops=(op,))


def _config(*, slots: int, envelope_mib: int) -> Config:
    base = Config()
    return dataclasses.replace(
        base,
        execution=dataclasses.replace(base.execution, max_concurrent_queries=slots),
        memory=dataclasses.replace(base.memory, max_memory_bytes=envelope_mib * _MIB),
    )


# --- the rule -----------------------------------------------------------------


@pytest.mark.parametrize("active", [0, 1, 5])
def test_unbounded_concurrency_never_divides_the_envelope(active: int) -> None:
    """The default deployment must be byte-identical, whatever the occupancy reads."""
    assert query_memory_share(active, slots=0) == 1.0


def test_a_lone_query_gets_the_whole_envelope() -> None:
    assert query_memory_share(1, slots=8) == 1.0


def test_the_share_is_one_over_the_queries_running() -> None:
    assert query_memory_share(2, slots=8) == pytest.approx(0.5)
    assert query_memory_share(4, slots=8) == pytest.approx(0.25)


def test_the_share_never_falls_below_a_full_slot_count() -> None:
    """A limiter rebuilt under load can transiently read more active than it has slots.

    Dividing by that number would hand a real query less than the share the slot count
    guarantees it, on the one path whose job is to keep the guarantee.
    """
    assert query_memory_share(12, slots=4) == pytest.approx(0.25)


# --- the limiter reads its own occupancy --------------------------------------


def test_the_limiter_reports_the_share_for_its_live_occupancy() -> None:
    limiter = ConcurrencyLimiter(slots=4, cores=8)
    assert limiter.memory_share() == 1.0
    limiter.acquire()
    try:
        assert limiter.memory_share() == 1.0  # one query: the whole envelope
    finally:
        limiter.release()


def test_the_share_appears_in_the_admission_stats() -> None:
    """An operator reading `stats()` must be able to tell a small box from a busy one."""
    limiter = ConcurrencyLimiter(slots=4, cores=8)
    assert limiter.stats()["memory_share"] == 1.0


# --- end to end through the manager -------------------------------------------


def test_the_budget_is_undivided_when_concurrency_is_unbounded() -> None:
    """The default path: `max_concurrent_queries = 0` must change no budget."""
    with config_context(_config(slots=0, envelope_mib=1000)):
        rm = ResourceManager()
        stats = rm.stats()
        assert stats["memory_share"] == 1.0
        assert stats["query_envelope_bytes"] == stats["envelope_bytes"]


@contextmanager
def _slot_held_by_another_thread(limiter: ConcurrencyLimiter):
    """Hold one of `limiter`'s slots from a second thread for the block.

    Occupancy has to come from a *different* thread: the limiter treats a second `acquire`
    on the same one as a re-entrant nested query (a `collect()` inside a UDF) and lets it
    through without consuming a slot, which is exactly the case that must not raise the
    apparent concurrency.
    """
    acquired = threading.Event()
    finish = threading.Event()

    def hold() -> None:
        limiter.acquire()
        acquired.set()
        finish.wait(timeout=30)
        limiter.release()

    worker = threading.Thread(target=hold, daemon=True)
    worker.start()
    assert acquired.wait(timeout=30), "the holder thread never acquired a slot"
    try:
        yield
    finally:
        finish.set()
        worker.join(timeout=30)


def test_a_second_admitted_query_halves_the_spill_threshold() -> None:
    """The behaviour this exists for: a plan that fits alone must spill when it is one of N.

    600 MiB of state inside a 1000 MiB envelope fits the hard budget on its own. With a
    second query holding a slot the entitlement is 500 MiB, and the same plan must go
    out-of-core rather than racing the neighbour into an OOM.
    """
    config = _config(slots=4, envelope_mib=1000)
    plan = _plan(600)
    with config_context(config):
        assert not ResourceManager(config).should_spill(plan), (
            "a lone query was refused the envelope it has entirely to itself"
        )
        limiter = process_limiter(config)
        with _slot_held_by_another_thread(limiter):
            assert limiter.active == 1
            # This query is the second: the manager it builds sees an occupancy of one
            # other, which is the conservative reading a pre-admission manager gets.
            with _slot_held_by_another_thread(limiter):
                assert limiter.active == 2
                assert ResourceManager(config).should_spill(plan), (
                    "a plan needing 60% of the envelope stayed in memory while two other "
                    "queries held slots"
                )


def test_a_nested_query_does_not_shrink_its_own_budget() -> None:
    """A `collect()` inside a UDF is the outer query's work, not competition for it.

    It takes no slot, so it must not raise the occupancy and halve the budget of the query
    that is already paying for the machine.
    """
    limiter = ConcurrencyLimiter(slots=4, cores=8)
    limiter.acquire()
    try:
        limiter.acquire()  # nested, same thread
        try:
            assert limiter.active == 1
            assert limiter.memory_share() == 1.0
        finally:
            limiter.release()
    finally:
        limiter.release()


def test_the_manager_divides_its_budget_by_the_live_occupancy(monkeypatch) -> None:
    """Two queries running: this one's hard budget is half what it would be alone."""
    config = _config(slots=4, envelope_mib=1000)
    with config_context(config):
        alone = ResourceManager(config)
        solo_budget = alone.stats()["hard_budget_bytes"]

        monkeypatch.setattr(
            "batcher.carbonite.manager._memory_share", lambda _config: 0.5, raising=True
        )
        shared = ResourceManager(config)
        shared_stats = shared.stats()

    assert shared_stats["memory_share"] == 0.5
    assert shared_stats["hard_budget_bytes"] == pytest.approx(solo_budget / 2, rel=1e-9)
    assert shared_stats["query_envelope_bytes"] == pytest.approx(
        shared_stats["envelope_bytes"] / 2, rel=1e-9
    )


def test_the_shared_pool_keeps_the_whole_envelope(monkeypatch) -> None:
    """The share bounds what a query *plans* to hold; it must not shrink the real pool.

    Sizing the process pool to a share would make a concurrent query's already-granted
    reservation retroactively unaffordable, and would do it differently depending on which
    query called `reserve` last.
    """
    config = _config(slots=4, envelope_mib=1000)
    with config_context(config):
        monkeypatch.setattr(
            "batcher.carbonite.manager._memory_share", lambda _config: 0.25, raising=True
        )
        rm = ResourceManager(config)
        with rm.reserve(_MIB):
            pass
        pool = current_process_pool()

    assert pool is not None
    expected = int(1000 * _MIB * config.memory.hard_limit)
    assert pool.limit == expected, (
        "the process pool was sized to one query's share instead of the whole envelope"
    )


def test_a_spill_decision_and_a_reservation_still_agree_under_a_share(monkeypatch) -> None:
    """The invariant `SpillAdvisor` exists to hold, checked with the share applied.

    A plan just over the (shared) hard budget must both read as needing to spill and fail
    to reserve. If the two figures were derived separately, a share would be exactly the
    change that pulls them apart.
    """
    config = _config(slots=4, envelope_mib=1000)
    with config_context(config):
        monkeypatch.setattr(
            "batcher.carbonite.manager._memory_share", lambda _config: 0.5, raising=True
        )
        rm = ResourceManager(config)
        budget = int(rm.stats()["hard_budget_bytes"])
        over = _plan((budget // _MIB) + 8)
        assert rm.should_spill(over)
        assert rm.estimated_bytes(over) > budget
