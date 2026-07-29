"""Flow control, admission concurrency, and the scheduling grant.

These three policies share a failure shape: each one has a path where being wrong costs
throughput or stability and costs *nothing* a correctness test can see. A credit ceiling
that is too generous OOMs a node; a limiter that leaks a slot oversubscribes the box
forever; a fan-out derived from an unknown cardinality asks Ray for a million tasks.
"""

from __future__ import annotations

import threading

import pytest

from batcher.carbonite.policies.concurrency import (
    AdmissionTimeout,
    ConcurrencyLimiter,
    process_limiter,
    reset_process_limiter,
)
from batcher.carbonite.policies.flow_control import AIMDFlowControl, credit_ceiling
from batcher.carbonite.policies.scheduling import _MAX_TASK_FANOUT, DefaultSchedulingPolicy
from batcher.carbonite.policies.spill_shape import partitions_for_volume
from batcher.config import Config, ExecutionConfig, FlowControlConfig
from batcher.plan.physical import OpId, PhysicalOp, PhysicalPlan
from batcher.plan.resource import ResourceBounds

pytestmark = pytest.mark.unit


# --- flow control ------------------------------------------------------------


def test_the_byte_budget_never_raises_a_deliberately_tiny_configured_cap() -> None:
    """The floor used to be one morsel, which *raised* a cap tuned below it."""
    cfg = Config().replace(
        flow_control=FlowControlConfig(credit_byte_budget=4096),
        execution=ExecutionConfig(morsel_bytes=1 << 20),
    )
    from batcher.carbonite.policies.flow_control import _channel_byte_budget

    assert _channel_byte_budget(cfg) <= 4096


def test_the_byte_budget_divides_by_the_channels_that_actually_fetch() -> None:
    """`shuffle_fetch_fan_in` is a cap on concurrency, not a measurement of it."""
    from batcher.carbonite.policies.flow_control import _channel_byte_budget

    cfg = Config()
    few = _channel_byte_budget(cfg, channels=2)
    many = _channel_byte_budget(cfg, channels=64)
    assert few >= many, "fewer concurrent channels may each hold at least as much"
    assert credit_ceiling(cfg, channels=64) <= credit_ceiling(cfg, channels=2)


def test_cubic_recovers_to_the_measured_capacity_and_stays_bounded() -> None:
    cfg = Config().replace(flow_control=FlowControlConfig(default_credits=16))
    ctrl = AIMDFlowControl(cfg)
    for _ in range(6):  # slow-start up to the ceiling
        ctrl.observe(congested=False)
    at_ceiling = ctrl.window
    ctrl.observe(congested=True)
    backed_off = ctrl.window
    assert backed_off < at_ceiling

    for _ in range(50):
        ctrl.observe(congested=False)
    assert ctrl.window <= at_ceiling, "growth must never exceed the memory-safe ceiling"
    assert ctrl.window >= backed_off


def test_the_recovery_clock_is_bounded() -> None:
    """A streaming channel observes millions of rounds; `(t - k)**3` must not run away."""
    from batcher.carbonite.policies.flow_control import _MAX_RECOVERY_ROUNDS

    ctrl = AIMDFlowControl(Config())
    ctrl.observe(congested=True)
    for _ in range(200):
        ctrl.observe(congested=False)
    assert ctrl._rounds_since_backoff == 200

    # Jump the clock to the cap rather than looping a million times, then keep observing:
    # the counter must stop, and the window must stay pinned at the ceiling rather than
    # cubing an ever-growing integer.
    ctrl._rounds_since_backoff = _MAX_RECOVERY_ROUNDS
    for _ in range(10):
        ctrl.observe(congested=False)
    assert ctrl._rounds_since_backoff == _MAX_RECOVERY_ROUNDS
    assert ctrl.window == ctrl._ceiling


def test_the_controller_reports_its_backpressure_story() -> None:
    ctrl = AIMDFlowControl(Config())
    ctrl.observe(congested=False)
    ctrl.observe(congested=True)
    ctrl.observe(congested=False)
    stats = ctrl.stats()
    assert stats["rounds"] == 3
    assert stats["backoffs"] == 1
    assert stats["backoff_rate"] == pytest.approx(1 / 3)
    assert stats["peak_window"] >= stats["window"]
    assert ctrl.backoffs == 1


def test_rewindow_clears_the_cubic_recovery_state() -> None:
    """Leftover CUBIC state let one uncongested round snap back to the previous query."""
    ctrl = AIMDFlowControl(Config())
    for _ in range(5):
        ctrl.observe(congested=False)
    ctrl.observe(congested=True)
    for _ in range(20):
        ctrl.observe(congested=False)

    ctrl.rewindow(4)
    assert ctrl.window == 4
    assert ctrl.observe(congested=False) <= 5, "a re-granted channel grows additively"


# --- admission concurrency ---------------------------------------------------


def test_an_unbalanced_release_cannot_inflate_the_pool() -> None:
    """One stray release used to free a slot forever, permanently over-admitting."""
    limiter = ConcurrencyLimiter(slots=2, cores=8)
    limiter.release()  # never acquired on this thread
    limiter.release()
    assert limiter.active == 0

    limiter.acquire()
    assert limiter.active == 1
    limiter.release()
    limiter.release()  # a second, unbalanced release
    assert limiter.active == 0


def test_a_nested_acquire_takes_no_second_slot() -> None:
    limiter = ConcurrencyLimiter(slots=1, cores=8)
    limiter.acquire()
    limiter.acquire()  # a collect() inside a UDF — must not deadlock against itself
    assert limiter.active == 1
    limiter.release()
    assert limiter.active == 1  # the outer grant still holds it
    limiter.release()
    assert limiter.active == 0


def test_a_full_queue_sheds_and_counts_it() -> None:
    limiter = ConcurrencyLimiter(slots=1, queue_depth=1, cores=8)
    limiter.acquire()  # the main thread holds the only slot
    done = threading.Event()

    def _waiter():
        try:
            limiter.acquire(timeout=10.0)
            limiter.release()
        finally:
            done.set()

    t = threading.Thread(target=_waiter, daemon=True)
    t.start()
    for _ in range(1000):  # wait until the queue really has its one occupant
        if limiter.waiting == 1:
            break
        threading.Event().wait(0.005)
    assert limiter.waiting == 1

    # A *third* arrival, on its own thread so re-entrancy does not exempt it, finds the
    # queue full and is shed rather than joining an unbounded line.
    shed: list[BaseException] = []

    def _overflow():
        try:
            limiter.acquire(timeout=1.0)
        except BaseException as exc:
            shed.append(exc)

    o = threading.Thread(target=_overflow, daemon=True)
    o.start()
    o.join(timeout=5.0)
    assert shed and isinstance(shed[0], AdmissionTimeout)
    assert "already queued" in str(shed[0])
    assert limiter.stats()["rejected"] >= 1

    limiter.release()
    t.join(timeout=10.0)
    assert done.is_set()
    assert limiter.stats()["queued"] >= 1


def test_a_grant_reports_how_long_it_queued() -> None:
    limiter = ConcurrencyLimiter(slots=4, cores=8)
    grant = limiter.acquire()
    assert grant.queued_s >= 0.0
    assert grant.concurrent == 1
    limiter.release()


def test_changing_the_queue_depth_rebuilds_the_process_limiter() -> None:
    """The `AdmissionTimeout` hint tells you to raise this; it used to be ignored."""
    reset_process_limiter()
    try:
        base = Config().replace(execution=ExecutionConfig(max_concurrent_queries=2))
        first = process_limiter(base)
        assert first._queue_depth == ExecutionConfig().admission_queue_depth

        deeper = Config().replace(
            execution=ExecutionConfig(max_concurrent_queries=2, admission_queue_depth=17)
        )
        second = process_limiter(deeper)
        assert second._queue_depth == 17
    finally:
        reset_process_limiter()


def test_the_core_count_is_probed_once() -> None:
    limiter = ConcurrencyLimiter(slots=4)
    calls = {"n": 0}
    import batcher._internal.hardware as hw

    real = hw.available_cpu_count

    def counted():
        calls["n"] += 1
        return real()

    hw.available_cpu_count = counted
    try:
        for _ in range(20):
            limiter.width_for(4)
    finally:
        hw.available_cpu_count = real
    assert calls["n"] == 1


# --- the scheduling grant ----------------------------------------------------


def _plan(*, parallelism: int, cpu_shares: float = 1.0) -> PhysicalPlan:
    op = PhysicalOp(
        op_id=OpId(1),
        kind="Aggregate",
        backend="native",
        algorithm="hash",
        bounds=ResourceBounds(
            m_max_bytes=1 << 30,
            c_max_credits=8,
            n_max_parallelism=parallelism,
            c_cpu_shares=cpu_shares,
        ),
        inputs=(),
    )
    return PhysicalPlan(ir={}, output_schema=None, ops=(op,))


def _ctx():
    from batcher.carbonite.base import ResourceContext

    return ResourceContext(config=Config())


def test_an_unsized_plans_fanout_is_held_under_a_sanity_ceiling() -> None:
    """Kyber's unknown-cardinality placeholder turns into a request for millions of tasks."""
    env = DefaultSchedulingPolicy().envelope(
        _plan(parallelism=10_000_000),
        _ctx(),
        requested_workers=None,
        available_bytes=1 << 34,
    )
    assert env.n_tasks == _MAX_TASK_FANOUT


def test_an_explicit_worker_request_is_still_honored_verbatim() -> None:
    env = DefaultSchedulingPolicy().envelope(
        _plan(parallelism=4),
        _ctx(),
        requested_workers=_MAX_TASK_FANOUT * 2,
        available_bytes=1 << 34,
    )
    assert env.n_tasks == _MAX_TASK_FANOUT * 2


def test_a_zero_cpu_share_never_reaches_ray_as_num_cpus_zero() -> None:
    """Ray reads `num_cpus=0` as "packs without limit", which is the oversubscription bug."""
    env = DefaultSchedulingPolicy().envelope(
        _plan(parallelism=4, cpu_shares=0.0),
        _ctx(),
        requested_workers=None,
        available_bytes=1 << 34,
    )
    assert env.num_cpus >= ExecutionConfig().cpu_share_min


# --- spill shape -------------------------------------------------------------


def test_partitions_track_the_requested_bucket_target() -> None:
    basis = 1 << 30  # 1 GiB
    coarse = partitions_for_volume(basis, 128 << 20)
    fine = partitions_for_volume(basis, 32 << 20)
    assert fine > coarse
    assert partitions_for_volume(0) is None


def test_the_manager_shards_under_the_grace_recursion_ceiling() -> None:
    """Sharding above it produced buckets the reduce had to split again, re-reading them."""
    from batcher.carbonite import ResourceManager
    from batcher.config import MemoryConfig, config_context

    cfg = Config().replace(memory=MemoryConfig(spill_bucket_max_bytes=16 << 20))
    with config_context(cfg):
        rm = ResourceManager()
        assert rm._spill._bucket_target_bytes() == 16 << 20
