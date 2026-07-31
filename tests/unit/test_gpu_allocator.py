"""The device allocator plan: sizing, inertness, and how it degrades with no RAPIDS.

Every test here runs on a host with no GPU, which is the point — the sizing is arithmetic over
a config and a byte count, and the only part that needs a device is the one call that installs
the resource. That call is exercised for its *failure* direction, since a worker that cannot
build a pool must keep computing on the allocator it had.
"""

from __future__ import annotations

import pytest

from batcher.carbonite.accel import (
    MIN_POOL_BYTES,
    AllocatorPlan,
    configure_device_memory,
    device_allocator_state,
    plan_allocator,
    prepare_device_memory,
    reset_device_allocator,
)
from batcher.config import DeviceMemoryConfig

pytestmark = pytest.mark.unit

GIB = 1 << 30


@pytest.fixture(autouse=True)
def _clean_allocator():
    reset_device_allocator()
    yield
    reset_device_allocator()


def test_the_shipped_config_plans_a_stream_ordered_pool():
    """The default is a pool, because the driver allocator is 3.25x slower on a real chain.

    Measured on a T4: twenty rounds of filter → project → group-by-sum over 4M rows takes
    459 ms on the driver allocator and 141 ms on `async`, and the whole gap is `cudaMalloc`
    synchronizing the device once per intermediate column. `async` rather than `pool` because
    a stream-ordered pool returns freed memory to the driver, so a co-tenant on the same
    device still sees it — which is the objection that kept this off.
    """
    plan = plan_allocator(DeviceMemoryConfig(), 40 * GIB)
    assert not plan.is_inert
    assert plan.allocator == "async"
    assert plan.initial_bytes > 0
    # ...and statistics on, which the subdivision ladder needs to divide an over-large shard
    # in one round instead of three blind halvings that each re-read it from storage. Measured
    # cost on the same benchmark: none (142.1 ms either way).
    assert plan.statistics


def test_an_explicit_default_allocator_still_plans_nothing():
    """An operator who asks for the driver allocator gets it — the change is to the default,
    not to what the knob means."""
    plan = plan_allocator(DeviceMemoryConfig(allocator="default", statistics=False), 40 * GIB)
    assert plan.is_inert
    assert plan.allocator == "default"
    assert plan.initial_bytes == 0


def test_pool_is_sized_from_reservable_bytes_not_capacity():
    cfg = DeviceMemoryConfig(allocator="pool", pool_initial_fraction=0.5, pool_max_fraction=1.0)
    # 40 GiB is what the VRAM pool says is reservable, headroom already deducted.
    plan = plan_allocator(cfg, 40 * GIB)
    assert plan.maximum_bytes == 40 * GIB
    assert plan.initial_bytes == 20 * GIB


def test_max_fraction_leaves_the_remainder_to_a_co_tenant():
    cfg = DeviceMemoryConfig(allocator="pool", pool_initial_fraction=0.25, pool_max_fraction=0.5)
    plan = plan_allocator(cfg, 80 * GIB)
    assert plan.maximum_bytes == 40 * GIB
    assert plan.initial_bytes == 20 * GIB
    assert plan.maximum_bytes < 80 * GIB


def test_initial_never_exceeds_maximum():
    cfg = DeviceMemoryConfig(allocator="pool", pool_initial_fraction=1.0, pool_max_fraction=1.0)
    plan = plan_allocator(cfg, 8 * GIB)
    assert plan.initial_bytes <= plan.maximum_bytes


@pytest.mark.parametrize("usable", [0, 1 << 20, MIN_POOL_BYTES - 1])
def test_an_unmeasured_or_full_device_gets_no_pool(usable):
    """No measurement means no reservation — a pool sized from a guess takes a tenant's memory."""
    cfg = DeviceMemoryConfig(allocator="pool")
    plan = plan_allocator(cfg, usable)
    assert plan.allocator == "default"
    assert plan.maximum_bytes == 0


def test_spill_and_statistics_survive_an_unpoolable_device():
    """A device too full to pool can still spill and still be measured."""
    cfg = DeviceMemoryConfig(allocator="pool", spill_to_host=True, statistics=True)
    plan = plan_allocator(cfg, 0)
    assert plan.allocator == "default"
    assert plan.spill_to_host and plan.statistics
    assert not plan.is_inert


def test_pool_sizes_are_aligned_down():
    """RMM reserves in granules, so a pool asks for a multiple rather than an odd byte count."""
    cfg = DeviceMemoryConfig(allocator="pool")
    plan = plan_allocator(cfg, 40 * GIB + 12345)
    assert plan.maximum_bytes % (256 << 20) == 0
    assert plan.maximum_bytes <= 40 * GIB + 12345


@pytest.mark.parametrize("allocator", ["pool", "async", "managed"])
def test_every_allocator_mode_plans_a_pool(allocator):
    plan = plan_allocator(DeviceMemoryConfig(allocator=allocator), 40 * GIB)
    assert plan.allocator == allocator
    assert plan.initial_bytes >= MIN_POOL_BYTES


def test_configuring_an_inert_plan_is_a_no_op():
    assert configure_device_memory(AllocatorPlan()) is False
    assert device_allocator_state()["allocator"] == "default"


def test_no_rapids_keeps_the_worker_on_its_own_allocator():
    """The failure direction that matters: no RMM must not stop a worker from computing."""
    plan = AllocatorPlan(allocator="pool", initial_bytes=GIB, maximum_bytes=2 * GIB)
    assert configure_device_memory(plan) is False
    assert device_allocator_state() == {
        "allocator": "default",
        "pool_bytes": 0,
        "peak_bytes": 0,
    }


def test_state_reports_no_peak_rather_than_a_guess():
    assert device_allocator_state()["peak_bytes"] == 0


def test_prepare_is_inert_on_a_host_with_no_device():
    assert prepare_device_memory() is False


def test_applying_twice_reports_the_first_answer(monkeypatch):
    """A reused worker keeps its pool: re-applying would free it under the live columns."""
    import batcher.carbonite.accel.allocator as mod

    calls = []
    monkeypatch.setattr(mod, "_apply_resource", lambda plan: calls.append(plan) or True)
    plan = AllocatorPlan(allocator="pool", initial_bytes=GIB, maximum_bytes=2 * GIB)
    assert configure_device_memory(plan) is True
    assert configure_device_memory(plan) is True
    assert len(calls) == 1
    assert device_allocator_state()["pool_bytes"] == 2 * GIB


def test_unmeasured_subdivision_keeps_the_blind_halving():
    from batcher.dist.gpu import measured_parts

    assert measured_parts() == 2
    assert measured_parts(4) == 4


@pytest.mark.parametrize(
    ("peak", "ceiling", "expected"),
    [
        (60 * GIB, 40 * GIB, 2),  # just over: halving was already right
        (300 * GIB, 40 * GIB, 8),  # far over: halving would cost three failed re-reads
        (30 * GIB, 40 * GIB, 2),  # under the ceiling: the overflow was not this shard's size
        (10_000 * GIB, 40 * GIB, 16),  # a runaway measurement is capped, not obeyed
    ],
)
def test_a_measured_peak_picks_the_factor_that_clears_it(monkeypatch, peak, ceiling, expected):
    from batcher.dist.gpu import measured_parts

    # `measured_parts` resolves the reader from the package, which is the name it imports.
    monkeypatch.setattr(
        "batcher.carbonite.accel.device_allocator_state",
        lambda: {"allocator": "pool", "pool_bytes": ceiling, "peak_bytes": peak},
    )
    assert measured_parts() == expected
