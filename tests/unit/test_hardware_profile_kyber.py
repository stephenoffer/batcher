"""Kyber plans against detected hardware, and a cluster profile overrides the driver's.

The optimizer used to plan against fixed constants tuned on one machine — a 4 MiB broadcast
threshold whatever the cache, a plan identical on a laptop and a 128-core server. These pin
the pieces that make it hardware-adaptive: the neutral `HardwareProfile`, the L3-sized
broadcast threshold, and the plan cache distinguishing hardware so a driver's plan is not
replayed on differently-sized workers.
"""

from __future__ import annotations

import dataclasses

import pytest

from batcher.config import active_config
from batcher.plan.resource import HardwareProfile

pytestmark = pytest.mark.unit


def test_local_profile_detects_this_machine():
    hw = HardwareProfile.local()
    assert hw.cpu_cores >= 1
    assert hw.worker_count == 1
    # memory and L3 are >= 0; on Linux CI they are positive, but the contract only promises
    # "0 means unknown", so a portable assertion checks the type and non-negativity.
    assert hw.memory_bytes >= 0 and hw.l3_cache_bytes >= 0


def test_cluster_profile_binds_to_the_weakest_worker():
    # Sizing to the largest node would OOM/overshoot every smaller one; the profile takes the
    # binding worker so a plan is valid on every node it may land on.
    hw = HardwareProfile.for_cluster(
        cpu_cores=32, memory_bytes=128 << 30, worker_count=8, gpu_count=4, gpu_memory_bytes=16 << 30
    )
    assert (hw.cpu_cores, hw.worker_count, hw.gpu_memory_bytes) == (32, 8, 16 << 30)
    # worker_count floors at 1 so a per-worker division never hits zero.
    degenerate = HardwareProfile.for_cluster(cpu_cores=0, memory_bytes=0, worker_count=0)
    assert degenerate.worker_count == 1


def test_broadcast_threshold_scales_with_the_cache():
    o = active_config().optimizer
    mib = 1 << 20
    # The default is "auto"; a bigger cache admits a bigger broadcast, a smaller cache a smaller.
    assert o.resolved_broadcast_max_bytes(64 * mib) > o.resolved_broadcast_max_bytes(16 * mib)
    assert o.resolved_broadcast_max_bytes(1 * mib) < o.resolved_broadcast_max_bytes(16 * mib)
    # A 16 MiB L3 reproduces the historical 4 MiB, so a machine of that class is unchanged.
    assert o.resolved_broadcast_max_bytes(16 * mib) == 4 * mib
    # Unknown cache falls back to the historical default rather than to zero.
    assert o.resolved_broadcast_max_bytes(0) == 4 * mib
    # A pinned value always wins over detection.
    pinned = dataclasses.replace(o, broadcast_max_bytes=10 * mib)
    assert pinned.resolved_broadcast_max_bytes(1 * mib) == 10 * mib


def test_plan_cache_key_distinguishes_hardware():
    from batcher.kyber import plan_cache

    cfg = active_config()
    big_l3 = HardwareProfile(l3_cache_bytes=64 << 20)
    small_l3 = HardwareProfile(l3_cache_bytes=4 << 20)
    k_big = plan_cache.cache_key("P", None, cfg, None, hardware=big_l3)
    k_small = plan_cache.cache_key("P", None, cfg, None, hardware=small_l3)
    assert k_big != k_small, "a plan cached for one cache size must not serve another"
    # No hardware supplied → a stable, reproducible key (the single-node default path).
    assert plan_cache.cache_key("P", None, cfg, None) == plan_cache.cache_key("P", None, cfg, None)


def test_optimizer_context_defaults_to_the_local_machine():
    # A context built without an explicit profile still plans against real hardware, not zeros —
    # so a test or a single-node run is hardware-adaptive without any wiring.
    from batcher.kyber.cardinality import CardinalityEstimator
    from batcher.kyber.pass_base import OptimizerContext

    ctx = OptimizerContext(
        config=active_config(), sources=[], hub=None, estimator=CardinalityEstimator([])
    )
    assert ctx.hardware.cpu_cores >= 1
