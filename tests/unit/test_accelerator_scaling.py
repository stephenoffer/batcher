"""Adding accelerators must add throughput: the pool is sized by devices, never by the data.

The failure this guards is silent. A pool clamped to the input's partition count runs the
same number of actors on a 4-GPU cluster and a 64-GPU one, so a fleet that doubled changes
nothing and the only symptom is a bill. Both execution paths size against devices; these
assert that, and assert the linearity, with a stubbed Ray so no cluster is needed.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def cluster(monkeypatch):
    """A stub `ray.cluster_resources()` whose device counts a test sets.

    The modules under test are imported *first*, with the real Ray still in place: they
    decorate actor classes with `@ray.remote` at import time, so a stub installed beforehand
    would fail the import rather than the lookup. Only the resource reading is faked.
    """
    pytest.importorskip("ray", reason="ray not installed")
    import batcher.dist.executors.map
    import batcher.dist.streaming.consumers

    assert batcher.dist.executors.map and batcher.dist.streaming.consumers  # imported, not used

    resources: dict[str, float] = {}
    stub = types.ModuleType("ray")
    stub.cluster_resources = lambda: dict(resources)
    monkeypatch.setitem(sys.modules, "ray", stub)
    return resources


@dataclass
class _Stage:
    num_gpus: float = 1.0
    concurrency: object = None
    accelerator_type: str | None = None
    resources: tuple = ()


def _pool(devices: float, partitions: int, *, per_actor: float = 1.0, resources=None) -> int:
    from batcher.dist.streaming.consumers import consumer_pool_size

    stage = _Stage(num_gpus=per_actor, resources=resources or ())
    return consumer_pool_size(stage, workers=4, num_partitions=partitions)


def test_the_gpu_pool_grows_linearly_with_the_cluster(cluster):
    from batcher.ml.gpu import gpu_aware_pool_default

    sizes = []
    for devices in (1, 2, 4, 8, 16, 64):
        cluster["GPU"] = float(devices)
        sizes.append(gpu_aware_pool_default(1.0, 4, 1 << 30))
    assert sizes == [1, 2, 4, 8, 16, 64]


def test_fractional_requests_pack_linearly_too(cluster):
    from batcher.ml.gpu import gpu_aware_pool_default

    cluster["GPU"] = 8.0
    assert gpu_aware_pool_default(0.5, 4, 1 << 30) == 16
    assert gpu_aware_pool_default(0.25, 4, 1 << 30) == 32


def test_a_custom_accelerator_pool_grows_linearly(cluster):
    # A TPU pod, Trainium fleet or Gaudi node is a named Ray resource rather than `GPU`, and
    # the same "one actor per chip" rule has to hold or the pod runs one actor.
    from batcher.ml.gpu import gpu_aware_pool_default

    sizes = []
    for chips in (4, 8, 32):
        cluster["TPU"] = float(chips)
        sizes.append(gpu_aware_pool_default(0.0, 4, 1 << 30, resources={"TPU": 4.0}))
    assert sizes == [1, 2, 8]


def test_the_batch_path_raises_the_partition_count_to_match_the_devices(cluster):
    # The partition count is sized from data; the device count is not. Sizing the pool by
    # devices and then clamping it to partitions lets the data decide how many accelerators
    # are allowed to work.
    from batcher.dist.executors.map import _pool_partition_count

    cluster["GPU"] = 12.0
    assert _pool_partition_count(4, 1.0, None, None, None) == 12


def test_the_streaming_consumer_pool_is_not_capped_by_the_sources_partitions(cluster):
    # A consumer reads no partition: morsels arrive over Flight and `take_consumer` hands each
    # to whichever consumer is free. A four-partition corpus used to leave eight of twelve GPUs
    # with no actor at all.
    cluster["GPU"] = 12.0
    assert _pool(12.0, partitions=4) == 12


def test_the_streaming_consumer_pool_grows_linearly_with_devices(cluster):
    sizes = []
    for devices in (2, 4, 8, 16):
        cluster["GPU"] = float(devices)
        sizes.append(_pool(devices, partitions=4))
    assert sizes == [2, 4, 8, 16]


def test_an_explicit_concurrency_is_still_honored_exactly(cluster):
    from batcher.dist.streaming.consumers import consumer_pool_size

    cluster["GPU"] = 12.0
    assert consumer_pool_size(_Stage(concurrency=3), workers=4, num_partitions=4) == 3


def test_a_cpu_stage_keeps_the_worker_count(cluster):
    from batcher.dist.streaming.consumers import consumer_pool_size

    cluster["GPU"] = 12.0
    assert consumer_pool_size(_Stage(num_gpus=0.0), workers=4, num_partitions=64) == 4


def test_a_pinned_accelerator_class_sizes_against_that_class_only(cluster, monkeypatch):
    # A stage pinned to the 4 A100s must never spawn actors for the 8 T4s it cannot run on.
    from batcher.ml.gpu import gpu_aware_pool_default

    cluster["GPU"] = 12.0
    monkeypatch.setattr(
        "batcher.dist.executors.ray_runtime.fabric.topology.devices_of_class",
        lambda name: 4,
        raising=False,
    )
    assert gpu_aware_pool_default(1.0, 4, 1 << 30, "NVIDIA_A100") == 4


class TestUtilizationTarget:
    """The packing loop must converge to a fed device and then stop moving."""

    @staticmethod
    def _next(util, actors, requested=1.0, max_actors=None):
        from batcher.ml.gpu import recommend_num_gpus

        fraction = recommend_num_gpus(util, requested, actors, max_actors)
        return round(1.0 / fraction) if fraction > 0 else 0

    def test_a_starved_device_gets_more_actors(self):
        # One actor holding a device at 30% is leaving two thirds of it idle.
        assert self._next(0.30, actors=1) >= 3

    def test_the_loop_stops_once_the_device_is_fed(self):
        # At or above the satisfied band the loop holds: every change is a pool rebuild, and
        # a measured step past a fed device came out slower and less evenly spread.
        for util in (0.80, 0.85, 0.92, 0.99):
            assert self._next(util, actors=2) == 2

    def test_the_target_is_eighty_percent(self):
        from batcher.ml.gpu import _PACK_SATISFIED

        assert _PACK_SATISFIED == 0.8

    def test_the_loop_is_a_fixed_point_at_the_target(self):
        # A configuration that lands exactly on the target must not oscillate.
        for actors in (1, 2, 4):
            assert self._next(0.80, actors=actors) == actors

    def test_vram_bounds_the_packing_even_when_utilization_asks_for_more(self):
        # The utilization target must never win over what memory allows, or the loop packs a
        # fleet into an out-of-memory.
        assert self._next(0.20, actors=1, max_actors=2) == 2

    def test_an_unmeasured_stage_keeps_what_it_declared(self):
        from batcher.ml.gpu import recommend_num_gpus

        assert recommend_num_gpus(None, 1.0, 1) == 1.0
