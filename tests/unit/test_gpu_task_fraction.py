"""What a relational GPU task asks Ray for, once the shards are priced.

The fan-out cuts four times as many shards as there are devices on purpose, and then used to
ask for a whole device per shard — so Ray ran one per device and queued the rest, at full
reported utilization. These cover the decision that closes that gap: that a small shard is
granted a share, that a large one is not, that a broadcast join's replicated build side is
charged to every co-tenant rather than once, and that every unknown resolves to a whole device.

No Ray and no device: descriptors are dicts and the options are inspected directly, which is
also the only way to check the `num_gpus` that would have been requested.
"""

from __future__ import annotations

import dataclasses

import pyarrow as pa
import pytest

from batcher._internal.device_share import MAX_COTENANTS
from batcher.config import active_config, set_config
from batcher.dist.gpu.resources import (
    binding_device_bytes,
    descriptor_bytes,
    fleet_derate,
    gpu_shard_options,
    largest_shard_bytes,
    shard_task_share,
    share_for_bytes,
)

pytestmark = pytest.mark.unit

GIB = 1 << 30
SCHEMA = pa.schema([("a", pa.int64()), ("b", pa.float64())])


@pytest.fixture
def packing_config():
    """Set the packing knobs against a known 80 GB device, then restore what was there."""
    saved = active_config()

    def _apply(**kw):
        kw.setdefault("gpu_memory_gb", 80.0)
        kw.setdefault("gpu_pack_shards", True)
        kw.setdefault("gpu_task_fraction", 0.0)
        kw.setdefault("gpu_max_tasks_per_device", 4)
        kw.setdefault("gpu_shard_expansion", 2.0)
        cfg = active_config()
        set_config(cfg.replace(distributed=dataclasses.replace(cfg.distributed, **kw)))

    yield _apply
    set_config(saved)


def _batch_descriptor(rows: int) -> dict:
    return {
        "batches": [
            pa.RecordBatch.from_pydict({"a": list(range(rows)), "b": [1.0] * rows}, schema=SCHEMA)
        ]
    }


def test_an_in_memory_shard_is_measured_rather_than_estimated() -> None:
    desc = _batch_descriptor(1000)
    assert descriptor_bytes(desc, row_bytes=1.0) == desc["batches"][0].nbytes


def test_a_split_manifest_is_priced_from_its_captured_row_count() -> None:
    class _Split:
        rows = 1_000_000

    assert descriptor_bytes({"splits": [_Split(), _Split()]}, row_bytes=16) == 32_000_000


def test_an_unpriceable_shard_reports_zero_rather_than_a_guess() -> None:
    assert descriptor_bytes({}, row_bytes=16) == 0
    assert largest_shard_bytes([], row_bytes=16) == 0


def test_the_largest_shard_sets_the_share_not_the_average() -> None:
    # One fraction is chosen for the whole fan-out. Sizing it to the mean guarantees the shard
    # that most needed room is the one that does not get it.
    descs = [_batch_descriptor(10), _batch_descriptor(10), _batch_descriptor(10_000)]
    assert largest_shard_bytes(descs, 1.0) == descriptor_bytes(descs[-1], 1.0)


def test_a_small_shard_is_granted_a_share_of_a_device(packing_config) -> None:
    packing_config()
    packing = share_for_bytes(2 * GIB, concurrency=8, gpu_count=2)
    assert packing.fraction < 1.0
    assert packing.per_device > 1
    assert packing.devices <= 2


def test_a_shard_that_needs_the_whole_device_gets_it(packing_config) -> None:
    packing_config()
    packing = share_for_bytes(60 * GIB, concurrency=8, gpu_count=2)
    assert packing.fraction == 1.0
    assert packing.per_device == 1


def test_the_expansion_factor_charges_the_intermediate_the_operator_builds(
    packing_config,
) -> None:
    # A shard at exactly a quarter of the device does not get a quarter: the hash table it
    # builds is resident at the same moment as the input it built it from.
    packing_config(gpu_shard_expansion=1.0)
    loose = share_for_bytes(16 * GIB, concurrency=8, gpu_count=2)
    packing_config(gpu_shard_expansion=4.0)
    tight = share_for_bytes(16 * GIB, concurrency=8, gpu_count=2)
    assert tight.fraction > loose.fraction


def test_turning_packing_off_restores_the_whole_device_behavior(packing_config) -> None:
    packing_config(gpu_pack_shards=False)
    packing = share_for_bytes(1 * GIB, concurrency=8, gpu_count=2)
    assert packing.fraction == 1.0
    assert packing.per_device == 1
    assert "gpu_pack_shards is off" in packing.reason


def test_a_pinned_fraction_is_applied_as_given(packing_config) -> None:
    packing_config(gpu_task_fraction=0.5)
    packing = share_for_bytes(60 * GIB, concurrency=8, gpu_count=2)
    assert packing.fraction == 0.5, "an operator statement outranks the estimate"
    assert "pinned" in packing.reason


def test_the_per_device_ceiling_floors_the_share(packing_config) -> None:
    packing_config(gpu_max_tasks_per_device=2)
    packing = share_for_bytes(1 * GIB, concurrency=8, gpu_count=2)
    assert packing.per_device <= 2
    assert packing.fraction >= 0.5

    packing_config(gpu_max_tasks_per_device=1)
    assert share_for_bytes(1 * GIB, concurrency=8, gpu_count=2).fraction == 1.0


def test_an_unmeasurable_shard_asks_for_a_whole_device(packing_config) -> None:
    packing_config()
    assert share_for_bytes(0, concurrency=8, gpu_count=2).fraction == 1.0
    assert shard_task_share([_batch_descriptor(10)], None, gpu_count=2).fraction == 1.0


def test_device_demand_is_bounded_by_the_fleet_not_by_the_shard_count(
    packing_config,
) -> None:
    # A thousand shards over two devices still occupies two devices; the surplus queues, as it
    # always did. Reporting a thousand would ask an autoscaler for a cluster nobody needs.
    packing_config()
    packing = share_for_bytes(1 * GIB, concurrency=1000, gpu_count=2)
    assert packing.devices <= 2


def test_a_replicated_build_side_is_charged_to_every_cotenant(packing_config) -> None:
    # Four tasks on one device hold four copies of a broadcast join's build side, not one
    # between them. Charging it once is how packing OOMs the one shape it was safest on.
    packing_config()
    alone = share_for_bytes(1 * GIB, concurrency=8, gpu_count=2)
    with_build = share_for_bytes(1 * GIB, concurrency=8, gpu_count=2, resident_bytes=20 * GIB)
    assert with_build.fraction > alone.fraction


def test_the_options_carry_the_decided_fraction_into_ray(packing_config) -> None:
    packing_config()
    descs = [_batch_descriptor(64) for _ in range(8)]
    opts, packing = gpu_shard_options(descs, SCHEMA, gpu_count=2)
    assert opts["num_gpus"] == packing.fraction
    assert "max_retries" in opts


def test_a_gpu_task_never_leaves_ray_with_a_zero_device_request() -> None:
    # Ray reads `num_gpus=0` as a CPU task, which would schedule a cuDF kernel onto a node
    # with no device — a failure far from the mis-derived share that caused it.
    from batcher.dist.gpu.tasks import gpu_task_options

    assert gpu_task_options(num_gpus=0.0)["num_gpus"] == 1.0
    assert gpu_task_options(num_gpus=-1.0)["num_gpus"] == 1.0
    assert gpu_task_options()["num_gpus"] == 1.0
    assert gpu_task_options(num_gpus=0.25)["num_gpus"] == 0.25


def test_the_share_is_sized_against_the_cluster_s_device_not_the_driver_s(
    packing_config, monkeypatch
) -> None:
    """The whole feature turns on this number.

    A fan-out is scheduled from a head node that usually has no GPU, where the local probe
    returns a 12 GB T4 constant. Packing an 80 GB fleet against 12 GB grants every shard a
    whole device it needed an eighth of — silently the behavior the packing replaced.
    """
    packing_config(gpu_memory_gb=12.0)
    monkeypatch.setattr(
        "batcher.dist.executors.ray_runtime.accelerators.cluster_gpu_memory_gb", lambda: 80.0
    )
    assert binding_device_bytes() == pytest.approx(80.0 * 1e9)
    fleet = share_for_bytes(4 * GIB, concurrency=8, gpu_count=2)

    monkeypatch.setattr(
        "batcher.dist.executors.ray_runtime.accelerators.cluster_gpu_memory_gb", lambda: None
    )
    assert binding_device_bytes() == pytest.approx(12.0 * 1e9)
    driver_only = share_for_bytes(4 * GIB, concurrency=8, gpu_count=2)

    assert fleet.per_device > driver_only.per_device
    assert fleet.fraction < driver_only.fraction


def test_an_unreadable_cluster_falls_back_to_the_configured_device(
    packing_config, monkeypatch
) -> None:
    packing_config(gpu_memory_gb=40.0)
    monkeypatch.setattr(
        "batcher.dist.executors.ray_runtime.accelerators.cluster_gpu_memory_gb", lambda: None
    )
    assert binding_device_bytes() == pytest.approx(40.0 * 1e9)


def _health(devices: int, quarantined: int = 0, degraded: int = 0) -> tuple[dict, ...]:
    return (
        {
            "devices": devices,
            "quarantined": list(range(quarantined)),
            "degraded": list(range(100, 100 + degraded)),
        },
    )


def test_a_healthy_or_unprobed_fleet_packs_exactly_as_before(monkeypatch) -> None:
    monkeypatch.setattr(
        "batcher.dist.executors.ray_runtime.hardware_probe.cluster_device_health",
        lambda: _health(8),
    )
    assert fleet_derate() == 1.0
    monkeypatch.setattr(
        "batcher.dist.executors.ray_runtime.hardware_probe.cluster_device_health", lambda: ()
    )
    assert fleet_derate() == 1.0, "no telemetry is not a reason to stop packing"


def test_one_sick_device_in_a_large_fleet_barely_moves_the_cotenancy(monkeypatch) -> None:
    # A large fleet always has a sick device somewhere. Disabling packing fleet-wide for one of
    # five hundred would make the feature evaporate on exactly the clusters it exists for.
    monkeypatch.setattr(
        "batcher.dist.executors.ray_runtime.hardware_probe.cluster_device_health",
        lambda: _health(500, degraded=1),
    )
    assert fleet_derate() == pytest.approx(499 / 500)


def test_a_widely_degraded_fleet_packs_less(monkeypatch) -> None:
    monkeypatch.setattr(
        "batcher.dist.executors.ray_runtime.hardware_probe.cluster_device_health",
        lambda: _health(8, degraded=4),
    )
    assert fleet_derate() == pytest.approx(0.5)


def test_quarantined_devices_count_on_neither_side(monkeypatch) -> None:
    # A fleet is not penalized for correctly taking a broken board out of rotation: a shard
    # cannot land on a quarantined device, so it is not a device the share must be safe on.
    monkeypatch.setattr(
        "batcher.dist.executors.ray_runtime.hardware_probe.cluster_device_health",
        lambda: _health(8, quarantined=4),
    )
    assert fleet_derate() == 1.0


def test_the_derate_never_falls_low_enough_to_refuse_placement(monkeypatch) -> None:
    """The worst this can do is stop packing; it must never report a fleet as unplaceable."""
    monkeypatch.setattr(
        "batcher.dist.executors.ray_runtime.hardware_probe.cluster_device_health",
        lambda: _health(8, degraded=8),
    )
    assert fleet_derate() == pytest.approx(1.0 / MAX_COTENANTS)


def test_a_failing_health_probe_does_not_fail_the_fan_out(monkeypatch) -> None:
    def _boom():
        raise RuntimeError("ray is down")

    monkeypatch.setattr(
        "batcher.dist.executors.ray_runtime.hardware_probe.cluster_device_health", _boom
    )
    assert fleet_derate() == 1.0
