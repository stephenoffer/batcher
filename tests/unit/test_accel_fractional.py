"""Packing several claimants onto one device: the decision, not the arithmetic.

`test_device_share` covers the rounding. What is tested here is everything the rounding cannot
see — a resident co-tenant, a derated device, an available hardware partition — and the
standing rule that every unknown resolves to one claimant per device, which is what the fleet
did before fractional packing existed. A test that only checked the happy path would pass while
the module quietly packed four tasks onto a quarantined device.
"""

from __future__ import annotations

import pytest

from batcher.carbonite.accel.fractional import (
    TaskPacking,
    derated_cotenants,
    external_headroom_bytes,
    packing_summary,
    plan_task_packing,
    shard_fraction,
    whole_device_packing,
)

pytestmark = pytest.mark.unit

GIB = 1 << 30


def test_the_refusal_answer_is_a_whole_device_never_a_none() -> None:
    plan = whole_device_packing(4, "because")
    assert plan.fraction == 1.0
    assert plan.per_device == 1
    assert plan.devices == 4
    assert not plan.packed
    assert plan.reason == "because"


def test_external_residency_is_subtracted_before_anything_is_granted() -> None:
    # 80 GiB device, 15% headroom -> 68 usable; an actor holding 40 leaves 28.
    left = external_headroom_bytes(80 * GIB, 40 * GIB, 0.15)
    assert left == int(80 * GIB * 0.85) - 40 * GIB
    assert external_headroom_bytes(80 * GIB, 80 * GIB, 0.15) == 0, "a full device offers nothing"
    assert external_headroom_bytes(80 * GIB, 0, 0.15) > 0, "unmeasured reads as empty"


def test_a_derate_reduces_the_cotenancy_and_a_quarantine_empties_it() -> None:
    assert derated_cotenants(4, 1.0) == 4
    assert derated_cotenants(4, 0.5) == 2
    assert derated_cotenants(4, 0.1) == 1, "a schedulable device still takes one claimant"
    assert derated_cotenants(4, 0.0) == 0, "a quarantined device takes none"
    assert derated_cotenants(1, 0.5) == 1


def test_a_quarantined_device_is_planned_for_no_devices_at_all() -> None:
    plan = plan_task_packing(3 * GIB, device_bytes=80 * GIB, concurrency=8, derate=0.0)
    assert plan.devices == 0
    assert "quarantin" in plan.reason


def test_a_small_claimant_on_a_large_device_is_packed() -> None:
    plan = plan_task_packing(3 * GIB, device_bytes=80 * GIB, concurrency=8, prefer_isolation=False)
    assert plan.packed
    assert plan.fraction <= 0.25
    assert plan.per_device >= 4
    assert plan.devices == 2, "eight quarter-device claimants occupy two devices"
    assert plan.share_bytes_ >= 3 * GIB


def test_packing_never_exceeds_the_concurrency_actually_requested() -> None:
    # A device that could hold four claimants must not be reported as holding four when the
    # stage only ever runs two: the device count would then be a fiction the caller schedules
    # against.
    plan = plan_task_packing(1 * GIB, device_bytes=80 * GIB, concurrency=2, prefer_isolation=False)
    assert plan.per_device <= 2
    assert plan.devices == 1


def test_a_resident_cotenant_shrinks_the_share_that_is_granted() -> None:
    empty = plan_task_packing(
        10 * GIB, device_bytes=80 * GIB, concurrency=8, prefer_isolation=False
    )
    busy = plan_task_packing(
        10 * GIB,
        device_bytes=80 * GIB,
        concurrency=8,
        used_bytes=50 * GIB,
        prefer_isolation=False,
    )
    assert busy.per_device < empty.per_device
    assert busy.devices > empty.devices


def test_a_full_device_declines_to_pack_rather_than_dividing_nothing() -> None:
    plan = plan_task_packing(3 * GIB, device_bytes=80 * GIB, concurrency=4, used_bytes=80 * GIB)
    assert plan.fraction == 1.0
    assert "fully resident" in plan.reason


def test_a_claimant_larger_than_a_device_takes_whole_devices() -> None:
    plan = plan_task_packing(
        200 * GIB, device_bytes=80 * GIB, concurrency=2, prefer_isolation=False
    )
    assert plan.fraction >= 1.0
    assert plan.per_device == 1
    assert plan.devices >= 2


def test_an_unknown_device_or_need_keeps_the_unpacked_behavior() -> None:
    assert plan_task_packing(3 * GIB, device_bytes=0, concurrency=4).fraction == 1.0
    assert plan_task_packing(0, device_bytes=80 * GIB, concurrency=4).fraction == 1.0
    assert plan_task_packing(0, device_bytes=80 * GIB, concurrency=4).devices == 4


def test_a_partitionable_part_is_isolated_rather_than_merely_shared() -> None:
    # The point of preferring MIG is not density — both schedule the same worker count — it is
    # that a co-tenant's allocation spike stops being this worker's OOM.
    plan = plan_task_packing(
        3 * GIB,
        device_bytes=80 * GIB,
        concurrency=7,
        accelerator_type="NVIDIA_H100",
        prefer_isolation=True,
    )
    assert plan.isolated
    assert plan.per_device > 1
    shared = plan_task_packing(
        3 * GIB,
        device_bytes=80 * GIB,
        concurrency=7,
        accelerator_type="NVIDIA_H100",
        prefer_isolation=False,
    )
    assert not shared.isolated


def test_an_unlabelled_fleet_falls_through_to_the_quanta_rather_than_refusing() -> None:
    plan = plan_task_packing(
        3 * GIB, device_bytes=80 * GIB, concurrency=8, accelerator_type="", prefer_isolation=True
    )
    assert not plan.isolated
    assert plan.packed, "no partition table is a reason to share, not a reason to stop packing"


def test_a_shard_that_needs_a_whole_device_asks_for_one() -> None:
    assert shard_fraction(70 * GIB, 80 * GIB) == 1.0
    assert shard_fraction(200 * GIB, 80 * GIB) == 1.0


def test_a_shard_task_never_asks_for_zero_gpus() -> None:
    # A `num_gpus=0` GPU task is a GPU task scheduled onto a CPU, which fails far from here.
    assert shard_fraction(0, 80 * GIB) == 1.0
    assert shard_fraction(3 * GIB, 0) == 1.0


def test_a_small_shard_is_packed_but_not_below_the_caller_s_ceiling() -> None:
    assert shard_fraction(3 * GIB, 80 * GIB) == 0.25
    assert shard_fraction(3 * GIB, 80 * GIB, max_per_device=2) == 0.5
    assert shard_fraction(3 * GIB, 80 * GIB, max_per_device=1) == 1.0


def test_the_summary_reports_total_device_demand_not_any_one_fraction() -> None:
    packings = [
        TaskPacking(fraction=0.25, per_device=4, devices=2),
        TaskPacking(fraction=1.0, per_device=1, devices=3),
        TaskPacking(fraction=0.5, per_device=2, devices=1, isolated=True),
    ]
    summary = packing_summary(packings)
    assert summary["stages"] == 3
    assert summary["devices"] == 6
    assert summary["packed_stages"] == 2
    assert summary["isolated"] == 1
    assert summary["min_fraction"] == 0.25


def test_the_summary_of_nothing_is_zeroed_rather_than_absent() -> None:
    assert packing_summary(()) == {
        "stages": 0,
        "devices": 0,
        "packed_stages": 0,
        "isolated": 0,
        "min_fraction": 0.0,
    }


def test_the_packing_serializes_to_a_flat_event_payload() -> None:
    payload = TaskPacking(fraction=0.25, per_device=4, devices=2, share_bytes_=17).as_dict()
    assert payload["share_bytes"] == 17
    assert payload["fraction"] == 0.25
    assert set(payload) == {
        "fraction",
        "per_device",
        "devices",
        "share_bytes",
        "isolated",
        "reason",
    }
