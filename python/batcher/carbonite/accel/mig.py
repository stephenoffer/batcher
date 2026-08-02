"""Choosing a MIG partitioning — Carbonite turning device profiles into a resource plan.

`_internal.hardware.mig` says which partitionings a device offers. This decides which one a
stage should ask for and how many devices that needs, which is a resource decision and so
belongs here rather than beside the hardware table.

A three-gigabyte embedding model on an eighty-gigabyte device uses four percent of the memory
and the whole schedulable unit. Fractional `num_gpus` fixes the scheduling half but not the
isolation half: co-tenants share one memory space and one scheduler, so one task's allocation
spike is every other task's OOM. A partition gives each worker its own memory and its own fault
domain, which is why it is preferred whenever a model fits one.
"""

from __future__ import annotations

from dataclasses import dataclass

from batcher._internal.hardware.mig import (
    MigProfile,
    mig_profiles,
    mig_supported,
    smallest_profile_for,
)

__all__ = [
    "MigPlan",
    "MigProfile",
    "mig_plan",
    "mig_profiles",
    "mig_supported",
    "smallest_profile_for",
]


@dataclass(frozen=True, slots=True)
class MigPlan:
    """How a stage should be laid out across partitioned devices.

    Attributes:
        profile: The chosen profile, or `None` when the stage should hold whole devices.
        instances_per_device: Concurrent instances one device yields under this plan.
        devices_needed: Devices the requested concurrency needs.
        gpu_fraction: The fractional `num_gpus` a scheduler should request per worker.
        reason: One line explaining the choice, for the decision log.
    """

    profile: MigProfile | None
    instances_per_device: int
    devices_needed: int
    gpu_fraction: float
    reason: str


def mig_plan(
    model_gib: float,
    accelerator_type: str | None,
    concurrency: int = 1,
    *,
    headroom: float | None = None,
) -> MigPlan:
    """Lay a stage's requested concurrency out across whole or partitioned devices.

    The decision this exists for: a fleet running twenty small inference stages on whole H100s
    is running at a fraction of its capacity while its power budget is fully committed, and the
    fix is to co-locate them. Partitioning is preferred whenever a model fits an instance,
    because a MIG instance gives isolation that fractional scheduling does not.

    Args:
        model_gib: The model's resident footprint per worker.
        accelerator_type: A Ray accelerator-type name.
        concurrency: Concurrent workers the stage wants.
        headroom: Fraction of an instance's memory left free, or `None` for the configured
            `accelerator.vram_headroom`. A literal default here was one of the private copies
            of that knob, so a fleet that raised it still chose partition profiles against the
            memory the knob had just reserved — and a MIG instance sized that way is the one
            share on a device that cannot borrow from its neighbour when it turns out short.

    Returns:
        The plan; `profile` is `None` when whole devices are the right answer.
    """
    from batcher._internal.device_share import device_headroom

    want = max(1, concurrency)
    room = device_headroom() if headroom is None else headroom
    profile = smallest_profile_for(model_gib, accelerator_type, headroom=room)
    if profile is None:
        return MigPlan(
            profile=None,
            instances_per_device=1,
            devices_needed=want,
            gpu_fraction=1.0,
            reason=(
                "whole devices: model does not fit a partition"
                if mig_supported(accelerator_type)
                else "whole devices: device model is not partitionable"
            ),
        )
    per_device = profile.instances
    devices = -(-want // per_device)  # ceiling division
    return MigPlan(
        profile=profile,
        instances_per_device=per_device,
        devices_needed=devices,
        gpu_fraction=profile.gpu_fraction,
        reason=(
            f"{profile.name}: {per_device} isolated instances per device, "
            f"{want} workers on {devices} device(s) instead of {want}"
        ),
    )
