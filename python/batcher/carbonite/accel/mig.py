"""Multi-Instance GPU: cutting one device into several, so a small model stops holding a big one.

A 3 GiB embedding model on an 80 GiB H100 uses 4% of the device and 100% of the schedulable
unit. Ray's fractional `num_gpus` fixes the *scheduling* half — several tasks can share a
device — but not the isolation half: they share one memory space and one scheduler, so one
task's allocation spike is every co-tenant's OOM and one task's kernel is every co-tenant's
latency. MIG is the hardware answer, partitioning a device into instances with their own SMs,
their own memory, and their own fault domain.

This module is the planning side: which partitioning a model should ask for, and how many
instances that yields. Creating instances is a privileged driver operation that a datacenter
performs at provisioning time, so nothing here reconfigures a device — the plan is what a
scheduler requests and what an operator provisions against.

**The profile family is derived, not tabulated.** NVIDIA's naming (`1g.10gb`, `2g.20gb`,
`3g.40gb`, `7g.80gb`) follows one rule on every MIG-capable part: the device has `S` compute
slices (seven on the A100/H100 class, four on the A30) and its memory divides into the next
power of two at or above `S`; a profile takes `k` compute slices and gets the next power of two
at or above `k` of those memory units. That reproduces the published A100-40, A100-80, H100-80,
and A30 tables exactly, and it extends to a future part without waiting for a table update —
which is the failure mode a hardcoded list has, and it fails by silently refusing to partition
new hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "MigPlan",
    "MigProfile",
    "mig_plan",
    "mig_profiles",
    "mig_supported",
    "smallest_profile_for",
]

#: Compute-slice counts a profile may take, per device slice count. These are the published
#: families, and they are listed rather than derived because the seven-slice parts offer a `3g`
#: (7 packs as 3+4) and a `4g` while the four-slice A30 offers neither — a rule that produces
#: both from arithmetic alone would be a rule invented to fit two data points.
_PROFILE_SLICES: dict[int, tuple[int, ...]] = {
    7: (1, 2, 3, 4, 7),  # A100, H100, H200, B200, GB200
    4: (1, 2, 4),  # A30
}


@dataclass(frozen=True, slots=True)
class MigProfile:
    """One MIG partitioning of a device.

    Attributes:
        name: NVIDIA profile name, `"{compute}g.{memory}gb"`.
        compute_slices: Compute slices the instance holds, out of seven.
        memory_gib: Device memory the instance holds.
        instances: Instances of this profile one device yields.
        device: The accelerator model this profile was derived for.
    """

    name: str
    compute_slices: int
    memory_gib: int
    instances: int
    device: str

    #: Compute slices the whole device has, so the instance's share is `compute_slices/total`.
    device_slices: int = 7

    @property
    def gpu_fraction(self) -> float:
        """The instance's share of the device, as the fractional `num_gpus` a scheduler asks for."""
        return self.compute_slices / max(1, self.device_slices)


def mig_supported(accelerator_type: str | None) -> bool:
    """Whether a device model can be partitioned at all.

    Args:
        accelerator_type: A Ray accelerator-type name.

    Returns:
        True for a MIG-capable model; False for PCIe inference parts, pre-Ampere devices, and
        every unrecognized name.
    """
    from batcher._internal.device_specs import device_mig_slices

    return device_mig_slices(accelerator_type) > 0


def mig_profiles(accelerator_type: str | None) -> tuple[MigProfile, ...]:
    """Every partitioning a device model supports, smallest instance first.

    Args:
        accelerator_type: A Ray accelerator-type name.

    Returns:
        The profile family, or an empty tuple when the device cannot be partitioned.
    """
    from batcher._internal.device_specs import device_spec

    spec = device_spec(accelerator_type)
    if spec is None or spec.mig_slices <= 0:
        return ()
    slices = spec.mig_slices
    memory_slices = 1 << (slices - 1).bit_length()  # next power of two >= slices
    # A slice count this build has not seen a published family for falls back to the powers of
    # two plus the whole device, which is the shape every published family shares. Fewer
    # profiles than the hardware really offers is the safe direction: each one it does offer is
    # real, and a caller only ever loses a packing option.
    candidates = _PROFILE_SLICES.get(slices) or tuple(
        sorted({1, 2, 4, slices} & set(range(1, slices + 1)))
    )
    out: list[MigProfile] = []
    for k in candidates:
        instances = slices // k
        mem_units = 1 << (k - 1).bit_length()
        memory_gib = spec.memory_gib * mem_units // memory_slices
        out.append(
            MigProfile(
                name=f"{k}g.{memory_gib}gb",
                compute_slices=k,
                memory_gib=memory_gib,
                instances=instances,
                device=spec.name,
                device_slices=slices,
            )
        )
    return tuple(out)


def smallest_profile_for(
    model_gib: float,
    accelerator_type: str | None,
    *,
    headroom: float = 0.15,
) -> MigProfile | None:
    """The smallest instance that holds a model, or `None` when partitioning does not help.

    `None` covers three distinct cases, all of which mean "do not partition": the device is not
    MIG-capable, the model needs the whole device anyway, or the device model is unrecognized.
    A caller treats all three the same way — request a whole device — so they are one return
    rather than an enumeration nobody would branch on.

    Args:
        model_gib: The model's resident footprint, before headroom.
        accelerator_type: A Ray accelerator-type name.
        headroom: Fraction of the instance's memory left free for the CUDA context,
            activations, and fragmentation.

    Returns:
        The smallest profile that fits, or `None`.
    """
    if model_gib <= 0:
        return None
    need = model_gib / (1.0 - min(0.9, max(0.0, headroom)))
    for profile in mig_profiles(accelerator_type):
        if profile.memory_gib >= need and profile.instances > 1:
            return profile
    return None


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
    headroom: float = 0.15,
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
        headroom: Fraction of an instance's memory left free.

    Returns:
        The plan; `profile` is `None` when whole devices are the right answer.
    """
    want = max(1, concurrency)
    profile = smallest_profile_for(model_gib, accelerator_type, headroom=headroom)
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
