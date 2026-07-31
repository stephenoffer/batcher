"""Putting several claimants on one device — Carbonite's half of fractional scheduling.

`_internal.device_share` is the arithmetic: bytes in, a quantum out. That is the easy half, and
on its own it is dangerous. A fraction is a *promise about a shared resource*, and the promise
is only as good as the three things the arithmetic cannot see:

* **Who else is on the device.** A quarter of a device is a quarter of what is *left*, and a
  fleet running an inference actor beside a relational shard has two claimants that each
  measured an empty device before the other started.
* **Whether the device is healthy.** A part clamped to half its clocks by a hardware thermal
  slowdown still reports its full memory, and four tenants on it are four tenants running at
  half rate with the failure that caused the clamp still in progress.
* **Whether isolation is available.** A MIG partition and a `0.25` request schedule the same
  number of workers onto a device and mean entirely different things when one of them
  over-allocates: the partition's neighbour is unaffected, the co-tenant's neighbour dies.

So this module decides, and `_internal.device_share` computes. `plan_task_packing` is the one
entry point most callers want: a byte figure and a concurrency in, a `TaskPacking` out that
names the fraction to request, how many tasks that puts on a device, how many devices the stage
then occupies, and whether the answer carries isolation. `dist` turns it into Ray options and
never re-derives any of it.

The refusal direction is deliberate throughout. Anything unknown — an unreadable device, an
unmeasured working set, a fleet whose health cannot be probed — packs *one* claimant per device,
which is exactly the behavior a fleet had before fractional packing existed. Nothing here can
make a working cluster worse by declining to answer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from batcher._internal.device_share import (
    MAX_COTENANTS,
    balanced_fraction,
    cotenants_per_device,
    devices_for,
    pack_fraction,
    quantize_fraction,
    share_bytes,
    usable_bytes,
)

__all__ = [
    "TaskPacking",
    "derated_cotenants",
    "external_headroom_bytes",
    "packing_summary",
    "plan_task_packing",
    "shard_fraction",
    "whole_device_packing",
]


@dataclass(frozen=True, slots=True)
class TaskPacking:
    """How a stage's concurrent claimants are laid out across devices.

    Attributes:
        fraction: The `num_gpus` each claimant requests. `1.0` is the unpacked answer and the
            one every unknown resolves to.
        per_device: Claimants one device holds concurrently.
        devices: Devices the requested concurrency occupies.
        share_bytes_: Device memory one claimant may use, `0` when the device size was unknown.
        isolated: Whether the share is a hardware partition (MIG) rather than a co-tenancy. An
            isolated share has its own memory and its own fault domain; a co-tenancy does not,
            and a caller that reports one as the other is claiming a guarantee it lacks.
        reason: One line for the decision log, naming the numbers that produced the answer.
    """

    fraction: float = 1.0
    per_device: int = 1
    devices: int = 1
    share_bytes_: int = 0
    isolated: bool = False
    reason: str = ""

    @property
    def packed(self) -> bool:
        """Whether more than one claimant shares a device under this plan."""
        return self.per_device > 1

    def as_dict(self) -> dict:
        """The packing as a flat record, for an event payload or a report.

        Returns:
            Every field under its own key, with `share_bytes` spelled without the trailing
            underscore the dataclass needs to avoid shadowing the builtin.
        """
        return {
            "fraction": self.fraction,
            "per_device": self.per_device,
            "devices": self.devices,
            "share_bytes": self.share_bytes_,
            "isolated": self.isolated,
            "reason": self.reason,
        }


def whole_device_packing(concurrency: int = 1, reason: str = "") -> TaskPacking:
    """The unpacked answer: one claimant per device.

    Every refusal in this module returns this rather than a `None` a caller has to remember to
    handle. It is the behavior a fleet had before fractional packing existed, so a caller that
    ignores the distinction is merely not getting the improvement, never getting a wrong device.

    Args:
        concurrency: How many claimants run at once.
        reason: Why packing was declined, for the decision log.

    Returns:
        A whole-device packing over `max(1, concurrency)` devices.
    """
    want = max(1, concurrency)
    return TaskPacking(fraction=1.0, per_device=1, devices=want, reason=reason)


def external_headroom_bytes(device_bytes: float, used_bytes: float, headroom: float) -> int:
    """The device memory actually available, after headroom *and* whoever is already on it.

    The figure a second claimant must size against, and the one a bare capacity lookup gets
    wrong in the only case that matters. A device with an inference actor holding 40 of its
    80 GiB reports 80 to anything that asks the driver for its capacity, and a shard packed
    against 80 discovers the other 40 as an out-of-memory error rather than as a smaller share.

    Args:
        device_bytes: The device's total memory.
        used_bytes: Bytes already resident across every process on the device, typically
            `DeviceTelemetry.memory_used_bytes`. `0` when nothing was measured, which is
            treated as an empty device — the pre-measurement behavior.
        headroom: Fraction of the device held back for the context and fragmentation.

    Returns:
        Bytes, never negative. `0` means the device is already full, which callers read as
        "do not place here" rather than as an unknown.
    """
    usable = usable_bytes(device_bytes, headroom)
    return max(0, usable - max(0, int(used_bytes)))


def derated_cotenants(per_device: int, derate: float) -> int:
    """Co-tenants a device should hold once its health verdict is applied.

    A derate is a statement about *throughput*, and packing is a statement about *memory*, so
    the two look independent. They are not: a device clamped to half its clocks runs each
    co-tenant at half rate, and four of them then take four times as long to release the memory
    they hold. Packing a clamped device as though it were healthy converts a slow device into a
    device that is both slow and out of memory.

    Args:
        per_device: Co-tenants the memory arithmetic allowed.
        derate: The device's health derate in `[0, 1]`, `1.0` for a healthy device.

    Returns:
        Co-tenants to actually place, at least `1` for any schedulable device. A derate of
        `0.0` is a quarantined device and returns `0` — nothing should be placed there at all.
    """
    if derate <= 0.0:
        return 0
    if per_device <= 1:
        return 1
    return max(1, int(per_device * min(derate, 1.0)))


def shard_fraction(
    shard_bytes: float,
    device_bytes: float,
    *,
    headroom: float = 0.15,
    max_per_device: int = MAX_COTENANTS,
) -> float:
    """The `num_gpus` one relational shard task should request.

    The relational counterpart of the inference-stage packing Kyber does. A GPU fan-out
    oversubscribes shards past the device count so each one is small and a lost one is cheap;
    every one of those shards then asks for a whole device, and Ray runs exactly one per device
    while the rest queue. When a shard's working set is a quarter of a device, that is a
    four-fold under-use of a fleet whose shard count already says it expected to fit.

    Args:
        shard_bytes: The largest shard's estimated device working set. The *largest*, not the
            mean: the fraction is one number for the whole fan-out, and sizing it to the mean
            guarantees the biggest shard does not fit the share it was granted.
        device_bytes: One device's total memory, the smallest on a mixed fleet.
        headroom: Fraction of the device held back.
        max_per_device: Ceiling on co-tenants, which floors the fraction. A caller that knows
            the device is derated or already occupied passes a lower number here rather than
            adjusting the byte figure, so the reason survives into the decision log.

    Returns:
        A fraction from the packing ladder, `1.0` whenever nothing could be decided or the
        shard needs a whole device or more. Never `0.0`: a shard task with no GPU request is a
        GPU task scheduled onto a CPU, so the no-opinion answer here is the whole device.
    """
    raw = pack_fraction(shard_bytes, device_bytes, headroom=headroom)
    if raw <= 0.0 or raw >= 1.0:
        return 1.0
    floor = balanced_fraction(max(1, min(max_per_device, MAX_COTENANTS)))
    return max(raw, floor)


def plan_task_packing(
    need_bytes: float,
    *,
    device_bytes: float,
    concurrency: int = 1,
    accelerator_type: str = "",
    headroom: float | None = None,
    derate: float = 1.0,
    used_bytes: float = 0.0,
    prefer_isolation: bool | None = None,
) -> TaskPacking:
    """Lay `concurrency` claimants of `need_bytes` out across devices.

    The entry point. It composes the three things the raw arithmetic cannot see — what else is
    resident, how healthy the device is, and whether the part can be partitioned — and returns
    one record naming the fraction, the co-tenancy, and the device count together, so no caller
    has to recombine them and none of them can disagree.

    A MIG partition wins whenever the claimant fits one and the configuration asks for it. That
    is not a tie-break on efficiency: a partition and a co-tenancy schedule the same number of
    workers, and only one of them keeps a neighbour's allocation spike from being this worker's
    out-of-memory error.

    Args:
        need_bytes: Device memory one claimant will hold.
        device_bytes: One device's total memory, the smallest on a mixed fleet.
        concurrency: Claimants running at once.
        accelerator_type: The binding device's model (`"NVIDIA_H100"`), `""` on an unlabelled
            or mixed fleet — where the partition question cannot be answered and the quanta
            are used instead.
        headroom: Fraction of the device held back; the configured `accelerator.vram_headroom`
            when omitted.
        derate: The device's health derate in `[0, 1]`. Below `1.0` it reduces the co-tenancy;
            at `0.0` the device is quarantined and nothing is placed on it.
        used_bytes: Bytes already resident on the device across every process.
        prefer_isolation: Whether to take a MIG partition when one fits; the configured
            `accelerator.prefer_mig` when omitted.

    Returns:
        The packing. Every unknown resolves to `whole_device_packing`, which is what the fleet
        did before this existed.
    """
    from batcher.config import active_config

    cfg = active_config().accelerator
    room = cfg.vram_headroom if headroom is None else headroom
    isolate = cfg.prefer_mig if prefer_isolation is None else prefer_isolation
    want = max(1, concurrency)

    if derate <= 0.0:
        return TaskPacking(
            fraction=1.0, per_device=1, devices=0, reason="device quarantined; nothing placed"
        )
    if need_bytes <= 0 or device_bytes <= 0:
        return whole_device_packing(want, "working set or device size unknown; whole devices")

    available = external_headroom_bytes(device_bytes, used_bytes, room)
    if available <= 0:
        return whole_device_packing(want, "device already fully resident; whole devices")

    isolated_plan = _isolated_packing(need_bytes, accelerator_type, want) if isolate else None
    if isolated_plan is not None:
        return isolated_plan

    # Packed against what is *left*, not against nameplate capacity: a resident co-tenant is
    # exactly the case a fraction chosen from the device's total size gets wrong.
    raw = quantize_fraction(need_bytes / available)
    if raw <= 0.0 or raw > 1.0:
        devices = max(1, int(raw)) * want if raw > 1.0 else want
        return TaskPacking(
            fraction=max(1.0, raw),
            per_device=1,
            devices=devices,
            share_bytes_=share_bytes(device_bytes, max(1.0, raw), room),
            reason=f"{need_bytes / 1e9:.1f}GB exceeds one device; whole devices",
        )

    per_device = derated_cotenants(min(cotenants_per_device(raw), want), derate)
    if per_device <= 1:
        return whole_device_packing(
            want,
            f"{need_bytes / 1e9:.1f}GB of {available / 1e9:.1f}GB available; one per device"
            + ("" if derate >= 1.0 else f" (derate {derate:.2f})"),
        )
    fraction = balanced_fraction(per_device)
    return TaskPacking(
        fraction=fraction,
        per_device=per_device,
        devices=devices_for(fraction, want),
        share_bytes_=share_bytes(device_bytes, fraction, room),
        reason=(
            f"{need_bytes / 1e9:.1f}GB of {available / 1e9:.1f}GB available → "
            f"{fraction} x {per_device} per device"
            + ("" if derate >= 1.0 else f", derated {derate:.2f}")
        ),
    )


def _isolated_packing(need_bytes: float, accelerator_type: str, want: int) -> TaskPacking | None:
    """A MIG-partitioned packing when the part offers one that fits, else `None`.

    Kept separate because the failure path matters more than the success path: a device model
    the profile table does not recognize, a part with no partitioning, or a claimant too large
    for the biggest instance must all fall through to the quanta rather than refuse to place
    the stage.
    """
    from batcher.carbonite.accel.mig import mig_plan

    plan = mig_plan(need_bytes / (1 << 30), accelerator_type or None, want)
    if plan.profile is None:
        return None
    return TaskPacking(
        fraction=plan.gpu_fraction,
        per_device=plan.instances_per_device,
        devices=plan.devices_needed,
        share_bytes_=int(plan.profile.memory_gib * (1 << 30)),
        isolated=True,
        reason=plan.reason,
    )


def packing_summary(packings: Sequence[TaskPacking]) -> dict:
    """What a set of packing decisions costs the fleet, as one record.

    The figure worth reporting is not any single fraction but the *total* device demand: a job
    whose stages each packed sensibly can still ask for more devices than the cluster has, and
    that is visible only in the sum.

    Args:
        packings: The decisions taken, one per stage.

    Returns:
        `stages`, `devices` (summed), `packed_stages` (those sharing a device), `isolated`
        (those on a hardware partition), and `min_fraction` — the tightest share granted, which
        is the one an out-of-memory report should be read against. Zeroed for an empty input.
    """
    if not packings:
        return {"stages": 0, "devices": 0, "packed_stages": 0, "isolated": 0, "min_fraction": 0.0}
    return {
        "stages": len(packings),
        "devices": sum(p.devices for p in packings),
        "packed_stages": sum(1 for p in packings if p.packed),
        "isolated": sum(1 for p in packings if p.isolated),
        "min_fraction": min(p.fraction for p in packings),
    }
