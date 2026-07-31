"""How much of one accelerator a claimant gets — the fractional-scheduling vocabulary.

A scheduler hands out *devices*; a workload consumes *device memory*. On a fleet of 80 GiB
parts those two units are an order of magnitude apart, and the gap is where a GPU cluster
loses most of its capacity: a 3 GiB model, a 2 GiB shard, or a small join build side each
occupies four percent of a device and one hundred percent of the schedulable unit. Ray answers
this with a fractional `num_gpus`, and the only hard part is choosing the fraction.

This module is that choice, as pure arithmetic over bytes, with no opinion about who is asking.
It lives at layer 0 for a reason that is structural rather than tidy: **Kyber sizes inference
stages, Carbonite admits co-tenants onto a device, and `dist` turns the answer into Ray
options** — three subsystems that must not import one another, and which were therefore one
copy-paste away from rounding the same fraction three different ways. A stage packed at `0.25`
by the optimizer and admitted at `0.2` by the resource manager is not a rounding difference; it
is five tenants on a device sized for four.

Two rules run through all of it:

* **Round up, never down.** A fraction is a *reservation*, so the error that costs a device
  OOM is under-asking. Every quantum boundary resolves upward, and a need larger than one
  device resolves to whole devices rather than to a fraction of several.
* **Zero means no opinion.** An unknown device size or an unmeasured need returns `0.0`, which
  every caller reads as "keep whatever you had". A fabricated denominator is how a fleet ends
  up packed against a device it does not have.

A neutral utility: any layer may import `_internal`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "MAX_COTENANTS",
    "PACK_QUANTA",
    "DeviceShare",
    "balanced_fraction",
    "cotenants_per_device",
    "devices_for",
    "fits_one_device",
    "pack_fraction",
    "quantize_fraction",
    "share_bytes",
    "usable_bytes",
]

#: The fractions of a device a claimant may be granted, ascending, ending at the whole device.
#: A ladder rather than a continuous value because Ray packs by *summing* the requests on a
#: node: arbitrary fractions leave slivers no task fits into (three tasks at 0.31 strand 0.07 of
#: every device), while powers-of-two-ish quanta tile a device exactly. Ending at `1.0` is what
#: lets `pack_fraction` guarantee a match for any need at or below one device.
#:
#: The ladder stops at quarters deliberately. An eighth of a device is a legitimate ask for a
#: tiny model, and it is also eight CUDA contexts on one device, eight processes contending for
#: one copy engine, and eight tenants whose combined allocation spike is one OOM. Below a
#: quarter, a MIG partition is the answer that carries isolation with it — see
#: `carbonite.accel.mig`.
PACK_QUANTA: tuple[float, ...] = (0.25, 0.5, 1.0)

#: Most claimants one device may hold, the reciprocal of the smallest quantum. Named rather
#: than recomputed because it also bounds things that are not fractions — the number of shard
#: tasks queued against a device, the number of streams a staging plan opens — and those
#: callers must not each pick their own ceiling.
MAX_COTENANTS: int = round(1.0 / PACK_QUANTA[0])


@dataclass(frozen=True, slots=True)
class DeviceShare:
    """One claimant's slice of the fleet: how much of a device, and how many devices.

    Attributes:
        fraction: The `num_gpus` value to request. In `(0, 1]` for a claimant that shares a
            device, a whole number above `1.0` for one that needs several, and `0.0` when
            nothing could be decided — which every caller must treat as "keep the default".
        bytes_: Device memory this share may actually use, after headroom. `0` when the
            device's capacity was unknown.
        per_device: Claimants of this share one device holds. `1` for a whole-device or
            multi-device share.
        devices: Whole devices this share occupies, `1` for any fraction of one.
        reason: One line for the decision log, naming the numbers that produced the answer.
    """

    fraction: float
    bytes_: int = 0
    per_device: int = 1
    devices: int = 1
    reason: str = ""

    @property
    def is_fractional(self) -> bool:
        """Whether this share leaves room on its device for another claimant."""
        return 0.0 < self.fraction < 1.0

    @property
    def decided(self) -> bool:
        """Whether anything was decided at all — `False` is the no-opinion answer."""
        return self.fraction > 0.0


def usable_bytes(device_bytes: float, headroom: float) -> int:
    """Device memory a claimant may plan against, after the untouchable part.

    The CUDA context, allocator fragmentation, and the activation peaks no declared footprint
    includes are real and are not in anyone's byte count. Sizing against nameplate capacity is
    how a plan that "fits" OOMs on its first dispatch.

    Args:
        device_bytes: The device's total memory.
        headroom: Fraction held back. At least a tenth of the device is always left usable —
            a headroom above nine tenths describes a device nothing can run on, which is a
            config error rather than a plan.

    Returns:
        Bytes, `0` when the capacity is unknown or non-positive.
    """
    if device_bytes <= 0:
        return 0
    # Clamping the *retained* share rather than the headroom keeps the boundary exact: the
    # complement of a clamped 0.9 is 0.09999999999999998, which floors a tenth of a device to
    # one byte below a tenth and would make the clamp itself look like an off-by-one.
    return int(device_bytes * max(0.1, min(1.0, 1.0 - headroom)))


def quantize_fraction(fraction: float, quanta: Sequence[float] = PACK_QUANTA) -> float:
    """Round a raw device fraction up to the next schedulable quantum.

    Args:
        fraction: The share a need works out to, as a fraction of one usable device.
        quanta: The ladder to round against, ascending and ending at `1.0`.

    Returns:
        The smallest quantum at or above `fraction`; `math.ceil(fraction)` as a float when the
        need exceeds a whole device, since a claimant larger than one device holds whole ones;
        and `0.0` for a non-positive need, the no-opinion answer.
    """
    if fraction <= 0.0 or not quanta:
        return 0.0
    if fraction > 1.0:
        return float(math.ceil(fraction))
    return next((q for q in quanta if q >= fraction), 1.0)


def pack_fraction(
    need_bytes: float,
    device_bytes: float,
    *,
    headroom: float = 0.15,
    quanta: Sequence[float] = PACK_QUANTA,
) -> float:
    """The `num_gpus` a claimant needing `need_bytes` should request.

    The one function this module exists for. Everything else here is either an input to it or a
    consequence of its answer, and every subsystem that packs a device goes through it so that
    "a quarter of a device" means the same number of bytes everywhere.

    Args:
        need_bytes: Device memory the claimant will hold.
        device_bytes: One device's total memory — the *smallest* device on a mixed fleet, since
            a fraction chosen against the largest is an over-commitment on every other one.
        headroom: Fraction of the device held back from the calculation.
        quanta: The packing ladder.

    Returns:
        A fraction from `quanta`, a whole number of devices when the need exceeds one, or `0.0`
        when either input is unknown — which leaves the caller's own default in place rather
        than packing against a fabricated device.

    Examples:
        .. doctest::

            >>> from batcher._internal.device_share import pack_fraction
            >>> gib = 1 << 30
            >>> pack_fraction(3 * gib, 80 * gib)  # a small model on a large device
            0.25
            >>> pack_fraction(120 * gib, 80 * gib)  # larger than one device
            2.0
            >>> pack_fraction(3 * gib, 0)  # nothing known about the device
            0.0
    """
    usable = usable_bytes(device_bytes, headroom)
    if need_bytes <= 0 or usable <= 0:
        return 0.0
    return quantize_fraction(need_bytes / usable, quanta)


def share_bytes(device_bytes: float, fraction: float, headroom: float = 0.15) -> int:
    """The memory one share of `fraction` may actually use.

    The inverse of `pack_fraction`, and the figure a claimant must size its *own* buffers
    against. Sizing them against the whole device instead is the failure the fraction was
    chosen to prevent: four tenants at `0.25` each budgeting for a full device demand four
    devices' worth of memory from one, at exactly the packing factor that was supposed to make
    them fit.

    Args:
        device_bytes: One device's total memory.
        fraction: The granted share; values above `1.0` are whole devices and multiply.
        headroom: Fraction of the device held back.

    Returns:
        Bytes, `0` when the device's capacity or the fraction is unknown.
    """
    usable = usable_bytes(device_bytes, headroom)
    return int(usable * fraction) if usable > 0 and fraction > 0 else 0


def cotenants_per_device(fraction: float) -> int:
    """How many claimants of this share one device holds.

    Args:
        fraction: A granted share.

    Returns:
        `floor(1 / fraction)` for a fraction of a device, `1` for a whole-device or
        multi-device share, and `0` for the no-opinion `0.0` — a count of zero being the honest
        answer to "how many fit" when nothing was decided.
    """
    if fraction <= 0.0:
        return 0
    if fraction >= 1.0:
        return 1
    return max(1, int(1.0 / fraction))


def devices_for(fraction: float, claimants: int) -> int:
    """Whole devices `claimants` of this share occupy at once.

    The figure an autoscaler request and a feasibility check are both sized from. A stage of
    sixteen workers at `0.25` needs four devices, not sixteen, and asking for sixteen is how a
    fleet grows to a size its own packing decision says it does not need.

    Args:
        fraction: The granted share per claimant.
        claimants: How many run concurrently.

    Returns:
        Devices, `0` when nothing was decided or nothing is running.
    """
    if fraction <= 0.0 or claimants <= 0:
        return 0
    if fraction >= 1.0:
        return math.ceil(fraction) * claimants
    return math.ceil(claimants / cotenants_per_device(fraction))


def balanced_fraction(claimants: int, quanta: Sequence[float] = PACK_QUANTA) -> float:
    """The largest quantum at which `claimants` still fit on one device.

    The complement of `pack_fraction`: that one starts from a byte count and asks what fits,
    this one starts from a desired concurrency and asks what may be granted. A caller that
    wants four shards resident on a device at once needs `0.25` regardless of how small they
    are, because Ray admits by summing requests and a fifth tenant at `0.25` simply queues.

    Args:
        claimants: Concurrent claimants one device should hold.
        quanta: The packing ladder.

    Returns:
        A quantum, `1.0` for one claimant or fewer, and the smallest available quantum when
        more claimants are asked for than the ladder can divide a device into — packing beyond
        `MAX_COTENANTS` is refused rather than approximated, since the extra tenants queue
        instead of running and the caller should see the ceiling it hit.
    """
    if claimants <= 1 or not quanta:
        return 1.0
    fitting = [q for q in quanta if q * claimants <= 1.0]
    return max(fitting) if fitting else min(quanta)


def fits_one_device(need_bytes: float, device_bytes: float, headroom: float = 0.15) -> bool:
    """Whether a claimant fits a single device, headroom included.

    The routing question one step above the fraction: a need that fits is *packed*, one that
    does not is *sharded*, and the two paths are different code. Answering it from the same
    usable-bytes arithmetic the fraction uses keeps the boundary in one place.

    Args:
        need_bytes: Device memory the claimant will hold.
        device_bytes: One device's total memory.
        headroom: Fraction of the device held back.

    Returns:
        True when it fits. An unknown device reports `False`, which routes to the sharded path
        — the one that is correct at any size, merely slower when it was not needed.
    """
    usable = usable_bytes(device_bytes, headroom)
    return usable > 0 and 0 < need_bytes <= usable
