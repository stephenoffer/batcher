"""What spilling costs on *this* machine's storage.

Whether to accept a plan that spills is the optimizer's most consequential memory decision,
and the right answer depends on what the plan spills *to*. Local flash sustains gigabytes per
second at microsecond latency, so a spilled plan is frequently better than a contorted one
that avoids spilling. A network-attached volume delivers a fraction of that with a queue in
front of it, and a spinning disk punishes the concurrent run reads an external merge produces.
The spread is roughly thirty-fold, and it runs in the direction that changes the decision.

Charging one number across all of them makes the optimizer confidently right on one class of
machine and confidently wrong on the rest, with nothing in the plan to indicate which. Reading
the device is what turns that constant into a measurement.

Separate from `cost` because it answers a different question — what this hardware is, rather
than what this plan does — and because the cost model should read a factor, not carry a table
of storage-device trivia.
"""

from __future__ import annotations

from batcher._internal.hardware.storage import (
    SPILL_DEVICE_FACTOR,
    SPILL_DEVICE_FACTOR_DEFAULT,
    device_cost_factor,
)

__all__ = ["SPILL_DEVICE_FACTOR", "SPILL_DEVICE_FACTOR_DEFAULT", "spill_device_factor"]

# The device-class cost table lives at layer 0 (`_internal.hardware.storage`), not here.
# Carbonite needs the same figures to decide whether compressing a spilled byte pays, and the
# two subsystems are forbidden to import each other — so the fact sits below both rather than
# being pasted into each. Re-exported under the names this module has always published, so a
# caller and the surface diff see no change.


def spill_device_factor(storage_class: str = "") -> float:
    """How much a spilled byte costs on the spill device, relative to local flash.

    Read from the device backing the directory the engine will actually spill to, which is
    the same three-step resolution the spill paths themselves use: the configured
    `spill_dir`, else the node's measured local scratch volume, else a system tempdir. Asking
    only the first and last of those would price a spill against the container's overlay while
    it lands on the node's NVMe — a factor of ten in the wrong direction on exactly the
    machines where an out-of-core plan is worth ranking carefully.

    **Whose device, though.** Resolving it in this process describes the *driver*, and on a
    cluster the driver spills nothing: the workers do, to their own volumes. A driver on local
    NVMe planning for workers on a network volume under-states a spilled byte tenfold, in the
    one term that decides whether an out-of-core plan is acceptable at all. A caller with a
    `HardwareProfile` passes the binding worker's measured class instead.

    Cheap enough for the planning path: the device probe behind it memoizes per resolved
    directory, so this costs one `stat` the first time a process plans a spilling query and a
    dict lookup after.

    Examples:
        .. doctest::

            >>> from batcher.kyber.storage_cost import spill_device_factor
            >>> spill_device_factor() >= 1.0
            True

    Args:
        storage_class: The measured device class of the node that will spill, from
            `HardwareProfile.storage_class`. `""` — what every caller without a profile
            passes — resolves this process's own spill directory, which is exactly right
            single-node and is what this always did.

    Returns:
        The cost multiplier for spilled bytes, at least 1.0.
    """
    if storage_class:
        return SPILL_DEVICE_FACTOR.get(storage_class, SPILL_DEVICE_FACTOR_DEFAULT)
    from batcher._internal.site import spill_scratch_dir

    return device_cost_factor(spill_scratch_dir())
