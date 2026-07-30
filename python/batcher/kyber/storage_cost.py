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

import tempfile

from batcher._internal.hardware import device_class
from batcher.config import active_config

__all__ = ["SPILL_DEVICE_FACTOR", "SPILL_DEVICE_FACTOR_DEFAULT", "spill_device_factor"]

# What a spilled byte costs relative to one on local flash, by the class of device it lands on.
#
# **Local flash is the omitted baseline, because that is where the spill cost term was
# calibrated.** This table only ever charges *more*, and only where a device was positively
# identified as slower than that baseline. A class nobody could read keeps the factor the model
# has always used, so an unreadable `/sys` re-ranks no plan: a measurement moves a decision
# here, never the absence of one.
#
# The multipliers are conservative ratios of sustained sequential throughput.
SPILL_DEVICE_FACTOR = {
    # Roughly a tenth of local flash's bandwidth, with request latency on top.
    "network": 10.0,
    # An order of magnitude worse again once access stops being purely sequential, which an
    # external merge's concurrent run reads guarantee.
    "rotational": 30.0,
    # A file-backed device: the host filesystem's own cost plus a layer of indirection.
    "loopback": 2.0,
}

# Every other class — `nvme`, `ssd`, `raid`, `mapped`, `memory`, and `unknown`. Local flash is
# the calibration point, and an unidentified device is left at it rather than guessed at, so
# this model is a strict refinement of the constant it replaces rather than a re-tuning of it.
#
# `memory` (a tmpfs spill target) belongs here too, and deliberately: it is genuinely fast, so
# costing it as slow would be a lie. It is also a trap — spilling to RAM relieves no memory
# pressure at all — but that is a *feasibility* question for Carbonite, not a ranking question
# for the cost model, and encoding the warning as a fake throughput number would bury it in the
# one place nobody reads.
SPILL_DEVICE_FACTOR_DEFAULT = 1.0


def spill_device_factor() -> float:
    """How much a spilled byte costs on this machine's spill device, relative to local flash.

    Read from the device backing the directory the engine will actually spill to, which is
    the same three-step resolution the spill paths themselves use: the configured
    `spill_dir`, else the node's measured local scratch volume, else a system tempdir. Asking
    only the first and last of those would price a spill against the container's overlay while
    it lands on the node's NVMe — a factor of ten in the wrong direction on exactly the
    machines where an out-of-core plan is worth ranking carefully.

    Cheap enough for the planning path: `device_class` memoizes per resolved directory, so this
    costs one `stat` the first time a process plans a spilling query and a dict lookup after.

    Examples:
        .. doctest::

            >>> from batcher.kyber.storage_cost import spill_device_factor
            >>> spill_device_factor() >= 1.0
            True

    Returns:
        The cost multiplier for spilled bytes, at least 1.0.
    """
    from batcher._internal.site import local_scratch_root

    spill_dir = active_config().memory.spill_dir or local_scratch_root() or tempfile.gettempdir()
    return SPILL_DEVICE_FACTOR.get(device_class(spill_dir), SPILL_DEVICE_FACTOR_DEFAULT)
