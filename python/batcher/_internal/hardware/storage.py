"""The block device behind a directory — what spilling to it will actually cost.

Spilling is a decision about a trade between memory and a device, and the device half of that
trade spans three orders of magnitude: a local NVMe sustains gigabytes per second at
microsecond latency, a network-attached volume delivers a fraction of that with a queue in
front of it, and a spinning disk punishes the random access an external merge produces. A
spill policy tuned against one of them is wrong against the others, and the failure is
asymmetric — treating NVMe as a spinning disk leaves memory pressure unrelieved for no
reason, while treating a network volume as NVMe turns a slow query into a stalled one.

None of this can be assumed from the operating system or the instance type. It has to be read
off the device, which is what this module does.

Linux-only (`/sys/block`); reports the neutral "unknown" answer elsewhere, which callers treat
as "keep the configured default".
"""

from __future__ import annotations

import functools
import os

__all__ = [
    "SPILL_DEVICE_FACTOR",
    "SPILL_DEVICE_FACTOR_DEFAULT",
    "device_class",
    "device_cost_factor",
]

# Device-name prefixes that identify a class without any /sys lookup. Ordered longest-first so
# a more specific prefix wins; `nvme` before `nbd` matters because both start the same way for
# a two-character match.
_NAME_CLASSES: tuple[tuple[str, str], ...] = (
    ("nvme", "nvme"),
    ("nbd", "network"),
    ("rbd", "network"),
    ("drbd", "network"),
    ("loop", "loopback"),
    ("md", "raid"),
    ("dm-", "mapped"),
)


def _sys_block_name(path: str) -> str:
    """The `/sys/block` entry backing `path`, or `""` when it cannot be resolved.

    Resolves the filesystem's device number to a name via ``/sys/dev/block``, which works for
    a partition (``nvme0n1p1``) as well as a whole disk, then walks up to the parent disk
    because the rotational and queue attributes live on the disk rather than the partition.
    """
    try:
        st = os.stat(path)
    except OSError:
        return ""
    major, minor = os.major(st.st_dev), os.minor(st.st_dev)
    if major == 0:
        return ""  # a virtual filesystem (tmpfs, overlayfs) has no backing block device
    link = f"/sys/dev/block/{major}:{minor}"
    try:
        target = os.path.realpath(link)
    except OSError:
        return ""
    if not os.path.isdir(target):
        return ""
    # A partition directory contains a `partition` file and sits under its disk.
    if os.path.exists(os.path.join(target, "partition")):
        target = os.path.dirname(target)
    return os.path.basename(target)


def _read_int(path: str) -> int | None:
    """An integer from a `/sys` file, or `None` when absent or unparseable."""
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


@functools.lru_cache(maxsize=8)
def device_class(path: str) -> str:
    """A coarse class for the device behind `path`, for sizing and for fingerprinting.

    One of ``nvme``, ``ssd``, ``rotational``, ``network``, ``loopback``, ``raid``, ``mapped``,
    ``memory`` (a tmpfs or other device-less filesystem), or ``unknown``. Coarse on purpose:
    the class is an input to the hardware fingerprint, so it must be stable across reboots and
    across instances of the same shape, which a device name or serial number would not be.

    Memoized per path, since a directory's backing device does not change under a running
    process and spill sizing asks on every admission decision.

    Args:
        path: Any path on the filesystem to classify.

    Returns:
        The device class name, or ``unknown`` when it cannot be determined.
    """
    name = _sys_block_name(path)
    if not name:
        # No backing block device: either a memory-backed filesystem, or not Linux. A tmpfs
        # spill target is a real configuration (and a trap — spilling to RAM relieves nothing),
        # so it is worth naming distinctly rather than folding into "unknown".
        return "memory" if os.path.exists("/sys/dev/block") else "unknown"
    for prefix, cls in _NAME_CLASSES:
        if name.startswith(prefix):
            return cls
    rotational = _read_int(f"/sys/block/{name}/queue/rotational")
    if rotational is None:
        return "unknown"
    return "rotational" if rotational else "ssd"


# What a byte written to a device class costs relative to one on local flash.
#
# **Local flash is the omitted baseline**, because that is where every spill constant in this
# engine was calibrated. This table only ever charges *more*, and only where a device was
# positively identified as slower than that baseline: a class nobody could read keeps the
# factor the callers have always used, so an unreadable `/sys` re-ranks no plan and re-tunes
# no policy. The multipliers are conservative ratios of sustained sequential throughput.
#
# It lives here, at layer 0 beside `device_class`, because *two* subsystems need it and they
# are forbidden to import each other: Kyber prices a spilled byte with it, and Carbonite
# decides whether compressing that byte pays. Copy-paste is the only wrong way to share
# between them, so the fact and its scale sit below both.
SPILL_DEVICE_FACTOR: dict[str, float] = {
    # Roughly a tenth of local flash's bandwidth, with request latency on top.
    "network": 10.0,
    # An order of magnitude worse again once access stops being purely sequential, which an
    # external merge's concurrent run reads guarantee.
    "rotational": 30.0,
    # A file-backed device: the host filesystem's own cost plus a layer of indirection.
    "loopback": 2.0,
}

# Every other class — `nvme`, `ssd`, `raid`, `mapped`, `memory`, and `unknown`. An
# unidentified device is left at the calibration point rather than guessed at.
#
# `memory` (a tmpfs target) belongs here too, and deliberately: it is genuinely fast, so
# costing it as slow would be a lie. That it relieves no memory pressure at all is a
# *feasibility* question for whoever chose the directory, not a throughput one, and encoding
# the warning as a fake bandwidth number would bury it where nobody reads.
SPILL_DEVICE_FACTOR_DEFAULT = 1.0


def device_cost_factor(path: str) -> float:
    """How much a byte written under `path` costs relative to one on local flash.

    Args:
        path: Any path on the filesystem in question; it need not exist.

    Returns:
        The multiplier, at least 1.0. `1.0` for local flash, for an unidentified device, and
        off Linux — a measurement moves a decision here, never the absence of one.
    """
    return SPILL_DEVICE_FACTOR.get(device_class(path), SPILL_DEVICE_FACTOR_DEFAULT)
