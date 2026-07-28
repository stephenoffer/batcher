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
    "device_class",
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
