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
    "device_queue_depth",
    "filesystem_free_bytes",
    "is_rotational",
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


def is_rotational(path: str) -> bool | None:
    """Whether the device backing `path` spins, or `None` when it cannot be determined.

    The single most consequential storage fact for spill planning, because it decides whether
    random access is affordable. An external merge reads many runs concurrently, which is
    sequential per run and random in aggregate; on flash that costs nothing extra, and on a
    spinning disk it collapses to seek time. `None` rather than `False` when unknown, so a
    caller can keep its configured default instead of acting on a fabricated answer.

    Args:
        path: Any path on the filesystem whose device should be classified.

    Returns:
        `True` for a spinning disk, `False` for solid state, `None` when undetectable.
    """
    name = _sys_block_name(path)
    if not name:
        return None
    value = _read_int(f"/sys/block/{name}/queue/rotational")
    return None if value is None else bool(value)


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


def device_queue_depth(path: str) -> int:
    """Requests the device behind `path` accepts in flight, or `0` when undetectable.

    The parallelism a spill or scan can actually use. Issuing 64 concurrent reads to a device
    with a queue depth of 1 does not make them concurrent — it makes them queue, adding
    latency to every one of them without adding throughput. Reading the real depth is how the
    reader fan-out stops being a guess.

    Args:
        path: Any path on the filesystem whose device should be inspected.

    Returns:
        The device's request queue depth, or `0` when it cannot be determined.
    """
    name = _sys_block_name(path)
    if not name:
        return 0
    return _read_int(f"/sys/block/{name}/queue/nr_requests") or 0


def filesystem_free_bytes(path: str) -> int:
    """Free bytes on the filesystem holding `path`, or `0` when it cannot be measured.

    The hard ceiling on how much an operator may spill. Without it, a spill decision that
    correctly relieves memory pressure can fail late with a disk-full error, having already
    written most of the data — strictly worse than having refused to admit the query.

    Args:
        path: Any path on the filesystem to measure.

    Returns:
        Free bytes available to an unprivileged writer, or `0` when unmeasurable.
    """
    try:
        st = os.statvfs(path)
    except OSError:
        return 0
    # f_bavail, not f_bfree: the reserved blocks a non-root process cannot touch are not free
    # for our purposes, and on a default ext4 that reservation is 5% of the volume.
    return int(st.f_bavail) * int(st.f_frsize)
