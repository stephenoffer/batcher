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
import glob
import os

__all__ = [
    "FLASH_SPILL_MBPS",
    "SPILL_DEVICE_FACTOR",
    "SPILL_DEVICE_FACTOR_DEFAULT",
    "device_class",
    "device_cost_factor",
]

# Device-name prefixes that identify a class outright, with no `/sys` lookup and no backing
# device to look through. No prefix here is a prefix of another, so the order is presentation
# only. (`drbd` does not match `rbd`: these are tested with `startswith`, not as substrings.)
_NAME_CLASSES: tuple[tuple[str, str], ...] = (
    ("nvme", "nvme"),
    ("nbd", "network"),
    ("rbd", "network"),
    ("drbd", "network"),
    ("loop", "loopback"),
)

# Virtual devices that are a *view* of other devices rather than a medium of their own, and the
# class to report when the devices underneath them cannot be resolved. Neither says anything
# about speed on its own: an `md` RAID0 of four local NVMe and an `md` RAID1 of two iSCSI
# targets are the same prefix and thirty times apart, and a `dm-` mapper sits over whatever LVM
# was pointed at — which on a cloud instance is very often a network-attached volume.
#
# Reporting the prefix alone is what made that invisible. `mapped` and `raid` both carry the
# default cost factor, so LVM over EBS — an ordinary, extremely common root-and-scratch layout
# — was priced as local flash, understating a spilled byte tenfold in the one term that decides
# whether an out-of-core plan is acceptable at all.
_COMPOSITE_CLASSES: dict[str, str] = {"md": "raid", "dm-": "mapped"}

# How far to follow `slaves/` before giving up. LVM over mdraid over partitions is three, and
# nothing real is deeper; the bound is what makes a `/sys` tree with a cycle in it terminate
# rather than hang a planning call.
_SLAVE_DEPTH_MAX = 4


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

    A **composite** device (LVM's `dm-*`, mdraid's `md*`) is resolved through its `slaves/`
    to what actually stores the bytes, and reported as the *slowest* class underneath it — the
    binding direction, since a spill striped across one local NVMe and one network volume runs
    at the network volume's rate. Only when the backing devices cannot be read does it fall
    back to naming the mapper itself.

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
    return _class_of_device(name, _SLAVE_DEPTH_MAX)


#: Transports an NVMe namespace can be reached over that are **not** a local PCIe link. NVMe
#: over Fabrics presents an ordinary `nvme0n1` whose bytes cross a network, so the device name
#: — the one signal used before — reports the fastest class in the table for storage that
#: belongs in the slowest. `pcie` is the local case and is deliberately absent.
_NVME_FABRIC_TRANSPORTS = frozenset({"rdma", "tcp", "fc", "loop"})

#: Substrings that appear in a SCSI device's `/sys` path only when the LUN is remote: an iSCSI
#: session, or a Fibre Channel / FCoE remote port. Such a device answers `rotational = 0` and
#: was therefore classified `ssd`, at a tenth of its real cost, on exactly the SAN-backed
#: deployments where an out-of-core plan most needs pricing correctly.
_REMOTE_SCSI_MARKERS = ("/session", "/rport-", "/fc_remote_ports")


def _class_of_device(name: str, depth: int) -> str:
    """The class of one `/sys/block` entry, following a composite device to its backing store."""
    for prefix, cls in _NAME_CLASSES:
        if name.startswith(prefix):
            # An NVMe *name* is not evidence of an NVMe *link*: NVMe-oF namespaces are named
            # identically to local ones. Ask the driver which transport this namespace uses.
            if cls == "nvme" and _nvme_transport(name) in _NVME_FABRIC_TRANSPORTS:
                return "network"
            return cls
    for prefix, fallback in _COMPOSITE_CLASSES.items():
        if name.startswith(prefix):
            return _slowest_backing_class(name, depth) or fallback
    if _is_remote_scsi(name):
        return "network"
    rotational = _read_int(f"/sys/block/{name}/queue/rotational")
    if rotational is None:
        return "unknown"
    return "rotational" if rotational else "ssd"


def _nvme_transport(name: str) -> str:
    """The transport an NVMe namespace is reached over (``pcie``/``tcp``/``rdma``/``fc``).

    `""` when the driver does not publish one, which every consumer reads as "assume local" —
    the pre-existing answer, so a kernel too old to expose `transport` prices exactly as before.
    """
    try:
        with open(f"/sys/block/{name}/device/transport") as f:
            return f.read().strip().lower()
    except OSError:
        return ""


def _is_remote_scsi(name: str) -> bool:
    """Whether a SCSI block device is a remote LUN (iSCSI, Fibre Channel, FCoE).

    Decided from where the device sits in the `/sys` device tree, which is a positive
    identification rather than an inference from the name: a local SAS disk and an iSCSI LUN
    are both `sd*` and both answer the same `rotational`, and only the path distinguishes them.
    """
    try:
        resolved = os.path.realpath(f"/sys/block/{name}/device")
    except OSError:
        return False
    return any(marker in resolved for marker in _REMOTE_SCSI_MARKERS)


def _slowest_backing_class(name: str, depth: int) -> str:
    """The costliest class among a composite device's backing devices, `""` when unresolvable.

    The costliest rather than the commonest, because a stripe finishes at the rate of its
    slowest member: an LVM volume group spanning local flash and a network disk delivers the
    network disk's throughput for anything that touches both, which an external merge's
    concurrent run reads reliably do.

    An `unknown` member is skipped rather than treated as fast or slow — it carries no
    measurement, and letting it win either way would turn one unreadable device into a verdict
    about the whole array.
    """
    if depth <= 0:
        return ""
    try:
        slaves = os.listdir(f"/sys/block/{name}/slaves")
    except OSError:
        return ""
    classes = [
        found
        for slave in slaves
        # A slave entry may be a partition (`nvme0n1p1`), whose queue attributes live on the
        # disk; `_sys_block_name` handles that for a path, and here the parent is found by the
        # same rule — a partition directory has no `slaves` and no `queue` of its own.
        if (found := _class_of_device(_parent_disk(slave), depth - 1)) not in ("", "unknown")
    ]
    if not classes:
        return ""
    return max(classes, key=lambda c: SPILL_DEVICE_FACTOR.get(c, SPILL_DEVICE_FACTOR_DEFAULT))


def _parent_disk(name: str) -> str:
    """The whole-disk name backing a `/sys/block` entry, which may itself be a partition."""
    if os.path.exists(f"/sys/block/{name}"):
        return name
    # A partition is not its own `/sys/block` entry; it lives under its disk. `sdb3` -> `sdb`,
    # `nvme0n1p2` -> `nvme0n1`. Resolved through the tree rather than by string surgery, so a
    # naming scheme this code has not seen degrades to "unknown" instead of to a wrong disk.
    for disk in glob.glob("/sys/block/*"):
        if os.path.isdir(os.path.join(disk, name)):
            return os.path.basename(disk)
    return name


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

# The sustained sequential throughput, in MB/s, that factor `1.0` above *means*.
#
# This is not a new assumption. Every entry in the table is already a ratio against local
# flash's bandwidth ("roughly a tenth", "an order of magnitude worse again"), so the anchor
# has always existed — it was simply implicit, which made the ratios impossible to check
# against a measurement. Naming it is what lets a *measured* spill rate be compared with what
# a device class claims, so a misclassified device can be caught instead of silently
# re-ranking every out-of-core plan on the machine.
#
# Deliberately conservative: a mid-range NVMe sustains well above this, so a device that
# measures faster than the anchor is unambiguously flash-class rather than borderline. Being
# conservative here only ever makes the measured correction *less* eager to fire, which is the
# right direction for a figure that decides whether a plan may spill.
FLASH_SPILL_MBPS = 1000.0


def device_cost_factor(path: str) -> float:
    """How much a byte written under `path` costs relative to one on local flash.

    Args:
        path: Any path on the filesystem in question; it need not exist.

    Returns:
        The multiplier, at least 1.0. `1.0` for local flash, for an unidentified device, and
        off Linux — a measurement moves a decision here, never the absence of one.
    """
    return SPILL_DEVICE_FACTOR.get(device_class(path), SPILL_DEVICE_FACTOR_DEFAULT)
