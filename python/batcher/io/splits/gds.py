"""Whether a file's bytes can reach a device without a detour through host memory.

A device-native Parquet read is worth doing because the decode happens on the thing that is
going to compute. It is worth *much* more when the bytes also travel storage-to-device by DMA
— GPUDirect Storage — because then the host never touches them at all. The two are separate
questions, and conflating them is how a "GPU read" ends up slower than the host reader it
replaced: on a path GDS cannot serve, the same call stages every byte through a host bounce
buffer, adding a copy to a read that was already going to cross PCIe once.

Three things have to hold, and none of them can be assumed:

* **The cuFile library has to be present.** It ships with the CUDA toolkit and is absent from
  the runtime-only container images most inference workloads run in.
* **The path has to be a local file.** An object-store URI is fetched by the client library
  into host memory first, whatever reader is nominally in front of it, so the device-direct
  argument does not apply and the second Parquet implementation is being taken on for nothing.
* **The filesystem has to be one that supports the DMA path.** A block-backed local filesystem
  does; an overlay, a tmpfs, or a FUSE mount does not, and those are exactly what a container's
  root and a mounted object-store cache are.

**The classification is conservative in one direction on purpose.** An unrecognized filesystem
reports *not* eligible. Being wrong that way costs a fallback to a path that already works;
being wrong the other way costs a read that silently doubles its own memory traffic.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass

__all__ = [
    "GDS_FILESYSTEMS",
    "GdsEligibility",
    "cufile_available",
    "filesystem_type",
    "gds_eligible",
    "gds_summary",
    "reset_gds_probe",
]

#: Filesystems whose GPUDirect Storage support is documented and general. Deliberately short:
#: `ext4` and `xfs` are the local block-backed cases, and the parallel filesystems listed carry
#: vendor GDS support on the clusters that run them. Anything not here reads as unsupported,
#: which costs a fallback rather than a silent bounce-buffered read.
GDS_FILESYSTEMS = frozenset({"ext4", "xfs", "lustre", "gpfs", "beegfs", "wekafs", "nfs"})

#: Filesystems worth naming as *known* to be unservable, so the reason a path was declined is
#: specific rather than "unrecognized". These are the ones that actually show up under a GPU
#: container: the image's own overlay, a shared-memory mount, and a FUSE-mounted object store.
_KNOWN_UNSUPPORTED = frozenset({"overlay", "overlayfs", "tmpfs", "fuse", "fuseblk", "squashfs"})

#: URI schemes that are not local files. A path carrying one of these is fetched by a client
#: library into host memory before any reader sees it.
_REMOTE_SCHEMES = (
    "s3://",
    "gs://",
    "gcs://",
    "az://",
    "abfs://",
    "abfss://",
    "https://",
    "http://",
    "hdfs://",
)


@dataclass(frozen=True, slots=True)
class GdsEligibility:
    """Whether one path can be read storage-to-device, and why not when it cannot.

    Attributes:
        eligible: Whether the DMA path applies.
        reason: A short machine-readable code when it does not (`"no_cufile"`, `"remote"`,
            `"filesystem"`, `"missing"`), `""` when it does. Carried because "the GPU read was
            not used" is otherwise indistinguishable from "the GPU read was not tried", and
            those have different fixes.
        filesystem: The filesystem type behind the path, `""` when it could not be resolved.
    """

    eligible: bool
    reason: str = ""
    filesystem: str = ""


@functools.lru_cache(maxsize=1)
def cufile_available() -> bool:
    """Whether the cuFile library this host would DMA through is loadable.

    Memoized: a shared library does not appear under a running process, and the callers ask
    per read. `reset_gds_probe()` clears it.

    Returns:
        True when cuFile can be loaded. False on a host without the CUDA toolkit, inside a
        runtime-only container image, and off Linux.
    """
    import ctypes

    for name in ("libcufile.so", "libcufile.so.0"):
        try:
            ctypes.CDLL(name)
            return True
        except OSError:
            continue
    return False


@functools.lru_cache(maxsize=1)
def _mount_table() -> tuple[tuple[str, str], ...]:
    """Mount points and their filesystem types, longest path first.

    Longest-first is what makes the lookup a *containing mount* rather than a prefix match:
    `/mnt/nvme` and `/mnt` are both prefixes of `/mnt/nvme/data`, and only the first is the
    filesystem the file is actually on.
    """
    entries: list[tuple[str, str]] = []
    try:
        with open("/proc/self/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    # The kernel escapes spaces in mount points as octal; only that escape
                    # appears in practice and leaving it unhandled would misclassify a path.
                    entries.append((parts[1].replace("\\040", " "), parts[2]))
    except OSError:
        return ()
    return tuple(sorted(entries, key=lambda kv: -len(kv[0])))


def filesystem_type(path: str) -> str:
    """The filesystem type behind a path.

    Args:
        path: A local path. It need not exist; the containing mount is what is resolved.

    Returns:
        The type as the kernel names it (`"ext4"`, `"overlay"`, `"nfs"`), or `""` off Linux
        and when the mount table cannot be read.
    """
    try:
        resolved = os.path.realpath(path)
    except OSError:
        return ""
    for mount, kind in _mount_table():
        if resolved == mount or resolved.startswith(mount.rstrip("/") + "/"):
            return kind
    return ""


def _is_remote(path: str) -> bool:
    """Whether a path names something other than a local file."""
    lowered = path.lower()
    return lowered.startswith(_REMOTE_SCHEMES)


def gds_eligible(path: str) -> GdsEligibility:
    """Whether `path` can be read storage-to-device, and why not when it cannot.

    Args:
        path: The file's path or URI, as a split holds it.

    Returns:
        The verdict. A `file://` URI is treated as the local path it names; every other scheme
        is remote. An unrecognized filesystem is reported ineligible with reason
        `"filesystem"`, which is the conservative direction.
    """
    if _is_remote(path):
        return GdsEligibility(False, "remote")
    if not cufile_available():
        return GdsEligibility(False, "no_cufile")
    local = path[7:] if path.lower().startswith("file://") else path
    kind = filesystem_type(local)
    if not kind:
        return GdsEligibility(False, "missing")
    if kind in _KNOWN_UNSUPPORTED or kind not in GDS_FILESYSTEMS:
        return GdsEligibility(False, "filesystem", kind)
    return GdsEligibility(True, "", kind)


def gds_summary(paths: tuple[str, ...]) -> dict:
    """What the DMA path would do with a set of files, for the decision log.

    Args:
        paths: The files a read is about to touch.

    Returns:
        Whether cuFile is present, how many paths are eligible, and the reasons the rest were
        not, counted. A reader of a slow GPU scan can tell from this whether the bytes reached
        the device directly or were bounced through the host, which is otherwise invisible.
    """
    verdicts = [gds_eligible(p) for p in paths]
    reasons: dict[str, int] = {}
    for verdict in verdicts:
        if not verdict.eligible:
            reasons[verdict.reason] = reasons.get(verdict.reason, 0) + 1
    return {
        "cufile": cufile_available(),
        "paths": len(paths),
        "eligible": sum(1 for v in verdicts if v.eligible),
        "reasons": reasons,
    }


def reset_gds_probe() -> None:
    """Forget the memoized cuFile and mount-table readings, so the next call re-probes."""
    for fn in (cufile_available, _mount_table):
        clear = getattr(fn, "cache_clear", None)
        if clear is not None:
            clear()
