"""The node-local fast filesystem — where a spill should actually go.

A GPU node ships with terabytes of local NVMe. Almost none of it is at `/tmp`. On a container
runtime `/tmp` is an overlay on the root filesystem, which is commonly 20-100 GB and shared
with the image, the model cache, and every other tenant of the node; the NVMe is mounted
somewhere else under a name that differs by platform (`/ephemeral`, `/scratch`, `/raid`,
`/mnt/local_disk`, `/mnt/resource`). A spill that defaults to a tempdir finds the overlay, and
the query dies of `ENOSPC` beside seven unused terabytes.

Nothing here is decided from the name alone. A candidate has to *be* there, be writable, be on
a real block device rather than tmpfs or an overlay, and have room. The ordering that results
puts a measured NVMe ahead of a measured SATA disk ahead of whatever the platform hinted at,
and `None` when no candidate survives — at which point the caller keeps the tempdir it was
always going to use.

**Writability is tested by writing.** `os.access` answers about the process's credentials, not
about the mount, and read-only mounts, full filesystems, and a root-owned directory in an
unprivileged container all pass it. A probe file is the only answer that means anything.
"""

from __future__ import annotations

import functools
import os
import tempfile
from dataclasses import dataclass

__all__ = [
    "SCRATCH_CANDIDATES",
    "ScratchVolume",
    "local_scratch_root",
    "reset_scratch_probe",
    "scratch_volumes",
]

#: Mount points that hold node-local fast storage across the platforms Batcher runs on, in no
#: particular order — the ordering is done by *measurement*, not by this list. Provider hints
#: from `site.provider` are checked first, then these, so a platform that mounts its NVMe
#: somewhere unusual is covered by its own hint and everything else is covered here.
SCRATCH_CANDIDATES = (
    "/ephemeral",
    "/scratch",
    "/raid",
    "/local",
    "/mnt/local_disk",
    "/mnt/local_storage",
    "/mnt/localdisk",
    "/mnt/nvme",
    "/mnt/disks/ssd0",
    "/mnt/resource",
    "/nvme",
    "/opt/dlami/nvme",
)

#: The environment override, checked first and used without a device-class preference: an
#: operator who names a directory has already decided.
_SCRATCH_OVERRIDE = "BATCHER_SCRATCH_DIR"

#: Device classes ranked by what a spill costs on them, best first, using the vocabulary
#: `hardware.storage.device_class` reports. `unknown` sits mid-table rather than last because
#: an unclassified device is far more likely to be a local disk the `/sys` probe could not
#: resolve than a network volume.
_CLASS_RANK = {
    "nvme": 0,
    "ssd": 1,
    "raid": 2,
    "mapped": 3,
    "unknown": 4,
    "rotational": 5,
    "network": 6,
}

#: Classes that are never scratch, whatever their size or speed. A tmpfs is the trap worth
#: naming: it is fast, it is often large, and spilling to it relieves no memory pressure at all
#: because the bytes stay in RAM — an out-of-core query that "spills" to `/dev/shm` OOMs at
#: exactly the point it was spilling to avoid. A loopback file inherits the properties of
#: whatever it sits on, which this probe cannot see through.
_EXCLUDED_CLASSES = frozenset({"memory", "loopback"})

#: A volume smaller than this is not worth preferring over a tempdir: it is a config mount or
#: a small ephemeral partition, and a spill that fills it fails the same way `/tmp` would.
_MIN_USEFUL_BYTES = 16 * 1024**3


@dataclass(frozen=True, slots=True)
class ScratchVolume:
    """One candidate scratch directory that survived probing.

    Attributes:
        path: The directory.
        device_class: What `hardware.storage` made of the block device behind it
            (`"nvme"`, `"ssd"`, `"raid"`, `"rotational"`, `"network"`, `"unknown"`).
        free_bytes: Space available at probe time.
        total_bytes: Size of the filesystem.
    """

    path: str
    device_class: str = "unknown"
    free_bytes: int = 0
    total_bytes: int = 0

    @property
    def rank(self) -> tuple[int, int]:
        """Sort key: device class first, then free space descending.

        Class before capacity is deliberate. A 2 TB NVMe beats a 30 TB network volume for
        spill by a margin no capacity difference makes up, because an external merge is
        random-access and that is exactly what a network volume is worst at.
        """
        return (_CLASS_RANK.get(self.device_class, 4), -self.free_bytes)


def _writable(path: str) -> bool:
    """Whether this process can actually create a file in `path`.

    Tested by creating and removing one. `os.access` reports on credentials rather than on the
    mount, so it passes on a read-only mount, on a full filesystem, and inside a container
    whose user does not match the directory's owner.
    """
    try:
        with tempfile.NamedTemporaryFile(dir=path, prefix=".batcher-probe-"):
            return True
    except OSError:
        return False


def _measure(path: str) -> ScratchVolume | None:
    """Probe one candidate, or `None` when it is not usable as scratch."""
    if not os.path.isdir(path) or not _writable(path):
        return None
    from batcher._internal.hardware.storage import device_class

    try:
        stat = os.statvfs(path)
    except OSError:
        return None
    free = stat.f_bavail * stat.f_frsize
    total = stat.f_blocks * stat.f_frsize
    if total <= 0:
        return None
    klass = device_class(path) or "unknown"
    if klass in _EXCLUDED_CLASSES:
        return None
    return ScratchVolume(path=path, device_class=klass, free_bytes=free, total_bytes=total)


@functools.lru_cache(maxsize=1)
def scratch_volumes() -> tuple[ScratchVolume, ...]:
    """Every usable node-local scratch directory, best first.

    Memoized: mounts do not appear under a running process, and the callers ask per spilling
    query. `reset_scratch_probe()` clears it.

    Returns:
        The candidates that exist, are writable, and are large enough to be worth using,
        ordered by device class then free space. Empty on a node with no fast local storage
        mounted, which callers read as "use a tempdir".
    """
    from batcher._internal.site.provider import site_profile

    seen: set[str] = set()
    out: list[ScratchVolume] = []
    for path in (*site_profile().scratch_hints, *SCRATCH_CANDIDATES):
        real = os.path.realpath(path)
        if real in seen:
            continue
        seen.add(real)
        volume = _measure(path)
        if volume is not None and volume.total_bytes >= _MIN_USEFUL_BYTES:
            out.append(volume)
    return tuple(sorted(out, key=lambda v: v.rank))


def local_scratch_root() -> str | None:
    """The directory node-local spilling should use, or `None` to keep the tempdir default.

    Resolution order:

    1. `BATCHER_SCRATCH_DIR`, used as given when it is writable. An operator who names a
       directory has already made the decision this function otherwise makes.
    2. The best measured local volume, by device class then free space.
    3. `None` — no fast local storage is mounted, so the caller's tempdir is already right.

    Returns:
        A directory path, or `None`. Never a path that was not verified writable at probe
        time, because a spill that discovers an unwritable scratch mid-query has already lost
        the work it spilled.
    """
    override = os.environ.get(_SCRATCH_OVERRIDE, "").strip()
    if override:
        return override if os.path.isdir(override) and _writable(override) else None
    volumes = scratch_volumes()
    return volumes[0].path if volumes else None


def reset_scratch_probe() -> None:
    """Forget the memoized scratch probe, so the next call re-measures the mounts."""
    scratch_volumes.cache_clear()
