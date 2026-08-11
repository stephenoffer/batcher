"""The scratch volume, measured — free space, budget clamping, and the IPC codec.

Everything the tiered spill store needs to know about the *filesystem* it is writing
to, kept apart from the store so the store reads as tier policy rather than as OS
archaeology. Three questions live here:

- **How much room is there really?** A configured budget is a guess about a disk it has
  never seen; `clamp_to_free_disk` holds it under a fraction of the measured free space,
  and `free_disk_bytes` re-measures during the query so a volume filled by a *co-tenant*
  still triggers overflow.
- **Was that failure the disk filling up?** `is_out_of_space` separates `ENOSPC`/`EDQUOT`
  from every other IO error, because only those two have an actionable answer.
- **Which codec?** The local (NVMe) and remote (object storage) tiers make opposite
  trade-offs, so each gets its own rule.

The free-space reading is TTL-cached. The store asks once per bucket, and a partitioned
spill opens thousands of buckets — a `statvfs` per bucket is a syscall storm measuring a
number that cannot meaningfully move between two buckets of the same query.
"""

from __future__ import annotations

import errno
import os
import shutil
import time
from enum import IntEnum

import pyarrow as pa

from batcher._internal.errors import IOError as BatcherIOError
from batcher._internal.logging import note_suppressed

__all__ = [
    "DISK_FLOOR_BYTES",
    "FREE_DISK_TTL_SECONDS",
    "SPILL_DISK_FRACTION",
    "DiskPressure",
    "clamp_to_free_disk",
    "disk_floor_bytes",
    "disk_pressure",
    "free_disk_bytes",
    "fsspec_open",
    "ipc_options",
    "is_out_of_space",
    "read_free_disk_bytes",
    "read_total_disk_bytes",
    "remote_ipc_options",
    "reset_disk_sampling",
    "scratch_disk_stats",
    "total_disk_bytes",
]


def fsspec_open(path: str, mode: str):
    """Open `path` on the remote tier through `fsspec`, naming the extra if it is absent.

    Args:
        path: Any `fsspec` URL (``s3://``, ``gs://``, ``memory://``, ...).
        mode: The file mode, as `fsspec.open` takes it.

    Returns:
        The `fsspec` `OpenFile` for `path`.

    Raises:
        IOError: If `fsspec` is not installed.
    """
    try:
        import fsspec
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise BatcherIOError(
            f"spilling to object storage ({path!r}) needs fsspec — install the 'cloud' extra"
        ) from exc
    return fsspec.open(path, mode)


# Never let the local spill tier fill the scratch filesystem past this fraction of its
# measured free space: the last sliver keeps room for other tenants, log/tmp writes, and
# the filesystem metadata that a 100%-full disk starves — a full scratch disk is a hard
# query failure, not a slow one.
SPILL_DISK_FRACTION = 0.9

# Free bytes below which the local tier is treated as exhausted regardless of how much of
# its own budget the store has used. The budget accounts for *this store's* bytes, so it is
# blind to anyone else consuming the same volume — a co-tenant process, another query's
# scratch, or a log that grew. Re-measuring catches that; this floor is the headroom the
# measurement is compared against, matching `SPILL_DISK_FRACTION`'s intent of never taking
# the last sliver.
DISK_FLOOR_BYTES = 256 << 20

# Ceiling on the floor above, as a fraction of the volume's *capacity*. A fixed 256 MiB is
# right headroom on a 1 TB NVMe and is 25% of a 1 GiB container scratch mount, where it
# would declare the tier exhausted before a single bucket landed and push every spill to
# object storage (or, with no remote tier, keep the whole guard permanently tripped). The
# floor is therefore whichever is *smaller*: the fixed reserve, or this share of capacity.
_DISK_FLOOR_CAPACITY_FRACTION = 0.05

# How long a measured free-space reading is reused. The store consults it once per bucket
# open, and a 4,096-way partitioned spill therefore asked the kernel 4,096 times for a
# figure that changes on a human timescale. One window still catches a disk filling during
# the query (which is the whole point of re-measuring), while collapsing the syscall storm.
FREE_DISK_TTL_SECONDS = 0.1

# Single-slot TTL cache keyed by the directory being measured: `(path, deadline, free)`.
# One slot, not a dict, because a store writes to exactly one scratch directory — a dict
# would only add a hash lookup and an unbounded key set for no hit-rate gain.
_free_cache: tuple[str, float, int | None] | None = None

# The same, for the volume's *capacity*. It needs its own slot rather than riding along with
# the free reading because the two are asked for by different callers at different rates —
# and because without it the TTL cache above was being routed around: `disk_pressure` reads
# the free bytes through the cache and then takes `total` twice with raw `shutil.disk_usage`
# calls (once directly, once inside `disk_floor_bytes`). The store consults the pressure once
# per bucket open, so a 4,096-way partitioned spill made 8,192 uncached `statvfs` calls —
# more than the storm the cache was added to stop, for a number that moves even less than
# free space does.
_total_cache: tuple[str, float, int | None] | None = None


def reset_disk_sampling() -> None:
    """Drop the cached volume readings, so the next call re-measures.

    For tests that patch the underlying `shutil.disk_usage` and need the change observed
    immediately rather than up to one TTL window later.
    """
    global _free_cache, _total_cache
    _free_cache = None
    _total_cache = None


def read_free_disk_bytes(path: str) -> int | None:
    """One uncached reading of free bytes on the filesystem holding `path`.

    Args:
        path: Any path on the volume of interest.

    Returns:
        Free bytes, or `None` when the volume cannot be stat'd.
    """
    try:
        return shutil.disk_usage(path).free
    except OSError:  # pragma: no cover - unstat-able volume
        return None


def free_disk_bytes(path: str) -> int | None:
    """Measured free bytes on the filesystem holding `path`, TTL-cached.

    Args:
        path: Any path on the volume of interest.

    Returns:
        Free bytes, or `None` if the volume can't be stat'd.
    """
    global _free_cache

    now = time.monotonic()
    cached = _free_cache
    if cached is not None and cached[0] == path and now < cached[1]:
        return cached[2]
    value = read_free_disk_bytes(path)
    _free_cache = (path, now + FREE_DISK_TTL_SECONDS, value)
    return value


def read_total_disk_bytes(path: str) -> int | None:
    """One uncached reading of the capacity of the filesystem holding `path`.

    Args:
        path: Any path on the volume of interest.

    Returns:
        Total bytes, or `None` when the volume cannot be stat'd.
    """
    try:
        return shutil.disk_usage(path).total
    except OSError:  # pragma: no cover - unstat-able volume
        return None


def total_disk_bytes(path: str) -> int | None:
    """Capacity of the filesystem holding `path`, TTL-cached like the free reading.

    Args:
        path: Any path on the volume of interest.

    Returns:
        Total bytes, or `None` if the volume can't be stat'd.
    """
    global _total_cache

    now = time.monotonic()
    cached = _total_cache
    if cached is not None and cached[0] == path and now < cached[1]:
        return cached[2]
    value = read_total_disk_bytes(path)
    _total_cache = (path, now + FREE_DISK_TTL_SECONDS, value)
    return value


def disk_floor_bytes(path: str) -> int:
    """The free-space floor below which the local tier counts as exhausted.

    `DISK_FLOOR_BYTES`, held under a small share of the volume's own capacity so a tiny
    container scratch mount is not declared full from the moment it is created.

    Args:
        path: Any path on the volume of interest.

    Returns:
        The floor in bytes (never below one MiB, so a pathologically small volume still
        gets *some* headroom rather than zero).
    """
    total = total_disk_bytes(path)
    if total is None:  # pragma: no cover - unstat-able volume: keep the fixed reserve
        return DISK_FLOOR_BYTES
    share = int(total * _DISK_FLOOR_CAPACITY_FRACTION)
    return max(1 << 20, min(DISK_FLOOR_BYTES, share))


class DiskPressure(IntEnum):
    """How full the scratch volume is, ordered so callers can compare with ``>=``.

    The disk analogue of `PressureLevel`, and the gap it closes: Carbonite governs memory
    with a four-level ladder every component reads, and governs disk with a single boolean
    consulted at one place — so the only thing that ever happens about a filling scratch
    volume is that a bucket routes elsewhere, and only if a remote tier happens to be
    configured. A ladder lets the same volume be *reported* while there is still room to
    act on it, which for disk is the whole game: an out-of-space write cannot be retried
    or degraded, it fails the query.
    """

    NORMAL = 0  # ample room — nothing to do
    ELEVATED = 1  # under a quarter free — prefer the remote tier for new buckets
    FULL = 2  # under the reserve floor — the local tier is exhausted


# Free-space fraction below which the volume reads as ELEVATED. Deliberately well above
# the reserve floor: the point of the middle rung is to be reached while there is still
# room to react, and the floor is by construction the point where there is not.
_ELEVATED_FREE_FRACTION = 0.25


def disk_pressure(path: str) -> DiskPressure:
    """Classify how full the volume holding `path` is.

    An unstat-able volume reads as `NORMAL`, matching every other probe here: a
    measurement that could not be taken is not evidence of a problem, and treating it as
    one would push every spill on an exotic filesystem to object storage.

    Args:
        path: Any path on the volume of interest.

    Returns:
        The pressure level.
    """
    free = free_disk_bytes(path)
    if free is None:
        return DiskPressure.NORMAL
    if free < disk_floor_bytes(path):
        return DiskPressure.FULL
    total = total_disk_bytes(path)
    if total is None:  # pragma: no cover - stat'd for free but not for total
        return DiskPressure.NORMAL
    if total > 0 and free < total * _ELEVATED_FREE_FRACTION:
        return DiskPressure.ELEVATED
    return DiskPressure.NORMAL


def scratch_disk_stats() -> dict[str, int | str]:
    """One reading of the volume a spill would land on, for telemetry and `explain`.

    Carbonite reports memory in detail and reported disk not at all, so a query that spilled
    slowly — or failed with `ENOSPC` — carried nothing in its profile about the volume it
    spilled to. The disk ladder exists (`DiskPressure`) and only the spill store consulted
    it, which means the one component that could act on a filling volume was also the only
    one that could see it.

    Cheap: two `statvfs` calls behind the same TTL cache the store reads through, so adding
    it to a per-query snapshot costs nothing measurable.

    Returns:
        The resolved scratch `path`, its measured `pressure` level, and `free_bytes` /
        `total_bytes` (`-1` for either the volume cannot report). Never raises — a probe that
        fails yields `UNKNOWN` and `-1`, which is honestly distinct from a healthy reading.
    """
    from batcher._internal.site import spill_scratch_dir

    try:
        path = spill_scratch_dir()
        free = free_disk_bytes(path)
        total = total_disk_bytes(path)
        return {
            "path": path,
            "pressure": disk_pressure(path).name,
            "free_bytes": -1 if free is None else free,
            "total_bytes": -1 if total is None else total,
        }
    except Exception as exc:  # pragma: no cover - a probe must never break a query
        note_suppressed("carbonite", "read the scratch volume", exc)
        return {"path": "", "pressure": "UNKNOWN", "free_bytes": -1, "total_bytes": -1}


def clamp_to_free_disk(local_dir: str, budget: int | None) -> int | None:
    """Clamp a configured local spill budget to a safe fraction of *measured* free disk.

    The static config budget is a guess about a disk it has never seen: on a
    smaller-than-configured scratch volume it lets the local tier fill the filesystem
    before overflow to the remote tier ever triggers (a hard OOM-on-disk); on a larger one
    it overflows to slow object storage prematurely. Measuring the scratch filesystem's
    free space makes overflow track the disk that actually exists — and derives a budget at
    all when none was configured. Best-effort: if the volume can't be stat'd, the configured
    value stands unchanged, so behavior only ever gets safer, never more fragile.

    Args:
        local_dir: The scratch directory (it need not exist yet).
        budget: The configured local budget in bytes, or `None` to derive one.

    Returns:
        The clamped budget, or `None` when neither a budget nor a measurement exists.
    """
    # The spill dir is created lazily (on first spill), so stat the nearest *existing*
    # ancestor — the same filesystem the dir will live on — instead of losing the disk-aware
    # clamp entirely on the common first-use path (which would silently keep the too-large
    # configured budget, the OOM-on-disk this guards against).
    free = None
    path = os.path.abspath(local_dir)
    while True:
        free = read_free_disk_bytes(path)
        if free is not None:
            break
        parent = os.path.dirname(path)
        if parent == path:  # reached the root and still couldn't stat
            break
        path = parent
    if free is None:  # pragma: no cover - even the root failed to stat
        return budget
    safe = int(free * SPILL_DISK_FRACTION)
    return safe if budget is None else min(budget, safe)


def is_out_of_space(exc: OSError) -> bool:
    """Whether `exc` is the volume running out of room, as opposed to any other IO failure.

    Covers both ways a write is refused for space: `ENOSPC` (the filesystem is full) and
    `EDQUOT` (this user's quota is full on a filesystem that is not). They have the same
    cause from the query's point of view and the same fix, and treating a quota stop as an
    unclassified `OSError` is what makes it surface as a bare `[Errno 122]` from inside the
    Arrow writer with nothing naming the spill tier.

    Args:
        exc: The `OSError` a write raised.

    Returns:
        True when the write failed for lack of space or quota.
    """
    return exc.errno in (errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC))


def ipc_options(compression: str | None) -> pa.ipc.IpcWriteOptions | None:
    """Arrow-IPC write options for the configured codec, or `None` if unavailable.

    Spilled data is transient, so a cheap-fast codec (LZ4) trades CPU for disk I/O
    and footprint. ``"auto"`` (the datatype-aware default) is uncompressed here: this
    Python tier spills to fast local disk, where compressing numeric/string state
    costs more CPU than the I/O it saves (the native Rust path's per-batch classifier
    compresses only blob payloads, where it pays). Set ``"lz4"``/``"zstd"`` explicitly
    to force compression on this tier (worthwhile for the slow remote-overflow path).
    Degrades to uncompressed if the codec isn't built into this pyarrow, so spilling never
    fails on a missing optional codec — but the fallback is *recorded*, because a silent
    one turns a typo (``"zstandard"``) or a stripped-down wheel into a bucket that is
    quietly ten times its intended size on a slow tier.

    Args:
        compression: The configured codec name, ``"auto"``, or `None`.

    Returns:
        The write options, or `None` to write uncompressed.
    """
    if not compression or compression == "auto":
        return None
    try:
        return pa.ipc.IpcWriteOptions(compression=compression)
    except (ValueError, pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
        note_suppressed("carbonite", f"use spill compression codec {compression!r}", exc)
        return None


def remote_ipc_options(compression: str | None) -> pa.ipc.IpcWriteOptions | None:
    """IPC write options for the REMOTE tier — **always compressed**.

    Object storage is slow and priced by bytes transferred, so the CPU of a cheap codec
    always pays there even when the fast local NVMe tier stays uncompressed. An unset or
    ``"auto"`` codec is upgraded to LZ4; an explicit ``"lz4"``/``"zstd"`` is honored.
    Degrades to uncompressed only if the codec isn't built into this pyarrow.

    Args:
        compression: The configured codec name, ``"auto"``, or `None`.

    Returns:
        The write options, or `None` if no codec is available at all.
    """
    codec = compression if compression in ("lz4", "zstd") else "lz4"
    return ipc_options(codec)
