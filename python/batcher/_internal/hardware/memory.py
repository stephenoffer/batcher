"""The memory ceiling and page geometry this process runs under.

The host's RAM is not the limit a containerized process lives within, and the page size is not
always 4 KiB. Both facts are read here once so every sizing decision above can share one
answer rather than each guessing.
"""

from __future__ import annotations

import functools
import os

from batcher._internal.hardware.cgroup import cgroup_v2_dirs, read_cgroup_bytes

__all__ = [
    "machine_memory_bytes",
    "page_size_bytes",
]


@functools.lru_cache(maxsize=1)
def machine_memory_bytes() -> int:
    """The memory ceiling this process runs under: `min(host RAM, cgroup limit)`, or `0`.

    The neutral hardware fact behind every memory-sizing decision. `min` because a container's
    cgroup cap — not the host's RAM — is the real ceiling: sizing to host RAM over-commits and
    gets the cgroup OOM-killed. Fixed for the process's lifetime, so memoized.

    (Carbonite's `pressure.total_memory_bytes` computes the same ceiling for its own live
    pressure sensing; this is the copy the layers that cannot import Carbonite — notably Kyber
    — read, so the planner can size to real memory without reaching across a subsystem boundary.)

    Returns:
        The binding memory ceiling in bytes, or `0` when undetectable.
    """
    host = 0
    try:
        host = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        host = 0
    limits = [host] if host > 0 else []
    for base in cgroup_v2_dirs():
        cap = read_cgroup_bytes(os.path.join(base, "memory.max"))
        if cap is not None:
            limits.append(cap)
    v1 = read_cgroup_bytes("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if v1 is not None:
        limits.append(v1)
    return min(limits) if limits else 0


@functools.lru_cache(maxsize=1)
def page_size_bytes() -> int:
    """The virtual-memory page size in bytes, or `4096` when unreportable.

    Not a constant across the hardware Batcher targets: 4 KiB on x86-64, but 16 KiB on Apple
    silicon and configurable to 64 KiB on ARM server parts. It sets the granularity of every
    fault the per-operator counters report, so converting a fault count into bytes of resident
    memory needs the real figure rather than an assumed one.

    Returns:
        The page size in bytes.
    """
    try:
        return int(os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return 4096
