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
    "hugepage_bytes",
    "machine_memory_bytes",
    "page_size_bytes",
    "swap_configured",
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


@functools.lru_cache(maxsize=1)
def hugepage_bytes() -> int:
    """The default transparent-hugepage size in bytes, or `0` when THP is off/unavailable.

    A 2 MiB page maps 512x the memory per TLB entry, which is the difference between a large
    hash table's probes hitting the TLB and missing it on nearly every row. Whether the
    machine offers them is therefore a real performance property of the box, not a detail:
    the same join can be TLB-bound on one host and not on another with identical cores.

    Returns:
        The hugepage size in bytes, or `0` when transparent hugepages are disabled.
    """
    enabled = ""
    try:
        with open("/sys/kernel/mm/transparent_hugepage/enabled") as f:
            enabled = f.read().strip()
    except OSError:
        return 0
    # The active mode is the one in brackets; "[never]" means no hugepages will be handed out.
    if "[never]" in enabled or not enabled:
        return 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("Hugepagesize:"):
                    parts = line.split()
                    return int(parts[1]) * 1024  # reported in kB
    except (OSError, ValueError, IndexError):
        return 0
    return 0


@functools.lru_cache(maxsize=1)
def swap_configured() -> bool:
    """Whether this machine has swap the kernel can push our pages into.

    Changes what memory pressure *means*. Without swap, exceeding the limit is an immediate
    OOM kill, so the memory budget must be respected exactly. With swap, the same overshoot
    degrades into disk latency instead — survivable, but it makes a query mysteriously slow
    with healthy-looking CPU and no spill of our own. The two failure modes want different
    responses, and nothing else distinguishes them.

    Returns:
        `True` when any swap device is configured.
    """
    try:
        with open("/proc/swaps") as f:
            # Line 1 is the header; any line after it is a configured swap device.
            return len(f.read().strip().splitlines()) > 1
    except OSError:
        return False
