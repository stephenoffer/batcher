"""The memory ceiling and page geometry this process runs under.

The host's RAM is not the limit a containerized process lives within, the page size is not
always 4 KiB, and whether overshooting the ceiling means *slow* or *dead* depends on whether
the machine has swap. All of it is read here once so every sizing decision above can share one
answer rather than each guessing.
"""

from __future__ import annotations

import functools
import glob
import os

from batcher._internal.hardware.cgroup import cgroup_v2_dirs, read_cgroup_bytes
from batcher._internal.hardware.sysfs import read_int

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

    Five ceilings bind, and the tightest wins:

    * **Host RAM**, less any memory reserved into an explicit hugepage pool. Reserved
      hugepages are carved out of general-purpose memory and are unreachable by an ordinary
      allocation, so `SC_PHYS_PAGES` alone over-states what the heap can have by the whole
      reservation — 32 GiB of it on a database- or DPDK-tuned node is not unusual, and the
      engine would size a working set against memory that structurally cannot hold it.
    * **`memory.max`** (v2) / **`memory.limit_in_bytes`** (v1): the hard cap, past which the
      kernel OOM-kills.
    * **`memory.high`** (v2): the *throttle* threshold. It is not a kill boundary, which is
      why it was skipped, but a cgroup above it is put into synchronous reclaim and made to
      crawl — so a budget sized to `memory.max` on a cgroup whose `high` sits below it buys a
      running-but-thrashing query rather than a spilling one. Spilling earlier is the strictly
      better trade, and it is the trade an operator asked for by setting `high` at all.
    * **The batch scheduler's grant** (`site.scheduler.scheduler_memory_bytes`). A Slurm
      allocation is not a cgroup unless the site configured confinement, and plenty do not: a
      job granted 16 GiB on a 512 GiB node *sees* 512 GiB, sizes a hash table against it, and
      is killed for exceeding a grant nothing in the process ever read. This is the memory
      half of the core bound `available_cpu_count` applies, and it fails the same way.
    * **`RLIMIT_AS`** (`site.container.address_space_limit_bytes`): how Grid Engine enforces
      `h_vmem`, LSF `-M`, PBS `pvmem`, and what `ulimit -v` sets. It binds harder than a
      cgroup — overshooting a cgroup gets the process OOM-killed, while overshooting this
      makes the allocator return NULL and the query die of `MemoryError` inside a kernel that
      had no chance to spill instead.

    This is the one implementation. Carbonite's `memory.probe.total_memory_bytes` delegates to
    it rather than computing its own, which is what keeps the planner's view of the machine and
    the admission controller's from drifting — they had already diverged on reserved hugepages
    and on `memory.high`, and the layer that was wrong was the one that decides whether to spill.

    Returns:
        The binding memory ceiling in bytes, or `0` when undetectable.
    """
    host = 0
    try:
        host = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        host = 0
    if host > 0:
        # Never below zero, and never *raised* by a bad reading: a hugepage figure larger than
        # the host's own RAM is a parse error, not a machine with negative memory.
        host = max(0, host - min(hugepage_bytes(), host))
    limits = [host] if host > 0 else []
    for base in cgroup_v2_dirs():
        for name in ("memory.max", "memory.high"):
            cap = read_cgroup_bytes(os.path.join(base, name))
            if cap is not None:
                limits.append(cap)
    v1 = read_cgroup_bytes("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if v1 is not None:
        limits.append(v1)
    # Imported inside the call: `site` reaches back into this package for the storage probe,
    # so a module-level edge would make that a cycle.
    from batcher._internal.site.container import address_space_limit_bytes
    from batcher._internal.site.scheduler import scheduler_memory_bytes

    grant = scheduler_memory_bytes()
    if grant:
        limits.append(grant)
    address_space = address_space_limit_bytes()
    if address_space:
        limits.append(address_space)
    return min(limits) if limits else 0


@functools.lru_cache(maxsize=1)
def hugepage_bytes() -> int:
    """Bytes reserved into explicit hugepage pools, or `0` when none are or it is unreadable.

    Every size class the kernel exposes, summed — a node commonly reserves 2 MiB pages for one
    tenant and 1 GiB pages for another, and counting only `/proc/meminfo`'s default class would
    miss whichever one the operator actually used.

    This is memory that *exists and cannot be allocated*. The hugetlb pool is carved out of the
    general allocator at reservation time and handed back only by an explicit `hugetlbfs`
    mapping, which nothing in this engine makes. So it is subtracted from the host figure in
    [`machine_memory_bytes`] rather than merely reported: a 256 GiB node with 64 GiB reserved
    has 192 GiB for a hash table, and sizing to 256 is how an operator that Kyber predicted
    would fit gets OOM-killed on a node with plenty of "free" memory in `free -g`.

    Returns:
        Reserved hugepage bytes, or `0` on a node with no pools and on any platform without
        `/sys/kernel/mm/hugepages`.
    """
    total = 0
    for pool in glob.glob("/sys/kernel/mm/hugepages/hugepages-*kB"):
        # The directory name carries the page size; `nr_hugepages` carries the count. Both are
        # needed, and a pool missing either contributes nothing rather than a partial figure.
        try:
            page_kib = int(os.path.basename(pool).removeprefix("hugepages-").removesuffix("kB"))
        except ValueError:
            continue
        count = read_int(os.path.join(pool, "nr_hugepages"))
        if page_kib > 0 and count > 0:
            total += page_kib * 1024 * count
    return total


@functools.lru_cache(maxsize=1)
def swap_configured() -> bool:
    """Whether this machine has any swap space active.

    The fact that decides what running out of memory *means*, and the two meanings want
    opposite policies. With swap, overshooting the budget degrades: pages go out, the query
    slows, and it finishes. Without it — the default on Kubernetes, on most container runtimes,
    and on every Ray worker pod this engine normally runs in — overshooting is terminal: the
    kernel OOM-kills the largest process, which is the engine, and the whole query is lost
    along with every partition it had already computed.

    A swapless node arguably wants to spill *earlier* and admit *less*, because it has no soft
    landing between "fits" and "dead".

    **Reported, not yet acted on.** No budget in the engine reads this: `memory.soft_limit` and
    `memory.hard_limit` are the same on both kinds of node, and an operator who wants the
    earlier line sets them. Saying otherwise here would be describing an intention — the
    reading is surfaced in `ResourceManager.stats()` so a person tuning those fractions can see
    which kind of node they are on. Making the engine choose is a throughput trade in the
    direction of spilling more, so it wants a benchmark rather than an argument.

    Reads the cgroup's own swap allowance first where one is published, because a container
    with host swap can still be denied it (`memory.swap.max = 0`, which is what Kubernetes
    writes) — and there the host's swap partitions are present and irrelevant.

    Returns:
        True when swap is available to this process; False when it demonstrably is not, and on
        any platform that cannot report it — the conservative reading, since it selects the
        earlier-spill policy.
    """
    for base in reversed(cgroup_v2_dirs()):  # leaf-most first: our own slice is what binds
        try:
            with open(os.path.join(base, "memory.swap.max")) as f:
                raw = f.read().strip()
        except OSError:
            continue
        if raw == "max":
            break  # unlimited by the cgroup — fall through to whether the host has any
        try:
            return int(raw) > 0
        except ValueError:
            continue
    try:
        with open("/proc/swaps") as f:
            # A header line is always present; a swapless machine has nothing after it.
            return len([line for line in f.read().splitlines() if line.strip()]) > 1
    except OSError:
        return False


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
