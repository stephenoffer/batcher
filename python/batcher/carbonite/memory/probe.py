"""What this process may actually allocate — host RAM, the cgroup cap, and live headroom.

"How much memory is there" is three different questions on a modern box, and conflating
them has bitten the engine each way. The host's RAM is not the ceiling when a cgroup limit
is lower. `memory.current` is not usage, because it counts reclaimable page cache. And the
free figure `psutil` reports is the *machine's*, not the container's.

Each answer is cached at the lifetime it is stable for: host RAM and the cgroup cap cannot
change for the process, so both are memoized; the live available reading and the reclaimable
page-cache term are re-sampled on a short (50 ms) TTL; the cgroup's raw charge — the figure
the OOM-killer acts on — is re-read on a 1 ms window, wide enough only to let the components
deciding about one query share a read and far too narrow to hide a real change.

Separated from `pressure` so the classification ladder there reads as policy, with the OS
archaeology it rests on in one place beneath it.
"""

from __future__ import annotations

import functools
import os
import time

from batcher._internal.hardware import cgroup_v2_dirs, machine_memory_bytes, read_cgroup_bytes
from batcher.config import active_config

__all__ = [
    "ROUND_COALESCE_SECONDS",
    "SAMPLE_TTL_SECONDS",
    "available_bytes",
    "cap_to_cgroup_headroom",
    "cgroup_current_bytes",
    "cgroup_limit_bytes",
    "effective_limit_bytes",
    "proc_meminfo_available",
    "process_rss_bytes",
    "read_available_bytes",
    "reset_memory_sampling",
    "total_memory_bytes",
]

# How long a live OS memory reading (`psutil.virtual_memory().available`) is reused
# before it is re-sampled. Reading it costs a `/proc` parse (~60 µs) — a real slice of
# the control plane's per-query cost on sub-second queries — yet the figure it returns
# does not move on a µs/ms timescale, and every Carbonite memory decision reacts at
# pipeline-breaker granularity (coarser than this by orders of magnitude). Sharing one
# sample across the decisions of a query, and across the back-to-back queries of an
# interactive session, therefore costs nothing in accuracy: a change is still observed
# within one window, and the live buffer-pool / footprint pressure reads (cheap, and
# left un-cached) still catch a real spike instantly. A stale reading can only make
# admission a touch conservative (older = smaller available), never over-admit unsafely.
SAMPLE_TTL_SECONDS = 0.05

# Single-slot TTL cache for the live available-bytes reading: `(monotonic_deadline, value)`.
_available_cache: tuple[float, int] | None = None

# Single-slot TTL cache for the cgroup page-cache term, on the same window and for the same
# reason: `(monotonic_deadline, value)`. See `_cgroup_file_cache_bytes`.
_file_cache_cache: tuple[float, int] | None = None

# How long the *raw* cgroup charge is reused. Two orders of magnitude below
# `SAMPLE_TTL_SECONDS`, because this is not a sampling window: it is only wide enough to
# let the components deciding about one query share a single read. See `_cgroup_total_bytes`.
ROUND_COALESCE_SECONDS = 0.001

# Single-slot coalescing cache for the raw charge: `(monotonic_deadline, value)`.
_total_cache: tuple[float, int | None] | None = None


def reset_memory_sampling() -> None:
    """Drop every memoized memory reading, so the next sample re-reads the OS.

    For tests that patch the underlying OS readers. This must clear every memoized *ceiling*
    too, not just the live samples: they are the figures memoized with `functools.lru_cache`,
    so a reset that only cleared the module globals left them pinned to whatever the first
    test in the process observed. Since the ceiling is a min over all of them, a stale one
    silently overrode every later patch — the reset appeared to work while the number it was
    supposed to refresh never moved.
    """
    global _available_cache, _file_cache_cache, _total_cache
    from batcher.carbonite.memory.kernel import reset_kernel_sampling

    _available_cache = None
    _file_cache_cache = None
    _total_cache = None
    cgroup_limit_bytes.cache_clear()
    # `total_memory_bytes` now delegates to the neutral probe, which memoizes the whole
    # ceiling — so clearing only this module's caches would leave it pinned.
    machine_memory_bytes.cache_clear()
    # The kernel snapshot (`memory.high`, `memory.events`, PSI) is sampled on its own TTL and
    # now feeds `effective_limit_bytes`, so a reset that left it cached would keep the same
    # silent-override failure this function's docstring describes for the cgroup cap.
    reset_kernel_sampling()


@functools.lru_cache(maxsize=1)
def cgroup_limit_bytes() -> int | None:
    """The container memory limit from cgroup v2 (`memory.max`) or v1
    (`memory.limit_in_bytes`), or `None` when unlimited / not in a cgroup.

    A container's cgroup cap is the *real* ceiling — the host's RAM is not — so
    honoring it is what stops the engine over-admitting and getting OOM-killed by
    the kernel.

    Cached for the process: the cgroup cap is fixed for a container's lifetime, while
    this is read on every admission check — re-opening `memory.max` per query is pure
    hot-path I/O. (The *current* usage, which does change, is read live and uncached.)

    Like the CPU quota, the limit can be set at any level of the cgroup v2 hierarchy — the
    process's own leaf (a namespaced pod), a parent slice (a non-namespaced Ray worker), or
    the mount root — so the effective cap is the tightest `memory.max` across the whole
    ancestry (`cgroup_v2_dirs`), not just the root. v1 keeps its single well-known path.
    """
    limits = [
        v for d in cgroup_v2_dirs() if (v := read_cgroup_bytes(os.path.join(d, "memory.max")))
    ]
    if limits:
        return min(limits)
    return read_cgroup_bytes("/sys/fs/cgroup/memory/memory.limit_in_bytes")  # cgroup v1


def _cgroup_file_cache_bytes() -> int:
    """Page cache charged to this cgroup — `file` (v2) or `total_cache` (v1); 0 if unknown.

    Re-sampled on `SAMPLE_TTL_SECONDS`, the same window `available_bytes` uses, because
    reading it means opening and line-scanning `memory.stat` (~50 fields) and the two
    readers that want a pressure level — morsel sizing and the spill gate — ask on every
    terminal op. Only the *subtracted* term is windowed: `memory.current` itself, the
    number the OOM-killer acts on, is still read fresh on every call, so the safety-critical
    signal keeps full resolution. A stale cache figure can only shift the reported usage by
    however much reclaimable cache moved inside one 50 ms window, and cache is by
    definition the part the kernel gives back before anything is killed.
    """
    global _file_cache_cache

    now = time.monotonic()
    if _file_cache_cache is not None and now < _file_cache_cache[0]:
        return _file_cache_cache[1]
    value = _read_cgroup_file_cache_bytes()
    _file_cache_cache = (now + SAMPLE_TTL_SECONDS, value)
    return value


def _read_cgroup_file_cache_bytes() -> int:
    """The uncached `memory.stat` read behind `_cgroup_file_cache_bytes`."""
    for path, key in (
        ("/sys/fs/cgroup/memory.stat", "file"),
        ("/sys/fs/cgroup/memory/memory.stat", "total_cache"),
    ):
        try:
            with open(path) as f:
                lines = f.read().splitlines()
        except OSError:
            # Not this cgroup version, or not in a cgroup at all. Both paths being absent
            # is the normal case off Linux, so trying the next one is the answer rather
            # than a reportable failure.
            continue
        for line in lines:
            field, _, raw = line.partition(" ")
            if field == key:
                try:
                    return max(0, int(raw))
                except ValueError:
                    return 0
    return 0


def cgroup_current_bytes() -> int | None:
    """The cgroup's *unreclaimable* memory, or `None` when not in a cgroup.

    `memory.current` is routinely mistaken for the OOM number, and is not: it counts
    anonymous memory **plus every clean file page the kernel happens to be caching**. Cache is
    reclaimable — dropped long before anything is OOM-killed — so charging it as pressure
    reads a box that has merely *read files* as one about to die. Measured on a 30 GiB host
    after loading TPC-H: 24.3 GiB "current", of which 15.3 GiB was cache. That pinned the
    monitor at ELEVATED, which halves every morsel (`_MORSEL_PRESSURE_FACTORS`) — a 7%
    throughput loss for the whole run, invisible to every correctness test.

    Subtracting `file` costs the guard nothing, because the two are disjoint: what it exists
    to see (the Flight shuffle store, off-pool pyarrow buffers) is **anonymous** and stays
    counted; what is subtracted is cache that was never the engine's. The remainder
    (anon + slab + sock) is what the kernel cannot get back — the figure the OOM-killer acts on.
    """
    total = _cgroup_total_bytes()
    if total is None:
        return None
    return max(0, total - _cgroup_file_cache_bytes())


def _cgroup_total_bytes() -> int | None:
    """The raw cgroup charge, anonymous **and** cached — `memory.current` / `usage_in_bytes`.

    Not a pressure signal on its own; see `cgroup_current_bytes`.

    Coalesced over `ROUND_COALESCE_SECONDS`. This is deliberately *not* the 50 ms sampling
    window the available-bytes reading uses: it exists only so the several components that
    each ask for the pressure level while making decisions about the *same* query — morsel
    sizing and the spill gate, microseconds apart — read the file once between them instead
    of once each. At a millisecond the reading is still fresh by every timescale a memory
    decision reacts on, and nothing can allocate enough to change the verdict inside it.
    """
    global _total_cache

    now = time.monotonic()
    if _total_cache is not None and now < _total_cache[0]:
        return _total_cache[1]
    value: int | None = None
    for path in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        value = read_cgroup_bytes(path)
        if value is not None:
            break
    _total_cache = (now + ROUND_COALESCE_SECONDS, value)
    return value


def cap_to_cgroup_headroom(host_available: int) -> int:
    """Clamp a host-wide available figure to what this container may still allocate.

    `psutil`/`/proc` report the *machine's* free RAM, but a cgroup-limited container (the norm
    under Kubernetes/Ray) OOMs at `memory.max`, not at host exhaustion — on a 184 GB host an
    8 GB container would otherwise read ~180 GB free and over-admit into a kill. The real
    headroom is `limit - current`, where `current` excludes the reclaimable page cache (see
    `cgroup_current_bytes`) — cache the kernel will evict on demand is headroom, not usage.
    Take the smaller of that and the host figure. No cgroup cap (bare metal / unlimited)
    leaves the reading untouched.

    The `limit` here is the **effective** ceiling, not `memory.max` alone: where cgroup v2
    publishes a lower `memory.high` the kernel throttles every allocating task into direct
    reclaim at that point, so memory above it is not headroom in any sense a query can use.
    Under Kubernetes memory QoS the two differ by the whole request-to-limit gap, and budgeting
    to `memory.max` there plans a query into a band it will spend its entire life being slept
    in. `memory.respect_cgroup_high` turns this off; it is inert where `memory.high` is unset.
    """
    limit = effective_limit_bytes()
    if limit is None:
        return host_available
    current = cgroup_current_bytes()
    if current is None:
        return host_available
    return min(host_available, max(0, limit - current))


def effective_limit_bytes() -> int | None:
    """The cgroup ceiling that binds this process, or `None` when none does.

    `memory.max` alone unless the config opts into the `memory.high` throttle threshold and
    one is published, in which case the lower of the two. Kept beside the other cgroup readers
    rather than in `kernel` so the one function that clamps headroom reads a single figure.

    Returns:
        The binding ceiling in bytes, or `None` outside a cgroup.
    """
    limit = cgroup_limit_bytes()
    if not active_config().memory.respect_cgroup_high:
        return limit
    from batcher.carbonite.memory.kernel import cgroup_high_bytes

    high = cgroup_high_bytes()
    candidates = [v for v in (limit, high) if v is not None]
    return min(candidates) if candidates else None


def process_rss_bytes() -> int | None:
    """This process's resident set size (RSS) via `psutil`, or `None` without it.

    RSS captures the engine's true footprint — the Flight `PartitionStore`, pyarrow
    buffers, everything — not just the buffer pool's accounted reservations. Falls back
    to `/proc/self/statm` on Linux, so a container without the optional dependency still
    gets a real footprint reading rather than losing the safety-critical half of
    `_engine_used_fraction` (which then cannot see anything the pool does not account).

    Returns:
        Resident bytes, or `None` when no reader is available.
    """
    try:
        import psutil
    except ImportError:
        return _proc_statm_rss_bytes()
    try:
        return int(psutil.Process().memory_info().rss)
    except (OSError, ValueError, AttributeError):
        return _proc_statm_rss_bytes()


def _proc_statm_rss_bytes() -> int | None:
    """RSS from `/proc/self/statm` (Linux), in bytes, or `None` if unreadable.

    Field two of `statm` is the resident set in pages. It costs one small read and needs
    no dependency, which is exactly the situation the psutil-less container is in.
    """
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def total_memory_bytes() -> int:
    """The memory ceiling every admission and pressure decision is taken against.

    Delegates to `_internal.hardware.machine_memory_bytes`, which is the one implementation
    of "how much memory may this process actually have": host RAM less reserved hugepages,
    every cgroup cap in the ancestry including `memory.high`, the batch scheduler's grant, and
    `RLIMIT_AS`.

    It used to compute its own, and the two had drifted — this copy missed reserved hugepages
    and the `memory.high` throttle threshold, so on a node with either one Carbonite admitted
    against memory the planner already knew was not there. The layer that was wrong is the one
    that decides whether to spill, which is the worst place for the two views to differ.

    Falls back to `MemoryConfig.default_total_bytes` (one home for the fallback) when the OS
    reports nothing at all. The underlying probe is memoized for the process — every figure it
    reads is fixed for a container's lifetime — while this is read on every admission and
    pressure check.
    """
    return machine_memory_bytes() or active_config().memory.default_total_bytes


def proc_meminfo_available() -> int | None:
    """`MemAvailable` from `/proc/meminfo` (Linux), or `None` if unreadable.

    The without-psutil fallback so memory governance still senses real pressure on
    Linux containers where the optional dep isn't installed."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024  # kB → bytes
    except (OSError, ValueError, IndexError):
        return None
    return None


def available_bytes() -> int:
    """Available RAM, served from a short-TTL sample (see `SAMPLE_TTL_SECONDS`).

    The underlying `psutil.virtual_memory()` (or `/proc` fallback) read is the single most
    expensive step in a per-query Carbonite decision. A change is still picked up within one
    TTL window, so amortizing the read across a query's decisions — and across a session's
    back-to-back queries — costs nothing in accuracy: a stale reading is an older, smaller
    figure, which makes admission slightly conservative and can never over-admit.

    Returns:
        Available bytes, capped to this container's cgroup headroom.
    """
    global _available_cache
    now = time.monotonic()
    cached = _available_cache
    if cached is not None and now < cached[0]:
        return cached[1]
    value = read_available_bytes()
    _available_cache = (now + SAMPLE_TTL_SECONDS, value)
    return value


def read_available_bytes() -> int:
    """One live, uncached reading of available RAM, capped to the cgroup headroom.

    `psutil` when present, else Linux `/proc/meminfo`, else a large sentinel meaning
    "assume headroom" so a platform that reports nothing does not read as full.

    Returns:
        Available bytes for this process.
    """
    try:
        import psutil

        host = int(psutil.virtual_memory().available)
    except ImportError:
        proc = proc_meminfo_available()
        host = proc if proc is not None else (1 << 62)
    return cap_to_cgroup_headroom(host)
