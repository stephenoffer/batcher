"""Effective hardware detection — the CPU, memory, and cache budget this process really has.

`os.cpu_count()` reports the *host's* logical cores, which over-counts inside a container:
a Kubernetes/Ray pod is throttled by a cgroup CPU quota (CFS bandwidth) or pinned to a
cpuset, so sizing thread pools and task fan-out to the host count over-subscribes — the
scheduler thrashes on context switches for cores the process will never get. This resolves
the real budget: the CPU-affinity mask (cpuset) capped by the CFS quota, falling back to the
host count when neither is discoverable. A neutral utility (any layer may import `_internal`).

Device *inventory* — what accelerator is attached and how to reach it — lives one module
over in `accelerators`, beside the model-to-VRAM table it belongs with, and is re-exported
here so every existing caller and probe-reset hook keeps its import path.
"""

from __future__ import annotations

import contextlib
import functools
import glob
import os

from batcher._internal.accelerators import (
    accelerator_backend,
    gpu_devices_absent,
    gpu_inventory,
    reset_accelerator_probes,
)
from batcher._internal.mathx import ceil_div

__all__ = [
    "INFERENCE_INFLIGHT_DEPTH_MAX",
    "accelerator_backend",
    "available_cpu_count",
    "cgroup_v2_dirs",
    "cpu_contention",
    "gpu_devices_absent",
    "gpu_inventory",
    "l3_cache_bytes",
    "machine_memory_bytes",
    "reset_hardware_probes",
]

# Hard ceiling on how many partitions an inference actor keeps in flight at once (submit-ahead
# depth), and the mirror ceiling on the intra-worker autobatch pipeline. One home in a neutral
# layer both the ML autobatcher (`ml.gpu`, layer 6) and the distributed actor pool (`dist`,
# layer 4) import, rather than two copies that drift — `dist` cannot import `ml`, so a shared
# copy was the ONLY correct way to share this, and it used to be pasted instead. Past ~16
# in-flight the GPU is already saturated and deeper submit-ahead only grows resident memory.
INFERENCE_INFLIGHT_DEPTH_MAX = 16


def reset_hardware_probes() -> None:
    """Forget every memoized hardware reading, so the next call re-probes the OS.

    These probes answer questions a running process cannot see change — its cgroup
    ancestry and CPU quota, the machine's RAM and L3, the attached accelerators — so each
    is read once and remembered, which is what keeps them off the per-query path. The one
    caller that needs them re-read is a test faking the underlying `/proc`, `/sys`, or
    device-node state; this is its hook, the counterpart of
    `carbonite.memory.probe.reset_memory_sampling`. A name currently bound to a
    test stand-in has no cache to clear, and is skipped.
    """
    for probe in (
        cgroup_v2_dirs,
        _read_cgroup_v2_quota,
        _cfs_quota_count,
        l3_cache_bytes,
        machine_memory_bytes,
    ):
        clear = getattr(probe, "cache_clear", None)
        if clear is not None:
            clear()
    reset_accelerator_probes()


def _affinity_count() -> int | None:
    """Cores in this process's scheduling-affinity mask (cpuset), or `None` if unavailable."""
    getaffinity = getattr(os, "sched_getaffinity", None)
    if getaffinity is None:  # not Linux (macOS/Windows expose no affinity mask)
        return None
    try:
        n = len(getaffinity(0))
    except OSError:
        return None
    return n if n > 0 else None


def _quota_cores(quota: int, period: int) -> int | None:
    """`ceil(quota / period)` cores, or `None` when either is non-positive (unlimited)."""
    if quota > 0 and period > 0:
        return max(1, ceil_div(quota, period))  # ceil-div
    return None


@functools.lru_cache(maxsize=8)
def _read_cgroup_v2_quota(base: str) -> int | None:
    """Cores the cgroup v2 ``<base>/cpu.max`` permits, or `None` when unlimited/absent.

    ``cpu.max`` is ``"<quota> <period>"``; a ``max`` quota means unlimited.

    Memoized per directory: the CFS quota is cgroup *configuration*, fixed for the
    process's lifetime in every deployment that sets one (a K8s limit, a Ray worker's
    slice), and `available_cpu_count` runs on every terminal op. Re-reading the file
    each time cost one `open`+`read` syscall pair per ancestry level per query for a
    value that never moves.
    """
    try:
        with open(os.path.join(base, "cpu.max")) as f:
            parts = f.read().split()
    except OSError:
        return None
    if len(parts) >= 1 and parts[0] != "max":
        try:
            period = int(parts[1]) if len(parts) > 1 else 100_000
            return _quota_cores(int(parts[0]), period)
        except ValueError:
            return None
    return None


@functools.lru_cache(maxsize=1)
def cgroup_v2_dirs() -> tuple[str, ...]:
    """Every cgroup v2 dir whose ``cpu.max`` can bind this process: mount root through leaf.

    The leaf comes from the process's own ``/proc/self/cgroup``. Root and leaf coincide inside a
    K8s pod (a cgroup *namespace* maps the pod's cgroup to the mount root) but diverge in a
    *delegated* cgroup with no namespace (a Ray worker under a systemd slice). cgroup v2 enforces
    the CFS bandwidth quota at **every** level, so the effective limit is the tightest ``cpu.max``
    anywhere in the chain — checking only the ends would miss a quota set on a parent slice.

    Resolved once and memoized: the CPU and memory probes read it on every terminal op, and a
    process cannot change the cgroup it belongs to while it runs. A tuple, not a list, so the
    memo cannot hand a mutable object to every caller (`reset_hardware_probes` clears it).
    """
    dirs = ["/sys/fs/cgroup"]
    sub = ""
    try:
        with open("/proc/self/cgroup") as f:
            for line in f:
                if line.startswith("0::"):  # the unified-hierarchy (v2) line
                    sub = line.rstrip().split("::", 1)[1]
                    break
    except OSError:
        return tuple(dirs)
    parts = [p for p in sub.split("/") if p]
    # Leaf first (most specific) down to the root; `_cfs_quota_count` mins over all anyway.
    for i in range(len(parts), 0, -1):
        dirs.append("/sys/fs/cgroup/" + "/".join(parts[:i]))
    return tuple(dirs)


@functools.lru_cache(maxsize=1)
def _cfs_quota_count() -> int | None:
    """Whole cores the cgroup CFS bandwidth quota permits, or `None` when unlimited/unavailable.

    cgroup v2 first (the tightest ``cpu.max`` across every dir in [`cgroup_v2_dirs`]), then
    v1 (``cpu.cfs_quota_us`` / ``cpu.cfs_period_us``).

    Memoized for the same reason as `_read_cgroup_v2_quota`: off a quota'd cgroup this
    otherwise falls all the way through to two more `open` attempts on the v1 paths, on
    every terminal op, to conclude "no quota" again. The affinity mask — which *can* change
    at runtime — is deliberately left un-memoized in `available_cpu_count`.
    """
    v2 = [q for d in cgroup_v2_dirs() if (q := _read_cgroup_v2_quota(d)) is not None]
    if v2:
        return min(v2)  # the most restrictive limit in the hierarchy is the effective one
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:  # cgroup v1
            quota = int(f.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read().strip())
        return _quota_cores(quota, period)
    except (OSError, ValueError):
        pass  # no cgroup v1 quota files (or unparseable) -> fall through to the next probe
    return None


def available_cpu_count() -> int:
    """The number of CPUs this process may actually use — never fewer than 1.

    The minimum of the affinity-mask size (cpuset pin) and the CFS-quota core count (bandwidth
    throttle), floored by `os.cpu_count()` and finally 1. Prefer this over `os.cpu_count()`
    anywhere thread pools or task fan-out are sized, so a container throttled to N cores fans
    out to N — not to the host core count it will never receive (which over-subscribes and
    thrashes the scheduler).

    Examples:
        .. doctest::

            >>> from batcher._internal.hardware import available_cpu_count
            >>> available_cpu_count() >= 1
            True
    """
    candidates = [c for c in (_affinity_count(), _cfs_quota_count()) if c is not None]
    host = os.cpu_count() or 1
    return max(1, min([host, *candidates]))


def _cgroup_throttled_ratio() -> float | None:
    """Share of CFS periods in which this cgroup was throttled, or `None` if unreadable.

    ``cpu.stat``'s ``nr_throttled``/``nr_periods`` are monotonic process-lifetime counters, so
    this is a lifetime average rather than a live reading — enough to answer "is the quota
    binding at all", which is the question that distinguishes a throttled container from an
    under-parallelized query.
    """
    for base in cgroup_v2_dirs():
        try:
            with open(os.path.join(base, "cpu.stat")) as f:
                stat = {
                    parts[0]: int(parts[1])
                    for line in f
                    if len(parts := line.split()) == 2 and parts[1].isdigit()
                }
        except (OSError, ValueError):
            continue
        periods = stat.get("nr_periods", 0)
        if periods > 0:
            return stat.get("nr_throttled", 0) / periods
    return None


@functools.lru_cache(maxsize=1)
def l3_cache_bytes() -> int:
    """Bytes of last-level (L3) cache in this core's cache domain, or `0` if undetectable.

    The physical quantity a broadcast-join threshold actually depends on: a broadcast builds
    one hash table probed from every core, so the strategy wins only while that table stays
    L3-resident. A fixed byte threshold is therefore wrong by the ratio of the real cache to
    the assumed one — ~1 MiB on a small ARM core to 32+ MiB per CCX on an EPYC, an 8x spread
    in both directions. Reading it lets the optimizer size the threshold to the machine.

    Reads ``cpu0``'s own cache hierarchy, so on a chiplet design it reports the L3 *shared by
    the cores that probe one bucket* (per-CCX), which is exactly the residency that matters,
    not the socket total. Linux-only (`/sys`); `0` elsewhere so the caller keeps its default.
    """
    best = 0
    for idx in sorted(glob.glob("/sys/devices/system/cpu/cpu0/cache/index*")):
        try:
            with open(os.path.join(idx, "level")) as f:
                if f.read().strip() != "3":
                    continue
            with open(os.path.join(idx, "size")) as f:
                best = max(best, _parse_cache_size(f.read().strip()))
        except (OSError, ValueError):
            continue
    return best


def _parse_cache_size(raw: str) -> int:
    """Parse a `/sys` cache size like ``"16384K"`` / ``"32M"`` / ``"1G"`` into bytes."""
    raw = raw.strip()
    if not raw:
        return 0
    units = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    mult = units.get(raw[-1].upper(), 1)
    return int(raw[:-1] if mult > 1 else raw) * mult


@functools.lru_cache(maxsize=1)
def machine_memory_bytes() -> int:
    """The memory ceiling this process runs under: `min(host RAM, cgroup limit)`, or `0`.

    The neutral hardware fact behind every memory-sizing decision. `min` because a container's
    cgroup cap — not the host's RAM — is the real ceiling: sizing to host RAM over-commits and
    gets the cgroup OOM-killed. Fixed for the process's lifetime, so memoized.

    (Carbonite's `pressure.total_memory_bytes` computes the same ceiling for its own live
    pressure sensing; this is the copy the layers that cannot import Carbonite — notably Kyber
    — read, so the planner can size to real memory without reaching across a subsystem boundary.)
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


def read_cgroup_bytes(path: str) -> int | None:
    """A byte-valued cgroup file, or `None` when absent, unlimited (`max`), or a v1 sentinel.

    Public within `_internal` because Carbonite's pressure monitor reads the same files.
    The *policy* above it is deliberately duplicated per layer (see `machine_memory_bytes`),
    but this parser is pure file-format mechanics with no layer-specific meaning — one copy.
    """
    try:
        with open(path) as f:
            raw = f.read().strip()
    except OSError:
        return None
    if raw in ("", "max"):
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    # cgroup v1 reports a near-2^63 sentinel when unlimited; treat huge/non-positive as none.
    if value <= 0 or value >= (1 << 62):
        return None
    return value


def cpu_contention() -> dict[str, float]:
    """How much of this machine's CPU is being taken by *other* work, right now.

    Low CPU utilization during a query has two completely different causes with opposite
    fixes: the engine failed to parallelize (fix the plan), or the cores were never available
    because something else was using them (fix the placement, or stop trusting the timing).
    Nothing distinguishes them from inside the process — a query pinned to 1 of 15 cores by a
    noisy co-tenant looks exactly like a query that only asked for 1 core. This reports the
    outside view so a diagnosis can tell them apart.

    Keys: ``load_per_core`` (1-minute run-queue length over [`available_cpu_count`]; ``1.0``
    means the box is exactly saturated, ``>1.0`` oversubscribed) and ``throttled_ratio``
    (share of CFS periods the cgroup quota throttled us). Omits any key the platform cannot
    report, rather than substituting a zero that would read as "no contention".

    Deliberately does **not** try to net out this process's own contribution. Load average
    counts *runnable* tasks while a process can only cheaply report its total thread count,
    most of them asleep; subtracting one from the other would over-subtract an idle worker
    pool into a negative reading. The caller already knows its own CPU use — comparing that
    against `load_per_core` is the sound way to attribute the difference.

    Returns:
        The measurable contention signals, each omitted when unavailable.
    """
    out: dict[str, float] = {}
    with contextlib.suppress(OSError, AttributeError):  # not Linux, or /proc unavailable
        out["load_per_core"] = os.getloadavg()[0] / available_cpu_count()
    throttled = _cgroup_throttled_ratio()
    if throttled is not None:
        out["throttled_ratio"] = throttled
    return out


def process_start_method_context():
    """A `multiprocessing` context using the best start method this platform offers.

    Preference order is ``forkserver`` (cheap children, no fork-safety hazards from the
    parent's threads), then ``fork``, then ``spawn``. Picking the *first available* rather
    than assuming a fallback matters on Windows, which offers only ``spawn``: the previous
    ``forkserver`` -> ``fork`` fallback raised `ValueError` there, so any pipeline reaching
    a process pool crashed on a platform the package otherwise supports.

    Returns:
        A `multiprocessing` context whose start method this platform actually provides.
    """
    import multiprocessing as mp

    available = mp.get_all_start_methods()
    for method in ("forkserver", "fork", "spawn"):
        if method in available:
            return mp.get_context(method)
    return mp.get_context()  # platform default; every platform provides one
