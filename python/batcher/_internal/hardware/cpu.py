"""The CPU budget this process really has, and how much of it something else is taking.

`os.cpu_count()` reports the *host's* logical cores, which over-counts inside a container: a
Kubernetes/Ray pod is throttled by a cgroup CPU quota (CFS bandwidth) or pinned to a cpuset, so
sizing thread pools and task fan-out to the host count over-subscribes — the scheduler thrashes
on context switches for cores the process will never get. `available_cpu_count` resolves the
real budget: the CPU-affinity mask capped by the CFS quota, falling back to the host count when
neither is discoverable.

The second half answers the question a core count cannot: whether those cores are *available*.
A query pinned to one busy core and a query that failed to parallelize look identical from
inside the process, and they have opposite fixes.
"""

from __future__ import annotations

import contextlib
import os

from batcher._internal.hardware.cgroup import (
    cfs_quota_count,
    cgroup_pressure,
    cgroup_throttled_ratio,
)

__all__ = [
    "INFERENCE_INFLIGHT_DEPTH_MAX",
    "available_cpu_count",
    "cpu_contention",
    "cpu_oversubscription",
    "process_start_method_context",
]

# Hard ceiling on how many partitions an inference actor keeps in flight at once (submit-ahead
# depth), and the mirror ceiling on the intra-worker autobatch pipeline. One home in a neutral
# layer both the ML autobatcher (`ml.gpu`, layer 6) and the distributed actor pool (`dist`,
# layer 4) import, rather than two copies that drift — `dist` cannot import `ml`, so a shared
# copy was the ONLY correct way to share this, and it used to be pasted instead. Past ~16
# in-flight the GPU is already saturated and deeper submit-ahead only grows resident memory.
INFERENCE_INFLIGHT_DEPTH_MAX = 16


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

    Returns:
        The effective logical-core budget, at least 1.
    """
    candidates = [c for c in (_affinity_count(), cfs_quota_count()) if c is not None]
    host = os.cpu_count() or 1
    return max(1, min([host, *candidates]))


def cpu_contention() -> dict[str, float]:
    """How much of this machine's CPU is being taken by *other* work, right now.

    Low CPU utilization during a query has two completely different causes with opposite
    fixes: the engine failed to parallelize (fix the plan), or the cores were never available
    because something else was using them (fix the placement, or stop trusting the timing).
    Nothing distinguishes them from inside the process — a query pinned to 1 of 15 cores by a
    noisy co-tenant looks exactly like a query that only asked for 1 core. This reports the
    outside view so a diagnosis can tell them apart.

    Keys: ``load_per_core`` (1-minute run-queue length over [`available_cpu_count`]; ``1.0``
    means the box is exactly saturated, ``>1.0`` oversubscribed), ``throttled_ratio`` (share
    of CFS periods the cgroup quota throttled us), and the PSI stall shares ``cpu_stall``,
    ``memory_stall``, and ``io_stall`` when the kernel publishes them. Omits any key the
    platform cannot report, rather than substituting a zero that would read as "no
    contention".

    The stall shares are the sharper signal of the set. Load average and throttling both say a
    resource was *busy*; a stall share says a runnable thread was *waiting for it*, which is
    the thing worth acting on. They are also the only one of these that attributes memory and
    I/O contention rather than CPU alone.

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
    throttled = cgroup_throttled_ratio()
    if throttled is not None:
        out["throttled_ratio"] = throttled
    out.update(cgroup_pressure())
    return out


def cpu_oversubscription() -> float:
    """How far past this process's core budget the machine is committed, as a multiplier.

    ``1.0`` means the run queue exactly matches the cores this process may use; ``2.0`` means
    twice as much runnable work as there are cores to run it, so every thread the engine adds
    lengthens the queue instead of shortening the query. The single number worth gating
    fan-out on, distilled from whichever signals the platform offers: the run-queue ratio, the
    CFS throttle share, and the CPU stall share.

    Returns `1.0` — "assume the budget is real" — when nothing is measurable, so a platform
    without these counters plans exactly as it did before.

    Returns:
        The oversubscription multiplier, at least 1.0.
    """
    signals = cpu_contention()
    ratio = max(1.0, signals.get("load_per_core", 1.0))
    # Throttling and stalling both mean a share of wall time bought no progress, so the cores
    # effectively available shrink by that share; 0.9 stalled is a 10x effective loss, capped
    # so a pathological reading cannot collapse fan-out to a single thread.
    lost = max(signals.get("throttled_ratio", 0.0), signals.get("cpu_stall", 0.0))
    ratio /= max(0.1, 1.0 - min(0.9, lost))
    return max(1.0, ratio)


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
