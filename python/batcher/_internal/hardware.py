"""Effective hardware detection — the CPU parallelism the process may actually use.

`os.cpu_count()` reports the *host's* logical cores, which over-counts inside a container:
a Kubernetes/Ray pod is throttled by a cgroup CPU quota (CFS bandwidth) or pinned to a
cpuset, so sizing thread pools and task fan-out to the host count over-subscribes — the
scheduler thrashes on context switches for cores the process will never get. This resolves
the real budget: the CPU-affinity mask (cpuset) capped by the CFS quota, falling back to the
host count when neither is discoverable. A neutral utility (any layer may import `_internal`).
"""

from __future__ import annotations

import os

__all__ = ["available_cpu_count", "cgroup_v2_dirs"]


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
        return max(1, -(-quota // period))  # ceil-div
    return None


def _read_cgroup_v2_quota(base: str) -> int | None:
    """Cores the cgroup v2 ``<base>/cpu.max`` permits, or `None` when unlimited/absent.

    ``cpu.max`` is ``"<quota> <period>"``; a ``max`` quota means unlimited.
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


def cgroup_v2_dirs() -> list[str]:
    """Every cgroup v2 dir whose ``cpu.max`` can bind this process: the mount root and each
    ancestor from the process's own leaf (``/proc/self/cgroup``) up to it.

    The root and leaf coincide inside a K8s pod (a cgroup *namespace* maps the pod's cgroup to
    the mount root) but diverge for a process in a *delegated* cgroup with no namespace — a Ray
    worker under a systemd slice, a nested container. cgroup v2 enforces the CFS bandwidth quota
    at **every** level, so the effective limit is the tightest ``cpu.max`` anywhere in the chain:
    a quota set on a parent slice rather than the leaf would be missed by checking only the ends.
    Walking the full ancestry and taking the minimum is correct for any topology.
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
        return dirs
    parts = [p for p in sub.split("/") if p]
    # Leaf first (most specific) down to the root; `_cfs_quota_count` mins over all anyway.
    for i in range(len(parts), 0, -1):
        dirs.append("/sys/fs/cgroup/" + "/".join(parts[:i]))
    return dirs


def _cfs_quota_count() -> int | None:
    """Whole cores the cgroup CFS bandwidth quota permits, or `None` when unlimited/unavailable.

    cgroup v2 first (the tightest ``cpu.max`` across every dir in [`cgroup_v2_dirs`]), then
    v1 (``cpu.cfs_quota_us`` / ``cpu.cfs_period_us``).
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
        pass
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
