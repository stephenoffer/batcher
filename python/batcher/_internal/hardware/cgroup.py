"""cgroup file-format mechanics — the container limits that override what the host reports.

Everything a process can learn about its *own* restriction, as opposed to the machine it
happens to sit on. A Kubernetes pod or a Ray worker slice sees the host's core count and RAM
through every ordinary API while being throttled to a fraction of both, so each probe here
reads the cgroup hierarchy directly and reports the binding limit.

Pure file parsing plus memoization; the *policy* built on it (how many threads to start, how
much memory to budget) lives in `cpu` and `memory` beside it.
"""

from __future__ import annotations

import functools
import os

from batcher._internal.mathx import ceil_div

__all__ = [
    "cfs_quota_count",
    "cgroup_memory_events",
    "cgroup_pressure",
    "cgroup_throttled_ratio",
    "cgroup_v2_dirs",
    "read_cgroup_bytes",
]


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

    Returns:
        Candidate cgroup v2 directories, mount root first and leaf-most last.
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
    # Leaf first (most specific) down to the root; `cfs_quota_count` mins over all anyway.
    for i in range(len(parts), 0, -1):
        dirs.append("/sys/fs/cgroup/" + "/".join(parts[:i]))
    return tuple(dirs)


@functools.lru_cache(maxsize=1)
def cfs_quota_count() -> int | None:
    """Whole cores the cgroup CFS bandwidth quota permits, or `None` when unlimited/unavailable.

    cgroup v2 first (the tightest ``cpu.max`` across every dir in [`cgroup_v2_dirs`]), then
    v1 (``cpu.cfs_quota_us`` / ``cpu.cfs_period_us``).

    Memoized for the same reason as `_read_cgroup_v2_quota`: off a quota'd cgroup this
    otherwise falls all the way through to two more `open` attempts on the v1 paths, on
    every terminal op, to conclude "no quota" again. The affinity mask — which *can* change
    at runtime — is deliberately left un-memoized in `available_cpu_count`.

    Returns:
        The quota in whole cores, or `None` when no quota binds this process.
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


def _cgroup_stat(base: str, name: str) -> dict[str, int]:
    """Parse a two-column ``key value`` cgroup stat file into a dict (empty when unreadable)."""
    try:
        with open(os.path.join(base, name)) as f:
            return {
                parts[0]: int(parts[1])
                for line in f
                if len(parts := line.split()) == 2 and parts[1].isdigit()
            }
    except (OSError, ValueError):
        return {}


def cgroup_throttled_ratio() -> float | None:
    """Share of CFS periods in which this cgroup was throttled, or `None` if unreadable.

    ``cpu.stat``'s ``nr_throttled``/``nr_periods`` are monotonic process-lifetime counters, so
    this is a lifetime average rather than a live reading — enough to answer "is the quota
    binding at all", which is the question that distinguishes a throttled container from an
    under-parallelized query.

    Returns:
        Throttled period fraction in [0, 1], or `None` when the counters are unavailable.
    """
    for base in cgroup_v2_dirs():
        stat = _cgroup_stat(base, "cpu.stat")
        periods = stat.get("nr_periods", 0)
        if periods > 0:
            return stat.get("nr_throttled", 0) / periods
    return None


def cgroup_pressure() -> dict[str, float]:
    """Live PSI stall shares for cpu, memory, and io — the kernel's own contention verdict.

    Pressure Stall Information (``<cgroup>/{cpu,memory,io}.pressure``) is the one signal that
    says *a resource was short* rather than *a resource was busy*. High CPU utilization is
    healthy; a high CPU stall share means runnable threads sat waiting for a core that never
    came. The same distinction separates a memory-tight query still making progress from one
    thrashing reclaim, and a saturated disk from a fast one merely kept busy. Nothing else in
    the process tells those apart, which is why an over-parallelized plan and a co-tenant
    stealing the box look identical without it.

    Reads the ``some`` line's 10-second average from the leaf-most cgroup that reports one, as
    a fraction in [0, 1]. Keys are ``cpu_stall``, ``memory_stall``, and ``io_stall``. A
    resource whose file is absent (PSI off, cgroup v1, not Linux) is omitted rather than
    reported as zero, which would read as "no pressure" when it means "no measurement".

    Returns:
        The available stall shares, each in [0, 1], keyed by resource.
    """
    out: dict[str, float] = {}
    # Leaf-most first: pressure on our own slice is what binds us, and a parent's figure folds
    # in every sibling's stalls, which we neither cause nor can act on.
    for base in reversed(cgroup_v2_dirs()):
        for resource in ("cpu", "memory", "io"):
            key = f"{resource}_stall"
            if key in out:
                continue
            share = _psi_some_avg10(os.path.join(base, f"{resource}.pressure"))
            if share is not None:
                out[key] = share
        if len(out) == 3:
            break
    return out


def _psi_some_avg10(path: str) -> float | None:
    """The ``some avg10`` share from a PSI file as a fraction, or `None` when unreadable.

    PSI reports ``avg10`` as a *percentage*; this normalizes to [0, 1] so every contention
    signal in the codebase carries the same units.
    """
    try:
        with open(path) as f:
            for line in f:
                if not line.startswith("some "):
                    continue
                for field in line.split():
                    if field.startswith("avg10="):
                        return max(0.0, min(1.0, float(field[6:]) / 100.0))
    except (OSError, ValueError):
        return None
    return None


def cgroup_memory_events() -> dict[str, int]:
    """Reclaim and OOM counters from ``memory.events`` — how close the cgroup came to dying.

    ``max`` counts the times allocation hit the hard limit and had to reclaim; ``oom`` and
    ``oom_kill`` count the times it could not. A run whose ``max`` counter moved spent real
    time in direct reclaim, which reads as unexplained slowness with normal CPU and no spill —
    the memory-side analogue of CFS throttling, and equally invisible without the counter.

    Returns:
        The available event counters (``high``, ``max``, ``oom``, ``oom_kill``), empty when
        the cgroup does not publish them.
    """
    for base in reversed(cgroup_v2_dirs()):
        stat = _cgroup_stat(base, "memory.events")
        if stat:
            return {k: v for k, v in stat.items() if k in ("high", "max", "oom", "oom_kill")}
    return {}


def read_cgroup_bytes(path: str) -> int | None:
    """A byte-valued cgroup file, or `None` when absent, unlimited (`max`), or a v1 sentinel.

    Public within `_internal` because Carbonite's pressure monitor reads the same files.
    The *policy* above it is deliberately duplicated per layer (see `machine_memory_bytes`),
    but this parser is pure file-format mechanics with no layer-specific meaning — one copy.

    Args:
        path: The cgroup file to read.

    Returns:
        The byte value, or `None` when the file is absent, unlimited, or unparseable.
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
