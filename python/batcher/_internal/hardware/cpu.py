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
    "cpu_thermal_events",
    "cpu_thermal_throttle_count",
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


# Slurm's per-task CPU allocation, most specific first. `SLURM_CPUS_PER_TASK` is set when the
# job asked with `--cpus-per-task`; `SLURM_CPUS_ON_NODE` is the node's whole share of the
# allocation and is the fallback for a job that did not.
_SLURM_CPU_VARS = ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE")


def _slurm_expansion_min(raw: str) -> int | None:
    """The smallest per-node count in a Slurm `"4(x2),8"` expansion, or `None` if unparseable.

    A heterogeneous job's `SLURM_CPUS_ON_NODE` is a run-length list: `"4(x2),8"` means two
    nodes granted 4 cores and one granted 8. Which entry describes *this* node is not
    derivable from the variable alone, so the minimum is taken, for the reason every binding
    figure in this package takes the weakest: under-parallelizing costs throughput, while
    over-parallelizing on the node that got the small grant is what oversubscribes a shared
    node and gets the job killed at a site with enforcement.

    Previously any value with a repeat count fell through entirely, so a heterogeneous
    allocation had *no* Slurm bound at all and sized to the affinity mask — the whole node —
    which is precisely the failure the Slurm bound exists to prevent, arriving on exactly the
    layouts that need it most.
    """
    counts: list[int] = []
    for part in raw.split(","):
        head = part.strip().split("(", 1)[0].strip()
        if not head.isdigit():
            return None  # an unrecognized shape: no bound beats a wrong one
        value = int(head)
        if value > 0:
            counts.append(value)
    return min(counts) if counts else None


def _slurm_cpu_count() -> int | None:
    """Cores this Slurm allocation granted on this node, or `None` off Slurm.

    A container is confined by cgroups, which the affinity mask and CFS quota already
    report. A Slurm allocation is not, unless the site configured `task/cgroup` confinement
    — and plenty of HPC sites do not. There the affinity mask reports every core on a
    shared login-class node, so sizing to it fans a job allocated 8 cores out to 128
    threads: it oversubscribes the node, steals from the co-tenants Slurm placed there, and
    at a site with enforcement is exactly what gets the job killed. Slurm publishes the real
    grant in the environment, so this reads it as one more upper bound rather than trusting
    a mask the scheduler never narrowed.
    """
    for var in _SLURM_CPU_VARS:
        raw = os.environ.get(var, "").strip()
        if not raw:
            continue
        # A plain int is the common case; `_slurm_expansion_min` also handles it, so there is
        # one parse rather than a fast path and a fallback that can disagree.
        n = _slurm_expansion_min(raw)
        if n is not None and n > 0:
            return n
    return None


def available_cpu_count() -> int:
    """The number of CPUs this process may actually use — never fewer than 1.

    The minimum of the affinity-mask size (cpuset pin), the CFS-quota core count (bandwidth
    throttle), and the Slurm allocation, floored by `os.cpu_count()` and finally 1. Prefer
    this over `os.cpu_count()` anywhere thread pools or task fan-out are sized, so a
    container throttled to N cores fans out to N — not to the host core count it will never
    receive (which over-subscribes and thrashes the scheduler).

    Examples:
        .. doctest::

            >>> from batcher._internal.hardware import available_cpu_count
            >>> available_cpu_count() >= 1
            True

    Returns:
        The effective logical-core budget, at least 1.
    """
    bounds = (_affinity_count(), cfs_quota_count(), _slurm_cpu_count())
    candidates = [c for c in bounds if c is not None]
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
    means there is exactly as much runnable work as this process has cores, ``>1.0`` more),
    ``host_load_per_core`` (the same run queue over the *host's* core count),
    ``throttled_ratio`` (share of CFS periods the cgroup quota throttled us), and the PSI
    stall shares ``cpu_stall``, ``memory_stall``, and ``io_stall`` when the kernel publishes
    them. Omits any key the platform cannot report, rather than substituting a zero that
    would read as "no contention".

    **The two load figures answer different questions and only one is an oversubscription
    ratio.** `os.getloadavg()` counts runnable tasks across the whole host; Linux publishes no
    per-cgroup equivalent. Dividing that host-wide numerator by this process's *slice* is a
    category error the moment a cgroup quota or cpuset narrows the slice: a 4-core container
    on a 128-core host at a healthy 50% load reads ``load_per_core = 16``, which
    [`cpu_oversubscription`] then took as a sixteen-fold oversubscription and collapsed the
    fan-out to a single thread — on a box that was not busy. `host_load_per_core` divides the
    host-wide numerator by the host-wide denominator, so it is a true saturation ratio at any
    container size, and it is what the oversubscription verdict is built from.

    `load_per_core` is kept because it answers the question a *diagnostic* asks — "is there
    more runnable work here than my share of the cores" — and because on a process that owns
    the whole machine the two are identical.

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
        runnable = os.getloadavg()[0]
        out["load_per_core"] = runnable / available_cpu_count()
        out["host_load_per_core"] = runnable / max(1, os.cpu_count() or 1)
    throttled = cgroup_throttled_ratio()
    if throttled is not None:
        out["throttled_ratio"] = throttled
    out.update(cgroup_pressure())
    return out


def cpu_oversubscription() -> float:
    """How far past this process's core budget the machine is committed, as a multiplier.

    ``1.0`` means the machine's run queue exactly matches its cores; ``2.0`` means twice as
    much runnable work as there are cores to run it, so every thread the engine adds
    lengthens the queue instead of shortening the query. The single number worth gating
    fan-out on, distilled from whichever signals the platform offers: the run-queue ratio, the
    CFS throttle share, and the CPU stall share.

    Built from ``host_load_per_core`` rather than ``load_per_core``, because the run-queue
    numerator is host-wide and only the host-wide denominator makes it a ratio — see
    [`cpu_contention`]. Under the previous per-slice denominator a small container on a large,
    *idle* host reported an oversubscription equal to the ratio of the two, so the engine
    throttled its own fan-out hardest on precisely the deployment where it had the most
    headroom. The throttle and stall shares are cgroup-scoped already and are unaffected.

    Returns `1.0` — "assume the budget is real" — when nothing is measurable, so a platform
    without these counters plans exactly as it did before.

    Returns:
        The oversubscription multiplier, at least 1.0.
    """
    signals = cpu_contention()
    # `load_per_core` is the fallback only for a caller that supplied a signal map without the
    # host-scoped key; when both are present the host-scoped one is the honest ratio.
    queue = signals.get("host_load_per_core", signals.get("load_per_core", 1.0))
    ratio = max(1.0, queue)
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


#: Where Linux publishes the CPU's own thermal-throttle event counters. One directory per
#: logical CPU; `core_throttle_count` and `package_throttle_count` are monotonic counts of
#: times the silicon clamped itself because it was too hot.
#:
#: Present only on bare metal. A virtualized instance does not expose it at all — the
#: hypervisor owns the thermal domain and does not share it — which is exactly why this
#: reports `None` rather than zero there. "Not measurable" and "not throttling" look the same
#: to a caller that conflates them, and they are opposite facts about whether the reading may
#: be trusted.
_THERMAL_THROTTLE_GLOB = "/sys/devices/system/cpu/cpu[0-9]*/thermal_throttle"

#: The counter total at the last `cpu_thermal_events` call, so each call reports the events
#: since the previous one rather than since boot. A count since boot answers "has this machine
#: ever been hot", which is not the question — a box that throttled during last month's heatwave
#: is not throttling now, and treating it as though it were would suppress a learned CPU share
#: forever.
_THERMAL_BASELINE: int | None = None

#: The expanded per-CPU thermal-throttle directories, with the pattern they came from.
#: Keyed on the pattern so that repointing `_THERMAL_THROTTLE_GLOB` — which is how the tests
#: substitute a `tmp_path` tree — re-expands instead of serving the previous tree's answer. A
#: cache a test has to know to clear is a cache that will one day be read stale by a test that
#: does not.
_THERMAL_DIRS: tuple[str, tuple[str, ...]] | None = None


def _thermal_throttle_dirs() -> tuple[str, ...]:
    """The per-CPU `thermal_throttle` directories, globbed once per process.

    The *counters* have to be re-read on every call — the caller wants a delta, and
    `plan.feedback` takes a median over per-run readings, so a run that reported nothing
    because the sample was skipped would drag that median to zero on a genuinely
    throttling box. The directory *list* is a different matter: it is fixed by the CPU
    topology, and expanding it per call put a `glob` of `/sys/devices/system/cpu` on the
    critical path of every query.

    That is not a rounding error at this size. Profiled over 300 warm 20,000-row queries,
    `glob` was **0.119 s of the 0.120 s** the whole probe cost — the 32 counter reads it
    guards are almost free by comparison — and the probe itself was 9.8% of the query,
    against 5.6% for the engine call. A filesystem sweep to find files whose names cannot
    change is the entire expense.

    CPU hotplug would invalidate this, and nothing here re-expands. That is the same
    assumption `usable_cores` already makes about the core count, and a CPU appearing
    mid-process would at worst leave its counter out of a diagnostic sum.
    """
    global _THERMAL_DIRS
    if _THERMAL_DIRS is None or _THERMAL_DIRS[0] != _THERMAL_THROTTLE_GLOB:
        import glob

        _THERMAL_DIRS = (_THERMAL_THROTTLE_GLOB, tuple(sorted(glob.glob(_THERMAL_THROTTLE_GLOB))))
    return _THERMAL_DIRS[1]


def cpu_thermal_throttle_count() -> int | None:
    """Total hardware thermal-throttle events across every CPU since boot, or `None`.

    `None` means the counters are unreadable — a virtualized guest, a non-Linux host, or a
    kernel without the `therm_throt` driver. That is deliberately distinct from `0`, which is
    a real reading that the silicon has never clamped itself.

    Returns:
        The summed event count, or `None` when no CPU exposes the counters.
    """
    total = 0
    seen = False
    for directory in _thermal_throttle_dirs():
        for name in ("core_throttle_count", "package_throttle_count"):
            try:
                with open(f"{directory}/{name}") as handle:
                    total += int(handle.read().strip())
                seen = True
            except (OSError, ValueError):
                continue
    return total if seen else None


def cpu_thermal_events() -> int:
    """Hardware thermal-throttle events since the previous call, `0` when unmeasurable.

    The *delta* rather than the total, because the question a caller asks is whether the CPU
    is clamping itself **now**, and a since-boot count answers whether it ever has. The first
    call establishes the baseline and reports `0`: there is no earlier reading to difference
    against, and guessing that everything since boot happened in this window would report a
    long-cool machine as burning.

    A counter that goes backwards (a CPU hot-unplugged, the baseline captured against a wider
    set) reports `0` and re-baselines, rather than a negative count.

    Returns:
        Events observed since the last call, or `0` when the counters cannot be read.
    """
    global _THERMAL_BASELINE

    total = cpu_thermal_throttle_count()
    if total is None:
        return 0
    previous = _THERMAL_BASELINE
    _THERMAL_BASELINE = total
    if previous is None or total < previous:
        return 0
    return total - previous
