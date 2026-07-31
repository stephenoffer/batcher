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
import time

from batcher._internal.mathx import ceil_div

__all__ = [
    "cfs_quota_count",
    "cgroup_pressure",
    "cgroup_throttled_ratio",
    "cgroup_v2_dirs",
    "read_cgroup_bytes",
    "read_cgroup_stat",
    "read_psi",
]


#: How long a sampled contention reading is reused before the cgroup files are re-read.
#:
#: The probes below are the only ones here that read a *changing* value, so they cannot be
#: memoized for the process lifetime the way the quota and hierarchy probes are. They were
#: therefore not memoized at all — and they run on every terminal op, costing up to twelve
#: `open`+parse round trips (four cgroup levels x three resources) plus `cpu.stat`. On a
#: minimal warm query that measured **~0.2-0.35 ms, about what executing the query itself
#: cost**, which is a real tax on the sub-second-small-query mandate.
#:
#: A short TTL is not a precision trade here, because none of these values is instantaneous
#: to begin with: PSI reports a **10-second** rolling average, and `cpu.stat`'s counters are
#: monotonic over the process lifetime. Re-reading a 10-second average more than four times
#: a second cannot observe anything a quarter-second-old sample missed. 250 ms still
#: oversamples the underlying window by 40x, so every consumer sees the same verdict it did
#: before while the file reads collapse to a handful per second.
_CONTENTION_TTL_S = 0.25


def _ttl_cached(fn):
    """Memoize a zero-argument probe for `_CONTENTION_TTL_S`, `lru_cache`-shaped.

    Exposes `cache_clear` so `reset_hardware_probes` clears it through exactly the same
    mechanism as every `functools.lru_cache` probe in this package — a test faking `/sys`
    resets one way, not two.
    """
    state: dict[str, object] = {"expires": 0.0, "value": None, "primed": False}

    @functools.wraps(fn)
    def wrapper():
        now = time.monotonic()
        if state["primed"] and now < state["expires"]:
            return state["value"]
        value = fn()
        state.update(expires=now + _CONTENTION_TTL_S, value=value, primed=True)
        return value

    def cache_clear() -> None:
        state.update(expires=0.0, value=None, primed=False)

    wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]
    return wrapper


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


def read_cgroup_stat(base: str, name: str) -> dict[str, int]:
    """Parse a two-column ``key value`` cgroup stat file into a dict (empty when unreadable).

    Public within `_internal` because three unrelated readers want the same parser:
    `cgroup_throttled_ratio` here, Carbonite's page-cache term (``memory.stat``), and its
    OOM-kill / reclaim-throttle counters (``memory.events``). Those files share exactly this
    format, and a second copy of the parser is how one of them ends up handling a malformed
    line differently from the others.

    Args:
        base: The cgroup directory to read from.
        name: The stat file's name within it.

    Returns:
        The parsed counters, empty when the file is absent or unreadable.
    """
    try:
        with open(os.path.join(base, name)) as f:
            return {
                parts[0]: int(parts[1])
                for line in f
                if len(parts := line.split()) == 2 and parts[1].isdigit()
            }
    except (OSError, ValueError):
        return {}


@_ttl_cached
def cgroup_throttled_ratio() -> float | None:
    """Share of CFS periods in which this cgroup was throttled, or `None` if unreadable.

    ``cpu.stat``'s ``nr_throttled``/``nr_periods`` are monotonic process-lifetime counters, so
    this is a lifetime average rather than a live reading — enough to answer "is the quota
    binding at all", which is the question that distinguishes a throttled container from an
    under-parallelized query.

    Sampled at most every `_CONTENTION_TTL_S`; see that constant for why a lifetime average
    loses nothing to a quarter-second-old reading.

    Returns:
        Throttled period fraction in [0, 1], or `None` when the counters are unavailable.
    """
    for base in cgroup_v2_dirs():
        stat = read_cgroup_stat(base, "cpu.stat")
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

    Sampled at most every `_CONTENTION_TTL_S`; see that constant for why a 10-second rolling
    average loses nothing to a quarter-second-old reading. A fresh dict per call, because the
    memo must not hand the same mutable mapping to every caller.

    Returns:
        The available stall shares, each in [0, 1], keyed by resource.
    """
    return dict(_cgroup_pressure_sampled())


@_ttl_cached
def _cgroup_pressure_sampled() -> dict[str, float]:
    """The TTL-sampled PSI read behind `cgroup_pressure` — never handed out directly."""
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
    """The ``some avg10`` share from a PSI file as a fraction, or `None` when unreadable."""
    return read_psi(path).get("some_avg10")


def read_psi(path: str) -> dict[str, float]:
    """Every ``some``/``full`` stall average in one PSI file, as fractions in [0, 1].

    PSI reports each average as a *percentage*; this normalizes to [0, 1] so every contention
    signal in the codebase carries the same units. Keys are ``<line>_avg<window>`` —
    ``some_avg10``, ``full_avg10``, ``some_avg60``, and so on.

    The ``full`` line is the one that matters for memory and has no reader anywhere else:
    ``some`` means *at least one* task stalled, which a healthy memory-tight process does
    constantly as it faults pages in, while ``full`` means **every** runnable task was stalled
    at once — the whole cgroup made no progress. A container thrashing reclaim on its way to an
    OOM kill shows a rising ``full`` share for seconds beforehand, and that is the only warning
    the kernel gives that is early enough to act on: `memory.current` is already at the limit by
    then, because the limit is what the reclaim is defending.

    Args:
        path: The PSI file to read (``<cgroup>/memory.pressure``, ``/proc/pressure/memory``).

    Returns:
        The available stall shares, empty when PSI is off, the file is absent, or this is not
        Linux. Empty is deliberately distinct from all-zeros, which would read as "measured, no
        pressure" for a kernel that measured nothing.
    """
    out: dict[str, float] = {}
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except OSError:
        return out
    for line in lines:
        fields = line.split()
        if not fields or fields[0] not in ("some", "full"):
            continue
        for field in fields[1:]:
            name, _, raw = field.partition("=")
            if not name.startswith("avg"):
                continue
            try:
                out[f"{fields[0]}_{name}"] = max(0.0, min(1.0, float(raw) / 100.0))
            except ValueError:
                continue
    return out


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
