"""Live memory-pressure sensing — Carbonite's view of how full RAM is.

The resource governor needs to know not just the machine's total RAM but how much
is *available right now* (other processes, the OS cache, this query's own working
set) to size its envelope and decide when to spill. `PressureMonitor` reads that
from `psutil` when present and degrades to the total-RAM figure (`os.sysconf`)
otherwise, so the engine runs with or without the optional dependency.

Pressure is classified against the configured soft/hard limits into a small ladder
of `PressureLevel`s (the architecture's three-threshold model); Core consumes the
level to throttle / spill / pause. The monitor only *measures* — it never acts.
"""

from __future__ import annotations

import functools
import os
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

from batcher.config import Config, active_config

if TYPE_CHECKING:
    from batcher.metadata import MetadataHub

__all__ = [
    "PressureLevel",
    "PressureMonitor",
    "hysteresis_alpha_from_flap",
    "load_flap_rate",
    "record_flap_rate",
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
_SAMPLE_TTL_SECONDS = 0.05

# Process-constant host RAM (`SC_PAGE_SIZE * SC_PHYS_PAGES`), memoized on first read: it
# cannot change for the process's lifetime, so re-running the two syscalls per call is
# pure waste. `None` until first read / when sysconf is unavailable.
_host_ram_bytes: int | None = None

# Single-slot TTL cache for the live available-bytes reading: `(monotonic_deadline, value)`.
_available_cache: tuple[float, int] | None = None


def reset_memory_sampling() -> None:
    """Drop the memoized host RAM and the live-reading TTL cache. For tests that
    patch the underlying OS readers and need the next sample to re-read them."""
    global _host_ram_bytes, _available_cache
    _host_ram_bytes = None
    _available_cache = None


# Learned-parameter namespace + key for the measured pressure-level flap rate (fraction
# of samples that reversed direction). One process-wide figure; the hysteresis adapts to it.
_FLAP_NS = "carbonite.pressure_flap"
_FLAP_KEY = "rate"
# The static default de-escalation weight (kept in sync with `PressureMonitor._EWMA_ALPHA`).
_DEFAULT_ALPHA = 0.5
# How strongly a high flap rate stiffens the hysteresis: a fully-flapping history (rate 1.0)
# cuts the de-escalation weight to `1 - _FLAP_STIFFEN` of the default (a much stickier level).
_FLAP_STIFFEN = 0.8


class PressureLevel(IntEnum):
    """Memory-pressure severity, ordered so callers can compare with ``>=``.

    Thresholds (fraction of the budget used) follow the architecture's envelope
    model: normal operation below the soft limit, spill between soft and hard,
    and an emergency stop once almost no headroom remains.
    """

    NORMAL = 0  # below the soft limit — run freely
    ELEVATED = 1  # approaching the soft limit — prefer spill-friendly plans
    SPILL = 2  # past the soft limit — spill stateful operators to disk
    CRITICAL = 3  # past the hard limit — pause producers, only drain


@functools.lru_cache(maxsize=1)
def _cgroup_limit_bytes() -> int | None:
    """The container memory limit from cgroup v2 (`memory.max`) or v1
    (`memory.limit_in_bytes`), or `None` when unlimited / not in a cgroup.

    A container's cgroup cap is the *real* ceiling — the host's RAM is not — so
    honoring it is what stops the engine over-admitting and getting OOM-killed by
    the kernel (C25).

    Cached for the process: the cgroup cap is fixed for a container's lifetime, while
    this is read on every admission check — re-opening `memory.max` per query is pure
    hot-path I/O. (The *current* usage, which does change, is read live and uncached.)

    Like the CPU quota, the limit can be set at any level of the cgroup v2 hierarchy — the
    process's own leaf (a namespaced pod), a parent slice (a non-namespaced Ray worker), or
    the mount root — so the effective cap is the tightest `memory.max` across the whole
    ancestry (`cgroup_v2_dirs`), not just the root. v1 keeps its single well-known path.
    """
    from batcher._internal.hardware import cgroup_v2_dirs

    limits = [
        v for d in cgroup_v2_dirs() if (v := _read_cgroup_bytes(os.path.join(d, "memory.max")))
    ]
    if limits:
        return min(limits)
    return _read_cgroup_bytes("/sys/fs/cgroup/memory/memory.limit_in_bytes")  # cgroup v1


def _read_cgroup_bytes(path: str) -> int | None:
    """A byte-valued cgroup file, or `None` when absent, unlimited (`max`/sentinel), or empty."""
    try:
        with open(path) as f:
            raw = f.read().strip()
    except OSError:
        return None
    if raw in ("max", ""):
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    # cgroup v1 reports a sentinel near 2^63 when unlimited; treat huge/non-positive as none.
    if value <= 0 or value >= (1 << 62):
        return None
    return value


def _cgroup_file_cache_bytes() -> int:
    """Page cache charged to this cgroup — `file` (v2) or `total_cache` (v1); 0 if unknown."""
    for path, key in (
        ("/sys/fs/cgroup/memory.stat", "file"),
        ("/sys/fs/cgroup/memory/memory.stat", "total_cache"),
    ):
        try:
            with open(path) as f:
                lines = f.read().splitlines()
        except OSError:
            continue
        for line in lines:
            field, _, raw = line.partition(" ")
            if field == key:
                try:
                    return max(0, int(raw))
                except ValueError:
                    return 0
    return 0


def _cgroup_current_bytes() -> int | None:
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

    Not a pressure signal on its own; see `_cgroup_current_bytes`.
    """
    for path in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        value = _read_cgroup_bytes(path)
        if value is not None:
            return value
    return None


def _cap_to_cgroup_headroom(host_available: int) -> int:
    """Clamp a host-wide available figure to what this container may still allocate.

    `psutil`/`/proc` report the *machine's* free RAM, but a cgroup-limited container (the norm
    under Kubernetes/Ray) OOMs at `memory.max`, not at host exhaustion — on a 184 GB host an
    8 GB container would otherwise read ~180 GB free and over-admit into a kill. The real
    headroom is `limit - current`, where `current` excludes the reclaimable page cache (see
    `_cgroup_current_bytes`) — cache the kernel will evict on demand is headroom, not usage.
    Take the smaller of that and the host figure. No cgroup cap (bare metal / unlimited)
    leaves the reading untouched.
    """
    limit = _cgroup_limit_bytes()
    if limit is None:
        return host_available
    current = _cgroup_current_bytes()
    if current is None:
        return host_available
    return min(host_available, max(0, limit - current))


def _process_rss_bytes() -> int | None:
    """This process's resident set size (RSS) via `psutil`, or `None` without it.

    RSS captures the engine's true footprint — the Flight `PartitionStore`, pyarrow
    buffers, everything — not just the buffer pool's accounted reservations."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def total_memory_bytes() -> int:
    """The memory ceiling: the min of host RAM and any cgroup/container limit.

    Falls back to `MemoryConfig.default_total_bytes` (one home for the fallback)
    when the OS won't report host RAM. Host RAM and the cgroup cap are both fixed for
    the process's lifetime, so both are memoized — this is read on every admission /
    pressure check, and re-running the syscalls each time is hot-path waste.
    """
    global _host_ram_bytes
    host = _host_ram_bytes
    if host is None:
        try:
            host = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            _host_ram_bytes = host
        except (ValueError, OSError, AttributeError):
            host = active_config().memory.default_total_bytes  # not memoized (config-derived)
    cgroup = _cgroup_limit_bytes()
    return min(host, cgroup) if cgroup is not None else host


def _proc_meminfo_available() -> int | None:
    """`MemAvailable` from `/proc/meminfo` (Linux), or `None` if unreadable.

    The without-psutil fallback so memory governance still senses real pressure on
    Linux containers where the optional dep isn't installed (C26)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024  # kB → bytes
    except (OSError, ValueError, IndexError):
        return None
    return None


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """One reading of memory state, in bytes (plus the fraction of budget used)."""

    total: int
    available: int
    used_fraction: float


class PressureMonitor:
    """Samples available memory and classifies it against the configured limits.

    Reads `psutil.virtual_memory().available` when the optional dependency is
    installed — the true free figure that accounts for other tenants on the box —
    and falls back to total RAM otherwise. The soft/hard limits come from
    `MemoryConfig` (default 0.85 / 0.90).
    """

    # Default smoothing factor for the de-escalation hysteresis (weight on the newest
    # reading). 0.5 relaxes the level over a few samples; escalation is never smoothed.
    _EWMA_ALPHA = 0.5

    def __init__(
        self, config: Config | None = None, *, hysteresis_alpha: float | None = None
    ) -> None:
        self._config = config or active_config()
        # Exponentially-weighted history of the used fraction, for the asymmetric
        # hysteresis in `level()`. `None` until the first reading.
        self._ewma: float | None = None
        # De-escalation smoothing weight. Lower = the level lingers longer on a falling
        # edge (heavier hysteresis), which the manager sets from a *measured* flap rate:
        # a channel that has been observed to flap SPILL↔NORMAL gets a stickier level so
        # its shuffle credit window stops oscillating. Clamped to (0, 1]; `None` keeps the
        # static default. Purely damps *when* the engine spills/throttles — never a result.
        if hysteresis_alpha is None:
            self._alpha = self._EWMA_ALPHA
        else:
            self._alpha = min(1.0, max(1e-3, hysteresis_alpha))

    def snapshot(self) -> MemorySnapshot:
        """Take a current reading of total/available memory and budget usage."""
        total = total_memory_bytes()
        available = self._available_bytes(total)
        used_fraction = 1.0 - (available / total) if total else 1.0
        return MemorySnapshot(total=total, available=available, used_fraction=used_fraction)

    def available_bytes(self) -> int:
        """Bytes of RAM available right now (psutil) or total RAM as a fallback."""
        return self._available_bytes(total_memory_bytes())

    def budget_bytes(self) -> int:
        """The soft envelope: the share of total RAM the engine aims to stay under."""
        return int(total_memory_bytes() * self._config.memory.soft_limit)

    def envelope_bytes(self) -> int:
        """The raw memory a query may draw on: the configured hard cap if set
        (honors a container/cgroup limit), else the RAM available right now.

        Sampled once per query by the `ResourceManager` and threaded through the
        `ResourceContext` so admission, spill, and reserve all reason about the
        same figure instead of each re-sampling live free RAM.
        """
        mem = self._config.memory
        if mem.max_memory_bytes is not None:
            return mem.max_memory_bytes
        return self.available_bytes()

    def level(self) -> PressureLevel:
        """Classify the **engine's** envelope usage against the soft/hard limits.

        The soft/hard limits are fractions of the *engine budget*, so the pressure
        level measures how full Carbonite's own envelope is (buffer-pool `used` /
        `limit`) — not how full the whole machine is. When no pool has been created
        yet, falls back to the machine's used fraction so a standalone monitor still
        reports something sensible.

        **Asymmetric hysteresis.** The level escalates *instantly* on the raw reading
        (a real reservation is real pressure — never delay protective spill) but
        de-escalates only as an EWMA of recent readings relaxes. Classifying on
        ``max(raw, ewma)`` gives exactly that: on a rising edge ``raw > ewma`` so the
        raw value drives the decision; on a falling edge ``raw < ewma`` so the lagging
        average holds the level up for a few samples. This stops a transient spike
        from flapping SPILL↔NORMAL and oscillating the shuffle's AIMD credit window,
        without ever under-reacting to growing pressure. Stateful by design (it
        updates the EWMA); it still never *acts*.
        """
        raw = self._engine_used_fraction()
        prev = self._ewma if self._ewma is not None else raw
        level = self._classify(max(raw, prev))
        self._ewma = self._alpha * raw + (1.0 - self._alpha) * prev
        return level

    def classify(self) -> PressureLevel:
        """The current pressure level **without** advancing the hysteresis average.

        `level()` is the *sampler*: each call folds one reading into the de-escalation
        EWMA, so its hysteresis is measured in samples. That only holds if exactly one
        component samples it per round. Readers that merely want to know the current level
        — morsel sizing, the shared-memory mirror check — must use this instead, or the
        EWMA advances several steps per round, collapses toward the raw reading, and the
        anti-flap smoothing the design exists for is defeated.
        """
        raw = self._engine_used_fraction()
        prev = self._ewma if self._ewma is not None else raw
        return self._classify(max(raw, prev))

    def _classify(self, used: float) -> PressureLevel:
        """Bucket a used-fraction into a level. Pure."""
        mem = self._config.memory
        if used >= mem.hard_limit:
            return PressureLevel.CRITICAL
        if used >= mem.soft_limit:
            return PressureLevel.SPILL
        if used >= mem.soft_limit * 0.9:
            return PressureLevel.ELEVATED
        return PressureLevel.NORMAL

    @staticmethod
    def _engine_used_fraction() -> float:
        """Fraction of the memory ceiling in use, by whichever measure is highest.

        Takes the MAX of the engine's reserved buffer-pool envelope and the process's
        *actual* footprint (the cgroup's unreclaimable usage, else RSS). Memory the pool
        does not track — the in-memory Flight shuffle `PartitionStore`, off-pool pyarrow
        buffers — therefore cannot let the monitor report NORMAL while the kernel
        OOM-kills a shuffle-heavy worker. The footprint term deliberately excludes the
        page cache: it is reclaimable, so counting it would report a box that has merely
        *read files* as one under pressure — which is not a safe over-read but a silent
        throttle (it halves every morsel). Falls back to the machine's used fraction when
        neither a pool nor a live reading exists.
        """
        from batcher.carbonite.memory.pool import current_process_pool

        candidates: list[float] = []
        pool = current_process_pool()
        if pool is not None and pool.limit > 0:
            candidates.append(pool.used / pool.limit)
        total = total_memory_bytes()
        if total:
            footprint = _cgroup_current_bytes() or _process_rss_bytes()
            if footprint is not None:
                candidates.append(footprint / total)
        if candidates:
            return max(candidates)
        # No pool and no process footprint reading — fall back to the machine fraction.
        if not total:
            return 1.0
        try:
            import psutil
        except ImportError:
            return 0.0  # no live reading and no pool — assume headroom
        return 1.0 - int(psutil.virtual_memory().available) / total

    @staticmethod
    def _available_bytes(total: int) -> int:
        """Available RAM, served from a short-TTL sample (see `_SAMPLE_TTL_SECONDS`).

        The underlying `psutil.virtual_memory()` (or `/proc` fallback) read is the
        single most expensive step in a per-query Carbonite decision; a change is still
        picked up within one TTL window, so amortizing the read across a query's
        decisions and a session's queries is free accuracy-wise (a stale reading only
        makes admission slightly conservative, never over-admits).
        """
        global _available_cache
        now = time.monotonic()
        cached = _available_cache
        if cached is not None and now < cached[0]:
            return min(cached[1], total)
        value = PressureMonitor._read_available_bytes()
        _available_cache = (now + _SAMPLE_TTL_SECONDS, value)
        return min(value, total)

    @staticmethod
    def _read_available_bytes() -> int:
        """One live reading of *available* RAM (uncached), capped to the container's cgroup
        headroom. `psutil` when present, else Linux `/proc/meminfo`, else a large sentinel
        meaning 'assume headroom'."""
        try:
            import psutil

            host = int(psutil.virtual_memory().available)
        except ImportError:
            # No psutil: read a real figure from /proc on Linux (C26); only as a
            # last resort assume the machine is otherwise idle.
            proc = _proc_meminfo_available()
            host = proc if proc is not None else (1 << 62)
        return _cap_to_cgroup_headroom(host)


def hysteresis_alpha_from_flap(flap_rate: float | None) -> float | None:
    """The de-escalation smoothing weight implied by a measured `flap_rate`, or `None`.

    `flap_rate` is the fraction of observed pressure samples that reversed direction
    (measured by Core, persisted via `record_flap_rate`). A quiet history (rate 0, or
    `None` for a cold store) keeps the static default weight (returns `None` → the
    monitor's default); a flappy history stiffens it toward `default x (1 - _FLAP_STIFFEN)`
    so the level lingers on falling edges and the shuffle credit window stops oscillating.
    Pure: it only damps *when* the engine spills/throttles, never a result.
    """
    if flap_rate is None:
        return None
    rate = min(1.0, max(0.0, flap_rate))
    return _DEFAULT_ALPHA * (1.0 - _FLAP_STIFFEN * rate)


def load_flap_rate(hub: MetadataHub | None) -> float | None:
    """The measured pressure-flap rate recorded to `hub`, or `None` (cold / unavailable).

    Best-effort: any read failure yields `None`, so the hysteresis keeps its static
    default and behavior is unchanged."""
    if hub is None:
        return None
    try:
        value = hub.get_keyed_param(_FLAP_NS, _FLAP_KEY)
    except Exception:  # pragma: no cover - metadata must never break a query
        return None
    return float(value) if value is not None else None


def record_flap_rate(
    hub: MetadataHub | None, flap_rate: float, config: Config | None = None
) -> None:
    """Persist a measured pressure-flap rate, exp-smoothed across runs. Best-effort.

    Core measures how often the pressure level reversed direction over a run and reports
    it here; the value is smoothed so a single noisy run doesn't jerk the hysteresis. A
    scheduling signal only — never a result."""
    if hub is None:
        return
    try:
        rate = min(1.0, max(0.0, flap_rate))
        alpha = (config or active_config()).optimizer.learning_smoothing_alpha
        prior = hub.get_keyed_param(_FLAP_NS, _FLAP_KEY)
        smoothed = rate if prior is None else alpha * rate + (1.0 - alpha) * float(prior)
        hub.put_keyed_param(_FLAP_NS, _FLAP_KEY, smoothed)
    except Exception:  # pragma: no cover - metadata must never break a query
        pass
