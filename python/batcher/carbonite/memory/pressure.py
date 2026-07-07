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
    "total_memory_bytes",
]

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
    """
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path) as f:
                raw = f.read().strip()
        except OSError:
            continue
        if raw in ("max", ""):
            return None
        try:
            value = int(raw)
        except ValueError:
            continue
        # cgroup v1 reports a sentinel near 2^63 when unlimited; treat huge as none.
        if value <= 0 or value >= (1 << 62):
            return None
        return value
    return None


def _cgroup_current_bytes() -> int | None:
    """The container's *current* memory usage from cgroup v2 (`memory.current`) or v1
    (`memory.usage_in_bytes`), or `None` when not in a cgroup.

    This is the figure the kernel OOM-killer watches — it counts *all* of the
    process's anonymous memory, including the in-memory Flight shuffle store and
    off-pool pyarrow buffers the engine's buffer pool does not track. Reading it is
    what lets the monitor see the real pressure instead of only the pool's reservations.
    """
    for path in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            with open(path) as f:
                raw = f.read().strip()
        except OSError:
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value > 0:
            return value
    return None


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
    when the OS won't report host RAM.
    """
    try:
        host = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        host = active_config().memory.default_total_bytes
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
        mem = self._config.memory
        raw = self._engine_used_fraction()
        prev = self._ewma if self._ewma is not None else raw
        used = max(raw, prev)  # escalate on raw, de-escalate on the lagging average
        self._ewma = self._alpha * raw + (1.0 - self._alpha) * prev
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
        *actual* footprint (the cgroup's current usage, else RSS). Memory the pool does
        not track — the in-memory Flight shuffle `PartitionStore`, off-pool pyarrow
        buffers — therefore cannot let the monitor report NORMAL while the kernel
        OOM-kills a shuffle-heavy worker. Over-reading only spills/throttles a little
        early (safe); under-reading risks the OOM-kill this guards against. Falls back
        to the machine's used fraction when neither a pool nor a live reading exists.
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
        try:
            import psutil
        except ImportError:
            # No psutil: read a real figure from /proc on Linux (C26); only as a
            # last resort assume the machine is otherwise idle.
            proc = _proc_meminfo_available()
            return min(proc, total) if proc is not None else total
        return min(int(psutil.virtual_memory().available), total)


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
