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

import os
from dataclasses import dataclass
from enum import IntEnum

from batcher.config import Config, active_config

__all__ = ["PressureLevel", "PressureMonitor", "total_memory_bytes"]


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


def _cgroup_limit_bytes() -> int | None:
    """The container memory limit from cgroup v2 (`memory.max`) or v1
    (`memory.limit_in_bytes`), or `None` when unlimited / not in a cgroup.

    A container's cgroup cap is the *real* ceiling — the host's RAM is not — so
    honoring it is what stops the engine over-admitting and getting OOM-killed by
    the kernel (C25).
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

    def __init__(self, config: Config | None = None) -> None:
        self._config = config or active_config()

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
        """
        mem = self._config.memory
        used = self._engine_used_fraction()
        if used >= mem.hard_limit:
            return PressureLevel.CRITICAL
        if used >= mem.soft_limit:
            return PressureLevel.SPILL
        if used >= mem.soft_limit * 0.9:
            return PressureLevel.ELEVATED
        return PressureLevel.NORMAL

    @staticmethod
    def _engine_used_fraction() -> float:
        """Fraction of the engine's buffer-pool envelope currently reserved.

        Falls back to the machine's used fraction when the pool isn't initialized.
        """
        from batcher.carbonite.memory.pool import current_process_pool

        pool = current_process_pool()
        if pool is not None and pool.limit > 0:
            return pool.used / pool.limit
        total = total_memory_bytes()
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
