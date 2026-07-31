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

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

from batcher.carbonite.memory import probe
from batcher.carbonite.memory.probe import reset_memory_sampling, total_memory_bytes
from batcher.config import Config, active_config
from batcher.metadata.smoothed import load_scalar, record_smoothed_scalar

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

# Learned-parameter namespace + key for the measured pressure-level flap rate (fraction
# of samples that reversed direction). One process-wide figure; the hysteresis adapts to it.
_FLAP_NS = "carbonite.pressure_flap"
_FLAP_KEY = "rate"
# The static default de-escalation weight (kept in sync with `PressureMonitor._EWMA_ALPHA`).
_DEFAULT_ALPHA = 0.5
# How strongly a high flap rate stiffens the hysteresis: a fully-flapping history (rate 1.0)
# cuts the de-escalation weight to `1 - _FLAP_STIFFEN` of the default (a much stickier level).
_FLAP_STIFFEN = 0.8
# Where ELEVATED begins, as a fraction of the soft limit. The level exists to give the
# engine a *warning* band it can act in before the soft limit forces spilling — 90% of soft
# is 76.5% of the envelope on the defaults, which is roughly one large morsel's worth of
# headroom on a typical budget. Named rather than inlined because two things must agree
# about it: the classifier here, and the morsel/cache ladders that react to the level.
_ELEVATED_FRACTION_OF_SOFT = 0.9


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
        # Flap accounting for `flap_rate` — the signal `hysteresis_alpha` is itself derived
        # from, closing that loop. Counts direction *reversals* across sampled levels: the
        # previous level, the direction of the last change (+1/-1/0), how many changes were
        # reversals, and how many samples were taken.
        self._prev_level: PressureLevel | None = None
        self._last_dir = 0
        self._reversals = 0
        self._samples = 0

    def snapshot(self) -> MemorySnapshot:
        """Take a current reading of total/available memory and budget usage."""
        total = probe.total_memory_bytes()
        available = self._available_bytes(total)
        used_fraction = 1.0 - (available / total) if total else 1.0
        return MemorySnapshot(total=total, available=available, used_fraction=used_fraction)

    def available_bytes(self) -> int:
        """Bytes of RAM available right now (psutil) or total RAM as a fallback."""
        return self._available_bytes(probe.total_memory_bytes())

    def budget_bytes(self) -> int:
        """The soft envelope: the share of total RAM the engine aims to stay under."""
        return int(probe.total_memory_bytes() * self._config.memory.soft_limit)

    def envelope_bytes(self) -> int:
        """The raw memory a query may draw on: the configured hard cap if set
        (honors a container/cgroup limit), else the RAM available right now.

        Sampled once per query by the `ResourceManager` and threaded through the
        `ResourceContext` so admission, spill, and reserve all reason about the
        same figure instead of each re-sampling live free RAM.

        An auto-sensed envelope is scaled down by `memory.oom_kill_backoff` when this cgroup's
        `memory.events` shows a task in it has already been OOM-killed. That counter is proof,
        not a forecast: the workload did not fit at the size it last ran, and a restarted
        worker re-deriving the envelope from the same box gets the same number and walks into
        the same kill. An explicitly configured `max_memory_bytes` is an instruction and is
        never scaled — an operator who pinned a cap gets exactly it.
        """
        mem = self._config.memory
        if mem.max_memory_bytes is not None:
            return mem.max_memory_bytes
        return int(self.available_bytes() * self._oom_history_factor())

    def _oom_history_factor(self) -> float:
        """`memory.oom_kill_backoff` when this cgroup has been OOM-killed before, else `1.0`."""
        backoff = self._config.memory.oom_kill_backoff
        if backoff >= 1.0:
            return 1.0
        from batcher.carbonite.memory.kernel import oom_kill_count

        return backoff if oom_kill_count() else 1.0

    def stall_floor(self) -> PressureLevel:
        """The lowest level the kernel's own memory PSI justifies, from its `full` share.

        The byte accounting this monitor otherwise runs on answers "how much is reserved",
        which is not the same question as "is the kernel coping". A cgroup can sit at 70% of
        its limit and still spend most of every second in direct reclaim — because the limit
        being defended is `memory.high`, or because the resident set is nearly all anonymous
        and there is no cache left to drop. In that state the byte reading says NORMAL while
        the container is seconds from a kill.

        PSI `full` is the share of the window in which *every* runnable task was stalled on
        memory, so it measures exactly the thing the byte reading cannot see. It is used only
        as a **floor**: it can raise the level the accounting reported and never lower it, so
        turning it on makes the engine spill sooner and never later.

        Returns:
            The floor, `NORMAL` when PSI is unavailable or the config opts out.
        """
        if not self._config.memory.stall_aware_pressure:
            return PressureLevel.NORMAL
        from batcher.carbonite.memory.kernel import (
            STALL_CRITICAL,
            STALL_ELEVATED,
            memory_stall_full,
        )

        stall = memory_stall_full()
        if stall is None:
            return PressureLevel.NORMAL
        # Deliberately capped at SPILL rather than CRITICAL. CRITICAL pauses producers
        # outright, and a stall share is a *rate*, not a headroom figure — a burst of reclaim
        # while the engine still has gigabytes free would otherwise stop the query dead. SPILL
        # is the strongest response that is always safe: it takes state out of core, which is
        # what relieves reclaim in the first place.
        if stall >= STALL_CRITICAL:
            return PressureLevel.SPILL
        if stall >= STALL_ELEVATED:
            return PressureLevel.ELEVATED
        return PressureLevel.NORMAL

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
        level = max(self._classify(max(raw, prev)), self.stall_floor())
        self._ewma = self._alpha * raw + (1.0 - self._alpha) * prev
        self._observe_flap(level)
        return level

    def _observe_flap(self, level: PressureLevel) -> None:
        """Fold one sampled level into the reversal count behind `flap_rate`.

        A *reversal* is a level change whose direction opposes the previous change — the
        SPILL->NORMAL->SPILL oscillation the hysteresis exists to damp. A monotonic climb or
        decay is not a flap however many steps it takes, so only direction changes count.
        """
        self._samples += 1
        if self._prev_level is None:
            self._prev_level = level
            return
        if level == self._prev_level:
            return
        direction = 1 if level > self._prev_level else -1
        if self._last_dir and direction != self._last_dir:
            self._reversals += 1
        self._last_dir = direction
        self._prev_level = level

    def flap_rate(self) -> float | None:
        """Fraction of sampled levels that reversed direction, or `None` before 2 samples.

        This is the measurement `hysteresis_alpha_from_flap` consumes to stiffen de-escalation
        for a workload observed to oscillate. Nothing produced it, so `load_flap_rate` returned
        `None` on every real run and the whole anti-oscillation mechanism was permanently
        inert at the static default. The monitor is the natural producer — it is the component
        that sees every level — and this only *measures*, consistent with the rest of the class.
        """
        return self._reversals / self._samples if self._samples >= 2 else None

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
        return max(self._classify(max(raw, prev)), self.stall_floor())

    def _classify(self, used: float) -> PressureLevel:
        """Bucket a used-fraction into a level. Pure.

        A config with `hard_limit < soft_limit` is a misconfiguration, and taken literally
        it deletes two of the four bands: every reading past the soft limit is also past
        the hard one, so the ladder jumps NORMAL → CRITICAL with no ELEVATED warning band
        and no SPILL band at all — the engine never spills, it only ever hits the emergency
        stop. `hard` is the cap and is authoritative, so `soft` is pulled *down* to it
        (never `hard` up, which would let a query run past a limit its operator set). The
        ladder is then well-formed for any config: an ELEVATED band below the cap, and
        CRITICAL at it.
        """
        mem = self._config.memory
        hard = mem.hard_limit
        soft = min(mem.soft_limit, hard)
        if used >= hard:
            return PressureLevel.CRITICAL
        if used >= soft:
            return PressureLevel.SPILL
        if used >= soft * _ELEVATED_FRACTION_OF_SOFT:
            return PressureLevel.ELEVATED
        return PressureLevel.NORMAL

    def headroom_bytes(self) -> int:
        """Bytes that may still be allocated before the *soft* limit is reached.

        The forward-looking companion to `level()`: a level says which band the engine is
        in, this says how much room is left in it. An operator deciding whether to
        materialize one more partition wants the byte figure, not the band — and computing
        it at the call site means re-deriving the soft threshold there, which is exactly
        how two components end up disagreeing about where the line is.

        Returns:
            Remaining bytes under the soft limit; `0` once it is reached or passed.
        """
        used = self._engine_used_fraction()
        envelope = self.envelope_bytes()
        return max(0, int(envelope * (self._config.memory.soft_limit - used)))

    def reset(self) -> None:
        """Forget the hysteresis average and the flap accounting.

        A monitor is per-`ResourceManager` and therefore per-query, but the flap statistics
        it accumulates describe a *channel's* behaviour over time. A long-lived session
        that reuses one monitor across unrelated queries otherwise carries the first
        query's oscillation into the second's de-escalation weight.
        """
        self._ewma = None
        self._prev_level = None
        self._last_dir = 0
        self._reversals = 0
        self._samples = 0

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
            candidates.append(pool.utilization)
        # Against the ceiling that actually binds, which is `memory.high` where one is set.
        # Dividing by `memory.max` reports a container as half-full at the exact moment the
        # kernel starts sleeping it in direct reclaim, so the level stays NORMAL through the
        # whole throttled band and the engine never spills its way out of it.
        total = probe.effective_limit_bytes() or probe.total_memory_bytes()
        if total:
            footprint = probe.cgroup_current_bytes() or probe.process_rss_bytes()
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
        """Available RAM, never reported as more than the `total` this monitor budgets to."""
        return min(probe.available_bytes(), total)


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
    return load_scalar(hub, _FLAP_NS, _FLAP_KEY)


def record_flap_rate(
    hub: MetadataHub | None, flap_rate: float, config: Config | None = None
) -> None:
    """Persist a measured pressure-flap rate, exp-smoothed across runs. Best-effort.

    Core measures how often the pressure level reversed direction over a run and reports
    it here; the value is smoothed so a single noisy run doesn't jerk the hysteresis. A
    scheduling signal only — never a result."""
    rate = min(1.0, max(0.0, flap_rate))
    record_smoothed_scalar(hub, _FLAP_NS, _FLAP_KEY, rate, config)
