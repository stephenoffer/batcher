"""How big a morsel should be, given memory pressure and the rows' measured width.

A morsel only *batches* data — it never changes the result — so shrinking one is always
safe, and that is what makes this a pure sizing rule rather than a decision with
consequences. Two independent levers reduce the configured target, and both exist because
the static `(morsel_rows, morsel_bytes)` pair is a guess about data it has never seen:

- **Pressure.** As the engine's envelope fills, a smaller morsel keeps the streaming
  working set tighter, so the query stays in memory longer before it must spill.
- **Measured row width.** A row-count target assumes a row width. A workload whose rows
  proved far wider than assumed (embeddings, blobs, large strings) fills a `morsel_rows`
  batch to many times `morsel_bytes`, so its true working set is bounded only once the
  *count* is capped by the measured width.

Split out of `ResourceManager` because it is arithmetic over a pressure level and a learned
width with no reference to the manager's state, and because keeping it there pushed that
module past the size limit while making it read as a sizing library with a governor
attached rather than the other way round.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher.carbonite.memory.pressure import PressureLevel

if TYPE_CHECKING:
    from collections.abc import Iterable

    from batcher.carbonite.memory.learned import LearnedMemoryModel
    from batcher.config import Config

__all__ = [
    "MIN_MORSEL_BYTES",
    "MIN_MORSEL_ROWS",
    "PRESSURE_FACTORS",
    "learned_row_cap",
    "morsel_target",
]

# How far to shrink the morsel target at each pressure level (adaptive morsel sizing).
# NORMAL keeps the configured target (no entry ⇒ factor 1.0); ELEVATED halves it;
# SPILL/CRITICAL quarter it so the streaming working set stays tight while the engine
# is already under pressure.
PRESSURE_FACTORS = {
    PressureLevel.ELEVATED: 0.5,
    PressureLevel.SPILL: 0.25,
    PressureLevel.CRITICAL: 0.25,
}
MIN_MORSEL_ROWS = 1024  # floor: a morsel never shrinks below a cache-efficient batch
MIN_MORSEL_BYTES = 64 * 1024  # 64 KiB floor (companion byte bound)


def learned_row_cap(
    config: Config, model: LearnedMemoryModel | None, families: Iterable[str] | None = None
) -> int | None:
    """Row cap that keeps a morsel's *measured* byte working set within the budget.

    Uses the widest learned per-row footprint (``rows = morsel_bytes / max_bytes_per_row``),
    restricted to `families` — this plan's operator kinds — when given, so a narrow plan is
    sized by its own data rather than throttled by an unrelated wide family measured in an
    earlier query.

    Args:
        config: The active config, for the byte and row targets.
        model: The learned memory model, or `None` on a cold store.
        families: The plan's operator kinds, or `None` for the global widest.

    Returns:
        The row cap, or `None` when nothing is learned yet or the learned width is no wider
        than the configured target already implies (so the common case adds no overhead and
        makes no change).
    """
    if model is None:
        return None
    width = model.max_bytes_per_row(families)
    if width is None or width <= 0:
        return None
    cap = int(config.execution.morsel_bytes / width)
    if cap >= config.execution.morsel_rows:
        return None  # learned width is no wider than assumed — nothing to tighten
    return max(MIN_MORSEL_ROWS, cap)


def morsel_target(
    config: Config,
    level: PressureLevel,
    model: LearnedMemoryModel | None = None,
    families: Iterable[str] | None = None,
) -> tuple[int, int] | None:
    """The per-morsel ``(rows, bytes)`` target for this pressure level and learned width.

    Args:
        config: The active config, carrying the configured target.
        level: The live pressure level. Must come from a *non-sampling* read
            (`PressureMonitor.classify`), because sizing a morsel is not the component
            that owns the de-escalation average's sampling cadence.
        model: The learned memory model, or `None`.
        families: The plan's operator kinds, narrowing the learned width to this plan.

    Returns:
        The recommended `(rows, bytes)`, or `None` to keep the configured target — the
        common, unpressured, nothing-learned case.
    """
    ex = config.execution
    factor = PRESSURE_FACTORS.get(level, 1.0)
    rows = int(ex.morsel_rows * factor)
    nbytes = int(ex.morsel_bytes * factor)
    cap = learned_row_cap(config, model, families)
    if cap is not None:
        rows = min(rows, cap)
    # Keep the configured target (fast path) only when neither lever moved anything.
    if factor >= 1.0 and rows >= ex.morsel_rows:
        return None
    return max(MIN_MORSEL_ROWS, rows), max(MIN_MORSEL_BYTES, nbytes)
