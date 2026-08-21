"""How big a morsel should be, given memory pressure and the rows' measured width.

A morsel only *batches* data — it never changes the result — so shrinking one is always
safe, and that is what makes this a pure sizing rule rather than a decision with
consequences. Two independent levers reduce the configured target, and both exist because
the static `(morsel_rows, morsel_bytes)` pair is a guess about data it has never seen:

- **Pressure.** As the engine's envelope fills, a smaller morsel keeps the streaming
  working set tighter, so the query stays in memory longer before it must spill.
- **Row width.** A row-count target assumes a row width, and the default pair assumes 64
  bytes. A workload whose rows are far wider — embeddings, blobs, large strings, and above
  all the decoded image, audio, and video tensors of a multimodal pipeline — fills a
  `morsel_rows` batch to many times `morsel_bytes`, so its true working set is bounded only
  once the *count* is capped by the width. Both widths are consulted, and the more binding
  wins: the **measured** one from the learned model, which is the only signal that can see
  a variable-length payload's real size, and the **planned** one from the plan's own
  schema, which is exact for a tensor column and, unlike a measurement, exists on the first
  run — the run with nothing learned yet, and the one that OOMs.

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
    "planned_row_cap",
    "row_floor",
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


def row_floor(byte_target: int, width: float) -> int:
    """The smallest row count a morsel may be cut to, for rows of `width` bytes.

    `MIN_MORSEL_ROWS` exists so a morsel never degrades into a batch too small to amortize
    per-batch overhead — a **narrow-row** concern, and on narrow rows it is exactly right.
    Applied unconditionally it silently overrides the byte bound it was meant to accompany,
    and the default targets put that crossover at **1,024 bytes per row**
    (`morsel_bytes / MIN_MORSEL_ROWS`), which is below essentially every unstructured or
    multimodal column in the engine:

    | column                          | width     | morsel at the flat floor    |
    |---------------------------------|-----------|-----------------------------|
    | 768-dim `float32` embedding     | 3 KiB     | 3 MiB — 3x the budget       |
    | 224x224x3 `uint8` image         | 147 KiB   | 147 MiB — **147x**          |
    | one 1080p RGB video frame       | 5.9 MiB   | 6 GiB — **6,000x**          |

    So the floor is itself bounded by the byte target: never demand more rows than fit in
    the budget. On narrow rows the byte term is the larger of the two and the floor is
    `MIN_MORSEL_ROWS` exactly as before; on wide rows it falls away, down to one row, which
    is the correct morsel for a frame that is larger than the whole budget on its own.

    Args:
        byte_target: The morsel's byte budget.
        width: Estimated or measured bytes per row.

    Returns:
        The row floor, at least 1.
    """
    if width <= 0:
        return MIN_MORSEL_ROWS
    return max(1, min(MIN_MORSEL_ROWS, int(byte_target / width)))


def _cap_for_width(
    config: Config, width: float | None, byte_target: int | None = None
) -> int | None:
    """Row cap keeping `width`-byte rows inside the byte budget, or `None` if not binding.

    `byte_target` defaults to the configured per-morsel byte target. A caller that has already
    tightened that budget — `morsel_target` under a configured memory envelope — passes the
    tightened figure, so the row count is cut against the budget that will actually be
    enforced rather than the one the config asked for.
    """
    if width is None or width <= 0:
        return None
    if byte_target is None:
        byte_target = config.execution.morsel_bytes
    cap = int(byte_target / width)
    if cap >= config.execution.morsel_rows:
        return None  # the width is no wider than assumed — nothing to tighten
    return max(row_floor(byte_target, width), cap)


def learned_row_cap(
    config: Config,
    model: LearnedMemoryModel | None,
    families: Iterable[str] | None = None,
    byte_target: int | None = None,
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
        byte_target: The per-morsel byte budget to fit inside; the configured target when
            omitted. See `_cap_for_width`.

    Returns:
        The row cap, or `None` when nothing is learned yet or the learned width is no wider
        than the configured target already implies (so the common case adds no overhead and
        makes no change).
    """
    if model is None:
        return None
    return _cap_for_width(config, model.max_bytes_per_row(families), byte_target)


def planned_row_cap(config: Config, plan: object, byte_target: int | None = None) -> int | None:
    """Row cap from the width the plan's **schema** implies.

    The companion to `learned_row_cap`, and the one that exists on the *first* run. A
    learned width only exists after a query of this shape has already run, and the first run
    of a multimodal pipeline is precisely the one with nothing measured and the one that
    OOMs. The width was knowable the whole time: it is a property of the column types, which
    the plan carries before a single row is read, and it is **exact** for the tensor columns
    that carry decoded images, audio, and video. A cold store was throwing that away and
    cutting a morsel by a row count that assumes 64 B/row.

    It is also keyed better than the learned width, which is filed by operator *family* and
    so cannot tell this plan's image scan from an earlier narrow one. `morsel_target` takes
    whichever of the two binds harder rather than preferring either.

    Takes the **widest** stage in the plan, because a morsel flowing through it must fit the
    widest one, not the average. Reads only the neutral `plan` layer, so Carbonite consults
    the schema without importing Kyber.

    Args:
        config: The active config, for the byte and row targets.
        plan: The logical plan about to run.

    Returns:
        The row cap, or `None` when no node carries a schema or the implied width is no
        wider than the configured target already assumes.
    """
    from batcher.plan.types import schema_row_bytes
    from batcher.plan.visitor import walk

    widest = 0.0
    for node in walk(plan):
        schema = getattr(node, "available_schema", None)
        resolved = schema() if callable(schema) else None
        arrow = getattr(resolved, "arrow", None)
        if arrow is not None:
            widest = max(widest, schema_row_bytes(arrow))
    return _cap_for_width(config, widest, byte_target) if widest > 0.0 else None


def _planned(config: Config, plan: object | None, byte_target: int | None = None) -> int | None:
    """`planned_row_cap` for an optional plan — `None` when the caller supplied none."""
    return planned_row_cap(config, plan, byte_target) if plan is not None else None


def _envelope_byte_cap(config: Config, plan: object | None) -> int | None:
    """Per-morsel byte budget implied by an **explicitly configured** memory envelope.

    `max_memory_bytes` is an instruction rather than a reading (`pressure.budget_bytes` says
    so), and admission bills a streaming pipeline one morsel per operator. So a small
    configured envelope and a default 16,384-row morsel are two settings that contradict each
    other, and the engine used to resolve the contradiction by *refusing the query*: a
    `UNION ALL` over 20,000 rows under a 1 MiB envelope was declined as "does not fit the
    memory envelope and has no out-of-core path" — for a pipeline that holds one morsel at a
    time and needs no spill path at all. Two operators x 16,384 rows x 52 bytes is 1.7 MB
    against a 1 MiB envelope, and every one of those rows was a row the engine chose to batch.

    The morsel is the knob that resolves it. A morsel only *batches* data — it never changes
    the result — so cutting it to fit is always available, and it is strictly better than
    declining. The budget is divided by the plan's node count because admission charges each
    node that holds a morsel and this layer must not import Kyber to learn which ones do:
    counting every node over-divides, which is the safe direction (a smaller morsel), and
    reads only the neutral `plan` layer.

    Returns `None` — changing nothing — whenever `max_memory_bytes` is unset, which is the
    default and the auto-sensed case. An envelope large enough that the quotient exceeds the
    configured `morsel_bytes` also changes nothing, since `morsel_target` takes the smaller.

    Args:
        config: The active config.
        plan: The logical plan about to run, or `None`.

    Returns:
        The per-morsel byte cap, or `None` to leave the configured target alone.
    """
    budget = config.memory.max_memory_bytes
    if budget is None or budget <= 0 or plan is None:
        return None
    from batcher.plan.visitor import walk

    nodes = sum(1 for _ in walk(plan))
    if nodes <= 0:
        return None
    return max(MIN_MORSEL_BYTES, int(budget / nodes))


def morsel_target(
    config: Config,
    level: PressureLevel,
    model: LearnedMemoryModel | None = None,
    families: Iterable[str] | None = None,
    plan: object | None = None,
) -> tuple[int, int] | None:
    """The per-morsel ``(rows, bytes)`` target for this pressure level and row width.

    Args:
        config: The active config, carrying the configured target.
        level: The live pressure level. Must come from a *non-sampling* read
            (`PressureMonitor.classify`), because sizing a morsel is not the component
            that owns the de-escalation average's sampling cadence.
        model: The learned memory model, or `None`.
        families: The plan's operator kinds, narrowing the learned width to this plan.
        plan: The logical plan about to run, whose schema sizes the morsel on a cold
            store. Measured width wins wherever there is one; this only covers the first
            run, which is the one that has no measurement and OOMs.

    Returns:
        The recommended `(rows, bytes)`, or `None` to keep the configured target — the
        common, unpressured, nothing-learned case.
    """
    ex = config.execution
    factor = PRESSURE_FACTORS.get(level, 1.0)
    nbytes = int(ex.morsel_bytes * factor)
    # A configured memory envelope is an instruction, and the morsel is the knob that honours
    # it for a streaming pipeline. Applied before the width caps below so they cut the row
    # count against the budget that will actually be enforced.
    envelope = _envelope_byte_cap(config, plan)
    if envelope is not None:
        nbytes = min(nbytes, envelope)
    # Pressure shrinks the row target but never past the cache-efficient batch: that floor
    # is about per-batch overhead, which pressure does not change.
    rows = max(MIN_MORSEL_ROWS, int(ex.morsel_rows * factor))
    # The two width signals cover each other's blind spots, so the morsel takes whichever
    # is more binding rather than preferring one outright:
    #
    #   - the **learned** width is a real measurement, but it is keyed by operator *family*
    #     ("Scan", "Aggregate"), which cannot tell this plan's image scan from yesterday's
    #     narrow one. It is the only signal that catches a wide *variable-length* payload,
    #     whose true width no schema can state.
    #   - the **planned** width is keyed by this plan's own schema and is *exact* for the
    #     fixed-width and tensor columns that carry multimodal data — and it exists on the
    #     first run, which is the one with no measurement and the one that OOMs.
    #
    # Taking the smaller cap is safe in both directions: a morsel only batches data, so an
    # over-tight one costs some throughput while an over-loose one costs the process.
    caps = [
        c
        for c in (
            learned_row_cap(config, model, families, nbytes),
            _planned(config, plan, nbytes),
        )
        if c is not None
    ]
    cap = min(caps) if caps else None
    # A width cap carries its own width-aware floor (`row_floor`), which is what lets it
    # take the count below `MIN_MORSEL_ROWS` — down to a single row for a video frame
    # wider than the whole budget — rather than being overridden by it.
    if cap is not None:
        rows = min(rows, cap)
    # Keep the configured target (fast path) only when no lever moved anything.
    if factor >= 1.0 and rows >= ex.morsel_rows and nbytes >= ex.morsel_bytes:
        return None
    return max(1, rows), max(MIN_MORSEL_BYTES, nbytes)
