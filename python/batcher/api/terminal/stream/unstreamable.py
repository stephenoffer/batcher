"""Why a plan cannot stream — the message `iter_batches` raises on an unbounded source.

Split out of `dispatch` because it is the one thing in that module that is *not* part of the
ordered branch sequence: the router's branches must be tested in a fixed order (a fused
`distinct().limit(n)` before the general dedup fold, and so on), and moving one across a
module boundary is how that ordering quietly stops holding. This function is order-independent
— it is called once, at the end, when nothing matched — so it is the safe thing to move when
the file needs room.
"""

from __future__ import annotations

from batcher.plan.logical import LogicalPlan

__all__ = ["_unstreamable_reason"]


def _unstreamable_reason(plan: LogicalPlan) -> str:
    """Why this plan cannot stream, naming the operator that stops it.

    The message used to name `type(plan).__name__` — the *top* node — which is the culprit
    only when the breaker happens to be at the root. ``ds.sort("t").group_by("a").agg(...)``
    reported "its top-level Aggregate forces the plan to materialize", and a streaming
    aggregate is exactly the shape that *does* stream: the reader was pointed at the one
    operator in their query that was fine, while the `sort` beneath it went unmentioned.
    Kyber already knows which nodes cannot emit under a stream
    (`kyber.streaming.blocking_operators`), so ask it rather than guess from the root.

    Args:
        plan: The plan the router found no streaming strategy for.

    Returns:
        A refusal naming the blocking operators, and the shapes that do stream.
    """
    from batcher.kyber.streaming import blocking_operators

    blocking = blocking_operators(plan)
    if blocking:
        # Deduplicated and ordered so a plan with three sorts reads as "sort", not
        # "sort / sort / sort", while a mixed plan still names each distinct offender.
        names = sorted({type(n).__name__.lower() for n in blocking})
        culprit = f"its {' and '.join(names)} cannot emit a row until the input ends"
    else:
        culprit = (
            f"its top-level {type(plan).__name__.lower()} forces the plan to materialize "
            "(a multi-source shape no streaming driver covers)"
        )
    return (
        f"this pipeline has an unbounded (streaming) source but {culprit}, so it cannot be "
        "streamed in bounded memory. Restructure to a streamable shape: filter / select / "
        "with_columns / map_batches, or a single top-level aggregate, distinct, limit or "
        "top-N over one of those."
    )
