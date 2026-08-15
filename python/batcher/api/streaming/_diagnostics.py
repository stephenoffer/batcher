"""What a streaming plan will do to memory, said at `start()` rather than at the OOM.

Kyber can already name the operators whose state nothing releases
(`kyber.streaming.retains_unbounded_state`): a grouped aggregate with no watermark holds
one entry per group for the life of the query, a `distinct()` one per distinct value, a
`transform_with_state` one per key until a TTL expires it. Each is *correct*. Each also
grows until the process dies, and the failure arrives days later, in production, on the
one input nobody tested — because a bounded input always ends and always releases the
state at end-of-input.

Until this module the analysis had no caller. The engine's only defence was the runtime
cap (`core.streaming.check_agg_state_bounded`), which fires when the retained state has
already reached the memory envelope: a real backstop, but it reports a query that has been
running for hours as a resource error, at the moment there is nothing left to do about it.
The same fact is knowable from the plan, before the first row, for the cost of a tree walk.

So this warns at `start()`. It does not refuse: the shape is legal, `complete` and `update`
output modes exist precisely to serve it, and a stream with a genuinely bounded key space
is a normal thing to run. What it does is make the memory profile a thing the author is
told once, up front, with the operator named and the bound that would fix it — which is
what turns "it died on Thursday" into a decision made on Monday.
"""

from __future__ import annotations

import warnings

from batcher.plan.logical import LogicalPlan

__all__ = ["warn_if_state_is_unbounded"]

#: How each leaking operator is bounded, keyed by node type name. The fix is the whole
#: value of the warning: "this leaks" without "and here is the operator that does not" is a
#: message whose only available response is to stop using streaming.
_REMEDIES = {
    "Aggregate": (
        "a grouped aggregate with no watermark keeps one entry per group forever. Add "
        ".with_watermark(...) and group by an event-time window (bt.window(col('ts'), "
        "'1h')) so closed windows evict, or narrow the group keys"
    ),
    "Distinct": (
        "distinct() keeps one entry per distinct row forever. Use "
        "drop_duplicates_within_watermark(subset, event_time=..., lateness=...), whose "
        "state the watermark bounds, or cap it with .limit(n) if a prefix will do"
    ),
    "Union": (
        "a distinct UNION dedupes the concatenation, so it keeps one entry per distinct "
        "row forever. Use UNION ALL (distinct=False) if the duplicates are acceptable"
    ),
    "Sort": (
        "a full sort buffers the whole stream. Add a limit to make it a top-N, whose "
        "state is that many rows"
    ),
    "Window": (
        "a window function holds every row of every partition it has seen, and a stream "
        "never closes a partition"
    ),
    "Join": (
        "a join buffers a whole side. Use join_stream(...) with an event-time interval, "
        "which evicts once the watermark guarantees no future match"
    ),
    "AsofJoin": "an ASOF join buffers its right side, which a stream never finishes",
    "RangeJoin": "a range join buffers its right side, which a stream never finishes",
    "TransformWithState": (
        "transform_with_state with ttl_micros=0 never expires a key's state. Pass a ttl so "
        "an idle key is dropped, which is what bounds it to the *active* key space"
    ),
}


def warn_if_state_is_unbounded(plan: LogicalPlan, sources: list) -> None:
    """Warn once per query when `plan` holds state over a stream that nothing releases.

    Silent for a wholly bounded plan: a bounded input ends, and end-of-input releases
    every operator's state, so the same node that leaks over a topic is simply a breaker
    over a file. That is the reason this cannot be a plan-build-time check.

    Args:
        plan: The streaming query's logical plan.
        sources: Its bound sources, to tell a stream from a bounded relation.
    """
    from batcher.io.source import is_bounded
    from batcher.kyber.streaming import unbounded_state_operators

    if all(is_bounded(s) for s in sources):
        return
    leaking = unbounded_state_operators(plan)
    if not leaking:
        return
    # Deduplicated by node type: a plan with three joins has one story, not three, and the
    # remedy is per operator kind. Sorted so the message is stable across runs.
    seen = sorted({type(node).__name__ for node in leaking})
    remedies = [_REMEDIES[name] for name in seen if name in _REMEDIES]
    if not remedies:  # pragma: no cover — every classified leaker has a remedy
        return
    from batcher._internal.errors import PerformanceWarning

    detail = "; ".join(remedies)
    warnings.warn(
        f"this streaming query retains state that nothing releases: {detail}. The query is "
        "correct and will keep running until the retained state reaches "
        "memory.streaming_state_max_bytes, at which point it raises. Nothing about that is "
        "visible in a bounded test, because a bounded input releases the state when it ends.",
        PerformanceWarning,
        stacklevel=3,
    )
