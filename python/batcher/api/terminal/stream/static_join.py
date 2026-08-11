"""The stream-static join — enrich a stream from a table that does not move.

The most common thing anyone does to a stream: clicks joined to a product catalogue,
events joined to a device registry, transactions joined to an account dimension. Spark
supports it directly. Batcher refused it — a `Join` is a pipeline breaker, so the router
saw an unbounded input beneath a breaker and raised "the plan must materialize" — and the
cookbook's advice was to hand-roll the lookup inside `map_batches`, which means writing the
join yourself, keeping the dimension in memory yourself, and refreshing it yourself.

It needs no new algebra, only the observation that makes it sound: **the static side is
bounded, so it can be read once, and joining each stream batch against the whole of it and
concatenating is exactly the join over the whole stream.** That holds because an equi-join
is per-row on the stream side — no stream row's result depends on another's.

The same observation says which join types are safe, and the answer is Spark's. A side is
*preserved* by an outer join if its unmatched rows must be emitted; a preserved side has to
be complete before you can know a row is unmatched. The stream never is:

* **inner** — safe either way round.
* **left**, stream on the left — safe: an unmatched stream row is unmatched the moment it
  arrives, because the static side is already whole.
* **right**, stream on the right — the mirror, safe for the same reason.
* **left with the stream on the right**, **right with the stream on the left**, and
  **full** — refused. They preserve the *static* side, which would mean holding every
  static row until the stream ends to find out which never matched.
* **semi** / **anti** with the stream on the left — safe; the stream row is the output and
  the static side answers immediately.
"""

from __future__ import annotations

from collections.abc import Iterator

import pyarrow as pa

from batcher.io.source import Source
from batcher.plan.logical import Join, LogicalPlan

__all__ = ["stream_static_join", "stream_static_sides"]

#: ``(join_type, streaming_side)`` pairs whose preserved side is the stream, or neither.
#: Everything absent from this set preserves the *static* side, which cannot be known
#: complete while the stream runs. Mirrors Spark's supported matrix exactly.
_SAFE = {
    ("inner", "left"),
    ("inner", "right"),
    ("left", "left"),
    ("right", "right"),
    ("semi", "left"),
    ("anti", "left"),
}


def stream_static_sides(plan: LogicalPlan, sources: list[Source]) -> tuple[str, LogicalPlan] | None:
    """Which side of `plan` is the stream, when it is a stream-static join at all.

    Returns None for anything else — two streams (which is `join_stream`'s interval join),
    two bounded sides (an ordinary join), or a plan that is not a top-level `Join`.

    Args:
        plan: The plan being routed.
        sources: The bound sources for the whole plan.

    Returns:
        ``(streaming_side, static_side_plan)``, or None.
    """
    from batcher.io.source import is_bounded
    from batcher.plan.visitor import scanned_source_ids

    if not isinstance(plan, Join):
        return None
    left_ids = scanned_source_ids(plan.left)
    right_ids = scanned_source_ids(plan.right)
    if not left_ids or not right_ids or left_ids & right_ids:
        return None
    left_streams = any(not is_bounded(sources[i]) for i in left_ids)
    right_streams = any(not is_bounded(sources[i]) for i in right_ids)
    if left_streams == right_streams:
        return None  # both stream (an interval join) or both bounded (an ordinary join)
    return ("left", plan.right) if left_streams else ("right", plan.left)


def refuse_reason(join_type: str, streaming_side: str) -> str | None:
    """Why this stream-static join cannot be run, or None when it can.

    Args:
        join_type: The join's type.
        streaming_side: ``"left"`` or ``"right"`` — which side is unbounded.

    Returns:
        An actionable message, or None.
    """
    if (join_type, streaming_side) in _SAFE:
        return None
    static_side = "right" if streaming_side == "left" else "left"
    return (
        f"a {join_type!r} stream-static join with the stream on the {streaming_side} preserves "
        f"the {static_side} (static) side, so an unmatched static row could only be emitted "
        "once the stream ended — which it does not. Supported: inner either way, left outer "
        "with the stream on the left, right outer with the stream on the right, and "
        "semi/anti with the stream on the left. (Spark draws the same line.)"
    )


def stream_static_join(
    plan: Join,
    sources: list[Source],
    streaming_side: str,
    batch_size: int | None,
) -> Iterator[pa.RecordBatch]:
    """Join each micro-batch of the streaming side against the whole static side.

    The static side is read **once**, before the first stream batch, and held for the life
    of the query. That is the operator's memory bound and it is the honest one: a dimension
    table small enough to join a stream against is small enough to hold, and one that is not
    is a design problem this operator cannot solve by being clever.

    It is deliberately *not* refreshed. A long-running query serves the snapshot it started
    with, which is what Spark does and is the only behavior with a defined answer — a
    dimension that changed mid-stream would otherwise make two rows of the same micro-batch
    disagree about the same key.

    Driver-executed today, including under `distributed=True`: the result is identical
    (the same rows through the same code), it simply does not fan out. Its mergeable form
    is the ordinary broadcast join — the static side is small by assumption, so shipping a
    copy to each worker and joining that worker's stream shard against it is exact for
    every join type in `_SAFE`, with no cross-worker state at all.

    Args:
        plan: The `Join` node.
        sources: The bound sources for the whole plan.
        streaming_side: ``"left"`` or ``"right"``.
        batch_size: Optional output rebatching.

    Yields:
        The joined rows, one micro-batch's worth at a time.
    """
    from batcher.api.session import from_arrow
    from batcher.api.terminal.stream.dispatch import _iter_batches, _iter_streaming
    from batcher.plan.logical import remap_sources
    from batcher.plan.visitor import scanned_source_ids

    static_plan = plan.right if streaming_side == "left" else plan.left
    stream_plan = plan.left if streaming_side == "left" else plan.right

    # Read the static side once, through the ordinary router so its own filters, projections
    # and even its own breakers run exactly as they would anywhere else.
    static_ids = sorted(scanned_source_ids(static_plan))
    static_batches = list(
        _iter_batches(
            remap_sources(static_plan, -static_ids[0]),
            [sources[i] for i in static_ids],
            static_plan.available_columns(),
            None,
        )
    )
    static_table = (
        pa.Table.from_batches(static_batches)
        if static_batches
        else pa.Table.from_batches([], schema=_static_schema(static_plan))
    )

    # The stream side goes through the same breaker-free router the stateless streaming
    # path uses, rather than a private optimize-and-execute loop. That is what makes a
    # `map_batches` beneath the join work: the UDF is Python and cannot be lowered, so
    # asking Kyber for a `PhysicalPlan` of a plan containing one raised
    # `NotImplementedError: map_batches is executed in Python, not lowered to the engine
    # IR` -- scoring a stream and then enriching it from a dimension is an obvious pipeline
    # and it could not run at all. `_iter_streaming` pushes the projection and predicate
    # down exactly as the private loop did, so nothing is given up for it.
    stream_ids = sorted(scanned_source_ids(stream_plan))
    stream_source = sources[stream_ids[0]]
    stream_local = remap_sources(stream_plan, -stream_ids[0])

    static_ds = from_arrow(static_table)
    aliases = [o.alias for o in plan.output]
    for produced in _iter_streaming(stream_local, [stream_source], None):
        if produced.num_rows == 0:
            continue
        batch_ds = from_arrow(pa.Table.from_batches([produced]))
        left_ds = batch_ds if streaming_side == "left" else static_ds
        right_ds = static_ds if streaming_side == "left" else batch_ds
        joined = left_ds.join(
            right_ds,
            left_on=list(plan.left_keys),
            right_on=list(plan.right_keys),
            how=plan.join_type,
        ).collect()
        if joined.num_rows == 0:
            continue
        # The declared output aliases, so the rows match the plan's schema exactly — the
        # join above is rebuilt from two fresh relations and would otherwise be free to
        # name its columns differently from what the plan promised.
        present = [a for a in aliases if a in joined.column_names]
        shaped = joined.select(present) if present else joined
        if batch_size is None:
            yield from shaped.to_batches()
        else:
            yield from shaped.to_batches(max_chunksize=batch_size)


def _static_schema(static_plan: LogicalPlan) -> pa.Schema:
    """The static side's schema, for the empty case.

    A dimension table that is empty is an ordinary state (a fresh deployment, a filter that
    matched nothing), and the join still has to know its column names or the output schema
    is not derivable at all.
    """
    declared = static_plan.available_schema()
    if declared is not None:
        return declared.arrow
    return pa.schema([pa.field(name, pa.null()) for name in static_plan.available_columns()])
