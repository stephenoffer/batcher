"""State shared by the event-time streaming operators: the watermark, eviction, and the cap.

Watermark dedup, the stream-stream interval join, and the session window all retain rows
between micro-batches and release them as event time advances, so they share four things
and nothing else: how far event time has actually advanced, how a state table is kept from
fragmenting, how an event-time column is normalized to the microseconds every bound is
expressed in, and what happens when a watermark stops advancing and the retained state stops
shrinking.

The first of those used to be three separate answers — each operator advanced its own
`max(event time) - lateness` scalar inline — and three copies of a rule is three chances to
get it wrong. They now share `plan.streaming.WatermarkTracker`, which is where the rule that
a stream's frontier is a *minimum over its partitions* is stated once.
"""

from __future__ import annotations

from collections.abc import Sequence

import pyarrow as pa

from batcher.plan.streaming import WatermarkTracker, event_micros
from batcher.plan.types import retained_bytes

#: Chunks a retained state table may accumulate before it is compacted.
#:
#: Streaming state grows by `concat_tables` — one chunk per micro-batch — and shrinks by
#: `filter`, which preserves the chunk structure. So a stream running for an hour at a 100ms
#: trigger carried a 36,000-chunk table into every anti-join and every eviction, and the
#: per-chunk dispatch cost grew without bound even while the *row* count stayed inside the
#: watermark window. Nothing caught it: the memory cap measures bytes, and the bytes were
#: fine. Compacting past this many chunks keeps the fragmentation bounded for the price of
#: one copy of state that is bounded by construction.
_MAX_STATE_CHUNKS = 64


def _compact(table: pa.Table | None) -> pa.Table | None:
    """Collapse a retained state table's chunks once it has fragmented too far."""
    if table is None or table.num_columns == 0:
        return table
    if table.column(0).num_chunks > _MAX_STATE_CHUNKS:
        return table.combine_chunks()
    return table


#: Event-time ticks as int64 **microseconds**, whatever the column's resolution.
#:
#: Watermarks, `within`, and `lateness` are all microseconds. Reading the raw int64 ticks of
#: a non-`us` timestamp (e.g. `timestamp[ns]`) would scale the watermark by up to 1000x —
#: evicting keys too early (re-emitting duplicates) or missing valid interval-join matches.
#: The definition lives beside the tracker that depends on it, so the operators and the
#: watermark cannot normalize differently; this alias keeps the local spelling.
_event_micros = event_micros


def _stream_tracker(source, lateness_micros: int) -> tuple[WatermarkTracker, Sequence[str]]:
    """A watermark tracker for `source`, and the columns that attribute a row to a partition.

    The columns are best-effort here in a way they are not for the windowed aggregate. That
    driver reads the source itself, so it can widen the projection to keep the partition
    columns alive; these operators consume the *output of a pipeline*, and a `select` that
    does not carry `partition` through leaves nothing to attribute by. The tracker degrades
    to a single partition in that case — today's behavior, and still an improvement on three
    hand-rolled scalars, because idleness, monotonicity, and unit normalization now come
    from one place.

    Args:
        source: The stream the operator reads.
        lateness_micros: Allowed lateness for this operator.

    Returns:
        `(tracker, partition columns to attribute rows by)`.
    """
    from batcher.io.source import watermark_partition_columns, watermark_partitions

    cols = watermark_partition_columns(source)
    tracker = WatermarkTracker(
        lateness_micros, expected_partitions=watermark_partitions(source) if cols else ()
    )
    return tracker, cols


def _check_stream_state(table: pa.Table | None, label: str) -> None:
    """Raise a clear `ResourceError` if a streaming operator's retained state has
    outgrown the configured cap.

    Watermark-bounded streaming state (dedup keys, stream-join buffers) is bounded by
    the watermark *advancing*; a stalled or one-sided stream lets it grow without
    bound. This turns that silent OOM into an actionable signal. A no-op for empty
    state; the cap derives from `memory.streaming_state_max_bytes`.
    """
    if table is None or table.num_rows == 0:
        return
    from batcher.config import active_config

    cap = active_config().memory.streaming_state_budget_bytes()
    # Retained, not logical: streaming state is built by filtering old rows out of a
    # larger table, which in Arrow can leave a window pinning the pre-eviction parent.
    # Measuring the window is measuring the wrong table — the state would read as
    # shrinking on every eviction while the process held everything it ever buffered.
    held = retained_bytes(table)
    if held > cap:
        from batcher._internal.errors import ResourceError

        raise ResourceError(
            f"{label} streaming state reached {held} bytes (cap {cap}): the "
            "watermark is not advancing (a stalled or one-sided stream), so old rows "
            "never evict. Advance event time, narrow the keys, or raise "
            "memory.streaming_state_max_bytes."
        )


def _optimized_streaming_node(plan, sources: list, hub, expect: type):
    """Kyber-optimize a streaming node in place, keeping its node type.

    The streaming drivers dispatch by `isinstance` on an exact plan shape, so an
    optimization that replaced the node with something else — however sound — would
    silently fall out of the streaming path and into a materializing one. Rather than
    forbid that, this verifies the result is still the node the caller is about to read
    and falls back to the unoptimized plan otherwise. A rewrite Kyber wants but the
    driver cannot dispatch is a missed optimization; a rewrite the driver mis-reads is a
    wrong answer.

    `logical_rewrite` is the required entry point, not `optimize`/`optimize_full`: those
    build a `PhysicalPlan`, and the streaming nodes deliberately define no `to_ir()`
    because they are executed by the driver rather than lowered to Rust.

    Args:
        plan: The streaming node to optimize.
        sources: The bound sources, for cardinality and boundedness analysis.
        hub: The metadata hub carrying learned statistics.
        expect: The node type the caller requires the result to still be.

    Returns:
        The optimized node, or `plan` unchanged when optimization changed its shape.
    """
    from batcher.kyber.optimizer import Optimizer

    logical = Optimizer(None, sources, hub).logical_rewrite(plan)
    return logical if isinstance(logical, expect) else plan
