"""Watermark deduplication — emit a key once, forget it when the watermark passes.

Spark's ``dropDuplicatesWithinWatermark``: the seen-key set is bounded by the watermark
rather than by the stream's lifetime, which is what makes dedup over an unbounded source
possible at all. Every value-touching step runs in the Rust engine; this advances the
shared `WatermarkTracker` and threads the small seen-keys table.
"""

from __future__ import annotations

from collections.abc import Iterator

import pyarrow as pa

from batcher.api.terminal.stream.watermark._state import (
    _check_stream_state,
    _compact,
    _event_micros,
    _optimized_streaming_node,
    _stream_tracker,
)
from batcher.io.source import Source

__all__ = ["stream_watermark_dedup"]


def stream_watermark_dedup(
    plan, source: Source, batch_size: int | None
) -> Iterator[pa.RecordBatch]:
    """Deduplicate a stream by `plan.subset`, evicting seen keys past the watermark.

    Per micro-batch: drop late rows, dedup the batch by `subset` (keep earliest by
    event time), anti-join against the running seen-keys table to emit only genuinely
    new keys, fold those keys into the seen set, advance the watermark, and evict seen
    keys older than it — so memory is bounded by the keys still inside the watermark
    window. Every value-touching step (filter, distinct, anti-join) runs in the Rust
    engine; this only advances the tracker and threads the small seen-keys table.
    """
    import pyarrow.compute as pc

    from batcher import core
    from batcher.api.session import from_arrow
    from batcher.api.terminal.stream.dispatch import _iter_streaming
    from batcher.plan.logical import WatermarkDedup

    hub = core.default_hub()
    # Optimize the dedup node *itself*, not just its input. Optimizing only the input
    # left the streaming operators — the two whose cost is dominated by retained state —
    # as the only nodes Kyber could never see, so a rule that shrinks the seen-key set
    # (`kyber.rules.streaming`) had nothing to fire on. The streaming rules preserve the
    # node type, so the driver below reads its fields exactly as before.
    plan = _optimized_streaming_node(plan, [source], hub, WatermarkDedup)
    subset = list(plan.subset)
    et = plan.event_time
    seen: pa.Table | None = None
    # One definition of how far event time has advanced, shared with the windowed
    # aggregate and the interval join: a minimum over the stream's partitions where the
    # rows still say which partition they came from, and a maximum where they do not.
    tracker, partition_cols = _stream_tracker(source, plan.lateness_micros)

    # Through the breaker-free router, which pushes the projection and predicate down --
    # reading with `iter_batches(None)` decoded every column of every message no matter how
    # narrow the dedup subset was, and a dedup is usually a two-column question asked of a
    # wide event. It also handles a `map_batches` beneath the dedup, which the private
    # optimize-and-execute loop this replaces could not: the UDF is Python and cannot be
    # lowered, so asking Kyber for a `PhysicalPlan` of a plan containing one raised
    # `NotImplementedError: map_batches is executed in Python, not lowered to the engine
    # IR`, and scoring a stream then deduplicating it could not run at all.
    for b in _iter_streaming(plan.input, [source], None):
        if b.num_rows == 0:
            continue
        if b.num_rows == 0:
            continue
        table = pa.Table.from_batches([b])
        # A watermark window is defined only for a row that has an event time, so
        # drop null-event-time rows uniformly. Post-watermark they were already
        # dropped as "late" (a null fails `>= wm`); pre-watermark they were kept,
        # folded into the seen set, then evicted on the very next batch (a null fails
        # the eviction `>= wm` too) — forgetting the key and re-emitting a later
        # duplicate as genuinely new. Dropping them keeps dedup sound and consistent.
        et_micros = _event_micros(table.column(et))
        mask = pc.is_valid(et_micros)
        # The frontier as it stood *before* this batch contributes to it, re-read from the
        # tracker so a partition that crossed its idleness threshold since the last batch
        # releases the minimum here rather than a batch later.
        wm = tracker.watermark
        if wm is not None:  # also drop rows below the watermark (late)
            mask = pc.and_kleene(mask, pc.greater_equal(et_micros, wm))
        table = table.filter(mask)
        if table.num_rows == 0:
            continue
        # Duplicate check against the seen-keys state *before* advancing the
        # watermark (a key is a duplicate while it is still in state).
        deduped = from_arrow(table).distinct(subset, keep="first", order_by=[(et, False)])
        if seen is not None:
            new = deduped.join(from_arrow(seen), on=subset, how="anti").collect()
        else:
            new = deduped.collect()
        # Advance the watermark from this batch's per-partition maxima, fold the new
        # keys into state, then evict keys the watermark has now passed — every
        # batch, so duplicates falling out of the window are forgotten (bounded).
        tracker.observe(table, et, partition_cols)
        wm = tracker.watermark
        if new.num_rows:
            # `dict.fromkeys` rather than a set literal: order is the state table's
            # column order and must be stable, and a dedup keyed *on* the event-time
            # column would otherwise select it twice and give the state two columns
            # with one name — which the anti-join then resolves ambiguously.
            fresh = new.select(list(dict.fromkeys([*subset, et])))
            seen = fresh if seen is None else pa.concat_tables([seen, fresh])
        if seen is not None and wm is not None:
            keep = pc.greater_equal(_event_micros(seen.column(et)), wm)
            seen = seen.filter(keep)
        seen = _compact(seen)
        _check_stream_state(seen, "watermark-dedup")
        if new.num_rows:
            rebatch = batch_size is not None
            yield from (new.to_batches(max_chunksize=batch_size) if rebatch else new.to_batches())
