"""The watermark-bounded stream-stream interval join, inner and outer.

A symmetric incremental hash join over two unbounded sources: each side buffers its rows,
an arriving batch joins against the *other* side's buffer, and a buffered row is evicted
once the opposite watermark guarantees no future match. The interval bound is what makes
the eviction sound, and the eviction is what keeps the state bounded.

It is also what makes an **outer** join expressible here at all. "This row will never
match" is not a decidable statement about an unbounded stream — but "this row can no
longer match, because any partner would have to carry an event time the watermark has
already passed" is. So an unmatched row is emitted null-padded at exactly the moment it
leaves the buffer, once, and never again. That is Spark's rule too, and it is why an
outer stream-stream join without a time bound is refused rather than approximated.
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

__all__ = ["stream_stream_join"]

#: Per-side bookkeeping columns carried inside the buffers and stripped from the output.
#: Distinct names on the two sides, so the join carries both without suffixing either.
_LEFT_ID = "__sj_left_id"
_RIGHT_ID = "__sj_right_id"
#: Whether a buffered row has already been emitted as part of a matched pair. Only an
#: outer join maintains it; an inner join never reads it and pays nothing for it.
_HIT = "__sj_hit"


def stream_stream_join(
    plan, sources: list[Source], batch_size: int | None
) -> Iterator[pa.RecordBatch]:
    """Watermark-bounded stream-stream interval join over two sources.

    Symmetric incremental hash join: each side's rows are buffered; an arriving
    batch from one side joins against the *other* side's buffer (so every matching
    pair is emitted exactly once), filtered to the event-time interval
    ``|left_time - right_time| <= within``. Per-side watermarks advance from the
    batches; a buffered row is evicted once the opposite side's watermark guarantees
    no future match (``time < other_watermark - within``), keeping state bounded. The
    joins/filters run in the Rust engine; this threads the small buffers and scalars.

    Under ``how="left"`` / ``"right"`` / ``"full"`` a row that reaches that eviction
    point without ever having matched is emitted once, padded with nulls on the other
    side. Anything still buffered when both streams end is flushed the same way, because
    end-of-stream is the last moment at which "no match will arrive" becomes true.
    """
    import pyarrow.compute as pc

    from batcher import core, kyber
    from batcher.api.session import from_arrow
    from batcher.plan.logical import WatermarkStreamJoin, remap_sources

    hub = core.default_hub()
    # Optimize the join node itself, not just each side in isolation. Optimizing the
    # sides separately meant no rule could ever reason *across* the join — a predicate
    # above it could not be pushed into the side it constrains, so rows were buffered,
    # matched, emitted, and only then discarded. Under a stream that buffer is the
    # query's memory bound, not a CPU detail (`kyber.rules.streaming`).
    plan = _optimized_streaming_node(plan, list(sources), hub, WatermarkStreamJoin)
    lt, rt = plan.left_time, plan.right_time
    within, lateness = plan.within_micros, plan.lateness_micros
    lk, rk = list(plan.left_keys), list(plan.right_keys)
    outer_left, outer_right = plan.emits_unmatched_left, plan.emits_unmatched_right
    tracking = outer_left or outer_right
    aliases = [o.alias for o in plan.output]
    out_schema = _output_schema(plan)
    # The interval filter differences the two event-time columns *of the join output*.
    # When both streams name their time column the same, the join suffixes the right
    # one (e.g. left `ts`, right `ts_right`); differencing the raw `left_time`/
    # `right_time` names would then read the left column twice, making every diff 0 so
    # every pair passes the interval filter. Resolve each side's real output alias.
    lt_out = next((o.alias for o in plan.output if o.side == "left" and o.name == lt), lt)
    rt_out = next((o.alias for o in plan.output if o.side == "right" and o.name == rt), rt)
    left_opt = kyber.optimize(plan.left, sources=[sources[0]], hub=hub)
    right_opt = kyber.optimize(remap_sources(plan.right, -1), sources=[sources[1]], hub=hub)

    state: dict = {"bufL": None, "bufR": None, "wmL": None, "wmR": None}
    counters = {"left": 0, "right": 0}
    # One tracker per *side*, because the two sides advance independently and the eviction
    # rule already reasons across them (a left row is evictable on the right's watermark).
    # Within a side the frontier is the minimum over that stream's partitions, which is the
    # part a single `max` got wrong: a two-partition topic whose second partition lagged had
    # its buffered rows evicted on the first partition's clock, so the pairs that would have
    # matched were gone before their partner arrived — an inner join quietly missing rows,
    # and an outer join emitting them null-padded as though nothing could ever match.
    trackers = {
        "left": _stream_tracker(sources[0], lateness),
        "right": _stream_tracker(sources[1], lateness),
    }

    def tag(table: pa.Table, *, left_side: bool) -> pa.Table:
        """Give each arriving row an id, and (for an outer join) an unmatched marker."""
        if not tracking:
            return table
        side = "left" if left_side else "right"
        start = counters[side]
        counters[side] = start + table.num_rows
        ids = pa.array(range(start, counters[side]), type=pa.int64())
        table = table.append_column(_LEFT_ID if left_side else _RIGHT_ID, ids)
        return table.append_column(_HIT, pa.array([False] * table.num_rows, type=pa.bool_()))

    def mark_matched(buf: pa.Table | None, id_column: str, matched) -> pa.Table | None:
        """Set the unmatched marker to False for every buffered row that just matched."""
        if buf is None or buf.num_rows == 0 or matched is None or len(matched) == 0:
            return buf
        hit = pc.or_(buf.column(_HIT), pc.is_in(buf.column(id_column), value_set=matched))
        return buf.set_column(buf.schema.get_field_index(_HIT), _HIT, hit)

    def unmatched_rows(dropped: pa.Table | None, *, left_side: bool) -> list[pa.RecordBatch]:
        """The null-padded output rows for buffered rows evicted without ever matching."""
        if dropped is None or dropped.num_rows == 0:
            return []
        still = dropped.filter(pc.invert(dropped.column(_HIT)))
        if still.num_rows == 0:
            return []
        padded = _pad(still, plan.output, out_schema, "left" if left_side else "right")
        return _rebatch(padded, batch_size)

    def evict(*, prune_left: bool, prune_right: bool) -> list[pa.RecordBatch]:
        """Drop buffered rows the opposite watermark has moved past.

        A buffered left row t can still match a future right row only while
        ``t >= wmR - within`` (and symmetrically), so eviction of one side depends on the
        *other* side's watermark and on that side's own new rows.

        Both sides used to be re-filtered on every batch of either stream. Half of that was
        always redundant: pushing a left batch cannot make a buffered right row evictable
        unless it advanced `wmL`, and a filter over the whole retained buffer is the most
        expensive thing this operator does per micro-batch. The side that just received rows
        is always pruned (an arriving row can already be outside the window — this path has
        no late filter); the opposite side only when its governing watermark moved.

        Eviction is also the only moment an outer join can speak: a row leaving the buffer
        unmatched is a row no future partner can reach, so it is emitted here.
        """
        emitted: list[pa.RecordBatch] = []
        if prune_left and state["wmR"] is not None and state["bufL"] is not None:
            keep = pc.greater_equal(_event_micros(state["bufL"].column(lt)), state["wmR"] - within)
            if outer_left:
                emitted.extend(
                    unmatched_rows(state["bufL"].filter(pc.invert(keep)), left_side=True)
                )
            state["bufL"] = _compact(state["bufL"].filter(keep))
        if prune_right and state["wmL"] is not None and state["bufR"] is not None:
            keep = pc.greater_equal(_event_micros(state["bufR"].column(rt)), state["wmL"] - within)
            if outer_right:
                emitted.extend(
                    unmatched_rows(state["bufR"].filter(pc.invert(keep)), left_side=False)
                )
            state["bufR"] = _compact(state["bufR"].filter(keep))
        return emitted

    def emit(side_table, other_buf, *, left_side):
        """Join `side_table` against the opposite buffer, interval-filtered.

        Returns the output batches and, for an outer join, the ids on each side that
        matched — which is how a buffered row learns it must not be emitted null-padded
        when its window closes.
        """
        if other_buf is None or other_buf.num_rows == 0:
            return [], None, None
        left_ds = from_arrow(side_table if left_side else other_buf)
        right_ds = from_arrow(other_buf if left_side else side_table)
        joined = left_ds.join(right_ds, left_on=lk, right_on=rk, how="inner")
        # Normalize event time to microseconds (`within` is micros) before differencing,
        # so a non-`us` timestamp (e.g. ns) is not compared 1000x off and missing matches.
        diff = joined[lt_out].cast("timestamp").cast("int64") - joined[rt_out].cast(
            "timestamp"
        ).cast("int64")
        res = joined.filter((diff <= within) & (diff >= -within)).collect()
        if res.num_rows == 0:
            return [], None, None
        left_hits = res.column(_LEFT_ID) if tracking else None
        right_hits = res.column(_RIGHT_ID) if tracking else None
        # The bookkeeping columns are the join's, not the query's: select the declared
        # output aliases so an outer join's rows and an inner join's are the same shape.
        return _rebatch(res.select(aliases), batch_size), left_hits, right_hits

    def push(raw, opt, *, left_side):
        out = []
        for b in core.execute_local(opt, [[raw]], feedback=hub):
            if b.num_rows == 0:
                continue
            table = tag(pa.Table.from_batches([b]), left_side=left_side)
            other = state["bufR"] if left_side else state["bufL"]
            batches, left_hits, right_hits = emit(table, other, left_side=left_side)
            out.extend(batches)
            key = "bufL" if left_side else "bufR"
            state[key] = table if state[key] is None else pa.concat_tables([state[key], table])
            if tracking:
                # After the append, so the arriving rows learn about their own matches too.
                state["bufL"] = mark_matched(state["bufL"], _LEFT_ID, left_hits)
                state["bufR"] = mark_matched(state["bufR"], _RIGHT_ID, right_hits)
            tracker, partition_cols = trackers["left" if left_side else "right"]
            tracker.observe(table, lt if left_side else rt, partition_cols)
            wk = "wmL" if left_side else "wmR"
            previous = state[wk]
            state[wk] = tracker.watermark
            advanced = state[wk] != previous
            # The side that just grew is always pruned; the other only when the watermark
            # that governs it actually moved.
            out.extend(
                evict(
                    prune_left=left_side or advanced,
                    prune_right=(not left_side) or advanced,
                )
            )
            # Either buffer grows unbounded if its opposite watermark stalls (a
            # one-sided stream), so cap both after eviction.
            _check_stream_state(state["bufL"], "stream-join")
            _check_stream_state(state["bufR"], "stream-join")
        return out

    def flush() -> list[pa.RecordBatch]:
        """Emit what is still buffered and unmatched once both streams have ended.

        The watermark can only ever declare a row unreachable *inside* the stream. A row
        buffered when the input ends is unreachable for a different reason — there is no
        more input — and without this it would simply never be emitted, so a left outer
        join would silently drop the tail of its preserved side.
        """
        emitted: list[pa.RecordBatch] = []
        if outer_left:
            emitted.extend(unmatched_rows(state["bufL"], left_side=True))
        if outer_right:
            emitted.extend(unmatched_rows(state["bufR"], left_side=False))
        return emitted

    # Both sides read through the pushdown their own `optimize` already computed; reading
    # with `iter_batches(None)` decoded every column of both streams, and on this path the
    # decoded columns are also *buffered* until the watermark releases them.
    from batcher.io.source import iter_source

    it_l = iter_source(
        sources[0], left_opt.source_projections.get(0), left_opt.source_predicates.get(0)
    )
    it_r = iter_source(
        sources[1], right_opt.source_projections.get(0), right_opt.source_predicates.get(0)
    )
    done_l = done_r = False
    while not (done_l and done_r):
        if not done_l:
            try:
                raw = next(it_l)
            except StopIteration:
                done_l = True
            else:
                if raw.num_rows:
                    yield from push(raw, left_opt, left_side=True)
        if not done_r:
            try:
                raw = next(it_r)
            except StopIteration:
                done_r = True
            else:
                if raw.num_rows:
                    yield from push(raw, right_opt, left_side=False)
    if tracking:
        yield from flush()


def _output_schema(plan) -> pa.Schema:
    """The join's declared output schema, which the null-padded rows must match exactly.

    A null-padded row is built column by column rather than by joining, so it has no
    schema of its own — and a batch whose types differ from the matched rows' cannot be
    concatenated with them by any consumer. Every field is nullable, because on this path
    every one of them can be the null half of a pair.
    """
    schema = plan.available_schema()
    if schema is None:  # pragma: no cover - an opaque side (a UDF) types nothing
        raise _unknown_schema()
    return pa.schema([field.with_nullable(True) for field in schema.arrow])


def _unknown_schema():
    """The error an outer stream join raises when its output schema is not knowable."""
    from batcher._internal.errors import PlanError

    return PlanError(
        "an outer stream-stream join needs a statically known output schema to build "
        "its null-padded rows; one side's schema is opaque (a map_batches UDF). Use "
        "how='inner', or give the UDF an explicit schema."
    )


def _pad(rows: pa.Table, output, schema: pa.Schema, side: str) -> pa.Table:
    """One side's rows in the join's output shape, with the other side's columns null."""
    arrays = []
    for field, out_col in zip(schema, output, strict=True):
        if out_col.side == side:
            arrays.append(rows.column(out_col.name).cast(field.type))
        else:
            arrays.append(pa.nulls(rows.num_rows, type=field.type))
    return pa.Table.from_arrays(arrays, schema=schema)


def _rebatch(table: pa.Table, batch_size: int | None) -> list[pa.RecordBatch]:
    """The table's batches, sliced to `batch_size` rows when one was asked for."""
    if table.num_rows == 0:
        return []
    return table.to_batches() if batch_size is None else table.to_batches(max_chunksize=batch_size)
