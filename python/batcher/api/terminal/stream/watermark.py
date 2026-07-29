"""Watermark-bounded streaming operators for `iter_batches` (control plane, `api`).

The seam: `dispatch` decides *which* streaming strategy a plan gets; this module owns
the two strategies whose bound on memory comes from **event time** rather than from the
plan's shape — watermark deduplication and the stream-stream interval join. Both retain
state across micro-batches and evict it as the watermark advances, so they share the
event-time normalization and the state-cap check that live here and nowhere else. Every
value-touching step still runs in the Rust engine; this only advances scalars and threads
the small state tables.
"""

from __future__ import annotations

from collections.abc import Iterator

import pyarrow as pa

from batcher.io.source import Source
from batcher.plan.types import retained_bytes

__all__ = ["stream_stream_join", "stream_watermark_dedup"]

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


def _event_micros(
    col: pa.Array | pa.ChunkedArray | pa.Scalar,
) -> pa.Array | pa.ChunkedArray | pa.Scalar:
    """Event-time ticks as int64 **microseconds**, whatever the column's resolution.

    Watermarks, `within`, and `lateness` are all microseconds. Reading the raw int64
    ticks of a non-`us` timestamp (e.g. `timestamp[ns]`) would scale the watermark by
    up to 1000x — evicting keys too early (re-emitting duplicates) or missing valid
    interval-join matches. Normalizing through `timestamp[us]` first keeps every
    comparison in the same unit. A column already in `us` (or int64) is unchanged.
    """
    import pyarrow.compute as pc

    return pc.cast(pc.cast(col, pa.timestamp("us")), pa.int64())


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


def stream_watermark_dedup(
    plan, source: Source, batch_size: int | None
) -> Iterator[pa.RecordBatch]:
    """Deduplicate a stream by `plan.subset`, evicting seen keys past the watermark.

    Per micro-batch: drop late rows, dedup the batch by `subset` (keep earliest by
    event time), anti-join against the running seen-keys table to emit only genuinely
    new keys, fold those keys into the seen set, advance the watermark, and evict seen
    keys older than it — so memory is bounded by the keys still inside the watermark
    window. Every value-touching step (filter, distinct, anti-join) runs in the Rust
    engine; this only advances a scalar and threads the small seen-keys table.
    """
    import pyarrow.compute as pc

    from batcher import core, kyber
    from batcher.api.session import from_arrow
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
    lateness = plan.lateness_micros
    opt = kyber.optimize(plan.input, sources=[source], hub=hub)
    seen: pa.Table | None = None
    wm: int | None = None

    # Read through Kyber's pushdown, which `optimize` has already computed. Reading with
    # `iter_batches(None)` decoded every column of every message no matter how narrow the
    # dedup subset was — the same gap the relational streaming path had, and worse here
    # because a dedup is usually a two-column question asked of a wide event.
    from batcher.io.source import iter_source

    for raw in iter_source(source, opt.source_projections.get(0), opt.source_predicates.get(0)):
        if raw.num_rows == 0:
            continue
        for b in core.execute_local(opt, [[raw]], feedback=hub):
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
            # Advance the watermark from this batch's max event time, fold the new
            # keys into state, then evict keys the watermark has now passed — every
            # batch, so duplicates falling out of the window are forgotten (bounded).
            hi = pc.max(table.column(et))
            if hi.is_valid:
                cand = _event_micros(hi).as_py() - lateness
                wm = cand if wm is None else max(wm, cand)
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
                yield from (
                    new.to_batches(max_chunksize=batch_size) if rebatch else new.to_batches()
                )


def stream_stream_join(
    plan, sources: list[Source], batch_size: int | None
) -> Iterator[pa.RecordBatch]:
    """Watermark-bounded stream-stream interval inner join over two sources.

    Symmetric incremental hash join: each side's rows are buffered; an arriving
    batch from one side joins against the *other* side's buffer (so every matching
    pair is emitted exactly once), filtered to the event-time interval
    ``|left_time - right_time| <= within``. Per-side watermarks advance from the
    batches; a buffered row is evicted once the opposite side's watermark guarantees
    no future match (``time < other_watermark - within``), keeping state bounded. The
    joins/filters run in the Rust engine; this threads the small buffers and scalars.
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
    # The interval filter differences the two event-time columns *of the join output*.
    # When both streams name their time column the same, the join suffixes the right
    # one (e.g. left `ts`, right `ts_right`); differencing the raw `left_time`/
    # `right_time` names would then read the left column twice, making every diff 0 so
    # every pair passes the interval filter. Resolve each side's real output alias.
    lt_out = next((o.alias for o in plan.output if o.side == "left" and o.name == lt), lt)
    rt_out = next((o.alias for o in plan.output if o.side == "right" and o.name == rt), rt)
    left_opt = kyber.optimize(plan.left, sources=[sources[0]], hub=hub)
    right_opt = kyber.optimize(remap_sources(plan.right, -1), sources=[sources[1]], hub=hub)

    state = {"bufL": None, "bufR": None, "wmL": None, "wmR": None}

    def micros(col):
        hi = pc.max(col)
        return _event_micros(hi).as_py() if hi.is_valid else None

    def evict(*, prune_left: bool, prune_right: bool) -> None:
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
        """
        if prune_left and state["wmR"] is not None and state["bufL"] is not None:
            keep = pc.greater_equal(_event_micros(state["bufL"].column(lt)), state["wmR"] - within)
            state["bufL"] = _compact(state["bufL"].filter(keep))
        if prune_right and state["wmL"] is not None and state["bufR"] is not None:
            keep = pc.greater_equal(_event_micros(state["bufR"].column(rt)), state["wmL"] - within)
            state["bufR"] = _compact(state["bufR"].filter(keep))

    def emit(side_table, other_buf, *, left_side):
        """Join `side_table` against the opposite buffer, interval-filtered."""
        if other_buf is None or other_buf.num_rows == 0:
            return []
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
            return []
        return res.to_batches() if batch_size is None else res.to_batches(max_chunksize=batch_size)

    def push(raw, opt, *, left_side):
        out = []
        for b in core.execute_local(opt, [[raw]], feedback=hub):
            if b.num_rows == 0:
                continue
            table = pa.Table.from_batches([b])
            other = state["bufR"] if left_side else state["bufL"]
            out.extend(emit(table, other, left_side=left_side))
            key = "bufL" if left_side else "bufR"
            state[key] = table if state[key] is None else pa.concat_tables([state[key], table])
            hi = micros(table.column(lt if left_side else rt))
            wk = "wmL" if left_side else "wmR"
            advanced = False
            if hi is not None:
                cand = hi - lateness
                previous = state[wk]
                state[wk] = cand if previous is None else max(previous, cand)
                advanced = state[wk] != previous
            # The side that just grew is always pruned; the other only when the watermark
            # that governs it actually moved.
            evict(
                prune_left=left_side or advanced,
                prune_right=(not left_side) or advanced,
            )
            # Either buffer grows unbounded if its opposite watermark stalls (a
            # one-sided stream), so cap both after eviction.
            _check_stream_state(state["bufL"], "stream-join")
            _check_stream_state(state["bufR"], "stream-join")
        return out

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
