"""Streaming terminal path for `Dataset.iter_batches`.

`_iter_batches` picks the most bounded-memory way to yield a plan's result as Arrow
batches, in order of preference:

1. a breaker-free pipeline streams one source batch at a time (`_iter_streaming`);
2. a top-level aggregate / distinct / top-N / limit over such a pipeline streams via
   the running-state drivers in `core.streaming`;
3. a top-level pipeline breaker (sort / join / window) over bounded sources streams
   from the out-of-core bucket pipeline in `dist.spill_breakers` — input consumed to
   disk, then the result yielded one bounded bucket at a time;
4. anything else materializes via `_collect` and re-chunks.

An unbounded (streaming) source whose plan must materialize raises `PlanError`
instead of hanging. Kept separate from the materializing terminals in `terminal` so
that file stays within its size budget; `terminal` re-exports `_iter_batches`.
"""

from __future__ import annotations

from collections.abc import Iterator

import pyarrow as pa

from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan

__all__ = ["_iter_batches", "_iter_streaming"]


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
    if table.nbytes > cap:
        from batcher._internal.errors import ResourceError

        raise ResourceError(
            f"{label} streaming state reached {table.nbytes} bytes (cap {cap}): the "
            "watermark is not advancing (a stalled or one-sided stream), so old rows "
            "never evict. Advance event time, narrow the keys, or raise "
            "memory.streaming_state_max_bytes."
        )


def _iter_batches(
    plan: LogicalPlan,
    sources: list[Source],
    columns: list[str],
    batch_size: int | None = None,
    *,
    distributed: bool = False,
    num_workers: int | None = None,
    transport: str = "auto",
) -> Iterator[pa.RecordBatch]:
    """Execute and yield the result as Arrow record batches.

    The bounded-memory streaming path is chosen automatically whenever the plan
    supports it (breaker-free pipeline, a top-level aggregate / distinct / top-N over
    one, or a top-level sort / join / window streamed from the out-of-core bucket
    pipeline); other plans materialize first. An unbounded source whose plan cannot
    stream raises `PlanError` instead of hanging on `_collect`.

    When `distributed`, a top-level breaker (sort / join / aggregate / window) fans out
    across Ray workers and its result streams back one reducer bucket at a time
    (`_iter_distributed`), so the driver never holds the whole distributed result.
    """
    from batcher.io.source import is_bounded
    from batcher.plan.logical import (
        Aggregate,
        Distinct,
        Limit,
        Sort,
        WatermarkDedup,
        WatermarkStreamJoin,
        is_streamable,
    )

    # A distributed breaker streams its result off the workers one bucket at a time,
    # bounding driver memory. A breaker-free pipeline already streams in bounded memory
    # single-node, so it stays on that path even when `distributed` is requested.
    if distributed and not is_streamable(plan) and all(is_bounded(s) for s in sources):
        from batcher.api.terminal.distributed_stream import iter_distributed

        yield from iter_distributed(plan, sources, columns, num_workers, transport, batch_size)
        return

    # A distributed breaker-free scan/filter/project over a SPLITTABLE source fans the
    # read out across workers AND streams each worker's output back one partition at a
    # time — parallel reads with the driver holding only one partition's result, the
    # bounded-memory way to pull a huge distributed scan. In-memory sources (which would
    # be shipped to workers) and `map_batches` pipelines stay on their existing paths.
    if distributed and is_streamable(plan):
        from batcher.api.terminal.distributed_stream import (
            distributable_scan_source,
            iter_distributed_scan,
        )

        if distributable_scan_source(plan, sources) is not None:
            yield from iter_distributed_scan(plan, sources, num_workers, batch_size)
            return

    # Stream-stream interval join: two streams, buffered + watermark-evicted.
    if isinstance(plan, WatermarkStreamJoin) and len(sources) == 2:
        yield from _stream_stream_join(plan, sources, batch_size)
        return

    if len(sources) == 1:
        if is_streamable(plan):
            yield from _iter_streaming(plan, sources, batch_size)
            return
        # Watermark-bounded streaming deduplication (bounded seen-key state).
        if isinstance(plan, WatermarkDedup) and is_streamable(plan.input):
            yield from _stream_watermark_dedup(plan, sources[0], batch_size)
            return
        # A top-level aggregate/distinct over a breaker-free relational input streams
        # with bounded memory: fold each micro-batch's partial into one running state.
        from batcher import core

        if (
            isinstance(plan, (Aggregate, Distinct))
            and is_streamable(plan.input)
            and not core.has_map_batches(plan.input)
        ):
            # A watermarked windowed aggregation emits each window as the watermark
            # closes it (bounded state); a plain aggregate folds one running state.
            if isinstance(plan, Aggregate) and plan.watermark is not None:
                from batcher.core.streaming import stream_windowed_aggregate

                yield from stream_windowed_aggregate(plan, sources[0], batch_size)
                return
            from batcher.core.streaming import stream_aggregate, stream_distinct

            driver = stream_distinct if isinstance(plan, Distinct) else stream_aggregate
            yield from driver(plan, sources[0], batch_size)
            return
        # Top-N (`head` over a sort) streams with memory bounded by N: keep only the
        # running best N rows.
        if (
            isinstance(plan, Limit)
            and plan.offset == 0
            and isinstance(plan.input, Sort)
            and is_streamable(plan.input.input)
            and not core.has_map_batches(plan.input.input)
        ):
            from batcher.core.streaming import stream_topn

            yield from stream_topn(plan.input, plan.n, sources[0], batch_size)
            return
        # A plain `Limit` over a breaker-free pipeline streams and stops early.
        if (
            isinstance(plan, Limit)
            and is_streamable(plan.input)
            and not core.has_map_batches(plan.input)
        ):
            from batcher.core.streaming import stream_limit

            yield from stream_limit(plan, sources[0], batch_size)
            return

    # Pipeline breakers (sort / join / window) over bounded sources stream their result
    # from the out-of-core bucket pipeline: the input is consumed to disk, then the
    # globally-ordered (sort), co-partition-joined (join), or per-partition-windowed
    # result is yielded one bounded bucket at a time — peak memory is a single bucket,
    # not the whole result. Each side must be a breaker-free single-source chain so the
    # per-batch map is valid.
    if all(is_bounded(s) for s in sources):
        from batcher.dist.spill_breakers import (
            stream_spilling_join,
            stream_spilling_sort,
            stream_spilling_window,
            supports_spilling_sort,
            supports_spilling_window,
        )
        from batcher.plan.logical import Join, Window

        gen = None
        if isinstance(plan, Sort) and supports_spilling_sort(plan) and is_streamable(plan.input):
            gen = stream_spilling_sort(plan, sources)
        elif isinstance(plan, Join) and is_streamable(plan.left) and is_streamable(plan.right):
            gen = stream_spilling_join(plan, sources)
        elif isinstance(plan, Window) and is_streamable(plan.input):
            # PARTITION BY window grace-partitions by those keys; a global window
            # (no PARTITION BY, single plain-column ORDER BY) streams via ordered-
            # bucket offsetting.
            if supports_spilling_window(plan):
                gen = stream_spilling_window(plan, sources)
            else:
                from batcher.dist.window_stream import (
                    stream_spilling_global_window,
                    supports_streaming_global_window,
                )

                if supports_streaming_global_window(plan):
                    gen = stream_spilling_global_window(plan, sources)
        if gen is not None:
            for b in gen:
                if batch_size is None:
                    yield b
                else:
                    for off in range(0, b.num_rows, batch_size):
                        yield b.slice(off, batch_size)
            return

    # No streaming path applies → materialize. An unbounded source would never
    # finish, so refuse instead of hanging.
    if any(not is_bounded(s) for s in sources):
        from batcher._internal.errors import PlanError

        raise PlanError(
            "this pipeline has an unbounded (streaming) source but a plan that must "
            "materialize (e.g. sort / join / window / multi-source), which cannot be "
            "streamed in bounded memory. Restructure to a streamable shape (filter / "
            "project / map_batches, or a single top-level aggregate / distinct / top-N)."
        )
    from batcher.api.terminal.core import _collect

    table = _collect(plan, sources, columns)
    batches = (
        table.to_batches() if batch_size is None else table.to_batches(max_chunksize=batch_size)
    )
    yield from batches


def _iter_streaming(
    plan: LogicalPlan, sources: list[Source], batch_size: int | None
) -> Iterator[pa.RecordBatch]:
    """Drive a breaker-free pipeline one source batch at a time."""
    from batcher import core, kyber
    from batcher.io.source import InMemorySource, iter_source

    source = sources[0]

    # map_batches pipelines are orchestrated in Python (no Kyber pass over the
    # opaque UDF), mirroring collect(); the relational path is optimized so the
    # source projection (and predicate, for capable sources) is pushed down.
    if core.has_map_batches(plan):

        def run(batch):
            return core.execute_with_udfs(plan, [InMemorySource([batch])])

        projection = None
        predicate = None
    else:
        hub = core.default_hub()
        opt_plan = kyber.optimize(plan, sources=sources, hub=hub)
        projection = opt_plan.source_projections.get(0)
        predicate = opt_plan.source_predicates.get(0)

        # Close the metadata loop on the streaming path too: each micro-batch's
        # per-operator stats feed the learner, so streaming queries also improve
        # future plans (cost calibration, cardinality, selectivity).
        def run(batch):
            return core.execute_local(opt_plan, [[batch]], feedback=hub)

    for batch in iter_source(source, projection, predicate):
        if batch.num_rows == 0:
            continue
        for b in run(batch):
            if b.num_rows == 0:
                continue
            if batch_size is None:
                yield b
            else:
                for off in range(0, b.num_rows, batch_size):
                    yield b.slice(off, batch_size)


def _stream_watermark_dedup(
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

    subset = list(plan.subset)
    et = plan.event_time
    lateness = plan.lateness_micros
    hub = core.default_hub()
    opt = kyber.optimize(plan.input, sources=[source], hub=hub)
    seen: pa.Table | None = None
    wm: int | None = None

    for raw in source.iter_batches(None):
        if raw.num_rows == 0:
            continue
        for b in core.execute_local(opt, [[raw]], feedback=hub):
            if b.num_rows == 0:
                continue
            table = pa.Table.from_batches([b])
            if wm is not None:  # drop rows below the watermark (late)
                table = table.filter(pc.greater_equal(pc.cast(table.column(et), pa.int64()), wm))
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
                cand = pc.cast(hi, pa.int64()).as_py() - lateness
                wm = cand if wm is None else max(wm, cand)
            if new.num_rows:
                fresh = new.select([*subset, et])
                seen = fresh if seen is None else pa.concat_tables([seen, fresh])
            if seen is not None and wm is not None:
                keep = pc.greater_equal(pc.cast(seen.column(et), pa.int64()), wm)
                seen = seen.filter(keep)
            _check_stream_state(seen, "watermark-dedup")
            if new.num_rows:
                rebatch = batch_size is not None
                yield from (
                    new.to_batches(max_chunksize=batch_size) if rebatch else new.to_batches()
                )


def _stream_stream_join(
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
    from batcher.plan.logical import remap_sources

    lt, rt = plan.left_time, plan.right_time
    within, lateness = plan.within_micros, plan.lateness_micros
    lk, rk = list(plan.left_keys), list(plan.right_keys)
    hub = core.default_hub()
    left_opt = kyber.optimize(plan.left, sources=[sources[0]], hub=hub)
    right_opt = kyber.optimize(remap_sources(plan.right, -1), sources=[sources[1]], hub=hub)

    state = {"bufL": None, "bufR": None, "wmL": None, "wmR": None}

    def micros(col):
        hi = pc.max(col)
        return pc.cast(hi, pa.int64()).as_py() if hi.is_valid else None

    def evict():
        # A buffered row can be dropped once the *other* stream's watermark has moved
        # past its match window: left row t can still match a future right row only if
        # t >= wmR - within (and symmetrically).
        if state["wmR"] is not None and state["bufL"] is not None:
            keep = pc.greater_equal(
                pc.cast(state["bufL"].column(lt), pa.int64()), state["wmR"] - within
            )
            state["bufL"] = state["bufL"].filter(keep)
        if state["wmL"] is not None and state["bufR"] is not None:
            keep = pc.greater_equal(
                pc.cast(state["bufR"].column(rt), pa.int64()), state["wmL"] - within
            )
            state["bufR"] = state["bufR"].filter(keep)

    def emit(side_table, other_buf, *, left_side):
        """Join `side_table` against the opposite buffer, interval-filtered."""
        if other_buf is None or other_buf.num_rows == 0:
            return []
        left_ds = from_arrow(side_table if left_side else other_buf)
        right_ds = from_arrow(other_buf if left_side else side_table)
        joined = left_ds.join(right_ds, left_on=lk, right_on=rk, how="inner")
        diff = joined[lt].cast("int64") - joined[rt].cast("int64")
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
            if hi is not None:
                wk = "wmL" if left_side else "wmR"
                cand = hi - lateness
                state[wk] = cand if state[wk] is None else max(state[wk], cand)
            evict()
            # Either buffer grows unbounded if its opposite watermark stalls (a
            # one-sided stream), so cap both after eviction.
            _check_stream_state(state["bufL"], "stream-join")
            _check_stream_state(state["bufR"], "stream-join")
        return out

    it_l, it_r = sources[0].iter_batches(None), sources[1].iter_batches(None)
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
