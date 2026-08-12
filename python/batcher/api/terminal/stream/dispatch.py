"""Streaming-strategy selection for `Dataset.iter_batches` (control plane, `api`).

The seam: this module is the *router*. It inspects a plan and picks the most
bounded-memory way to yield its result. Driving the two paths that are pure plan shape —
the breaker-free pipeline and the peeled row-wise re-application — lives next door in
`pipeline`, and the exact-size rebatcher in `rebatch`. Strategies with their own
retained state live beside it (`watermark`), as do their proof obligations (`union`) and
the distributed (`distributed_stream`), map (`map_stream`), running-state
(`core.streaming`), and out-of-core bucket (`dist.spill_breakers`) drivers it delegates
to. Preference order:

1. a breaker-free pipeline streams one source batch at a time (`_iter_streaming`);
2. a top-level aggregate / distinct / top-N / limit over such a pipeline streams via
   the running-state drivers in `core.streaming`;
3. a top-level pipeline breaker (sort / join / window) over bounded sources streams
   from the out-of-core bucket pipeline in `dist.spill_breakers` — input consumed to
   disk, then the result yielded one bounded bucket at a time;
4. row-wise operators stacked above a breaker are peeled and re-applied per batch;
5. anything else materializes via `_collect` and re-chunks.

A top-level UNION ALL is decomposed *before* any of these (and before the distributed
routes): each branch re-enters this router on its own, so it streams by whichever of the
five suits it, and the driver ever holds one branch's one batch (`union`).

An unbounded (streaming) source whose plan must materialize raises `PlanError`
instead of hanging.
"""

from __future__ import annotations

from collections.abc import Iterator

import pyarrow as pa

from batcher.api.terminal.stream.pipeline import _apply_peeled, _iter_streaming, _pushdown
from batcher.api.terminal.stream.rebatch import _rebatch_exact, _take
from batcher.api.terminal.stream.static_join import (
    refuse_reason as static_join_refusal,
)
from batcher.api.terminal.stream.static_join import (
    stream_static_join,
    stream_static_sides,
)
from batcher.api.terminal.stream.union import (
    interleave,
    union_branch_sources,
    union_streams_branchwise,
    union_streams_interleaved,
)
from batcher.api.terminal.stream.watermark import stream_stream_join, stream_watermark_dedup
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan

#: `_iter_streaming` and `_pushdown` are re-exported rather than merely imported: both are
#: reached by name from outside this package (`terminal.core`, the launcher-parity tests),
#: and this is the path they have always used. A move that changed it would also silently
#: defuse any monkeypatch aimed at the old name.
__all__ = ["_iter_batches", "_iter_streaming", "_pushdown"]


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
        StreamingSessionWindow,
        TransformWithState,
        Union,
        WatermarkDedup,
        WatermarkStreamJoin,
        is_partition_independent,
        is_streamable,
        remap_sources,
    )

    # `batch_size` is an *exact* output-granularity contract ("rebatch the output to
    # this many rows"), which the per-path chunkers below cannot honor: slicing each
    # engine batch/chunk independently flushes a short batch at every boundary (e.g. a
    # sorted result yields 1000, 1000, 651, 1000, … rows). Run the natural-batch path
    # and coalesce once at the boundary so every emitted batch is exactly `batch_size`
    # rows except the final remainder — matching the engine's own `map_batches`
    # rebatch. The inner call passes `None`, so the chunkers stay on their pass-through
    # branch and this delegation cannot recurse.
    if batch_size is not None:
        if batch_size < 1:
            from batcher._internal.errors import PlanError

            raise PlanError(f"iter_batches(): batch_size must be >= 1, got {batch_size}")
        raw = _iter_batches(
            plan,
            sources,
            columns,
            None,
            distributed=distributed,
            num_workers=num_workers,
            transport=transport,
        )
        yield from _rebatch_exact(raw, batch_size)
        return

    # UNION ALL is decomposed BEFORE the generic distributed-breaker route below, both
    # single-node and distributed. `_distributed_union` runs each branch to a driver table
    # and concatenates, so routing a union there materializes the whole result on the driver
    # — exactly what streaming exists to avoid. Decomposing first sends each branch through
    # this router on its own, where a branch that is itself a breaker takes the distributed
    # bucket-at-a-time path and a breaker-free branch takes the fan-out scan. The driver then
    # holds one branch's one bucket.
    if isinstance(plan, Union) and union_streams_branchwise(plan, sources):
        for branch, sid in union_branch_sources(plan):
            yield from _iter_batches(
                remap_sources(branch, -sid),
                [sources[sid]],
                branch.available_columns(),
                None,
                distributed=distributed,
                num_workers=num_workers,
                transport=transport,
            )
        return

    # A union over *streams* interleaves instead of concatenating: an unbounded branch
    # never ends, so concatenation would emit branch 0 forever and branch 1 never. UNION
    # ALL is a multiset union and makes no ordering claim, so a row from whichever branch
    # has one next is as correct as any other order — which is what makes this sound.
    if isinstance(plan, Union) and union_streams_interleaved(plan, sources):
        yield from interleave(
            [
                _iter_batches(
                    remap_sources(branch, -sid),
                    [sources[sid]],
                    branch.available_columns(),
                    batch_size,
                )
                for branch, sid in union_branch_sources(plan)
            ]
        )
        return

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

    # Stream-static join: one side is a table that does not move, so it is read once and
    # every micro-batch joins against the whole of it. A `Join` is a pipeline breaker, so
    # without this the router saw an unbounded input beneath a breaker and refused the most
    # common thing anyone does to a stream.
    static_sides = stream_static_sides(plan, sources)
    if static_sides is not None:
        from batcher._internal.errors import PlanError

        reason = static_join_refusal(plan.join_type, static_sides[0])
        if reason is not None:
            raise PlanError(reason)
        yield from stream_static_join(plan, sources, static_sides[0], batch_size)
        return

    # Stream-stream interval join: two streams, buffered + watermark-evicted.
    if isinstance(plan, WatermarkStreamJoin) and len(sources) == 2:
        yield from stream_stream_join(plan, sources, batch_size)
        return
    # A WatermarkStreamJoin only ever reaches this router with more than two stream
    # sources when `join_stream` calls were nested (`a.join_stream(b).join_stream(c)`).
    # The buffered symmetric hash join drives exactly two sides, so name the limit
    # rather than falling through to the generic "must materialize" error below.
    if isinstance(plan, WatermarkStreamJoin):
        from batcher._internal.errors import PlanError

        raise PlanError(
            f"join_stream() joins exactly two streams, but this plan chains "
            f"{len(sources)} stream sources through nested join_stream calls. A 3+-way "
            "stream-stream interval join is not supported. Materialize an intermediate "
            "result to a bounded source before joining the next stream, or restructure "
            "to a single two-stream join."
        )

    if len(sources) == 1:
        if is_streamable(plan):
            yield from _iter_streaming(plan, sources, batch_size)
            return
        # Watermark-bounded streaming deduplication (bounded seen-key state).
        if isinstance(plan, WatermarkDedup) and is_streamable(plan.input):
            yield from stream_watermark_dedup(plan, sources[0], batch_size)
            return
        # Arbitrary keyed state: the user function owns one key's state, the engine owns
        # when it is called and expired. State is bounded by the key space and the TTL.
        # Gap-based sessions: a session's end is only knowable once the gap has passed
        # with nothing arriving, so rows are held until the watermark says so.
        if isinstance(plan, StreamingSessionWindow) and is_streamable(plan.input):
            from batcher.api.terminal.stream.session import stream_session_window

            yield from stream_session_window(plan, sources[0], batch_size)
            return
        if isinstance(plan, TransformWithState) and is_streamable(plan.input):
            from batcher.core.streaming import stream_keyed_state

            yield from stream_keyed_state(plan, sources[0], batch_size)
            return
        # A top-level aggregate/distinct over a breaker-free relational input streams
        # with bounded memory: fold each micro-batch's partial into one running state.
        from batcher import core

        # A plain aggregate over a `map_batches` input streams too: apply the (parallel,
        # windowed) map in bounded memory and fold each mapped batch's partial into the
        # running state — so a large `map→agg` (Ray Data's bread-and-butter) never
        # materializes the whole mapped output on the driver.
        if (
            isinstance(plan, Aggregate)
            and plan.watermark is None
            and is_streamable(plan.input)
            and core.has_map_batches(plan.input)
        ):
            from batcher.api.terminal.map_stream import stream_map_aggregate

            yield from stream_map_aggregate(
                plan, _iter_streaming(plan.input, sources, None), batch_size
            )
            return
        # A `Distinct` carrying a fused `limit` is bounded by that limit and stops reading
        # once it has that many distinct rows, so it takes the capped driver rather than the
        # running fold below — which has no early exit and whose `as_aggregate` refuses a
        # fused limit outright. Tested before the general branch because it is a strictly
        # narrower case of the same node. This is the shape Kyber's `fuse_limit_into_distinct`
        # produces, so the router meets it whenever it re-enters on an optimized subtree.
        if (
            isinstance(plan, Distinct)
            and plan.limit is not None
            and not plan.keys
            and is_streamable(plan.input)
            and not core.has_map_batches(plan.input)
        ):
            from batcher.core.streaming import stream_distinct_limit

            yield from stream_distinct_limit(
                plan, sources[0], batch_size, projection=_pushdown(plan)
            )
            return
        # A *keyed* dedup is deliberately not routed here. It is not a group-by — its
        # surviving row carries columns the key does not determine — so the aggregate fold
        # below cannot express it, and it falls through to the ordinary materializing path
        # exactly as it did when it lowered to a window.
        if (
            isinstance(plan, (Aggregate, Distinct))
            and not (isinstance(plan, Distinct) and plan.keys)
            and is_streamable(plan.input)
            and not core.has_map_batches(plan.input)
        ):
            # A watermarked windowed aggregation emits each window as the watermark
            # closes it (bounded state); a plain aggregate folds one running state.
            if isinstance(plan, Aggregate) and plan.watermark is not None:
                from batcher.core.streaming import stream_windowed_aggregate

                yield from stream_windowed_aggregate(
                    plan, sources[0], batch_size, projection=_pushdown(plan)
                )
                return
            from batcher.core.streaming import stream_aggregate, stream_distinct

            driver = stream_distinct if isinstance(plan, Distinct) else stream_aggregate
            yield from driver(plan, sources[0], batch_size, projection=_pushdown(plan))
            return
        # `distinct().limit(n)` — the first `n` distinct rows, then stop reading. Bounded
        # state and a terminating read, yet the router refused it: the `Limit` branch below
        # needs a breaker-free input and a `Distinct` is a breaker, so the pair fell through
        # to "this plan must materialize" for the most ordinary way anyone inspects an
        # unfamiliar topic. Matched here on the *unfused* pair rather than waiting for
        # `fuse_limit_into_distinct`, which is gated on a cardinality estimate a stream does
        # not have — the streaming exit is sound for any cardinality, since it is the
        # operator's own input-order rule that makes it so.
        if (
            isinstance(plan, Limit)
            and plan.n > 0
            and isinstance(plan.input, Distinct)
            and not plan.input.keys
            and plan.input.limit is None
            and is_streamable(plan.input.input)
            and not core.has_map_batches(plan.input.input)
        ):
            import dataclasses

            from batcher.core.streaming import stream_distinct_limit

            capped = dataclasses.replace(plan.input, limit=plan.offset + plan.n)
            yield from _take(
                stream_distinct_limit(capped, sources[0], None, projection=_pushdown(plan)),
                plan.n,
                plan.offset,
                batch_size,
            )
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

            yield from stream_topn(
                plan.input, plan.n, sources[0], batch_size, projection=_pushdown(plan)
            )
            return
        # A plain `Limit` over a breaker-free pipeline streams and stops early.
        if (
            isinstance(plan, Limit)
            and is_streamable(plan.input)
            and not core.has_map_batches(plan.input)
        ):
            from batcher.core.streaming import stream_limit

            yield from stream_limit(plan, sources[0], batch_size, projection=_pushdown(plan))
            return
        # The same limit over a `map_batches` pipeline. `core.streaming.stream_limit`
        # cannot take it: it asks Kyber to lower the plan, and a UDF is Python. Scoring
        # the first hundred events off an unfamiliar topic is how anyone smoke-tests a
        # model against live data, and it answered "the plan must materialize" -- which a
        # limit does not, and which named a breaker the caller had not written.
        if isinstance(plan, Limit) and is_streamable(plan.input):
            yield from _take(
                _iter_streaming(plan.input, sources, None), plan.n, plan.offset, batch_size
            )
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
        if (
            isinstance(plan, Sort)
            and supports_spilling_sort(plan, sources)
            and is_streamable(plan.input)
        ):
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
                from batcher.dist.global_window import (
                    stream_spilling_global_window,
                    supports_ordered_bucket_offsets,
                )

                if supports_ordered_bucket_offsets(plan):
                    gen = stream_spilling_global_window(plan, sources)
        if gen is not None:
            for b in gen:
                if batch_size is None:
                    yield b
                else:
                    for off in range(0, b.num_rows, batch_size):
                        yield b.slice(off, batch_size)
            return

    # Row-wise operators stacked *above* a breaker — `group_by().agg().select()`, SQL
    # `HAVING` (`Filter(Aggregate)`), a renamed aggregate output, `group_by().agg().explode()`
    # — matched none of the exact-shape branches above, because each of those tests the top
    # node only. They therefore fell through to `_collect` and materialized the whole result,
    # even though the breaker underneath already streams and a row-wise op is per-batch valid.
    # Peel them, stream the breaker, and re-apply them to each emitted batch.
    #
    # The peelable set is `is_partition_independent`, not a hand-written tuple. That predicate
    # is the neutral definition of "safe to run per batch", which is the exact property this
    # re-application needs — and it already admitted `Unnest`/`Unpivot`/fraction-`Sample`
    # while the tuple here listed only `Project`/`Filter`, so an `explode` or a `melt` above an
    # aggregate materialized for no reason. Deferring to it also means a future row-wise node
    # is peelable the moment it is classified once, rather than in two places that drift.
    #
    # `Limit` is deliberately NOT peeled, though `dist.spill._peel_to_breaker` does peel it:
    # that path re-applies to one materialized table, where a limit is well defined. Here the
    # re-application is per batch, and `LIMIT n` applied per batch would keep n rows from
    # EVERY batch instead of n overall. `Sort` is likewise not peeled — it is a breaker, not
    # a row-wise op, and reordering per batch is not a global sort. Neither is partition-
    # independent, so the predicate excludes both without a special case.
    peeled: list[LogicalPlan] = []
    below: LogicalPlan = plan
    while is_partition_independent(below):
        peeled.append(below)
        below = below.input
    if peeled:
        # Recursion terminates: `below` is a strict subtree of `plan` and is neither a
        # `Project` nor a `Filter`, so it cannot re-enter this branch.
        inner = _iter_batches(
            below,
            sources,
            below.available_columns(),
            None,
            distributed=distributed,
            num_workers=num_workers,
            transport=transport,
        )
        yield from _apply_peeled(peeled, inner)
        return

    # No streaming path applies → materialize. An unbounded source would never
    # finish, so refuse instead of hanging.
    if any(not is_bounded(s) for s in sources):
        from batcher._internal.errors import PlanError

        raise PlanError(_unstreamable_reason(plan))
    from batcher.api.terminal.core import _collect

    table = _collect(plan, sources, columns)
    batches = (
        table.to_batches() if batch_size is None else table.to_batches(max_chunksize=batch_size)
    )
    yield from batches


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


