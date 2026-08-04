"""Streaming-strategy selection for `Dataset.iter_batches` (control plane, `api`).

The seam: this module is the *router*. It inspects a plan and picks the most
bounded-memory way to yield its result, then drives the two paths that are pure plan
shape — the breaker-free pipeline and the exact-size rebatcher. Strategies with their
own retained state live beside it (`watermark`), as do their proof obligations (`union`)
and the distributed (`distributed_stream`), map (`map_stream`), running-state
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

from batcher.api.terminal.stream.rebatch import _rebatch_exact
from batcher.api.terminal.stream.union import (
    interleave,
    union_branch_sources,
    union_streams_branchwise,
    union_streams_interleaved,
)
from batcher.api.terminal.stream.watermark import stream_stream_join, stream_watermark_dedup
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan

__all__ = ["_iter_batches", "_iter_streaming"]


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
        if (
            isinstance(plan, (Aggregate, Distinct))
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

        raise PlanError(
            f"this pipeline has an unbounded (streaming) source but its top-level "
            f"{type(plan).__name__} forces the plan to materialize (a pipeline breaker "
            "such as sort / join / window / multi-source), which cannot be streamed in "
            "bounded memory. Restructure to a streamable shape (filter / project / "
            "map_batches, or a single top-level aggregate / distinct / top-N)."
        )
    from batcher.api.terminal.core import _collect

    table = _collect(plan, sources, columns)
    batches = (
        table.to_batches() if batch_size is None else table.to_batches(max_chunksize=batch_size)
    )
    yield from batches


def _apply_peeled(
    peeled: list[LogicalPlan], batches: Iterator[pa.RecordBatch]
) -> Iterator[pa.RecordBatch]:
    """Re-apply row-wise operators peeled from above a breaker to each streamed batch.

    The caller guarantees every entry of `peeled` is `is_partition_independent` — a
    stateless, row-wise transform — which is exactly what makes per-batch application equal
    to whole-relation application. That admits the row-multiplying reshapers (`Unnest`,
    `Unpivot`) as well as `Project`/`Filter`: they hold no state, so a batch's output does
    not depend on how the input was split.

    The chain is rebuilt over a `Scan(0)` and optimized **once**, not per batch. Kyber's
    `optimize` is plan-shaped work that does not depend on the data, so running it inside
    the loop would pay the full optimizer cost per morsel — which for many small batches
    can cost more than the materialization this path exists to avoid.
    """
    import dataclasses

    from batcher import core, kyber
    from batcher.plan.logical import Scan
    from batcher.plan.schema import SchemaRef

    physical = None
    for batch in batches:
        if physical is None:
            chain: LogicalPlan = Scan(0, SchemaRef.from_arrow(batch.schema))
            for node in reversed(peeled):  # innermost (closest to the breaker) first
                chain = dataclasses.replace(node, input=chain)
            physical = kyber.optimize(chain)
        # `execute_local` takes already-resolved batch lists, one per source — not `Source`
        # objects. The batch IS the resolved single source here.
        out = core.execute_local(physical, [[batch]])
        # A `Filter` that matches nothing in this batch yields no batches; skip it rather
        # than emitting an empty batch, so a filtered stream does not pad the consumer with
        # zero-row batches. The final schema still comes from the batches that do match.
        for b in out:
            if b.num_rows:
                yield b


def _pushdown(plan: LogicalPlan) -> list[str] | None:
    """The columns source 0 must produce for `plan` — Kyber's answer, for a core driver.

    The bounded-state drivers in `core.streaming` read the source directly, and every one of
    them read it *whole*: a `group_by("user").sum("cents")` over a forty-column event decoded
    thirty-eight columns per micro-batch and threw them away. `_iter_streaming` on the
    neighbouring branch has always read through the pushdown; this is the same answer for the
    branches that bypass it.

    Computed over the plan the driver will actually execute, not over an optimized rewrite of
    it, so the projection is exactly the set that plan's own IR references. Asking Kyber keeps
    the decision in Kyber's lane; `core` only reads what it is handed.

    Args:
        plan: The logical plan the driver runs, rooted at the operator being streamed.

    Returns:
        The projection for source 0, or ``None`` when the plan does not narrow it.
    """
    from batcher import kyber

    return kyber.required_columns_per_source(plan).get(0)


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
        # Build the (class) UDFs once so a load-once inference model loads a single time
        # and is reused across every streamed batch, not rebuilt per batch.
        resident = core.prebuild_factories(plan)

        # Stream the source in *windows* of batches, not one batch at a time: a
        # `map_batches` UDF parallelizes across `num_workers` only when it is handed
        # several batches at once (a single batch is applied sequentially). Feeding one
        # source batch per call throws away all UDF parallelism — the difference between
        # a serial and an all-cores read→map→write. The window holds ~one morsel per
        # worker so the pool fills, and bounds driver memory to that window (+ its
        # output) — never the whole input.
        from batcher.api.terminal.map_stream import max_map_workers, stream_windowed
        from batcher.config import active_config

        workers = max(1, max_map_workers(resident))
        target_rows = workers * max(1, active_config().execution.morsel_rows)

        def run_window(window_batches):
            return core.execute_with_udfs(resident, [InMemorySource(window_batches)])

        try:
            yield from stream_windowed(source, run_window, target_rows, batch_size)
        finally:
            # `prebuild_factories` made this generator the models' owner, and `teardown_udf`
            # declines a prebuilt instance for exactly that reason — so without this a
            # streamed model held its GPU allocation for the life of the process.
            core.release_prebuilt(resident)
        return

    # Relational (no UDF) breaker-free pipeline: optimize once, then stream micro-batches.
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
