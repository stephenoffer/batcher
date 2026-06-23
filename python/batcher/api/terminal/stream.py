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


def _iter_batches(
    plan: LogicalPlan,
    sources: list[Source],
    columns: list[str],
    batch_size: int | None = None,
) -> Iterator[pa.RecordBatch]:
    """Execute and yield the result as Arrow record batches.

    The bounded-memory streaming path is chosen automatically whenever the plan
    supports it (breaker-free pipeline, a top-level aggregate / distinct / top-N over
    one, or a top-level sort / join / window streamed from the out-of-core bucket
    pipeline); other plans materialize first. An unbounded source whose plan cannot
    stream raises `PlanError` instead of hanging on `_collect`.
    """
    from batcher.plan.logical import Aggregate, Distinct, Limit, Sort, is_streamable

    if len(sources) == 1:
        if is_streamable(plan):
            yield from _iter_streaming(plan, sources, batch_size)
            return
        # A top-level aggregate/distinct over a breaker-free relational input streams
        # with bounded memory: fold each micro-batch's partial into one running state.
        from batcher import core

        if (
            isinstance(plan, (Aggregate, Distinct))
            and is_streamable(plan.input)
            and not core.has_map_batches(plan.input)
        ):
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

    from batcher.io.source import is_bounded

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
