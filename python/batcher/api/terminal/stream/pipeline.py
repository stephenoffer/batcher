"""How a streaming strategy is *driven*, once `dispatch` has chosen one.

`dispatch` is the router: it inspects a plan and decides which bounded-memory path can
yield its result. This module is the half that then does the yielding for the two paths
the router owns itself rather than delegating — the breaker-free pipeline
(`_iter_streaming`) and the per-batch re-application of row-wise operators peeled off a
breaker (`_apply_peeled`) — plus the projection they read through (`_pushdown`).

They live apart because they answer different questions and change for different reasons.
The router's content is a preference order over plan shapes; this module's is how a
morsel actually moves, which is where the UDF residency, the window sizing and the
pushdown live. Keeping them in one file also put the router past the module size limit,
and the seam the limit forced is the seam that was already there.

`dispatch` re-exports `_iter_streaming` and `_pushdown`, so the import paths callers and
tests already use keep working — and a monkeypatch aimed at one of those names still
lands, which a move that changed the paths would silently have broken.
"""

from __future__ import annotations

from collections.abc import Iterator

import pyarrow as pa

from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan

__all__ = ["_apply_peeled", "_iter_streaming", "_pushdown"]


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


def _window_latency(source: Source) -> float | None:
    """How long a `map_batches` window may wait before flushing, or `None` for no bound.

    Only an **unbounded** source gets a bound. `stream_windowed`'s row and byte budgets are
    size questions, and over a bounded input they are the right ones: a whole window's worth
    of rows is always on its way, so filling one costs nothing but memory the budget already
    caps. Over a stream the same budgets are a *duration* — 4,000,000 rows at 2,000 rows/s is
    33 minutes with no output and no diagnostic — because a stream is bounded in rate rather
    than in size. Returning `None` for a bounded source keeps the batch path byte-for-byte as
    it was, so the responsiveness is bought only where it is needed.

    Non-positive configuration disables the bound rather than flushing every batch, which is
    what makes `0` a way to restore the pure size-based window instead of a way to destroy
    the UDF parallelism the window exists to create.

    Args:
        source: The bound input the window reads.

    Returns:
        The latency bound in seconds, or `None` when the window should close on size alone.
    """
    from batcher.config import active_config
    from batcher.io.source import is_bounded

    if is_bounded(source):
        return None
    latency = float(active_config().streaming.max_window_latency_seconds)
    return latency if latency > 0.0 else None


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

        # The row target is a *cap*, not the operating point: `stream_windowed` also flushes on
        # a byte budget, and for ordinary (narrow) rows that is what binds. Sized so a window
        # still holds at least one morsel per worker — the pool has to have something to fan
        # across — but far enough above it that the fixed per-window cost is amortized instead
        # of paid every 245,760 rows. Wide rows flush on bytes long before this.
        workers = max(1, max_map_workers(resident))
        morsel = max(1, active_config().execution.morsel_rows)
        target_rows = max(workers * morsel, active_config().optimizer.target_rows_per_task)

        def run_window(window_batches):
            return core.execute_with_udfs(resident, [InMemorySource(window_batches)])

        try:
            # Read only the columns the pipeline needs. `collect()` has always done this
            # (`kyber.required_columns_per_source`, via the UDF executor) and so has the
            # relational branch below; the *streamed* map branch read the source whole. On a
            # 31-column corpus whose `fn` declares four `input_columns` that is 27 columns
            # decoded per window and discarded, measured at **4,425 ms against 733 ms for the
            # same query collected** — the streaming API, which exists for inputs too large to
            # collect, was the one paying for the widest read. An undeclared `fn` still yields
            # `None` here and still reads everything, which is the only safe answer for a
            # black box.
            yield from stream_windowed(
                source,
                run_window,
                target_rows,
                batch_size,
                projection=_pushdown(plan),
                latency_seconds=_window_latency(source),
            )
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
