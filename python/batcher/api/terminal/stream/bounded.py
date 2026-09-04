"""The shapes whose bounded-memory streaming driver is a running fold over the stream.

Three operators materialized under `iter_batches()` — `with_row_index` (and `tail`, which
lowers to the same node), a fixed-count `sample(n=)`, and a keyed `DISTINCT ON` — each for a
reason that is correct about *partitioning* and does not carry to a *stream*. They are grouped
here rather than inline in `dispatch` because each needs a paragraph saying why its fold is
sound, and that reasoning belongs beside the decision it justifies; the router keeps a
pointer.

`bounded_driver` returns a generator for a shape it serves and `None` for everything else, so
the router's branch is a single call with no knowledge of which shapes those are.
"""

from __future__ import annotations

from collections.abc import Iterator

import pyarrow as pa

from batcher.api.terminal.stream.pipeline import _iter_streaming, _pushdown
from batcher.io.source import Source, is_bounded
from batcher.plan.logical import (
    AsofJoin,
    Distinct,
    LogicalPlan,
    RowId,
    Sample,
    is_streamable,
)
from batcher.plan.visitor import scanned_source_ids

__all__ = ["asof_lookup_driver", "bounded_driver"]


def bounded_driver(
    plan: LogicalPlan, sources: list[Source], batch_size: int | None
) -> Iterator[pa.RecordBatch] | None:
    """A running-fold driver for `plan`, or `None` when this module does not serve it.

    Args:
        plan: The plan the streaming router is dispatching.
        sources: The query's bound inputs.
        batch_size: The caller's requested batch size, or `None`.

    Returns:
        The batch iterator, or `None` to let the router continue.
    """
    from batcher import core

    # The type test comes first because the two below dereference `plan.input`, and the router
    # calls this for *every* plan it is handed — including one whose root is a `Union`, which
    # has `inputs`. `rollup`/`cube` over an empty relation is such a plan (the grouping sets
    # fold to a union of constants), and `iter_batches` on it raised `AttributeError: 'Union'
    # object has no attribute 'input'` where `collect()` answered. Ordering the checks the
    # other way is not a narrowing: every shape this module serves is single-input, so a plan
    # that fails the test was going to be declined by the next line anyway.
    if not isinstance(plan, (RowId, Distinct, Sample)):
        return None
    if core.has_map_batches(plan.input) or not is_streamable(plan.input):
        return None

    # `with_row_index` / `tail` / `with_random` all lower to `RowId`, and it materialized —
    # the router had no branch for it at all. A row index is a *position*, so `RowId` is not
    # partition-independent and the peeling loop correctly refuses it: run per partition,
    # every partition restarts the counter at zero.
    #
    # A stream is where that difficulty does not arise, because batches arrive in the input's
    # own row order. `stream_row_index` says why that makes one counter sufficient, and why
    # the distributed path needs `preserve_order` and a driver-side assembly where this needs
    # neither. Unbounded is fine here: an index is assigned on arrival and never retracted.
    if isinstance(plan, RowId):
        from batcher.core.streaming import stream_row_index

        return stream_row_index(plan, _iter_streaming(plan.input, sources, batch_size))

    if not isinstance(plan, (Distinct, Sample)):
        return None
    keyed_dedup = isinstance(plan, Distinct) and bool(plan.keys)
    fixed_sample = isinstance(plan, Sample) and plan.n is not None
    if not (keyed_dedup or fixed_sample):
        return None

    # **Bounded sources only, and that is a correctness bound rather than caution.** Both
    # answers are final only once the input ends: which row wins a key is, in the sink path's
    # own words, "decided by an ordering over rows that have not arrived", and the `n`
    # smallest hashes of an unbounded relation are never settled. Each driver accumulates and
    # emits at the end, which an unbounded source never reaches.
    #
    # `core/streaming_query/processors.py` already refuses a keyed dedup on exactly this
    # ground, pointing the caller at `drop_duplicates_within_watermark` whose state a
    # watermark bounds — and `test_streaming_sink_matches_batch.py` asserts that it and this
    # router agree about what streams. Without this guard they would not, which is how one
    # query comes to have one capability through `iter_batches` and another through `write`.
    if not all(is_bounded(s) for s in sources):
        return None

    # Both are mergeable in their own right, which is what the router's aggregate/dedup branch
    # could not use: it folds through `Distinct.as_aggregate`, and a keyed dedup is not a
    # group-by — its survivor carries columns the key does not determine, so per-column
    # aggregates would build a row that was never in the input. Re-applying each *operator*
    # to (running + batch) has no such problem: the dedup keeps the minimum under `order` and
    # min associates, and `sample(n=)` keeps the `n` smallest-hash rows so the running `n` is
    # closed under adding a batch. Peak memory is the key count and `n` respectively.
    from batcher.core.streaming import stream_distinct_on, stream_sample_n

    driver = stream_distinct_on if keyed_dedup else stream_sample_n
    return driver(plan, sources[0], batch_size, projection=_pushdown(plan))


def asof_lookup_driver(
    asof: AsofJoin, sources: list[Source], batch_size: int | None
) -> Iterator[pa.RecordBatch] | None:
    """A **keyless** ASOF join, streamed by materializing only the right side.

    The `by`-keyed form does not come here: it co-partitions on `by` and rides the grace join
    (`spill_breakers.stream_spilling_join`), which is bounded on both sides and is the better
    plan when the right is large. A keyless ASOF has no group to hash, so that decomposition
    is unavailable and the shape materialized *both* sides instead.

    It does not need a decomposition, because it is not a fold. Each left row's match is a
    **lookup** into the right side — the nearest row at or before it, or after it, within
    `tolerance` — and it depends on nothing but that row's `on` value and the right side
    itself. No state carries from one left row to the next. So running the operator per left
    batch against the whole right side yields exactly the rows the collected form yields, in
    exactly the same order, and peak memory drops from the whole join to the right side alone.
    That is the ordinary shape for this operator: a large fact stream against a smaller quote
    or dimension table.

    Verified before it was written, on all four match settings (`backward`, `forward`,
    `nearest`, and a `tolerance`) and **row for row** rather than as a multiset — the output
    order is the property a per-batch rewrite is most likely to change and a sorted comparison
    would not see. Also verified with an *unsorted* left, since the operator emits in left-row
    order either way.

    The right side must be bounded, because it is materialized once up front; the left may be
    unbounded, which is the direction that matters. Returns `None` when either condition fails,
    leaving the caller's own path unchanged.
    """
    from batcher import core
    from batcher.api.terminal.core import _collect

    # The router calls this for **every** plan, above the single-source block, so the type
    # test comes first. Without it the very next line reaches for `left_by` on whatever node
    # is passing through — a `Limit`, a `Project` — and the router dies on an `AttributeError`
    # for a shape that has nothing to do with ASOF joins.
    if not isinstance(asof, AsofJoin):
        return None
    if asof.left_by or core.has_map_batches(asof) or not is_streamable(asof.left):
        return None
    right_ids = scanned_source_ids(asof.right)
    if len(right_ids) != 1:
        return None
    right_id = next(iter(right_ids))
    if not is_bounded(sources[right_id]):
        return None

    return _asof_batches(asof, sources, batch_size, _collect)


def _asof_batches(asof, sources, batch_size, collect) -> Iterator[pa.RecordBatch]:
    """Materialize the right once, then run the ASOF per left batch."""
    import dataclasses

    from batcher.io.source import InMemorySource
    from batcher.plan.logical import Scan
    from batcher.plan.schema import SchemaRef

    right = collect(asof.right, sources, asof.right.available_columns())
    right_source = InMemorySource(
        right.to_batches() or [pa.RecordBatch.from_pylist([], schema=right.schema)]
    )
    right_scan = Scan(1, SchemaRef.from_arrow(right.schema))

    for batch in _iter_streaming(asof.left, sources, batch_size):
        left_source = InMemorySource([batch])
        node = dataclasses.replace(
            asof, left=Scan(0, SchemaRef.from_arrow(batch.schema)), right=right_scan
        )
        table = collect(node, [left_source, right_source], node.available_columns())
        yield from table.to_batches()
