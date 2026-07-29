"""Bounded-memory drivers for a top-level operator over a streaming source.

Each driver folds or short-circuits so peak memory is a property of the *operator* rather
than of the input: an aggregate holds one running state, a distinct holds its distinct rows,
a limit stops reading, a top-N holds N. The result is identical to materializing the whole
input and running the operator, so batch is the bounded special case of streaming.

Every driver reads its source through the projection Kyber decided for the plan — the
conductor computes it and passes it in, keeping the decision in Kyber's lane.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pyarrow as pa

from batcher._internal.native import engine
from batcher.config import active_config
from batcher.core.streaming.folds import (
    _AggFold,
    _read,
    _rebatch,
    _window_key,
    _WindowedAggFold,
    empty_global_aggregate,
)
from batcher.io.source import Source
from batcher.plan.ir_specs import sort_keys_ir
from batcher.plan.logical import Aggregate, Distinct, Limit, Sort

__all__ = [
    "stream_aggregate",
    "stream_distinct",
    "stream_limit",
    "stream_topn",
    "stream_windowed_aggregate",
]


def stream_aggregate(
    agg: Aggregate,
    source: Source,
    batch_size: int | None = None,
    *,
    projection: list[str] | None = None,
) -> Iterator[pa.RecordBatch]:
    """Aggregate `source` incrementally, holding only one running partial state.

    `agg.input` must be a breaker-free relational pipeline over the single source
    (filter/project/scan); each source batch is run through it, partial-aggregated,
    and combined into the running state. Yields the finalized result once the source
    is exhausted (one logical result, optionally rebatched by `batch_size`).
    """
    fold = _AggFold(agg)
    for batch in _read(source, projection):
        fold.push(batch)
    result = fold.finalize()
    if result is None and not agg.group_keys:
        # A *global* aggregate over an empty input still yields exactly one row — `SUM` is
        # NULL, `COUNT` is 0 — which is what SQL, DuckDB, and `collect()` all produce. The
        # fold has no partial to finalize (it skips empty batches), so it would yield nothing
        # and silently disagree with the oracle. Ask the engine for the empty-input result
        # through the ordinary plan path, so the answer comes from the same operator.
        result = _empty_global_aggregate(agg, source)
    if result is not None:
        yield from _rebatch(result, batch_size)


def _empty_global_aggregate(agg: Aggregate, source: Source) -> pa.RecordBatch | None:
    """The one-row result of a keyless aggregate over an empty input, via the engine."""
    return empty_global_aggregate(agg, source.schema())


def stream_windowed_aggregate(
    agg: Aggregate,
    source: Source,
    batch_size: int | None = None,
    *,
    projection: list[str] | None = None,
) -> Iterator[pa.RecordBatch]:
    """Windowed aggregation over a stream, emitting each window as the watermark
    closes it and flushing the rest at end-of-stream (bounded state)."""
    key = _window_key(agg)
    if key is None or agg.watermark is None:  # not a watermarked windowed agg
        yield from stream_aggregate(agg, source, batch_size, projection=projection)
        return
    fold = _WindowedAggFold(agg, key[0], key[1])
    for batch in _read(source, projection):
        for result in fold.push(batch):
            yield from _rebatch(result, batch_size)
    final = fold.flush()
    if final is not None:
        yield from _rebatch(final, batch_size)


def stream_distinct(
    distinct: Distinct,
    source: Source,
    batch_size: int | None = None,
    *,
    projection: list[str] | None = None,
) -> Iterator[pa.RecordBatch]:
    """DISTINCT over a streaming source, with bounded memory.

    DISTINCT is a group-by over *all* columns with no aggregate functions, so it
    reuses the incremental aggregate driver verbatim: identical rows fold into the
    same running group, and the state is bounded by the number of distinct rows.
    """
    yield from stream_aggregate(distinct.as_aggregate(), source, batch_size, projection=projection)


def stream_limit(
    limit: Limit,
    source: Source,
    batch_size: int | None = None,
    *,
    projection: list[str] | None = None,
) -> Iterator[pa.RecordBatch]:
    """`Limit(n, offset)` over a streamable input, reading the source only until `n`
    rows are produced (then stopping) — IO- and memory-bounded by `n + offset`,
    never the source size. Ray Data's `limit(n)` processes the whole input; this
    short-circuits.

    `limit.input` must be a breaker-free pipeline over the single source
    (filter/project/scan/unnest/…): such ops preserve row order and are
    partition-independent, so taking the first `n` rows across source batches in
    iteration order equals applying the `Limit` to the whole pipeline.
    """
    nat = engine()

    input_ir = json.dumps(limit.input.to_ir())
    cfg = active_config().engine_config_json()
    remaining_skip = limit.offset
    remaining_take = limit.n
    if remaining_take <= 0:
        return
    for batch in _read(source, projection):
        if batch.num_rows == 0:
            continue
        for b in nat.execute_plan(input_ir, [[batch]], cfg):
            if b.num_rows == 0:
                continue
            if remaining_skip >= b.num_rows:
                remaining_skip -= b.num_rows
                continue
            start, remaining_skip = remaining_skip, 0
            take_n = min(b.num_rows - start, remaining_take)
            chunk = b.slice(start, take_n)
            remaining_take -= take_n
            if batch_size is None:
                yield chunk
            else:
                for off in range(0, chunk.num_rows, batch_size):
                    yield chunk.slice(off, batch_size)
            # Stop the instant the limit is met — `return` ends the generator without
            # advancing the source iterator again (the early-read short-circuit).
            if remaining_take <= 0:
                return


def stream_topn(
    sort: Sort,
    limit: int,
    source: Source,
    batch_size: int | None = None,
    *,
    projection: list[str] | None = None,
) -> Iterator[pa.RecordBatch]:
    """Top-N (`sort` + `limit`) over a streaming source, with memory bounded by N.

    Top-N is mergeable — top-N of (A concat B) equals top-N of (top-N of A, B) — so the driver keeps
    only the running best `limit` rows: each micro-batch is run through the sort
    sub-plan, merged with the running best, and re-trimmed to `limit`. The final
    running set is the global top-N, identical to sorting the whole input then
    taking the first `limit` rows.
    """
    nat = engine()

    sort_ir = json.dumps(
        {
            "op": "sort",
            "input": {"op": "scan", "source_id": 0},
            "keys": sort_keys_ir(sort.keys),
            "limit": limit,
        }
    )
    input_ir = json.dumps(sort.input.to_ir())

    running: list[pa.RecordBatch] = []
    # The engine config is constant for the query, so read and serialize it once — not once
    # per micro-batch inside the loop (`stream_limit` already hoists it the same way).
    cfg_json = active_config().engine_config_json()
    for batch in _read(source, projection):
        if batch.num_rows == 0:
            continue
        rows = [b for b in nat.execute_plan(input_ir, [[batch]], cfg_json) if b.num_rows]
        merged = running + rows
        if not merged:
            continue
        running = [b for b in nat.execute_plan(sort_ir, [merged], cfg_json) if b.num_rows]

    if not running:
        return
    result = pa.Table.from_batches(running)
    # `to_batches` yields `RecordBatch`es (the `iter_batches` contract); slicing the
    # `Table` directly would leak `pa.Table` objects to the caller.
    if batch_size is None:
        yield from result.to_batches()
    else:
        yield from result.to_batches(max_chunksize=batch_size)
