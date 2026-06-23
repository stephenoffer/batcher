"""Streaming (incremental) aggregation — bounded-memory group-by over a source.

An aggregation is mergeable (`partial → combine → finalize`), so it can run over an
unbounded / larger-than-memory source one micro-batch at a time: each batch's
partial state is folded into a single running state — bounded by the number of
groups, not the input size — via the native `combine`, and finalized once at the
end. The result is identical to materializing the whole input and aggregating, so
batch is the bounded special case of streaming.

Core's lane: this drives the engine (`batcher._native`) over the plan it is given;
it makes no optimization decisions.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pyarrow as pa

from batcher.config import active_config
from batcher.io.source import Source
from batcher.plan.expr_ir import Col
from batcher.plan.logical import Aggregate, Distinct, Limit, Projection, Sort


def stream_aggregate(
    agg: Aggregate, source: Source, batch_size: int | None = None
) -> Iterator[pa.RecordBatch]:
    """Aggregate `source` incrementally, holding only one running partial state.

    `agg.input` must be a breaker-free relational pipeline over the single source
    (filter/project/scan); each source batch is run through it, partial-aggregated,
    and combined into the running state. Yields the finalized result once the source
    is exhausted (one logical result, optionally rebatched by `batch_size`).
    """
    import batcher._native as nat

    group_keys_json = json.dumps(
        [{"expr": k.expr.to_ir(), "alias": k.alias} for k in agg.group_keys]
    )
    aggregates_json = json.dumps([s.agg.to_ir(s.alias) for s in agg.aggregates])
    input_ir = json.dumps(agg.input.to_ir())  # scans source 0

    running: pa.RecordBatch | None = None
    for batch in source.iter_batches(None):
        if batch.num_rows == 0:
            continue
        rows = nat.execute_plan(input_ir, [[batch]], active_config().engine_config_json())
        if not rows or sum(b.num_rows for b in rows) == 0:
            continue
        partial = nat.partial_aggregate(group_keys_json, aggregates_json, rows)
        running = (
            partial
            if running is None
            else nat.combine(group_keys_json, aggregates_json, [running, partial])
        )

    if running is None:
        return  # empty source → no groups → no rows
    result = nat.combine_finalize(group_keys_json, aggregates_json, [running])
    if result.num_rows == 0:
        return
    if batch_size is None:
        yield result
    else:
        for off in range(0, result.num_rows, batch_size):
            yield result.slice(off, batch_size)


def stream_distinct(
    distinct: Distinct, source: Source, batch_size: int | None = None
) -> Iterator[pa.RecordBatch]:
    """DISTINCT over a streaming source, with bounded memory.

    DISTINCT is a group-by over *all* columns with no aggregate functions, so it
    reuses the incremental aggregate driver verbatim: identical rows fold into the
    same running group, and the state is bounded by the number of distinct rows.
    """
    cols = distinct.input.available_columns()
    group_keys = tuple(Projection(c, Col(c)) for c in cols)
    agg = Aggregate(distinct.input, group_keys, ())
    yield from stream_aggregate(agg, source, batch_size)


def stream_limit(
    limit: Limit, source: Source, batch_size: int | None = None
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
    import batcher._native as nat

    input_ir = json.dumps(limit.input.to_ir())
    cfg = active_config().engine_config_json()
    remaining_skip = limit.offset
    remaining_take = limit.n
    if remaining_take <= 0:
        return
    for batch in source.iter_batches(None):
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
    sort: Sort, limit: int, source: Source, batch_size: int | None = None
) -> Iterator[pa.RecordBatch]:
    """Top-N (`sort` + `limit`) over a streaming source, with memory bounded by N.

    Top-N is mergeable — top-N of (A concat B) equals top-N of (top-N of A, B) — so the driver keeps
    only the running best `limit` rows: each micro-batch is run through the sort
    sub-plan, merged with the running best, and re-trimmed to `limit`. The final
    running set is the global top-N, identical to sorting the whole input then
    taking the first `limit` rows.
    """
    import batcher._native as nat

    sort_ir = json.dumps(
        {
            "op": "sort",
            "input": {"op": "scan", "source_id": 0},
            "keys": [
                {"expr": k.expr.to_ir(), "descending": k.descending, "nulls_first": k.nulls_first}
                for k in sort.keys
            ],
            "limit": limit,
        }
    )
    input_ir = json.dumps(sort.input.to_ir())

    running: list[pa.RecordBatch] = []
    for batch in source.iter_batches(None):
        if batch.num_rows == 0:
            continue
        cfg_json = active_config().engine_config_json()
        rows = [b for b in nat.execute_plan(input_ir, [[batch]], cfg_json) if b.num_rows]
        merged = running + rows
        if not merged:
            continue
        running = [b for b in nat.execute_plan(sort_ir, [merged], cfg_json) if b.num_rows]

    if not running:
        return
    result = pa.Table.from_batches(running)
    if batch_size is None:
        yield from result.to_batches()
    else:
        for off in range(0, result.num_rows, batch_size):
            yield result.slice(off, batch_size)
