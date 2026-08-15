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
from collections.abc import Iterator, Sequence

import pyarrow as pa

from batcher._internal.logging import note_suppressed
from batcher._internal.native import engine
from batcher.config import active_config
from batcher.core.streaming.folds import (
    _AggFold,
    _read,
    _rebatch,
    _window_key,
    _WindowedAggFold,
    check_agg_state_bounded,
    empty_global_aggregate,
    streaming_state_budget,
)
from batcher.io.source import Source
from batcher.plan.logical import Aggregate, Distinct, Limit, Sort
from batcher.plan.types import one_batch

#: How many rows a streaming top-N round buffers per row of running state before it merges.
#:
#: The merge re-reads the running `limit` rows, so this bounds that overhead at roughly its
#: reciprocal — a few percent — however large `limit` is, while keeping the round a small
#: multiple of the result. It matches the engine-side fold's ratio deliberately: the two are the
#: same reduction driven from different sides of the FFI boundary, and a reader comparing them
#: should not have to work out whether the difference is meaningful.
_TOPN_MERGE_RATIO = 16

__all__ = [
    "stream_aggregate",
    "stream_distinct",
    "stream_distinct_limit",
    "stream_keyed_state",
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

    A *keyed* aggregate over an *unbounded* source holds one entry per group for the life
    of the query and only emits when the source ends, which it never does — so the state is
    capped here exactly as it is on the sink path (`AggregateProcessor`). Only that path had
    the guard, so the driver that grows forever was the one nothing was watching.
    """
    fold = _AggFold(agg)
    guard = _unbounded_group_guard(agg, source)
    for batch in _read(source, projection):
        fold.push(batch)
        if guard is not None:
            check_agg_state_bounded(fold, guard[0], guard[1], label="streaming aggregate")
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


def _unbounded_group_guard(agg: Aggregate, source: Source) -> tuple[int, str] | None:
    """The `(cap, cause)` to check a running aggregate against, or None when unneeded.

    A keyless aggregate holds one row, and a bounded source ends — neither can leak, so
    neither pays for the check.
    """
    from batcher.io.source import is_bounded

    if not agg.group_keys or is_bounded(source):
        return None
    return (
        streaming_state_budget(),
        "this aggregate has no watermark, so no group is ever closed and evicted. Add "
        ".with_watermark(...) with a windowed group_by so closed windows evict, narrow the "
        "group keys, or raise memory.streaming_state_max_bytes",
    )


def stream_windowed_aggregate(
    agg: Aggregate,
    source: Source,
    batch_size: int | None = None,
    *,
    projection: list[str] | None = None,
) -> Iterator[pa.RecordBatch]:
    """Windowed aggregation over a stream, emitting each window as the watermark
    closes it and flushing the rest at end-of-stream (bounded state).

    A watermarked aggregate whose group keys contain no event-time window has nothing the
    watermark can close, so over an unbounded source it is refused rather than run: the
    fallback it used to take was an ordinary running aggregate, which on a stream that never
    ends emits nothing, evicts nothing, and grows until the memory cap fires. Over a bounded
    source the same fallback terminates and is correct, so it stays.
    """
    from batcher.io.source import (
        is_bounded,
        watermark_partition_columns,
        watermark_partitions,
    )

    key = _window_key(agg)
    if key is None or agg.watermark is None:  # not a watermarked windowed agg
        if agg.watermark is not None and not is_bounded(source):
            raise _unwindowed_watermark_error(agg)
        yield from stream_aggregate(agg, source, batch_size, projection=projection)
        return
    # The watermark is a minimum over the source's partitions, so the partition columns have
    # to survive the projection — and then be removed again before the batch reaches the
    # plan, which must compute what it would have computed without them.
    partition_cols = watermark_partition_columns(source)
    read_projection, extra = _widen(projection, partition_cols, source)
    fold = _WindowedAggFold(
        agg,
        key,
        partition_cols=partition_cols,
        expected_partitions=watermark_partitions(source),
        drop_columns=extra,
    )
    for batch in _read(source, read_projection):
        for result in fold.push(batch):
            yield from _rebatch(result, batch_size)
    final = fold.flush()
    if final is not None:
        yield from _rebatch(final, batch_size)


def _widen(
    projection: list[str] | None, partition_cols: Sequence[str], source: Source
) -> tuple[list[str] | None, tuple[str, ...]]:
    """Add the watermark's partition columns to `projection`, and name what was added.

    A projection of `None` already reads everything, so there is nothing to widen and
    nothing extra to strip. A column the source's schema does not carry is skipped rather
    than requested, so a projection is never widened into a read that fails.

    Args:
        projection: Kyber's source projection for the plan, or None to read everything.
        partition_cols: Columns the watermark needs to attribute rows to partitions.
        source: The stream, for the schema that says which of those columns exist.

    Returns:
        `(projection to read with, columns added purely for the watermark)`.
    """
    if projection is None or not partition_cols:
        return projection, ()
    try:
        available = set(source.schema().names)
    except Exception as exc:
        # A source that cannot describe itself before it is read gets the unwidened
        # projection: no per-partition watermark, rather than a read of a column that may
        # not exist. The tracker degrades to one partition, which is today's behavior.
        # Traced, because that degradation is invisible in the result: the query still runs
        # and still returns rows, it just attributes event time less precisely.
        note_suppressed("core", "widen projection for per-partition watermark", exc)
        return projection, ()
    extra = tuple(c for c in partition_cols if c in available and c not in projection)
    return ([*projection, *extra], extra) if extra else (projection, ())


def _unwindowed_watermark_error(agg: Aggregate):
    """The refusal for `.with_watermark(...)` on an aggregate with no event-time window."""
    from batcher._internal.errors import PlanError

    keys = ", ".join(repr(k.alias) for k in agg.group_keys) or "(none)"
    return PlanError(
        f"this aggregate sets a watermark on {agg.watermark.time_col!r} but groups by "
        f"{keys}, none of which is an event-time window — so the watermark has nothing to "
        "close and no state is ever released. Over an unbounded source that never emits a "
        "row and grows without limit. Group by a window "
        "(`group_by(w=bt.window(col('ts'), '1h'))`, or `bt.window(col('ts'), '1h', '30m')` "
        "exploded first for overlapping windows), or drop the watermark and write the "
        "running aggregate to a sink with output_mode='update'."
    )


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

    Whole-row only. A keyed dedup (`distinct(subset=...)`) is not a group-by — its
    surviving row carries columns the key does not determine — and the dispatcher keeps
    it off this driver; `as_aggregate` raises rather than approximate it here.
    """
    yield from stream_aggregate(distinct.as_aggregate(), source, batch_size, projection=projection)


def stream_distinct_limit(
    distinct: Distinct,
    source: Source,
    batch_size: int | None = None,
    *,
    projection: list[str] | None = None,
) -> Iterator[pa.RecordBatch]:
    """The first `distinct.limit` distinct rows of a stream, then stop reading.

    "Show me the first fifty distinct values off this topic" is how anyone inspects an
    unfamiliar stream, and it is bounded in every way that matters: the state is `limit`
    rows, and the read stops as soon as that many distinct rows exist. An uncapped
    `stream_distinct` can do neither — it holds one entry per distinct value forever and
    finalizes only at an end-of-input a stream never reaches — so the router refused the
    capped form along with the uncapped one, and a query that terminates in bounded memory
    was answered with "this plan must materialize".

    **The early exit is sound because the survivors are the first `limit` in input order.**
    Once that many distinct rows have been seen, every later row is either a duplicate (which
    changes nothing) or a new distinct row that arrived later and therefore cannot displace
    one of them. That is the same rule the engine's own fused `Distinct(limit)` follows and
    the same one `kyber.rules.extra.topn_limit.fuse_limit_into_distinct` documents, which is
    why the two agree on *which* rows come back and not merely on how many.

    The dedup itself is the engine's `Distinct` operator, run over the accumulated survivors
    concatenated with the newly-mapped batch. Re-running the real operator is what keeps this
    from being a second definition of what `DISTINCT` means: the alternative — folding
    through the group-by hash state the way `stream_distinct` does — returns the rows in
    hash-bucket order, which is not input order and so is not the same answer.

    Args:
        distinct: A whole-column `Distinct` carrying a `limit`, over a breaker-free input.
        source: The stream to read.
        batch_size: Optional output rebatching.
        projection: The columns the plan needs, from Kyber.

    Yields:
        One logical result of at most `distinct.limit` rows, rebatched by `batch_size`.
    """
    from batcher.plan.logical import Scan
    from batcher.plan.schema import SchemaRef

    cap = distinct.limit
    if cap is None or cap <= 0:
        return
    nat = engine()
    cfg = active_config().engine_config_json()
    input_ir = json.dumps(distinct.input.to_ir())

    survivors: pa.RecordBatch | None = None
    dedup_ir: str | None = None
    for batch in _read(source, projection):
        if batch.num_rows == 0:
            continue
        mapped = [b for b in nat.execute_plan(input_ir, [[batch]], cfg) if b.num_rows]
        if not mapped:
            continue
        if dedup_ir is None:
            # Built from the *mapped* schema, not the source's: the input pipeline may
            # project, rename, or compute columns, and the dedup runs over what it produced.
            capped = Distinct(Scan(0, SchemaRef.from_arrow(mapped[0].schema)), limit=cap)
            dedup_ir = json.dumps(capped.to_ir())
        rows = mapped if survivors is None else [survivors, *mapped]
        out = [b for b in nat.execute_plan(dedup_ir, [rows], cfg) if b.num_rows]
        # The operator caps its own output at `cap`, so concatenating is bounded by it.
        # `one_batch`, because `combine_chunks().to_batches()[0]` splits at the 32-bit
        # offset limit — a distinct over a text or blob column whose survivors exceed
        # 2 GiB came back as several batches and every row after the first was dropped
        # from the result, silently.
        survivors = one_batch(out) if out else survivors
        if survivors is not None and survivors.num_rows >= cap:
            break  # no later row can displace one of the first `cap` distinct rows

    if survivors is not None and survivors.num_rows:
        yield from _rebatch(survivors, batch_size)


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
    only the running best `limit` rows: batches are run through the sort sub-plan, merged
    with the running best, and re-trimmed to `limit`. The final running set is the global
    top-N, identical to sorting the whole input then taking the first `limit` rows.

    **The merge happens per round, not per micro-batch.** Merging on every batch re-reads
    and re-sorts the running `limit` rows once per batch, so a small source batch against a
    large `limit` pays far more for the merge than for the rows it contributed — and it pays
    two engine round-trips per batch to do it. A round buffers until it holds several times
    `limit` rows, which bounds the merge overhead at a fraction of the round regardless of
    `limit` while leaving peak memory at `limit` plus one round. It is the same reasoning,
    and the same ratio, the engine's own streaming top-N fold uses.

    Rounds cannot change the answer: the rows are still presented to the merge in arrival
    order, with the running best ahead of them, which is what fixes both the survivors and
    the order ties resolve in.
    """
    nat = engine()

    sort_ir = json.dumps(
        {
            **sort.shape_ir(),
            "input": {"op": "scan", "source_id": 0},
            # The driver trims each micro-batch to its own running limit, which is not the
            # plan's — the only field it overrides rather than carries.
            "limit": limit,
        }
    )
    input_ir = json.dumps(sort.input.to_ir())

    running: list[pa.RecordBatch] = []
    pending: list[pa.RecordBatch] = []
    pending_rows = 0
    round_rows = max(limit * _TOPN_MERGE_RATIO, active_config().execution.morsel_rows)
    # The engine config is constant for the query, so read and serialize it once — not once
    # per micro-batch inside the loop (`stream_limit` already hoists it the same way).
    cfg_json = active_config().engine_config_json()

    def merge(buffered: list[pa.RecordBatch]) -> list[pa.RecordBatch]:
        """Run one round's batches through the input pipeline and fold them into `running`."""
        rows = [b for b in nat.execute_plan(input_ir, [buffered], cfg_json) if b.num_rows]
        merged = running + rows
        if not merged:
            return running
        return [b for b in nat.execute_plan(sort_ir, [merged], cfg_json) if b.num_rows]

    for batch in _read(source, projection):
        if batch.num_rows == 0:
            continue
        pending.append(batch)
        pending_rows += batch.num_rows
        if pending_rows >= round_rows:
            running = merge(pending)
            pending, pending_rows = [], 0
    if pending:
        running = merge(pending)

    if not running:
        return
    result = pa.Table.from_batches(running)
    # `to_batches` yields `RecordBatch`es (the `iter_batches` contract); slicing the
    # `Table` directly would leak `pa.Table` objects to the caller.
    if batch_size is None:
        yield from result.to_batches()
    else:
        yield from result.to_batches(max_chunksize=batch_size)


def stream_keyed_state(
    node,
    source: Source,
    batch_size: int | None = None,
    *,
    projection: list[str] | None = None,
) -> Iterator[pa.RecordBatch]:
    """Drive a `TransformWithState` over `source`, one micro-batch at a time.

    Memory is bounded by the *key space* rather than by the input, and by the node's
    `state_ttl` rather than by the query's lifetime — which is the whole reason the TTL is
    part of the operator instead of advice in a docstring.

    Args:
        node: The `TransformWithState` to drive.
        source: Its single source.
        batch_size: Optional output rebatching.
        projection: The source projection Kyber decided for this plan.

    Yields:
        Whatever the user function emitted, rebatched if asked.
    """
    from batcher.core.streaming.keyed_state import KeyedStateFold

    fold = KeyedStateFold(node)
    for batch in _read(source, projection):
        for produced in fold.push(batch):
            yield from _rebatch(produced, batch_size)
