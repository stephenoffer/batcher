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

import datetime
import json
from collections.abc import Iterator

import pyarrow as pa

from batcher._internal.native import engine
from batcher.config import active_config
from batcher.core.mergeable import RunningAggregate
from batcher.io.source import Source
from batcher.plan.ir_specs import sort_keys_ir
from batcher.plan.logical import Aggregate, Distinct, Limit, Sort


def _rebatch(result: pa.RecordBatch, batch_size: int | None) -> Iterator[pa.RecordBatch]:
    """Yield `result` whole, or sliced into `batch_size`-row chunks."""
    if batch_size is None:
        yield result
    else:
        for off in range(0, result.num_rows, batch_size):
            yield result.slice(off, batch_size)


class _AggFold:
    """Running partial-aggregate state folded across micro-batches.

    Each pushed source batch is run through the breaker-free input pipeline, then
    `partial`-aggregated and `combine`d into one running state (bounded by the group
    count, not the input size) entirely in Rust. `finalize()` materializes the
    current result. This is the shared kernel under both the one-shot streaming
    aggregate driver and the long-running streaming-query engine's complete/update
    output modes — the running state is the same Arrow `RecordBatch` the engine
    snapshots for checkpoint recovery.
    """

    __slots__ = ("_fold", "_input_ir", "_nat")

    def __init__(self, agg: Aggregate) -> None:
        self._nat = engine()
        self._fold = RunningAggregate(agg)
        self._input_ir = json.dumps(agg.input.to_ir())  # scans source 0

    def push(self, batch: pa.RecordBatch) -> int:
        """Fold one source batch into the running state; return rows consumed."""
        if batch.num_rows == 0:
            return 0
        rows = self._nat.execute_plan(
            self._input_ir, [[batch]], active_config().engine_config_json()
        )
        if not rows or sum(b.num_rows for b in rows) == 0:
            return 0
        self._fold.push(rows)
        return batch.num_rows

    def finalize(self) -> pa.RecordBatch | None:
        """Materialize the current aggregate result, or None if no groups yet."""
        return self._fold.finalize()

    def state(self) -> pa.RecordBatch | None:
        """The running partial state, for a checkpoint snapshot (None if empty)."""
        return self._fold.state()

    def restore(self, state: pa.RecordBatch) -> None:
        """Seed the running partial state from a checkpoint snapshot."""
        self._fold.restore(state)


def stream_aggregate(
    agg: Aggregate, source: Source, batch_size: int | None = None
) -> Iterator[pa.RecordBatch]:
    """Aggregate `source` incrementally, holding only one running partial state.

    `agg.input` must be a breaker-free relational pipeline over the single source
    (filter/project/scan); each source batch is run through it, partial-aggregated,
    and combined into the running state. Yields the finalized result once the source
    is exhausted (one logical result, optionally rebatched by `batch_size`).
    """
    fold = _AggFold(agg)
    for batch in source.iter_batches(None):
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
    nat = engine()
    empty = pa.RecordBatch.from_pylist([], schema=source.schema())
    out = nat.execute_plan(json.dumps(agg.to_ir()), [[empty]], active_config().engine_config_json())
    return next((b for b in out if b.num_rows), None)


_EPOCH = datetime.datetime(1970, 1, 1)


def _td(micros: int) -> datetime.timedelta:
    """A timedelta of `micros` microseconds (added to `_EPOCH` to build a literal)."""
    return datetime.timedelta(microseconds=micros)


#: Schema-metadata key under which a windowed fold's watermark travels with its state batch.
#: The `StateStore` persists exactly one `RecordBatch`, and the watermark is a scalar — this is
#: how the scalar rides along without forking that contract.
_WATERMARK_META = b"batcher.watermark_micros"


def _window_key(agg: Aggregate) -> tuple[str, int] | None:
    """The (alias, width_micros) of the `window_start` group key, or None."""
    from batcher.plan.expr_ir.func_nodes import WindowStart

    for key in agg.group_keys:
        if isinstance(key.expr, WindowStart):
            return key.alias, key.expr.width_micros
    return None


def _scan_filter_ir(predicate) -> str:
    """A `filter(scan source 0, predicate)` plan as JSON IR."""
    return json.dumps(
        {"op": "filter", "input": {"op": "scan", "source_id": 0}, "predicate": predicate.to_ir()}
    )


class _WindowedAggFold:
    """Watermark-bounded windowed aggregation: fold, evict closed windows, emit.

    Holds one running partial state keyed by `window_start` plus a scalar watermark
    (`max event time minus lateness`). Per micro-batch it drops late rows, advances the
    watermark, folds the survivors, then **evicts** every window whose end is at or
    below the watermark — emitting those finalized rows and dropping them from state,
    so memory is bounded by the number of *open* windows (the Flink/Spark bound). All
    row-touching work (late filter, eviction split, max event time) runs in Rust /
    Arrow kernels; this only advances a scalar and orchestrates.
    """

    __slots__ = (
        "_cap",
        "_fold",
        "_input_ir",
        "_lateness",
        "_nat",
        "_time_col",
        "_w_alias",
        "_width",
        "_wm",
    )

    def __init__(self, agg: Aggregate, w_alias: str, width: int) -> None:
        self._nat = engine()
        self._fold = RunningAggregate(agg)
        self._input_ir = json.dumps(agg.input.to_ir())
        self._w_alias = w_alias
        self._width = width
        self._time_col = agg.watermark.time_col
        self._lateness = agg.watermark.lateness_micros
        self._wm: int | None = None
        # The retained open-window state is bounded by the watermark advancing; cap it
        # so a stalled watermark fails loudly instead of OOMing (read once here).
        self._cap = active_config().memory.streaming_state_budget_bytes()

    def _advance_watermark(self, batch: pa.RecordBatch) -> None:
        import pyarrow.compute as pc

        col = batch.column(self._time_col)
        hi = pc.max(col)
        if not hi.is_valid:
            return
        # The watermark, window widths, and `window_start` all live in microseconds
        # (the engine's `window_start` output is always `timestamp[us]`), but the event-
        # time column may be any timestamp resolution (s/ms/us/ns). Normalize to
        # microseconds first — reading the raw int64 ticks of a non-`us` column would
        # scale the watermark by 1000× (dropping every row, or overflowing the literal).
        micros = pc.cast(pc.cast(hi, pa.timestamp("us")), pa.int64()).as_py()
        candidate = micros - self._lateness
        self._wm = candidate if self._wm is None else max(self._wm, candidate)

    def push(self, batch: pa.RecordBatch) -> list[pa.RecordBatch]:
        from batcher.plan.expr_ir import col, lit

        if batch.num_rows == 0:
            return []
        cfg = active_config().engine_config_json()
        # Drop rows below the current watermark (late records) in Rust. The watermark
        # literal is microseconds (`timestamp[us]`); normalize the event-time column to
        # the same resolution so a non-`us` timestamp neither mis-compares nor raises a
        # unit-mismatch error in the engine.
        if self._wm is not None:
            on_time = col(self._time_col).cast("timestamp") >= lit(_EPOCH + _td(self._wm))
            kept = self._nat.execute_plan(_scan_filter_ir(on_time), [[batch]], cfg)
        else:
            kept = [batch]
        self._advance_watermark(batch)
        for b in kept:
            if b.num_rows == 0:
                continue
            self._fold.push(self._nat.execute_plan(self._input_ir, [[b]], cfg))
        out = self._evict(cfg)
        # After eviction, what remains is the open-window state; if it has outgrown the
        # cap the watermark is not closing windows (a stall), so fail clearly.
        self._check_state_bounded()
        return out

    def _check_state_bounded(self) -> None:
        size = self._fold.nbytes()
        if size > self._cap:
            from batcher._internal.errors import ResourceError

            raise ResourceError(
                f"windowed streaming aggregate state reached {size} bytes (cap "
                f"{self._cap}): the watermark on '{self._time_col}' is not advancing, so "
                "closed windows never evict (an event-time gap or an idle source), or the "
                "key space is too large. Advance event time, narrow the keys, or raise "
                "memory.streaming_state_max_bytes."
            )

    def _evict(self, cfg: str) -> list[pa.RecordBatch]:
        from batcher.plan.expr_ir import col, lit

        state = self._fold.state()
        if state is None or self._wm is None:
            return []
        thr = _EPOCH + _td(self._wm - self._width)  # window_start ≤ thr ⟺ window closed
        wk = col(self._w_alias)
        closed = [
            b
            for b in self._nat.execute_plan(_scan_filter_ir(wk <= lit(thr)), [[state]], cfg)
            if b.num_rows
        ]
        open_ = [
            b
            for b in self._nat.execute_plan(
                _scan_filter_ir(wk.is_null() | (wk > lit(thr))), [[state]], cfg
            )
            if b.num_rows
        ]
        self._fold.combine_all(open_)
        result = self._fold.finalize_partials(closed)
        return [result] if result is not None else []

    def flush(self) -> pa.RecordBatch | None:
        """Finalize and emit every remaining (open) window — the end-of-stream flush."""
        return self._fold.take()

    def state(self) -> pa.RecordBatch | None:
        """The open-window partials **and the watermark**, as one checkpointable batch.

        Both halves are state, and checkpointing only the partials would be worse than
        checkpointing nothing: on restore the engine would hold windows it could never close,
        and would re-admit as on-time the very rows the old watermark had already ruled late.
        So the watermark rides in the batch's schema metadata (Arrow IPC persists it), keeping
        the `StateStore`'s "state is one RecordBatch" contract intact.

        A watermark that has advanced with no open windows still has to survive, so that case
        snapshots a zero-column batch carrying only the metadata — dropping it would silently
        rewind event time to the next batch's maximum.
        """
        running = self._fold.state()
        if running is None and self._wm is None:
            return None
        meta = {_WATERMARK_META: b"" if self._wm is None else str(self._wm).encode()}
        if running is None:
            return pa.RecordBatch.from_pylist([], schema=pa.schema([], metadata=meta))
        return running.replace_schema_metadata({**(running.schema.metadata or {}), **meta})

    def restore(self, state: pa.RecordBatch) -> None:
        """Resume the open windows and the watermark from a checkpoint snapshot."""
        raw = (state.schema.metadata or {}).get(_WATERMARK_META)
        # `b""` means "no watermark yet"; `b"0"` and `b"-5"` are real watermarks, so test for
        # emptiness rather than truthiness of the decoded int.
        self._wm = int(raw) if raw else None
        self._fold.restore(state if state.num_columns else None)


def stream_windowed_aggregate(
    agg: Aggregate, source: Source, batch_size: int | None = None
) -> Iterator[pa.RecordBatch]:
    """Windowed aggregation over a stream, emitting each window as the watermark
    closes it and flushing the rest at end-of-stream (bounded state)."""
    key = _window_key(agg)
    if key is None or agg.watermark is None:  # not a watermarked windowed agg
        yield from stream_aggregate(agg, source, batch_size)
        return
    fold = _WindowedAggFold(agg, key[0], key[1])
    for batch in source.iter_batches(None):
        for result in fold.push(batch):
            yield from _rebatch(result, batch_size)
    final = fold.flush()
    if final is not None:
        yield from _rebatch(final, batch_size)


def stream_distinct(
    distinct: Distinct, source: Source, batch_size: int | None = None
) -> Iterator[pa.RecordBatch]:
    """DISTINCT over a streaming source, with bounded memory.

    DISTINCT is a group-by over *all* columns with no aggregate functions, so it
    reuses the incremental aggregate driver verbatim: identical rows fold into the
    same running group, and the state is bounded by the number of distinct rows.
    """
    yield from stream_aggregate(distinct.as_aggregate(), source, batch_size)


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
    nat = engine()

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
    for batch in source.iter_batches(None):
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
