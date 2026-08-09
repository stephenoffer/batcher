"""The running-state folds a streaming aggregate is built on, and their memory bound.

An aggregation is mergeable (`partial -> combine -> finalize`), so it can run over an
unbounded / larger-than-memory source one micro-batch at a time: each batch's partial state
is folded into a single running state — bounded by the number of groups, not the input size
— via the native `combine`. `_AggFold` is that running state; `_WindowedAggFold` adds the
watermark that *releases* it, evicting each window as it closes so memory is bounded by the
number of open windows rather than by the stream's lifetime.

The two release state differently, which is why they diagnose an over-budget state
differently while sharing one check (`check_agg_state_bounded`).

Core's lane: this drives the engine (`batcher._native`) over the plan it is given; it makes
no optimization decisions.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Iterator

import pyarrow as pa

from batcher._internal.native import engine
from batcher.config import active_config
from batcher.core.mergeable import RunningAggregate
from batcher.io.source import Source, iter_source
from batcher.plan.logical import Aggregate
from batcher.plan.streaming import StateOperatorProgress

__all__ = ["check_agg_state_bounded", "empty_global_aggregate", "streaming_state_budget"]


def empty_global_aggregate(agg: Aggregate, schema: pa.Schema) -> pa.RecordBatch | None:
    """The one row a *keyless* aggregate owes an input that had none.

    A global `count`/`sum` over zero rows still yields exactly one row — `0`, `NULL` — which
    is what SQL, DuckDB, and `collect()` all produce. The incremental fold cannot produce
    it: it skips empty batches, so with nothing to finalize it yields nothing at all and the
    stream silently disagrees with the oracle. Asking the engine for the empty-input result
    through the ordinary plan path means the identity element falls out of the mergeable
    algebra rather than being special-cased per aggregate function.

    Takes a schema rather than a source, so both the `iter_batches` driver and the
    micro-batch processor — which has a plan and no source — can reach the same answer.

    Args:
        agg: The keyless aggregate.
        schema: The input schema, to type the empty batch fed through the plan.

    Returns:
        The one-row result, or `None` if the engine produced none.
    """
    nat = engine()
    empty = pa.RecordBatch.from_pylist([], schema=schema)
    out = nat.execute_plan(json.dumps(agg.to_ir()), [[empty]], active_config().engine_config_json())
    return next((b for b in out if b.num_rows), None)


def _rebatch(result: pa.RecordBatch, batch_size: int | None) -> Iterator[pa.RecordBatch]:
    """Yield `result` whole, or sliced into `batch_size`-row chunks."""
    if batch_size is None:
        yield result
    else:
        for off in range(0, result.num_rows, batch_size):
            yield result.slice(off, batch_size)


def _read(source: Source, projection: list[str] | None) -> Iterator[pa.RecordBatch]:
    """Read `source` through the projection Kyber decided for this plan.

    Every driver in this module used to call ``source.iter_batches(None)`` — decoding *every*
    column of every message regardless of what the plan touched, while `_iter_streaming` on
    the neighbouring path already read through the pushdown. On a wide topic that is the
    dominant cost of a streaming aggregate: a `group_by("user").sum("cents")` over a
    forty-column event decoded thirty-eight columns it then discarded, per micro-batch,
    forever.

    Core does not compute the projection — the conductor asks Kyber for it and passes it in,
    keeping the decision in Kyber's lane. `iter_source` degrades safely, so a source that
    cannot narrow its read simply returns everything and the plan is unaffected.

    Args:
        source: The stream to read.
        projection: Columns the plan needs, or ``None`` to read everything.

    Returns:
        An iterator of the source's record batches, narrowed where the source can.
    """
    return iter_source(source, projection, None)


def streaming_state_budget() -> int:
    """The byte envelope a streaming operator's retained state must stay inside."""
    return active_config().memory.streaming_state_budget_bytes()


def check_agg_state_bounded(fold, cap: int, cause: str, *, label: str) -> None:
    """Raise an actionable `ResourceError` when a running aggregate has outgrown `cap`.

    Streaming state is only bounded by something that *releases* it, and the two folds here
    release differently — the windowed one by an advancing watermark, the plain one not at
    all — so the diagnosis differs while the check does not. Sharing the check is what keeps
    them from drifting into one operator that guards its state and one that does not, which
    is exactly what had happened.

    Args:
        fold: The running aggregate to measure.
        cap: The byte budget.
        cause: Why the state grew, and what the user can do about it.
        label: The operator's name, quoted back in the message.

    Raises:
        ResourceError: When the retained state exceeds `cap`.
    """
    size = fold.nbytes()
    if size > cap:
        from batcher._internal.errors import ResourceError

        raise ResourceError(f"{label} state reached {size} bytes (cap {cap}): {cause}.")


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

    __slots__ = ("_cfg", "_fold", "_input_ir", "_nat", "_updated")

    def __init__(self, agg: Aggregate) -> None:
        self._nat = engine()
        self._fold = RunningAggregate(agg)
        self._input_ir = json.dumps(agg.input.to_ir())  # scans source 0
        # Constant for the query, so read and serialize it once. `push` runs per micro-batch
        # and rebuilt this every time — a config lookup plus a JSON dump charged to the
        # latency of every epoch, which is exactly what S10 hoisted out of `stream_topn`.
        self._cfg = active_config().engine_config_json()
        # Partial rows this fold last absorbed, reported as `num_rows_updated`. See `metrics`.
        self._updated = 0

    def push(self, batch: pa.RecordBatch) -> int:
        """Fold one source batch into the running state; return rows consumed."""
        self._updated = 0
        if batch.num_rows == 0:
            return 0
        rows = self._nat.execute_plan(self._input_ir, [[batch]], self._cfg)
        if not rows or sum(b.num_rows for b in rows) == 0:
            return 0
        self._fold.push(rows)
        self._updated = sum(b.num_rows for b in rows)
        return batch.num_rows

    def metrics(self) -> StateOperatorProgress:
        """This fold's state after the last `push` — what the progress record reports.

        A streaming aggregation with no watermark never evicts, so `num_rows_removed` is
        always zero and `num_rows_total` is the whole answer: it is the number that grows
        without bound when the group key is too wide, and the one an operator watches to
        see it happening before the memory guard fires.
        """
        state = self._fold.state()
        return StateOperatorProgress(
            operator_name="aggregate",
            num_rows_total=0 if state is None else state.num_rows,
            num_rows_updated=self._updated,
            memory_used_bytes=self._fold.nbytes(),
        )

    def finalize(self) -> pa.RecordBatch | None:
        """Materialize the current aggregate result, or None if no groups yet."""
        return self._fold.finalize()

    def nbytes(self) -> int:
        """Bytes the running partial state currently holds — the streaming memory bound."""
        return self._fold.nbytes()

    def state(self) -> pa.RecordBatch | None:
        """The running partial state, for a checkpoint snapshot (None if empty)."""
        return self._fold.state()

    def restore(self, state: pa.RecordBatch) -> None:
        """Seed the running partial state from a checkpoint snapshot."""
        self._fold.restore(state)


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
        "_cfg",
        "_evicted_through",
        "_fold",
        "_input_ir",
        "_late_dropped",
        "_lateness",
        "_nat",
        "_removed",
        "_time_col",
        "_updated",
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
        # Constant for the query; rebuilt per micro-batch before (see `_AggFold`).
        self._cfg = active_config().engine_config_json()
        # The index of the last window the watermark had closed at the previous sweep. See
        # `_evict`; `None` means nothing has been swept yet.
        self._evicted_through: int | None = None
        # The retained open-window state is bounded by the watermark advancing; cap it
        # so a stalled watermark fails loudly instead of OOMing (read once here).
        self._cap = active_config().memory.streaming_state_budget_bytes()
        # Per-micro-batch counters, reported through `metrics`. `_late_dropped` is the one
        # that mattered and did not exist: a row below the watermark is filtered out in
        # Rust and simply never appears in the result, so a window that closed too early
        # left a total quietly short by an amount nothing anywhere recorded.
        self._late_dropped = 0
        self._removed = 0
        self._updated = 0

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

        self._late_dropped = self._removed = self._updated = 0
        if batch.num_rows == 0:
            return []
        cfg = self._cfg
        # Drop rows below the current watermark (late records) in Rust. The watermark
        # literal is microseconds (`timestamp[us]`); normalize the event-time column to
        # the same resolution so a non-`us` timestamp neither mis-compares nor raises a
        # unit-mismatch error in the engine.
        if self._wm is not None:
            on_time = col(self._time_col).cast("timestamp") >= lit(_EPOCH + _td(self._wm))
            kept = self._nat.execute_plan(_scan_filter_ir(on_time), [[batch]], cfg)
            # Counted here rather than inferred later: this is the only point at which
            # the difference between "arrived" and "counted" is still visible.
            self._late_dropped = batch.num_rows - sum(b.num_rows for b in kept)
        else:
            kept = [batch]
        self._advance_watermark(batch)
        for b in kept:
            if b.num_rows == 0:
                continue
            partial = self._nat.execute_plan(self._input_ir, [[b]], cfg)
            self._fold.push(partial)
            self._updated += sum(p.num_rows for p in partial)
        out = self._evict(cfg)
        # After eviction, what remains is the open-window state; if it has outgrown the
        # cap the watermark is not closing windows (a stall), so fail clearly.
        self._check_state_bounded()
        return out

    def _check_state_bounded(self) -> None:
        check_agg_state_bounded(
            self._fold,
            self._cap,
            f"the watermark on '{self._time_col}' is not advancing, so closed windows never "
            "evict (an event-time gap or an idle source), or the key space is too large. "
            "Advance event time, narrow the keys, or raise memory.streaming_state_max_bytes",
            label="windowed streaming aggregate",
        )

    def _evict(self, cfg: str) -> list[pa.RecordBatch]:
        """Emit and drop every window the watermark has closed since the last sweep.

        Skipped entirely when no *new* window can have closed, which is the common case:
        a window is closed exactly when ``window_start <= watermark - width``, and window
        starts are multiples of the width, so the closed set changes only when the threshold
        crosses a width boundary. With a ten-minute window and a per-second watermark, that
        is one micro-batch in six hundred. Each sweep costs *two* full relational passes over
        the whole open-window state (one for the closed side, one for the open side) plus a
        re-combine, so running it unconditionally made the per-epoch cost scale with the
        retained state rather than with the batch — the opposite of what bounded-state
        streaming is for.

        Keying on the boundary index rather than on the raw threshold is what makes the skip
        actually bite: the threshold itself advances with every batch, so comparing it would
        sweep every time and buy nothing.

        No surviving row can land in an already-swept window, which is what makes the skip
        safe rather than merely cheap: a row is kept only when its event time is at or above
        the watermark, so its ``window_start`` exceeds ``watermark - width``, putting it in a
        strictly later bucket than the one just swept.
        """
        from batcher.plan.expr_ir import col, lit

        state = self._fold.state()
        if state is None or self._wm is None:
            return []
        threshold = self._wm - self._width
        bucket = threshold // self._width  # the last window index the watermark has closed
        if bucket == self._evicted_through:
            return []
        self._evicted_through = bucket
        thr = _EPOCH + _td(threshold)  # window_start ≤ thr ⟺ window closed
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
        self._removed = sum(b.num_rows for b in closed)
        self._fold.combine_all(open_)
        result = self._fold.finalize_partials(closed)
        return [result] if result is not None else []

    def metrics(self) -> StateOperatorProgress:
        """This fold's state after the last `push` — what the progress record reports.

        `num_late_inputs_dropped` is the field this whole path exists to surface: rows
        that arrived below the watermark were filtered out in Rust and never appeared in
        the output, so the only symptom was a total that was slightly too low.
        """
        state = self._fold.state()
        return StateOperatorProgress(
            operator_name="windowed_aggregate",
            num_rows_total=0 if state is None else state.num_rows,
            num_rows_updated=self._updated,
            num_rows_removed=self._removed,
            memory_used_bytes=self._fold.nbytes(),
            num_late_inputs_dropped=self._late_dropped,
            watermark_micros=self._wm,
        )

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
        # A restored fold has swept nothing yet, so the next push must evict rather than
        # assume the recovered watermark's windows were already emitted by the dead run.
        self._evicted_through = None
        self._fold.restore(state if state.num_columns else None)
