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
from collections.abc import Iterator, Sequence
from typing import Any, NamedTuple

import pyarrow as pa

from batcher._internal.native import engine
from batcher.config import active_config
from batcher.core.mergeable import RunningAggregate
from batcher.io.source import Source, iter_source
from batcher.plan.logical import Aggregate
from batcher.plan.streaming import StateOperatorProgress, WatermarkTracker

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
#: The `StateStore` persists exactly one `RecordBatch`, and the watermark state is a small
#: JSON document — this is how it rides along without forking that contract.
_WATERMARK_META = b"batcher.watermark_micros"


class _WindowKey(NamedTuple):
    """Which group key is the event-time window, and the geometry that closes it.

    `hop` is the distance between consecutive window starts: equal to `width` for a
    tumbling window, and the slide for an overlapping one. The two are separate because
    eviction sweeps on *boundary crossings*, and a sliding window crosses a boundary every
    hop — sweeping on the width would skip most of them and hold closed windows in state.
    """

    alias: str
    width: int
    hop: int


def _window_key(agg: Aggregate) -> _WindowKey | None:
    """The event-time window this aggregate groups by, tumbling or sliding, or None.

    Two shapes express a window, and only one of them puts the geometry in the group key:

    * **Tumbling** — the key *is* `window_start(ts, width)`, a scalar start per row.
    * **Sliding** — a row belongs to several overlapping windows, so `window(ts, w, slide)`
      is the *list* of their starts, which `group_by` rejects outright: the caller explodes
      it first and groups by the resulting plain column. By the time the aggregate is built,
      its group key is an ordinary `Col` and the width and slide live two nodes down, in the
      `WindowBuckets` the `Unnest` fans out.

    Only the first shape was recognized, and the consequence was not a missing optimization.
    A sliding windowed aggregation over a stream fell through to the *unwatermarked* running
    aggregate: nothing was ever dropped as late, nothing was ever evicted, and nothing was
    emitted until a source that never ends ended. The window machinery was configured, the
    watermark was set, and none of it ran.

    Args:
        agg: The aggregate to inspect.

    Returns:
        The window key, or None when no group key is an event-time window.
    """
    from batcher.plan.expr_ir.func_nodes import WindowStart

    for key in agg.group_keys:
        if isinstance(key.expr, WindowStart):
            return _WindowKey(key.alias, key.expr.width_micros, key.expr.width_micros)
    for key in agg.group_keys:
        sliding = _sliding_window_key(agg.input, key)
        if sliding is not None:
            return sliding
    return None


def _sliding_window_key(plan, key) -> _WindowKey | None:
    """The width and slide behind an exploded `window(ts, width, slide)` group key.

    Walks down from the aggregate for the `Unnest` that produced `key`, then for the
    `Project` that built the list it exploded. Both hops tolerate intervening single-input
    nodes (a `Filter` between the explode and the group-by is ordinary), and either one
    failing to match simply means this key is not a sliding window.

    Args:
        plan: The aggregate's input.
        key: The group key being tested, whose `expr` must be a bare column reference.

    Returns:
        The window key, or None when `key` is not an exploded sliding window.
    """
    from batcher.plan.expr_ir import Col
    from batcher.plan.expr_ir.func_nodes import WindowBuckets
    from batcher.plan.logical import Project, Unnest

    if not (isinstance(key.expr, Col) and key.expr.name == key.alias):
        return None
    unnest = _find_below(plan, lambda n: isinstance(n, Unnest) and n.alias == key.alias)
    if unnest is None:
        return None
    project = _find_below(
        unnest.input,
        lambda n: isinstance(n, Project) and any(item.alias == unnest.column for item in n.items),
    )
    if project is None:
        return None
    item = next(i for i in project.items if i.alias == unnest.column)
    if not isinstance(item.expr, WindowBuckets):
        return None
    return _WindowKey(key.alias, item.expr.width_micros, item.expr.slide_micros)


def _find_below(plan, matches) -> object | None:
    """The nearest node at or below `plan` satisfying `matches`, down the single-input spine.

    Stops at the first node with no `input` (a scan) or with more than one — a join or a
    union means the column could have come from either side, and guessing which is how a
    window width gets read off an unrelated expression.
    """
    node = plan
    while node is not None:
        if matches(node):
            return node
        node = getattr(node, "input", None)
    return None


def _scan_filter_ir(predicate) -> str:
    """A `filter(scan source 0, predicate)` plan as JSON IR."""
    return json.dumps(
        {"op": "filter", "input": {"op": "scan", "source_id": 0}, "predicate": predicate.to_ir()}
    )


class _WindowedAggFold:
    """Watermark-bounded windowed aggregation: fold, evict closed windows, emit.

    Holds one running partial state keyed by the window start, plus a `WatermarkTracker`
    that says how far event time has advanced. Per micro-batch it drops late rows, folds
    this batch's event times into the tracker, folds the survivors into the state, then
    **evicts** every window whose end is at or below the watermark — emitting those
    finalized rows and dropping them from state, so memory is bounded by the number of
    *open* windows (the Flink/Spark bound). All row-touching work (late filter, eviction
    split, per-partition maxima) runs in Rust / Arrow kernels; this orchestrates.

    The watermark is the tracker's **minimum over partitions**, not a maximum over rows.
    A maximum is the claim that the fastest partition's progress is the whole stream's, and
    on a multi-partition topic it is false in the direction that loses data: the frontier
    runs ahead of the slow partitions and every row they then deliver is dropped as late,
    with only a quietly-short window total to show for it. `partition_cols` is what makes
    the minimum computable; a source that cannot attribute a row to a partition has one
    partition, where the two agree and nothing changes.
    """

    __slots__ = (
        "_cap",
        "_cfg",
        "_drop",
        "_evicted_through",
        "_fold",
        "_hop",
        "_input_ir",
        "_late_dropped",
        "_nat",
        "_partition_cols",
        "_removed",
        "_time_col",
        "_tracker",
        "_updated",
        "_w_alias",
        "_width",
        "_wm",
    )

    def __init__(
        self,
        agg: Aggregate,
        key: _WindowKey,
        *,
        partition_cols: Sequence[str] = (),
        expected_partitions: Sequence[Sequence[Any]] = (),
        drop_columns: Sequence[str] = (),
    ) -> None:
        """Fold `agg` into windows `key` describes, watermarked across `partition_cols`.

        Args:
            agg: The watermarked windowed aggregate to drive.
            key: Which group key is the window, and its width and hop.
            partition_cols: Columns attributing each row to a stream partition, so the
                watermark can be the minimum over them.
            expected_partitions: Partitions the source says it will read, so the minimum is
                not taken over a subset during startup.
            drop_columns: Columns present only to compute the watermark, removed before the
                batch reaches the plan. Reading a partition column the projection excluded
                must not change what the aggregate sees.
        """
        self._nat = engine()
        self._fold = RunningAggregate(agg)
        self._input_ir = json.dumps(agg.input.to_ir())
        self._w_alias = key.alias
        self._width = key.width
        self._hop = key.hop
        self._time_col = agg.watermark.time_col
        self._partition_cols = tuple(partition_cols)
        self._drop = tuple(drop_columns)
        self._tracker = WatermarkTracker(
            agg.watermark.lateness_micros, expected_partitions=expected_partitions
        )
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

    def push(self, batch: pa.RecordBatch) -> list[pa.RecordBatch]:
        from batcher.plan.expr_ir import col, lit

        self._late_dropped = self._removed = self._updated = 0
        if batch.num_rows == 0:
            return []
        cfg = self._cfg
        # Read the frontier *before* this batch contributes to it: a row is late relative
        # to what the stream had already claimed when it arrived, not to what it claims
        # afterwards. Re-read from the tracker rather than trusting the cached value,
        # because a partition can have crossed its idleness threshold since the last push
        # with no batch in between to notice.
        self._wm = self._tracker.watermark
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
        self._tracker.observe(batch, self._time_col, self._partition_cols)
        self._wm = self._tracker.watermark
        for b in kept:
            if b.num_rows == 0:
                continue
            partial = self._nat.execute_plan(self._input_ir, [[self._narrow(b)]], cfg)
            self._fold.push(partial)
            self._updated += sum(p.num_rows for p in partial)
        out = self._evict(cfg)
        # After eviction, what remains is the open-window state; if it has outgrown the
        # cap the watermark is not closing windows (a stall), so fail clearly.
        self._check_state_bounded()
        return out

    def _narrow(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        """The batch the plan sees — without the columns read only for the watermark.

        Zero-copy: dropping a column rebinds the array list and touches no buffer. It has
        to happen, though, because the alternative is a plan whose result depends on which
        columns the watermark happened to need.
        """
        if not self._drop:
            return batch
        present = [c for c in self._drop if c in batch.schema.names]
        return batch.drop_columns(present) if present else batch

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
        starts are multiples of the **hop**, so the closed set changes only when the
        threshold crosses a hop boundary. With a ten-minute tumbling window and a per-second
        watermark, that is one micro-batch in six hundred. Each sweep costs *two* full
        relational passes over the whole open-window state (one for the closed side, one for
        the open side) plus a re-combine, so running it unconditionally made the per-epoch
        cost scale with the retained state rather than with the batch — the opposite of what
        bounded-state streaming is for.

        Keying on the boundary index rather than on the raw threshold is what makes the skip
        actually bite: the threshold itself advances with every batch, so comparing it would
        sweep every time and buy nothing. It is the *hop* and not the width because a sliding
        window's starts fall every hop: dividing by the width would collapse several distinct
        boundary crossings into one index and hold closed windows in state until a coarser
        boundary happened to come round.

        No surviving row can land in an already-swept window, which is what makes the skip
        safe rather than merely cheap: a row is kept only when its event time is at or above
        the watermark, so its earliest containing window starts after ``watermark - width``,
        putting it in a strictly later bucket than the one just swept. That argument holds
        for overlapping windows too — the *earliest* of a row's several starts is still the
        one bounded below by ``event time - width``.
        """
        from batcher.plan.expr_ir import col, lit

        state = self._fold.state()
        if state is None or self._wm is None:
            return []
        threshold = self._wm - self._width
        bucket = threshold // self._hop  # the last window boundary the watermark has closed
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
        """The open-window partials **and the watermark state**, as one checkpointable batch.

        Both halves are state, and checkpointing only the partials would be worse than
        checkpointing nothing: on restore the engine would hold windows it could never close,
        and would re-admit as on-time the very rows the old watermark had already ruled late.
        So the watermark rides in the batch's schema metadata (Arrow IPC persists it), keeping
        the `StateStore`'s "state is one RecordBatch" contract intact.

        What rides there is the tracker's **whole** per-partition state, not the frontier
        alone. Restoring the frontier by itself would restart every partition's maximum from
        nothing, so the first partition to deliver after a restart would set the minimum on
        its own — reintroducing exactly the over-claim the tracker exists to prevent, once
        per restart, for as long as it took the other partitions to catch up.

        A watermark that has advanced with no open windows still has to survive, so that case
        snapshots a zero-column batch carrying only the metadata — dropping it would silently
        rewind event time to the next batch's maximum.
        """
        running = self._fold.state()
        if running is None and self._wm is None:
            return None
        meta = {_WATERMARK_META: self._tracker.to_json().encode()}
        if running is None:
            return pa.RecordBatch.from_pylist([], schema=pa.schema([], metadata=meta))
        return running.replace_schema_metadata({**(running.schema.metadata or {}), **meta})

    def restore(self, state: pa.RecordBatch) -> None:
        """Resume the open windows and the watermark state from a checkpoint snapshot.

        The payload is the tracker's JSON, and a bare integer written by an earlier version
        is read as a single global partition — so a query checkpointed before per-partition
        watermarks existed resumes rather than rewinding its event time.
        """
        raw = (state.schema.metadata or {}).get(_WATERMARK_META)
        self._tracker.restore(raw.decode() if raw else None)
        self._wm = self._tracker.watermark
        # A restored fold has swept nothing yet, so the next push must evict rather than
        # assume the recovered watermark's windows were already emitted by the dead run.
        self._evicted_through = None
        self._fold.restore(state if state.num_columns else None)
