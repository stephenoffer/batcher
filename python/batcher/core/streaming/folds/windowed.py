"""`_WindowedAggFold` — the watermark-bounded windowed aggregate, its spill tier and its
changelog.

The operator whose state has a *cold end*: the watermark only moves forward, so windows
evict in increasing order. That one property is what lets this fold spill to disk and read a
window back exactly once, and — because eviction removes a prefix of a totally ordered axis —
what lets it record a changelog whose whole tombstone is a single integer.
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
from batcher.core.streaming.folds.shared import check_agg_state_bounded
from batcher.plan.logical import Aggregate
from batcher.plan.streaming import StateOperatorProgress, WatermarkTracker

__all__ = ["_WindowedAggFold", "_window_key"]

_EPOCH = datetime.datetime(1970, 1, 1)


def _td(micros: int) -> datetime.timedelta:
    """A timedelta of `micros` microseconds (added to `_EPOCH` to build a literal)."""
    return datetime.timedelta(microseconds=micros)


#: Schema-metadata key under which a windowed fold's watermark travels with its state batch.
#: The `StateStore` persists exactly one `RecordBatch`, and the watermark state is a small
#: JSON document — this is how it rides along without forking that contract.
_WATERMARK_META = b"batcher.watermark_micros"

#: The highest window start this fold has already evicted, carried alongside the watermark.
#:
#: This one integer is the whole tombstone a changelog for a *windowed* aggregate needs. A
#: changelog records what was folded in and has no way to say what was taken out, which is
#: why the evicting operators keep whole snapshots — but eviction here is not an arbitrary
#: deletion. It removes windows whose start is at or below a threshold, on a totally ordered
#: axis, so what it removes is always a **prefix**. A prefix is described by its upper bound,
#: and replaying `combine` over the partials and then dropping everything at or below that
#: bound reconstructs exactly the open set, without recording a single key.
#:
#: Applying the bound to a *whole* snapshot is a no-op, because that state is already
#: post-eviction — so restore does it unconditionally rather than branching on which kind of
#: checkpoint it is reading.
_EVICTED_META = b"batcher.evicted_through_micros"


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


def _median_window(state: pa.RecordBatch, window_column: str) -> int | None:
    """The median window start in `state`, as int64 microseconds, or None if unusable.

    One Arrow kernel over the window column — no per-row Python, and no sort of the whole
    state, which is the cost the spill exists to avoid paying.
    """
    import pyarrow.compute as pc

    from batcher.plan.streaming import event_micros

    column = event_micros(state.column(window_column))
    if column.null_count == len(column):
        return None
    value = pc.approximate_median(column).as_py()
    return None if value is None else int(value)


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
        "_delta",
        "_drop",
        "_evicted_micros",
        "_evicted_through",
        "_fold",
        "_hop",
        "_input_ir",
        "_late_dropped",
        "_nat",
        "_partition_cols",
        "_removed",
        "_spill",
        "_spilled_rows",
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
        # The highest window start already evicted, and this epoch's changelog entry. See
        # `_EVICTED_META` and `take_delta`.
        self._evicted_micros: int | None = None
        self._delta: list[pa.RecordBatch] = []
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
        # The disk tier under the state cap, created only if the cap is ever reached. See
        # `_spill_cold_windows`.
        self._spill: Any = None
        self._spilled_rows = 0

    def push(self, batch: pa.RecordBatch) -> list[pa.RecordBatch]:
        from batcher.plan.expr_ir import col, lit

        self._late_dropped = self._removed = self._updated = 0
        self._delta = []
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
            folded = self._fold.push(partial)
            if folded is not None:
                self._delta.append(folded)
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
        """Keep resident state under the cap — by spilling cold windows, then by failing.

        Reaching the cap used to end the query. That is the wrong answer for the shape that
        reaches it most legitimately: an open set `allowed_lateness / hop` windows wide, one
        row per group key, behaving exactly as designed and simply large. Spilling the oldest
        windows to disk trades latency for survival, which is the trade every mature streaming
        engine makes, and the watermark's one-way motion is what makes it cheap here — a
        spilled window is read back once, when it closes, and never sought into.

        The `ResourceError` still exists and still means something, but it now means what it
        says: the state cannot be spilled far enough because the *newest* windows alone
        exceed the cap. No amount of disk fixes that; it is a key space too wide for the
        envelope, or a watermark that has stopped closing anything.
        """
        if self._fold.nbytes() > self._cap:
            self._spill_cold_windows()
        check_agg_state_bounded(
            self._fold,
            self._cap,
            f"the watermark on '{self._time_col}' is not advancing, so closed windows never "
            "evict (an event-time gap or an idle source), or the key space is too large — "
            "and the newest windows alone exceed the cap, so spilling the older ones to disk "
            "did not help. Advance event time, narrow the keys, or raise "
            "memory.streaming_state_max_bytes",
            label="windowed streaming aggregate",
        )

    def _spill_cold_windows(self) -> None:
        """Move the oldest half of the open windows to disk, repeatedly, until under the cap.

        The split point is the **median** window start, taken with an Arrow kernel over the
        state's window column. A median rather than a tuned fraction because it needs no
        tuning and halves resident state on every pass whatever the distribution: a run of
        passes reaches the target in a logarithmic number of steps, and each one leaves the
        newest windows — the ones incoming rows actually land in — resident.

        Stops when a pass moves nothing, which is the unspillable case: every remaining row
        shares the newest window start. The caller then raises, because that is genuinely a
        state too large rather than a state in the wrong place.
        """
        from batcher.core.streaming.spill import SpilledWindows
        from batcher.plan.expr_ir import col, lit

        target = self._cap // 2
        while self._fold.nbytes() > target:
            state = self._fold.state()
            if state is None or state.num_rows <= 1:
                return
            split = _median_window(state, self._w_alias)
            if split is None:
                return
            wk = col(self._w_alias)
            thr = _EPOCH + _td(split)
            cold = [
                b
                for b in self._nat.execute_plan(
                    _scan_filter_ir(wk <= lit(thr)), [[state]], self._cfg
                )
                if b.num_rows
            ]
            hot = [
                b
                for b in self._nat.execute_plan(
                    _scan_filter_ir(wk.is_null() | (wk > lit(thr))), [[state]], self._cfg
                )
                if b.num_rows
            ]
            if not cold or not hot:
                # Everything is on one side of its own median: one window start holds the
                # whole state, so there is no colder half to move.
                return
            if self._spill is None:
                self._spill = SpilledWindows(self._w_alias)
            for batch in cold:
                self._spill.spill(batch)
            self._fold.combine_all(hot)
            self._spilled_rows = self._spill.rows()

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
        self._evicted_micros = threshold
        thr = _EPOCH + _td(threshold)  # window_start ≤ thr ⟺ window closed
        wk = col(self._w_alias)
        # Everything that can hold a window this sweep closes: the resident state, plus the
        # spilled runs whose range reaches the threshold. A run comes back whole and is split
        # by the same predicate the resident state is, so there is one rule for what "closed"
        # means rather than two that have to agree.
        sources = [state]
        if self._spill is not None:
            sources.extend(self._spill.drain_through(threshold))
            self._spilled_rows = self._spill.rows()
        closed: list[pa.RecordBatch] = []
        open_: list[pa.RecordBatch] = []
        for source in sources:
            closed.extend(
                b
                for b in self._nat.execute_plan(_scan_filter_ir(wk <= lit(thr)), [[source]], cfg)
                if b.num_rows
            )
            open_.extend(
                b
                for b in self._nat.execute_plan(
                    _scan_filter_ir(wk.is_null() | (wk > lit(thr))), [[source]], cfg
                )
                if b.num_rows
            )
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
        resident = 0 if state is None else state.num_rows
        return StateOperatorProgress(
            operator_name="windowed_aggregate",
            # Rows on disk are still state. Reporting only the resident half would make a
            # spilling query look like it had *shed* state rather than moved it, which is
            # exactly backwards for the operator watching this number to decide whether the
            # key space is too wide.
            num_rows_total=resident + self._spilled_rows,
            num_rows_updated=self._updated,
            num_rows_removed=self._removed,
            memory_used_bytes=self._fold.nbytes(),
            num_late_inputs_dropped=self._late_dropped,
            watermark_micros=self._wm,
        )

    def flush(self) -> pa.RecordBatch | None:
        """Finalize and emit every remaining (open) window — the end-of-stream flush.

        Spilled runs come back first. A flush that ignored them would drop every window the
        memory pressure had moved to disk — silently, and only on the streams large enough to
        have spilled at all, which is the worst possible place for a quiet loss.
        """
        if self._spill is not None:
            self._fold.combine_all([*self._resident(), *self._spill.drain_all()])
            self._spill.close()
            self._spill = None
            self._spilled_rows = 0
        return self._fold.take()

    def _resident(self) -> list[pa.RecordBatch]:
        """The in-memory partial state as a list, empty when nothing has been folded."""
        state = self._fold.state()
        return [] if state is None else [state]

    def _drop_evicted(self, threshold: int) -> None:
        """Remove every window at or below `threshold` from the state, emitting nothing."""
        from batcher.plan.expr_ir import col, lit

        state = self._fold.state()
        if state is None:
            return
        wk = col(self._w_alias)
        thr = _EPOCH + _td(threshold)
        open_ = [
            b
            for b in self._nat.execute_plan(
                _scan_filter_ir(wk.is_null() | (wk > lit(thr))), [[state]], self._cfg
            )
            if b.num_rows
        ]
        self._fold.combine_all(open_)

    def close(self) -> None:
        """Release the spill scratch. Idempotent; safe on a fold that never spilled."""
        if self._spill is not None:
            self._spill.close()
            self._spill = None
            self._spilled_rows = 0

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
        meta = self._meta()
        if running is None:
            return pa.RecordBatch.from_pylist([], schema=pa.schema([], metadata=meta))
        return running.replace_schema_metadata({**(running.schema.metadata or {}), **meta})

    def _meta(self) -> dict[bytes, bytes]:
        """The scalars that must ride with any checkpointed batch of this fold.

        The tracker's whole per-partition state, not the frontier alone — restoring the
        frontier by itself restarts every partition's maximum from nothing, so the first
        partition to speak after a restart sets the minimum on its own — plus the eviction
        bound that makes a changelog expressible at all (see `_EVICTED_META`).
        """
        meta = {_WATERMARK_META: self._tracker.to_json().encode()}
        if self._evicted_micros is not None:
            meta[_EVICTED_META] = str(self._evicted_micros).encode()
        return meta

    def take_delta(self) -> pa.RecordBatch | None:
        """Take this micro-batch's changelog entry — what it folded in, plus the eviction bound.

        **Reading consumes it**, for the reason the running aggregate's does: the engine
        commits epochs that never push (the end-of-drain marker, an idle trigger), and an
        entry left in place is written a second time under a new batch id and replayed twice.

        The entry is the partial as it was folded *in*, before eviction ran. Replaying a
        chain therefore rebuilds the pre-eviction state, and the bound in the metadata is
        what turns that back into the open set — which is why this operator can offer a
        changelog at all despite removing state, and why the bound must travel with every
        entry rather than only with the base.

        Returns:
            One combined partial for the epoch, or None when it folded nothing in.
        """
        delta, self._delta = self._delta, []
        if not delta:
            return None
        combined = delta[0] if len(delta) == 1 else self._nat.combine(*self._fold.spec, delta)
        return combined.replace_schema_metadata(
            {**(combined.schema.metadata or {}), **self._meta()}
        )

    def state_parts(self) -> Iterator[pa.RecordBatch]:
        """The whole open-window state as a stream of parts — resident first, then spilled.

        `state()` returns the resident half, which is all there is until the fold spills. Once
        it has, a snapshot built from that alone would persist only what happened to be in
        memory and silently drop every window the memory pressure had moved to disk — on
        exactly the queries large enough to spill, which is the worst place for a quiet loss.

        A stream rather than one concatenated batch because a spilled fold's state is larger
        than the memory cap by construction; the snapshot writer consumes this run by run, so
        the peak is one part. The spilled runs are read **without** being consumed, so a
        snapshot leaves the fold exactly as it found it.

        Yields:
            The resident state (carrying the watermark in its schema metadata), then one
            batch per spilled run.
        """
        resident = self.state()
        if resident is not None:
            yield resident
        if self._spill is not None:
            yield from self._spill.iter_runs()

    def restore(self, state: pa.RecordBatch) -> None:
        """Resume the open windows and the watermark state from a checkpoint snapshot.

        The payload is the tracker's JSON, and a bare integer written by an earlier version
        is read as a single global partition — so a query checkpointed before per-partition
        watermarks existed resumes rather than rewinding its event time.
        """
        self.restore_parts([state])

    def restore_parts(self, parts: Sequence[pa.RecordBatch]) -> None:
        """Resume from a multi-part snapshot: the resident state plus any spilled runs.

        The parts are combined rather than concatenated, which is what makes the split into
        parts invisible to the result: they hold *partial* aggregate state and `combine` is
        associative and commutative (invariant #7), so where the window rows happened to sit
        when the snapshot was taken cannot change what they fold to.

        The watermark rides in the schema metadata of whichever part carries it — the
        resident one, written first — so it is recovered from the first part that has it
        rather than assumed to be on any particular one.

        Args:
            parts: The snapshot's batches, in the order they were written.
        """
        raw = next(
            (
                (part.schema.metadata or {}).get(_WATERMARK_META)
                for part in parts
                if (part.schema.metadata or {}).get(_WATERMARK_META)
            ),
            None,
        )
        self._tracker.restore(raw.decode() if raw else None)
        self._wm = self._tracker.watermark
        # A restored fold has swept nothing yet, so the next push must evict rather than
        # assume the recovered watermark's windows were already emitted by the dead run.
        self._evicted_through = None
        evicted = next(
            (
                (part.schema.metadata or {}).get(_EVICTED_META)
                for part in reversed(list(parts))
                if (part.schema.metadata or {}).get(_EVICTED_META)
            ),
            None,
        )
        rows = [part for part in parts if part.num_columns and part.num_rows]
        self._fold.combine_all(rows)
        if evicted is not None:
            # Re-apply the prefix eviction the chain's partials predate. A no-op on a whole
            # snapshot, whose state is already post-eviction — which is why this is
            # unconditional rather than branching on the checkpoint's kind. Silent by
            # design: these windows were emitted before the crash, so emitting them again
            # under a new batch id is a duplicate no sink's by-batch-id idempotency absorbs.
            self._evicted_micros = int(evicted)
            self._drop_evicted(self._evicted_micros)
            self._evicted_through = self._evicted_micros // self._hop
        # Recovery can land more state than the cap allows — the snapshot it came from was
        # spilled precisely because it did. Re-spill now rather than waiting for the next
        # push, so a restart cannot hold more than a live query would.
        if rows:
            self._check_state_bounded()
