"""What a micro-batch *becomes* — the per-batch processors and the routing that picks one.

`MicroBatchProcessor` is the whole seam between the engine's cadence and the query's
semantics: the engine pulls a batch and asks for the rows to emit, and each processor here
answers for one output mode. Stateless append runs the Kyber-optimized per-batch plan;
complete/update fold a running aggregate; append over a watermarked windowed aggregate emits
each window as the watermark closes it.

`make_processor` is where an impossible combination fails — at `start()`, with the
Spark-parity rule named, rather than mid-stream.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pyarrow as pa

from batcher.core.streaming import _AggFold
from batcher.plan.streaming import OutputMode, StateOperatorProgress

if TYPE_CHECKING:
    from batcher.plan.logical import Aggregate

__all__ = [
    "AggregateProcessor",
    "KeyedStateProcessor",
    "MicroBatchProcessor",
    "StatelessProcessor",
    "WindowedAggregateProcessor",
    "make_processor",
]


@runtime_checkable
class MicroBatchProcessor(Protocol):
    """Turns one source micro-batch into the rows to emit this micro-batch."""

    def process(self, batch: pa.RecordBatch) -> list[pa.RecordBatch]: ...


class StatelessProcessor:
    """Run a breaker-free pipeline per micro-batch (append output mode).

    `run_batch` is supplied by the conductor: it is the Kyber-optimized per-batch
    relational run (`core.execute_local` over the pushed-down plan), so this class
    holds no optimization logic — it just drops empty batches.
    """

    __slots__ = ("_run",)

    def __init__(self, run_batch: Callable[[pa.RecordBatch], list[pa.RecordBatch]]) -> None:
        self._run = run_batch

    def process(self, batch: pa.RecordBatch) -> list[pa.RecordBatch]:
        return [b for b in self._run(batch) if b.num_rows]


class AggregateProcessor:
    """Fold each micro-batch into a running aggregate and emit the current result.

    `complete` emits the whole running result every micro-batch. `update` emits **only
    the rows that changed since the last trigger**, which is what the mode means in Spark
    and the whole reason to pick it over `complete`: on a wide key space, re-sending every
    group every trigger is the traffic the mode exists to avoid. `append` on an unwindowed
    aggregate is rejected upstream — it needs a watermark to know a group is final.

    The changed rows are found by anti-joining this result against the previous one over
    every column, so a row is "changed" exactly when it is not present unaltered in what
    was last emitted. That is deliberately *value*-based rather than key-based: the group
    keys of an `Aggregate` can be expressions (`window(ts, "1h")`) that the input batch
    does not carry, so "which keys did this batch touch" is not answerable from the batch
    alone, while "which output rows differ" always is. It costs one extra copy of the
    result and one vectorized Arrow join per micro-batch, and it never *omits* a changed
    row — a null or NaN in an output column makes a row compare unequal to itself, so such
    a row is re-emitted every trigger, which an upsert sink absorbs and a missed update
    would not be.

    A `map_batches` input is handled by `run_batch`, which runs the map and hands back the
    mapped batches for the fold to consume. The fold is then built lazily against the
    *mapped* schema, because an `Aggregate` sitting on a `MapBatches` cannot be lowered at
    all: `MapBatches.to_ir()` raises, the UDF being Python is the whole point of it. Until
    this existed, `map → agg` -- inference followed by a rollup, the single most common
    shape an ML streaming job has -- streamed correctly through `iter_batches` and answered
    a streaming write with a bare `NotImplementedError` about the IR.
    """

    __slots__ = ("_agg", "_cap", "_emitted", "_fold", "_keyed", "_map", "_previous", "_update_only")

    def __init__(
        self,
        agg: Aggregate,
        *,
        update_only: bool = False,
        run_batch: Callable[[pa.RecordBatch], list[pa.RecordBatch]] | None = None,
    ) -> None:
        from batcher.core.streaming import streaming_state_budget

        self._map = run_batch
        # With a map in front, the fold's input schema is whatever the UDF returns, which
        # is knowable only once it has returned something. See `_folded`.
        self._fold = None if run_batch is not None else _AggFold(agg)
        self._agg = agg
        self._update_only = update_only
        #: The last result emitted, for the `update` diff. None until the first one.
        self._previous: pa.Table | None = None
        # Whether any micro-batch produced a result. See `finalize`.
        self._emitted = False
        # A *grouped* aggregate over a stream with no watermark holds one entry per group
        # for the life of the query — correct, and a memory leak measured in days. Kyber can
        # already name that shape (`kyber.streaming.retains_unbounded_state`) and the
        # windowed fold has been capped since it was written; this one had no cap at all, so
        # the query that leaks was the one nothing was watching. A keyless aggregate holds
        # exactly one row and needs no guard.
        self._keyed = bool(agg.group_keys)
        self._cap = streaming_state_budget()

    def process(self, batch: pa.RecordBatch) -> list[pa.RecordBatch]:
        for mapped in self._map(batch) if self._map is not None else [batch]:
            if mapped.num_rows or self._fold is not None:
                self._folded(mapped).push(mapped)
        if self._fold is None:  # the map produced nothing at all yet
            return []
        if self._keyed:
            from batcher.core.streaming import check_agg_state_bounded

            check_agg_state_bounded(
                self._fold,
                self._cap,
                "this aggregate has no watermark, so no group is ever closed and evicted. "
                "Add .with_watermark(...) with a windowed group_by so closed windows evict, "
                "narrow the group keys, or raise memory.streaming_state_max_bytes",
                label="streaming aggregate",
                extra_bytes=self._previous_bytes(),
            )
        result = self._fold.finalize()
        if result is None:
            return []
        self._emitted = True
        if not self._update_only:
            return [result]
        return self._changed_rows(result)

    def _folded(self, mapped: pa.RecordBatch) -> _AggFold:
        """The fold, built against the mapped schema the first time one is available.

        The map is already applied by then, so the fold's input is a plain scan of that
        schema — it partial-aggregates and combines, and never re-runs the UDF. This is the
        same substitution `api.terminal.map_stream` makes for `iter_batches`, which is what
        keeps the two terminals computing one aggregate rather than two.
        """
        import dataclasses

        from batcher.plan.logical import Scan
        from batcher.plan.schema import SchemaRef

        if self._fold is None:
            identity = dataclasses.replace(
                self._agg, input=Scan(0, SchemaRef.from_arrow(mapped.schema))
            )
            self._fold = _AggFold(identity)
        return self._fold

    def _changed_rows(self, result: pa.RecordBatch) -> list[pa.RecordBatch]:
        """The rows of `result` that differ from the last emitted result.

        Falls back to the whole result when a column's type cannot be a join key (a list,
        struct or map aggregate). Emitting more than changed is `complete`'s behavior,
        which is correct if wasteful; emitting less would lose an update.
        """
        current = pa.Table.from_batches([result])
        previous, self._previous = self._previous, current
        if previous is None:
            return [result]
        if not _joinable(current.schema):
            return [result]
        changed = current.join(previous, keys=current.schema.names, join_type="left anti")
        if changed.num_rows == 0:
            return []
        return changed.combine_chunks().to_batches()

    def finalize(self) -> list[pa.RecordBatch]:
        """The one row a *keyless* aggregate owes a stream that carried none.

        A global `count`/`sum` over zero rows yields exactly one row — `0`, `NULL` — which
        is what SQL, DuckDB, and `collect()` all produce. The incremental fold cannot: it
        skips empty batches, so a stream that never carried a row finalized nothing and the
        sink ended up **empty**, disagreeing with `collect()` on the same pipeline. An empty
        window is an ordinary streaming outcome — a quiet topic, a filter that matched
        nothing — so this was a silent wrong answer on a routine path. The `iter_batches`
        driver has had this fallback since S29; the micro-batch engine did not.

        A *grouped* aggregate over an empty input correctly yields no rows, and a stream
        that emitted anything has already answered, so both return nothing here.

        Returns:
            The identity row, or nothing when one is not owed.
        """
        if self._emitted or self._keyed:
            return []
        from batcher.core.streaming import empty_global_aggregate

        schema = self._agg.input.available_schema()
        if schema is None:  # an opaque input (a UDF) — nothing to type the empty batch from
            return []
        result = empty_global_aggregate(self._agg, schema.arrow)
        return [result] if result is not None else []

    def _previous_bytes(self) -> int:
        """Bytes held by the `update`-mode copy of the last emitted result.

        Retained state that the fold does not know about: `update` diffs against a full copy
        of what it last emitted, so a query in that mode holds the aggregate twice. Counting
        it is what makes `memory.streaming_state_max_bytes` mean what it says.
        """
        if self._previous is None:
            return 0
        from batcher.plan.types import retained_bytes

        return sum(retained_bytes(batch) for batch in self._previous.to_batches())

    def state_metrics(self) -> tuple[StateOperatorProgress, ...]:
        """This operator's retained state after the last micro-batch.

        `memory_used_bytes` counts the `update`-mode previous-result copy as well as the
        fold, because both are retained and an operator reading this number to decide
        whether the key space is too wide needs the total rather than half of it.
        """
        if self._fold is None:
            return ()
        import dataclasses

        metrics = self._fold.metrics()
        extra = self._previous_bytes()
        if extra:
            metrics = dataclasses.replace(
                metrics, memory_used_bytes=metrics.memory_used_bytes + extra
            )
        return (metrics,)

    def snapshot_state(self) -> pa.RecordBatch | None:
        """The running partial state for a checkpoint snapshot."""
        return None if self._fold is None else self._fold.state()

    def snapshot_delta(self) -> pa.RecordBatch | None:
        """Take this micro-batch's changelog entry, or None when there is nothing to record.

        Offering this is what lets the engine write an incremental checkpoint instead of
        rewriting the whole running state every epoch. A processor that evicts state must
        **not** offer one — see `_AggFold.take_delta` for why a delta chain cannot express a
        removal — which is why this is an opt-in method rather than a field on the protocol.

        Reading consumes the entry — see `_AggFold.take_delta` for the duplicate it
        prevents.

        Returns:
            The partial the last `process` folded in, or None.
        """
        return None if self._fold is None else self._fold.take_delta()

    def restore_state_chain(self, states: list[pa.RecordBatch]) -> None:
        """Rebuild the running state from a base snapshot plus the deltas after it.

        Args:
            states: The base snapshot followed by each recorded delta, oldest first.
        """
        self._fold.restore_chain(states)

    def restore_state(self, state: pa.RecordBatch) -> None:
        """Resume from a checkpointed running partial state.

        Never reached for a mapped aggregate: the fold is built against the schema the UDF
        returns, which is not knowable before the UDF has run, so `checkpoint=` is refused
        for that shape at `start()` rather than silently resuming from an empty aggregate.
        """
        self._fold.restore(state)


class WindowedAggregateProcessor:
    """Append-mode windowed aggregation: emit each window as the watermark closes it.

    Backed by the same `_WindowedAggFold` as the `iter_batches` windowed driver, so
    bounded streaming state and append output share one implementation — including the
    per-partition watermark, so the two terminals cannot disagree about which rows are
    late. `finalize` flushes any windows still open when the query stops.
    """

    __slots__ = ("_fold",)

    def __init__(self, agg: Aggregate, key, source=None) -> None:
        """Fold `agg` into the windows `key` describes over `source`'s partitions.

        Args:
            agg: The watermarked windowed aggregate.
            key: Which group key is the window, and its width and hop.
            source: The stream, asked which columns attribute a row to a partition and
                which partitions it expects. None (a plan with no bound source yet) leaves
                the watermark unpartitioned.
        """
        from batcher.core.streaming import _WindowedAggFold
        from batcher.io.source import watermark_partition_columns, watermark_partitions

        # The runner reads a windowed aggregate's source with no projection (Kyber's
        # pushdown is built only for the stateless path), so the partition columns are
        # already in the batch and none has to be stripped again.
        self._fold = _WindowedAggFold(
            agg,
            key,
            partition_cols=() if source is None else watermark_partition_columns(source),
            expected_partitions=() if source is None else watermark_partitions(source),
        )

    def process(self, batch: pa.RecordBatch) -> list[pa.RecordBatch]:
        return self._fold.push(batch)

    def finalize(self) -> list[pa.RecordBatch]:
        result = self._fold.flush()
        return [result] if result is not None else []

    def state_metrics(self) -> tuple[StateOperatorProgress, ...]:
        """This operator's open windows, watermark, and anything dropped as late."""
        return (self._fold.metrics(),)

    def snapshot_state(self) -> pa.RecordBatch | None:
        """The open windows and the watermark, for a checkpoint snapshot.

        This processor previously defined neither `snapshot_state` nor `restore_state`, and
        `StreamingRunner.has_state` duck-types on exactly this method — so it reported
        *stateless* and its state was never written. Offsets were committed regardless. A crash
        therefore resumed **past** consumed data with every open window and the watermark gone:
        those windows were silently never emitted. That is data loss in the one query shape the
        whole watermark machinery exists to serve, so these two methods are load-bearing, not a
        nicety.
        """
        return self._fold.state()

    def snapshot_state_parts(self) -> Iterator[pa.RecordBatch]:
        """The whole state as a stream of parts, so a spilled fold checkpoints in full.

        Offered only by a processor whose state can live partly on disk. The engine prefers
        it over `snapshot_state` when present, because that one reports the resident half —
        correct until the fold spills, and a silent loss of the rest afterwards.

        Yields:
            The resident state, then one batch per spilled run.
        """
        return self._fold.state_parts()

    def restore_state_parts(self, parts: list[pa.RecordBatch]) -> None:
        """Resume from a multi-part snapshot, combining the parts back into one state."""
        self._fold.restore_parts(parts)

    def snapshot_delta(self) -> pa.RecordBatch | None:
        """Take this micro-batch's changelog entry.

        A windowed aggregate *removes* state, which is normally what disqualifies an
        operator from a changelog — a chain cannot say what was taken out. It qualifies
        because its removal is not arbitrary: eviction drops a **prefix** of the window
        axis, so the whole tombstone is one integer that rides in the entry's metadata.
        See `_EVICTED_META`.

        Returns:
            The epoch's combined partial, or None when it folded nothing in.
        """
        return self._fold.take_delta()

    def restore_state_chain(self, states: list[pa.RecordBatch]) -> None:
        """Rebuild from a base snapshot plus the changelog after it.

        The same call as `restore_state_parts`: both are "combine these and re-apply the
        eviction bound", and keeping them one implementation is what stops a chain and a
        multi-part snapshot from drifting into two restore semantics.
        """
        self._fold.restore_parts(states)

    def close(self) -> None:
        """Release the fold's spill scratch. Idempotent; a no-op if it never spilled."""
        self._fold.close()

    def restore_state(self, state: pa.RecordBatch) -> None:
        """Resume the open windows and the watermark from a checkpointed snapshot."""
        self._fold.restore(state)


class KeyedStateProcessor:
    """`transform_with_state` as a micro-batch processor: the user function owns the state.

    A thin wrapper over `KeyedStateFold` — the same fold `iter_batches()` drives — so a
    query written against the sink and one consumed batch by batch cannot disagree about
    what the operator means. `snapshot_state`/`restore_state` are the fold's, so the key
    space rides through a restart in the checkpoint like any other streaming state.
    """

    __slots__ = ("_fold",)

    def __init__(self, node) -> None:
        from batcher.core.streaming import KeyedStateFold

        self._fold = KeyedStateFold(node)

    def process(self, batch: pa.RecordBatch) -> list[pa.RecordBatch]:
        return [b for b in self._fold.push(batch) if b.num_rows]

    def state_metrics(self) -> tuple[StateOperatorProgress, ...]:
        """The key count, what the TTL expired, and the footprint."""
        return (self._fold.metrics(),)

    def snapshot_state(self) -> pa.RecordBatch | None:
        """The whole key space, for a checkpoint snapshot."""
        return self._fold.state()

    def restore_state(self, state: pa.RecordBatch) -> None:
        """Rebuild the key space from a checkpoint snapshot."""
        self._fold.restore(state)


def make_processor(
    plan,
    output_mode: str,
    run_batch: Callable[[pa.RecordBatch], list[pa.RecordBatch]] | None,
    source=None,
) -> MicroBatchProcessor:
    """Pick the processor for `plan` under `output_mode` (built by the conductor).

    Stateless (breaker-free) plans require `append`; aggregates require
    `complete`/`update`. The mismatch cases raise `PlanError` with the Spark-parity
    rule, so an impossible query fails at `start()`, not mid-stream.

    `source` is passed through to a windowed aggregation, which needs to know how the
    stream is partitioned before it can take a watermark as the minimum over those
    partitions rather than the maximum over rows.

    The aggregate shape is recognized by `plan.logical.streaming_fold_target`, which is the
    same predicate the `iter_batches` router and the distributed epoch gate ask. The two used
    to disagree: this function accepted any top-level `Aggregate` **without checking what was
    beneath it**, while the router required a breaker-free input. So
    ``ds.distinct().agg(n=count())`` written to a sink built an `AggregateProcessor` whose
    fold re-ran the `Distinct` inside each micro-batch — deduplicating per batch and counting
    the results — and returned 3 where the same query collected returns 2. No error, no
    warning, and no bounded test can see it because `iter_batches` took the materializing path
    for the same plan and got the right answer.
    """
    from batcher._internal.errors import PlanError
    from batcher.plan.logical import (
        Aggregate,
        Distinct,
        TransformWithState,
        is_streamable,
        streaming_fold_target,
    )

    if isinstance(plan, TransformWithState):
        if output_mode != OutputMode.APPEND:
            raise PlanError(
                f"output_mode={output_mode!r} needs an aggregation; transform_with_state "
                "emits whatever its function returns, once per call, which is 'append'"
            )
        if not is_streamable(plan.input):
            raise PlanError(
                "transform_with_state needs a breaker-free input (filter / select / "
                "map_batches over one source); this plan has a pipeline breaker beneath it"
            )
        return KeyedStateProcessor(plan)
    if isinstance(plan, Distinct) and plan.keys:
        raise PlanError(
            "distinct(subset=...) cannot run as a streaming query: keeping one row per key "
            "needs the key set held for the life of the query, and which row wins is decided "
            "by an ordering over rows that have not arrived. Use "
            "drop_duplicates_within_watermark(subset, event_time=..., lateness=...), whose "
            "state the watermark bounds, or distinct() with no subset when the whole row is "
            "the key."
        )
    if isinstance(plan, (Aggregate, Distinct)):
        fold = streaming_fold_target(plan)
        if fold is None:
            raise PlanError(
                f"a streaming {type(plan).__name__.lower()} needs a breaker-free input "
                "(filter / select / map_batches over one source); this plan has another "
                f"pipeline breaker beneath it ({type(plan.input).__name__.lower()}), which "
                "the running fold would re-run inside every micro-batch — deduplicating, "
                "sorting or joining each batch on its own rather than the whole stream, and "
                "returning a different answer than the same query collected. Materialize the "
                "inner result to a bounded source first, or aggregate directly over the scan."
            )
        if output_mode == OutputMode.APPEND:
            from batcher.core.streaming import _window_key

            key = _window_key(plan) if isinstance(plan, Aggregate) else None
            if isinstance(plan, Aggregate) and plan.watermark is not None and key is not None:
                return WindowedAggregateProcessor(plan, key, source)
            raise PlanError(
                "output_mode='append' on a streaming aggregation needs a watermark and a "
                "windowed group key — an event-time window is the only thing a watermark "
                "can close, so it is the only thing that makes a row final. Use "
                ".with_watermark(...) with group_by(w=bt.window(col('ts'), '1h')) (or an "
                "exploded bt.window(col('ts'), '1h', '30m') for overlapping windows), or "
                "output_mode 'complete'/'update'"
            )
        agg = fold
        # `run_batch` is present here only for a `map → agg`: the conductor builds it from
        # the aggregate's *input* precisely so the UDF runs in Python and the fold sees
        # mapped batches. Without it the fold would try to lower a `MapBatches` node.
        return AggregateProcessor(
            agg, update_only=output_mode == OutputMode.UPDATE, run_batch=run_batch
        )
    if is_streamable(plan):
        if output_mode != OutputMode.APPEND:
            raise PlanError(
                f"output_mode={output_mode!r} requires an aggregation; a stateless "
                "streaming pipeline only supports 'append'"
            )
        if run_batch is None:  # pragma: no cover — conductor always supplies it
            raise PlanError("internal: stateless streaming processor needs a run_batch")
        return StatelessProcessor(run_batch)
    raise PlanError(
        "this plan cannot be streamed to a sink (it has a pipeline breaker other than "
        "a top-level aggregation); restructure to a streamable shape"
    )


def _joinable(schema: pa.Schema) -> bool:
    """Whether every column can be an Arrow join key (nested types cannot)."""
    return not any(
        pa.types.is_nested(field.type) or pa.types.is_dictionary(field.type) for field in schema
    )
