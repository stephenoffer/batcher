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

from collections.abc import Callable
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

    `complete` and `update` both emit the full running result here (a keyed sink
    upserts it idempotently); a later step narrows `update` to only the changed
    groups via a Rust delta path. `append` on an unwindowed aggregate is rejected
    upstream — it needs a watermark to know a group is final.
    """

    __slots__ = ("_agg", "_cap", "_emitted", "_fold", "_keyed")

    def __init__(self, agg: Aggregate) -> None:
        from batcher.core.streaming import streaming_state_budget

        self._fold = _AggFold(agg)
        self._agg = agg
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
        self._fold.push(batch)
        if self._keyed:
            from batcher.core.streaming import check_agg_state_bounded

            check_agg_state_bounded(
                self._fold,
                self._cap,
                "this aggregate has no watermark, so no group is ever closed and evicted. "
                "Add .with_watermark(...) with a windowed group_by so closed windows evict, "
                "narrow the group keys, or raise memory.streaming_state_max_bytes",
                label="streaming aggregate",
            )
        result = self._fold.finalize()
        if result is None:
            return []
        self._emitted = True
        return [result]

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

    def state_metrics(self) -> tuple[StateOperatorProgress, ...]:
        """This operator's retained state after the last micro-batch."""
        return (self._fold.metrics(),)

    def snapshot_state(self) -> pa.RecordBatch | None:
        """The running partial state for a checkpoint snapshot."""
        return self._fold.state()

    def restore_state(self, state: pa.RecordBatch) -> None:
        """Resume from a checkpointed running partial state."""
        self._fold.restore(state)


class WindowedAggregateProcessor:
    """Append-mode windowed aggregation: emit each window as the watermark closes it.

    Backed by the same `_WindowedAggFold` as the `iter_batches` windowed driver, so
    bounded streaming state and append output share one implementation. `finalize`
    flushes any windows still open when the query stops.
    """

    __slots__ = ("_fold",)

    def __init__(self, agg: Aggregate, w_alias: str, width: int) -> None:
        from batcher.core.streaming import _WindowedAggFold

        self._fold = _WindowedAggFold(agg, w_alias, width)

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
) -> MicroBatchProcessor:
    """Pick the processor for `plan` under `output_mode` (built by the conductor).

    Stateless (breaker-free) plans require `append`; aggregates require
    `complete`/`update`. The mismatch cases raise `PlanError` with the Spark-parity
    rule, so an impossible query fails at `start()`, not mid-stream.
    """
    from batcher._internal.errors import PlanError
    from batcher.plan.logical import Aggregate, Distinct, TransformWithState, is_streamable

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
    if isinstance(plan, (Aggregate, Distinct)):
        if output_mode == OutputMode.APPEND:
            from batcher.core.streaming import _window_key

            key = _window_key(plan) if isinstance(plan, Aggregate) else None
            if isinstance(plan, Aggregate) and plan.watermark is not None and key is not None:
                return WindowedAggregateProcessor(plan, key[0], key[1])
            raise PlanError(
                "output_mode='append' on a streaming aggregation needs a watermark "
                "(use .with_watermark(...) with a windowed group_by, or output_mode "
                "'complete'/'update')"
            )
        agg = plan if isinstance(plan, Aggregate) else _distinct_as_aggregate(plan)
        return AggregateProcessor(agg)
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


def _distinct_as_aggregate(distinct) -> Aggregate:
    """A `Distinct` is a group-by over all columns — reuse the aggregate fold."""
    return distinct.as_aggregate()
