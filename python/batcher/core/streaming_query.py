"""The streaming-query engine — the micro-batch loop behind a unified `ds.write`.

`core`'s lane is *making it happen and measuring*: this drives a long-running query
by pulling micro-batches from an unbounded `Source`, processing each through a
`MicroBatchProcessor` (built by the conductor in `api`, already Kyber-optimized for
the stateless case), emitting the result to a `StreamSink` per the `OutputMode`, and
firing on the `Trigger`'s cadence. It never optimizes (Kyber ran once at start) and
never owns resources (Carbonite) — it only runs and records `StreamingQueryProgress`.

The loop runs on a background thread so `start()` returns a handle the caller can
`stop()` / `await`; per-micro-batch metrics already flow to the `MetadataHub`
through `core.execute_local`, so a streaming query improves future plans the same
way a batch query does.
"""

from __future__ import annotations

import contextlib
import threading
from collections import deque
from collections.abc import Callable
from time import perf_counter, time
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pyarrow as pa

from batcher.core.streaming import _AggFold
from batcher.plan.streaming import (
    OutputMode,
    StreamingQueryProgress,
    StreamingQueryStatus,
    Trigger,
)

if TYPE_CHECKING:
    from batcher.io.source import Source
    from batcher.plan.logical import Aggregate

__all__ = [
    "AggregateProcessor",
    "MicroBatchProcessor",
    "StatelessProcessor",
    "StreamingQueryEngine",
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

    __slots__ = ("_fold",)

    def __init__(self, agg: Aggregate) -> None:
        self._fold = _AggFold(agg)

    def process(self, batch: pa.RecordBatch) -> list[pa.RecordBatch]:
        self._fold.push(batch)
        result = self._fold.finalize()
        return [result] if result is not None else []

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


class StreamingQueryEngine:
    """Drives a streaming query on a background thread; the `api` handle wraps it."""

    def __init__(
        self,
        *,
        name: str,
        source: Source,
        sink,
        processor: MicroBatchProcessor,
        trigger: Trigger,
        output_mode: str,
        checkpoint=None,
    ) -> None:
        self._name = name
        self._source = source
        self._sink = sink
        self._processor = processor
        self._trigger = trigger
        self._output_mode = output_mode
        self._checkpoint = checkpoint
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._progress: deque[StreamingQueryProgress] = deque(maxlen=100)
        self._batches = 0
        self._error: BaseException | None = None
        self._active = False

    # --- lifecycle --------------------------------------------------------
    def start(self) -> None:
        """Recover from the checkpoint (if any), open the sink, launch the loop."""
        self._active = True
        self._recover()
        self._sink.open()
        self._thread = threading.Thread(
            target=self._run, name=f"batcher-stream-{self._name}", daemon=True
        )
        self._thread.start()

    def _recover(self) -> None:
        """Restore source position, batch counter, and running state from a checkpoint."""
        if self._checkpoint is None:
            return
        from batcher.io.formats.streaming.checkpoint import recover
        from batcher.io.source import is_checkpointable

        plan = recover(self._checkpoint)
        self._batches = plan.start_batch
        if plan.seek and is_checkpointable(self._source) and 0 in plan.seek:
            self._source.seek(plan.seek[0])
        restore = getattr(self._processor, "restore_state", None)
        if plan.state is not None and restore is not None:
            restore(plan.state)

    def stop(self) -> None:
        """Signal the loop to halt at the next micro-batch boundary and join."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def await_termination(self, timeout: float | None = None) -> bool:
        """Block until the query stops (or `timeout` seconds); return whether it stopped."""
        if self._thread is None:
            return True
        self._thread.join(timeout)
        if self._thread.is_alive():
            return False
        if self._error is not None:
            raise self._error
        return True

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def exception(self) -> BaseException | None:
        return self._error

    def recent_progress(self) -> list[StreamingQueryProgress]:
        return list(self._progress)

    def status(self) -> StreamingQueryStatus:
        return StreamingQueryStatus(
            is_active=self._active,
            is_data_available=bool(self._progress) and self._progress[-1].num_input_rows > 0,
            is_trigger_active=self._active and not self._stop.is_set(),
            message="Waiting for data" if self._active else "Stopped",
            batches_processed=self._batches,
        )

    # --- the loop ---------------------------------------------------------
    def _run(self) -> None:
        try:
            self._loop()
            self._emit_finalize()
        except BaseException as exc:
            self._error = exc
        finally:
            self._active = False
            with contextlib.suppress(Exception):
                self._sink.close()

    def _emit_finalize(self) -> None:
        """Flush any windows still open when the loop ends (the final emission)."""
        finalize = getattr(self._processor, "finalize", None)
        if finalize is None:
            return
        for rows in finalize():
            if rows.num_rows:
                self._sink.write_batch(self._batches, pa.Table.from_batches([rows]))

    def _loop(self) -> None:
        kind = self._trigger.kind
        iterator = self._source.iter_batches(None)
        if kind == "once":
            self._process_next(iterator)
            return
        if kind in ("available_now", "continuous"):
            # Continuous: process micro-batches back-to-back with no inter-batch
            # delay (lowest latency), committing a checkpoint epoch per batch, until
            # the query is stopped or the source is exhausted. (`available_now` shares
            # the loop; it is simply expected to drain a finite source and end.)
            while not self._stop.is_set() and self._process_next(iterator):
                pass
            return
        # processing_time: fire a micro-batch, then sleep the remainder of the interval.
        interval = self._trigger.interval_seconds or 0.0
        while not self._stop.is_set():
            t0 = perf_counter()
            if not self._process_next(iterator):
                break  # bounded source exhausted
            remaining = interval - (perf_counter() - t0)
            if remaining > 0:
                self._stop.wait(remaining)

    def _process_next(self, iterator) -> bool:
        """Pull one source micro-batch, process and emit it; False if exhausted.

        With a checkpoint, the per-micro-batch commit ordering is: record the source
        offset (write-ahead), process and emit to the sink, snapshot the running
        state, then commit — so a crash leaves an uncommitted batch the next run
        replays idempotently (exactly-once for a replayable source + idempotent sink).
        """
        try:
            batch = next(iterator)
        except StopIteration:
            return False
        t0 = perf_counter()
        if self._checkpoint is not None:
            self._record_offset()
        out = self._processor.process(batch)
        emitted = 0
        for rows in out:
            if rows.num_rows:
                self._sink.write_batch(self._batches, pa.Table.from_batches([rows]))
                emitted += rows.num_rows
        if self._checkpoint is not None:
            self._commit_microbatch()
        self._progress.append(
            StreamingQueryProgress(
                batch_id=self._batches,
                num_input_rows=batch.num_rows,
                num_output_rows=emitted,
                duration_ms=(perf_counter() - t0) * 1000.0,
                timestamp=time(),
            )
        )
        self._batches += 1
        return True

    def _record_offset(self) -> None:
        """Write-ahead the current source position for this micro-batch."""
        from batcher.io.source import is_checkpointable

        positions = {0: self._source.snapshot_position()} if is_checkpointable(self._source) else {}
        self._checkpoint.record_offsets(self._batches, positions)

    def _commit_microbatch(self) -> None:
        """Snapshot running state (if any) and mark the micro-batch committed."""
        snap = getattr(self._processor, "snapshot_state", None)
        if snap is not None:
            self._checkpoint.snapshot_state(self._batches, snap())
        self._checkpoint.commit(self._batches)


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
    from batcher.plan.logical import Aggregate, Distinct, is_streamable

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
    from batcher.plan.expr_ir import Col
    from batcher.plan.logical import Aggregate, Projection

    cols = distinct.input.available_columns()
    group_keys = tuple(Projection(c, Col(c)) for c in cols)
    return Aggregate(distinct.input, group_keys, ())
