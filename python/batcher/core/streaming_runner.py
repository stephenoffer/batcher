"""How one micro-batch gets run — the seam between the loop and where the work happens.

`StreamingQueryEngine` owns the *cadence* of a streaming query: the trigger, the batch
counter, the offset log, recovery, progress. None of that depends on whether the rows are
processed on this thread or on fifty machines. What does depend on it is the middle: read
the epoch's data, turn it into output, make it visible. That is this module's Protocol, and
it has exactly two implementations — `LocalRunner` here, and the Ray fan-out in
`dist.streaming.microbatch`. One loop, two schedulings; the streaming *semantics* are not
forked (`.claude/rules/rust-engine.md`: single-node == distributed).

The Protocol is deliberately **two-phase**, because that is what an exactly-once
micro-batch needs:

* `stage()` reads the epoch and prepares its output without publishing anything;
* `positions()` then reports where the source now stands — the engine write-aheads this;
* `publish()` makes the epoch visible atomically.

The offset write-ahead has to land *between* those two, so a crash can only ever leave an
epoch that was staged but not published — one the next run replays into an idempotent sink.
Collapsing the phases (publishing as you read) is what makes a stream at-least-once, and
what makes a distributed one commit once per *worker* instead of once per *micro-batch*.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import pyarrow as pa

if TYPE_CHECKING:
    from batcher.core.streaming_query import MicroBatchProcessor
    from batcher.io.source import Source

__all__ = ["LocalRunner", "MicroBatchRunner"]


@runtime_checkable
class MicroBatchRunner(Protocol):
    """Runs one micro-batch: read it, then make it visible — as two separate steps."""

    def stage(self, batch_id: int) -> Any | None:
        """Read the epoch's data and prepare its output; None when the source is spent.

        Publishes nothing: whatever this returns is handed back to `publish` unchanged.
        """
        ...

    def positions(self) -> dict[int, dict]:
        """Per-source read positions after `stage` — what the engine write-aheads."""
        ...

    def publish(self, batch_id: int, staged: Any) -> tuple[int, int]:
        """Make the staged epoch visible; return ``(input_rows, output_rows)``."""
        ...

    def seek(self, position: dict) -> None:
        """Restore the source to a checkpointed position (recovery)."""
        ...


class LocalRunner:
    """Run the micro-batch on this thread — the single-node path, unchanged.

    `stage` pulls one batch from the source iterator, `publish` runs it through the
    processor and writes the result to the sink. The processor and sink are exactly the
    ones the conductor built, so this is the same code the engine ran before the seam
    existed; it is the reference the distributed runner has to agree with.

    **An idle unbounded source is not a finished one.** `iter_batches` on a source that
    discovers its work per pass — the incremental file source, whose pass ends when the
    directory holds nothing new — returns a generator that *ends*, and this runner read
    that as end-of-stream and stopped the query. An Auto Loader stream therefore ran its
    first discovery pass, silently terminated, and never ingested another file: no error,
    no log line, a `StreamingQuery` that simply reported `is_active == False` seconds
    after it started. The distributed runner never had the bug, because it asks the
    source for work each epoch instead of holding one iterator — so single-node and
    distributed streaming disagreed about when a stream is over, which is precisely the
    divergence the one-runner-protocol exists to prevent.

    An exhausted pass now re-opens the source and waits `streaming.idle_poll_seconds`
    before looking again, exactly as `dist.streaming.microbatch` does. Only a *drain*
    trigger (`once` / `available_now`) reads an empty pass as the end — which is what
    makes it a drain — along with a stop signal or a genuinely bounded source.
    """

    __slots__ = (
        "_drain",
        "_iterator",
        "_last_token",
        "_pass_yielded",
        "_predicate",
        "_processor",
        "_projection",
        "_should_stop",
        "_sink",
        "_source",
    )

    def __init__(
        self,
        source: Source,
        processor: MicroBatchProcessor,
        sink: Any,
        *,
        projection: list[str] | None = None,
        predicate: dict | None = None,
        should_stop: Callable[[], bool] | None = None,
        drain: bool = False,
    ) -> None:
        self._source = source
        # True under `Trigger.once()` / `Trigger.available_now()`: process what is
        # available and stop, rather than waiting for more. See the class docstring.
        self._drain = drain
        self._processor = processor
        self._sink = sink
        # Handed to a source that knows how to end its own poll loop. Without it, stopping a
        # query parked on an idle unbounded source did not merely take a while — `stage()`
        # sat inside `next()` waiting for data that might never come, and `stop()` blocked
        # forever joining the thread. A source that does not accept the signal is unaffected;
        # it simply keeps the old behavior.
        self._should_stop = should_stop
        attach = getattr(source, "set_stop_signal", None)
        if attach is not None and should_stop is not None:
            attach(should_stop)
        # Kyber's source pushdown for this plan, or None when there is none to push (a
        # `map_batches` pipeline, whose UDF is opaque to the optimizer).
        self._projection = projection
        self._predicate = predicate
        self._iterator: Iterator[pa.RecordBatch] | None = None
        # Whether the *current* pass has produced a batch. An exhausted pass that produced
        # rows may have more waiting behind it; one that produced none is the idle (or,
        # under a drain, the terminal) condition. See `stage`.
        self._pass_yielded = False
        # What the sink said it wrote for the last published epoch. See `last_sink_token`.
        self._last_token: str | None = None

    def stage(self, batch_id: int) -> pa.RecordBatch | None:  # noqa: ARG002
        """Pull this epoch's batch; None only when the stream is genuinely over.

        An exhausted iterator is the end of a *pass*. Whether it is also the end of the
        stream depends on the source:

        * a bounded source has one pass, so its first exhaustion is the end;
        * so does an unbounded source that does **not** declare
          `continues_across_passes` — re-opening one of those replays its rows from the
          beginning, which would write the whole stream to the sink again, forever;
        * a source that continues (an incremental directory listing, a broker) is
          re-opened. A drain trigger (`once` / `available_now`) then ends on the first
          *empty* pass, which is why `available_now` still drains a whole backlog the
          source hands over a few files at a time (`max_files_per_trigger`) instead of
          stopping after the first. Anything else waits `streaming.idle_poll_seconds`
          and looks again, so a directory-watching stream keeps ingesting for the life
          of the query.

        A pass that *did* produce rows is re-opened immediately with no wait, so a
        backlog drains at full speed and only genuine idleness costs latency.
        """
        import time

        from batcher.config import active_config
        from batcher.io.source import continues_across_passes, is_bounded

        nap = active_config().streaming.idle_poll_seconds
        resumable = not is_bounded(self._source) and continues_across_passes(self._source)
        while True:
            batch = self._next_batch()
            if batch is not None:
                self._pass_yielded = True
                return batch
            # Exhausted: every batch the source yielded has already been published (each
            # publish precedes the next stage), so the last file's rows are durable and
            # its `confirm` — which would otherwise never come — is safe here.
            _confirm(self._source)
            if not resumable:
                return None
            # Drop the spent iterator so the next one re-runs the source's discovery.
            productive, self._iterator, self._pass_yielded = self._pass_yielded, None, False
            if productive:
                continue  # more may already be waiting — look again without a wait
            if self._drain:
                return None
            if self._should_stop is not None and self._should_stop():
                return None
            time.sleep(nap)

    def _next_batch(self) -> pa.RecordBatch | None:
        """One batch from the current pass, opening the source's iterator if needed."""
        if self._iterator is None:
            # Read *through* the pushdown, exactly as `_iter_streaming` does. This was
            # `iter_batches(None)`, which decoded every column regardless of the plan's
            # projection — the batch and streaming paths disagreeing on pushdown for the
            # same pipeline. `iter_source` degrades safely: a source whose `iter_batches`
            # takes no predicate is called with the projection only, and the plan's own
            # `Filter` re-checks every batch, so correctness never depends on the source
            # honoring either.
            from batcher.io.source import iter_source

            self._iterator = iter_source(self._source, self._projection, self._predicate)
        return next(self._iterator, None)

    def positions(self) -> dict[int, dict]:
        from batcher.io.source import is_checkpointable

        if not is_checkpointable(self._source):
            return {}
        return {0: self._source.snapshot_position()}

    def publish(self, batch_id: int, staged: pa.RecordBatch) -> tuple[int, int]:
        """Write the micro-batch's whole output as **one** sink write.

        This used to call `write_batch` once per record batch the processor returned, all
        under the same `batch_id` — and a sink's exactly-once machinery is keyed on exactly
        that id. The second write of an epoch therefore found the epoch already written and
        *dropped its rows*: a `FileStreamSink` skipped the `part-batch<id>` file that already
        existed, and a `DeltaStreamSink` found its own `(app_id, batch_id)` transaction in the
        log and committed nothing. It was invisible because a processor usually returns one
        batch — a plan whose output exceeds a morsel, or a union, returns several, and only
        then did the tail of the epoch silently vanish.

        Concatenating first is also the cheaper spelling: one table, one write, one commit.
        """
        rows = [b for b in self._processor.process(staged) if b.num_rows]
        emitted = sum(b.num_rows for b in rows)
        self._last_token = None
        if rows:
            self._last_token = self._sink.write_batch(batch_id, pa.Table.from_batches(rows))
        _confirm(self._source)
        return staged.num_rows, emitted

    def last_sink_token(self) -> str | None:
        """What the sink reported writing for the epoch just published, if anything.

        Every `StreamSink` returns one — a `part-batch` path, a Delta
        ``(app_id, batch_id)`` marker, a `foreach_batch:{id}` receipt — and the commit log
        has a `sink_token` column to record it in. Nothing carried the value between them,
        so the column was NULL for every row ever written: the log could say *that* a batch
        committed and never *what* the sink did for it, which is the difference between a
        recoverable audit trail and a row of batch ids.
        """
        return self._last_token

    def seek(self, position: dict) -> None:
        from batcher.io.source import is_checkpointable

        if is_checkpointable(self._source):
            self._source.seek(position)
        # A source recovering into the middle of a run restarts its iterator from the
        # restored position — the old generator still holds the pre-crash cursor.
        self._iterator = None

    # --- delegated to the processor (the engine discovers these by duck-typing) ---
    def finalize(self) -> list[pa.RecordBatch]:
        finalize: Callable[[], list[pa.RecordBatch]] | None = getattr(
            self._processor, "finalize", None
        )
        return finalize() if finalize is not None else []

    def emit_final(self, batch_id: int, rows: pa.RecordBatch) -> None:
        self._sink.write_batch(batch_id, pa.Table.from_batches([rows]))

    def snapshot_state(self) -> pa.RecordBatch | None:
        snap = getattr(self._processor, "snapshot_state", None)
        return snap() if snap is not None else None

    def restore_state(self, state: pa.RecordBatch) -> None:
        restore = getattr(self._processor, "restore_state", None)
        if restore is not None:
            restore(state)

    def has_state(self) -> bool:
        return getattr(self._processor, "snapshot_state", None) is not None


def _confirm(source: Source) -> None:
    """Tell a source its staged position is now durable (see `IncrementalFileSource`).

    A source whose read position lives in its *own* durable store — rather than in the
    offset log — may only advance that store once the epoch it read is published. Sources
    that keep no such state (a broker: the log holds the offsets) do not define `confirm`
    and are unaffected.
    """
    confirm = getattr(source, "confirm", None)
    if confirm is not None:
        confirm()
