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
    """

    __slots__ = ("_iterator", "_predicate", "_processor", "_projection", "_sink", "_source")

    def __init__(
        self,
        source: Source,
        processor: MicroBatchProcessor,
        sink: Any,
        *,
        projection: list[str] | None = None,
        predicate: dict | None = None,
    ) -> None:
        self._source = source
        self._processor = processor
        self._sink = sink
        # Kyber's source pushdown for this plan, or None when there is none to push (a
        # `map_batches` pipeline, whose UDF is opaque to the optimizer).
        self._projection = projection
        self._predicate = predicate
        self._iterator: Iterator[pa.RecordBatch] | None = None

    def stage(self, batch_id: int) -> pa.RecordBatch | None:  # noqa: ARG002
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
        batch = next(self._iterator, None)
        if batch is None:
            # Exhausted: every batch the source yielded has already been published (each
            # publish precedes the next stage), so the last file's rows are durable and
            # its `confirm` — which would otherwise never come — is safe here.
            _confirm(self._source)
        return batch

    def positions(self) -> dict[int, dict]:
        from batcher.io.source import is_checkpointable

        if not is_checkpointable(self._source):
            return {}
        return {0: self._source.snapshot_position()}

    def publish(self, batch_id: int, staged: pa.RecordBatch) -> tuple[int, int]:
        emitted = 0
        for rows in self._processor.process(staged):
            if rows.num_rows:
                self._sink.write_batch(batch_id, pa.Table.from_batches([rows]))
                emitted += rows.num_rows
        _confirm(self._source)
        return staged.num_rows, emitted

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
