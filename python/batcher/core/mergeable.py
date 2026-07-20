"""The one running fold over the mergeable aggregate algebra.

An aggregation is `partial → combine → finalize` with an associative+commutative
`combine` (invariant #7). That single property is what lets *one* implementation serve
every schedule Batcher runs:

- **batch, single node** — one partial per morsel, combined, finalized once;
- **batch, distributed** — one partial per worker, combined at the reducer;
- **streaming** — one partial per micro-batch, combined into a running state that is
  bounded by the group count rather than the input size, finalized on demand.

Those are three *schedules* of the same algebra, so they must not be three
implementations of it. They were: the fold was re-derived in `core.streaming._AggFold`,
again in `core.streaming._WindowedAggFold`, again in `dist.streaming.microbatch`, and
again in the batch reducer. Four copies of an invariant means the invariant is only
tested where someone remembered to test it — and a divergence shows up as *wrong
results at scale*, never as a failing local test (`CLAUDE.md`, "a stateful operator
without a mergeable form").

Core's lane: this drives the engine over the spec it is handed. It makes no
optimization decisions and collects no plan metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from batcher._internal.native import engine
from batcher.plan.ir_specs import agg_spec_json

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.plan.logical import Aggregate

__all__ = ["RunningAggregate"]


class RunningAggregate:
    """A single running partial-aggregate state, folded across batches.

    The state is one Arrow `RecordBatch` of partials — bounded by the number of groups,
    not by how much input has been consumed — which is also exactly what a streaming
    checkpoint snapshots and restores.

    Args:
        agg: The aggregate whose group keys and aggregate functions define the fold.
    """

    __slots__ = ("_aggs", "_keys", "_nat", "_running")

    def __init__(self, agg: Aggregate) -> None:
        self._nat = engine()
        self._keys, self._aggs = agg_spec_json(agg)
        self._running: pa.RecordBatch | None = None

    @property
    def spec(self) -> tuple[str, str]:
        """The ``(group_keys_json, aggregates_json)`` pair this fold was built from."""
        return self._keys, self._aggs

    def is_empty(self) -> bool:
        """Whether no rows have been folded in yet."""
        return self._running is None

    def nbytes(self) -> int:
        """The in-memory size of the running state (0 when empty)."""
        return 0 if self._running is None else self._running.nbytes

    def partial(self, rows: Sequence[pa.RecordBatch]) -> pa.RecordBatch:
        """Partial-aggregate `rows` without folding them in.

        Used by the distributed mapper, which ships its partial across the shuffle
        instead of combining it locally.

        Args:
            rows: Input batches to aggregate.

        Returns:
            The partial-aggregate state for `rows`.
        """
        return self._nat.partial_aggregate(self._keys, self._aggs, list(rows))

    def absorb(self, partial: pa.RecordBatch) -> None:
        """Fold an already-computed partial into the running state.

        Args:
            partial: A partial-aggregate state over the same spec.
        """
        self._running = (
            partial
            if self._running is None
            else self._nat.combine(self._keys, self._aggs, [self._running, partial])
        )

    def push(self, rows: Sequence[pa.RecordBatch]) -> None:
        """Partial-aggregate `rows` and fold them into the running state.

        Empty input is a no-op, so a batch that filters away leaves the state — and
        therefore the finalized result — untouched.

        Args:
            rows: Input batches to fold in.
        """
        if not rows or not sum(b.num_rows for b in rows):
            return
        self.absorb(self.partial(rows))

    def combine_all(self, partials: Sequence[pa.RecordBatch]) -> None:
        """Replace the running state with the combination of `partials`.

        Args:
            partials: Partial states to merge; an empty sequence clears the state.
        """
        self._running = (
            self._nat.combine(self._keys, self._aggs, list(partials)) if partials else None
        )

    def finalize(self) -> pa.RecordBatch | None:
        """Materialize the current aggregate result.

        Returns:
            The finalized rows, or None when no groups have been accumulated.
        """
        if self._running is None:
            return None
        result = self._nat.combine_finalize(self._keys, self._aggs, [self._running])
        return result if result.num_rows else None

    def finalize_partials(self, partials: Sequence[pa.RecordBatch]) -> pa.RecordBatch | None:
        """Finalize `partials` directly, leaving the running state alone.

        The eviction path needs this: closed windows are finalized and emitted while the
        open windows stay in the running state.

        Args:
            partials: Partial states to combine and finalize.

        Returns:
            The finalized rows, or None when `partials` is empty or produces no rows.
        """
        if not partials:
            return None
        result = self._nat.combine_finalize(self._keys, self._aggs, list(partials))
        return result if result.num_rows else None

    def take(self) -> pa.RecordBatch | None:
        """Finalize and clear the state — the end-of-stream flush.

        Returns:
            The finalized rows, or None when no groups have been accumulated.
        """
        result = self.finalize()
        self._running = None
        return result

    def state(self) -> pa.RecordBatch | None:
        """The running partial state, for a checkpoint snapshot (None if empty)."""
        return self._running

    def restore(self, state: pa.RecordBatch | None) -> None:
        """Seed the running partial state from a checkpoint snapshot.

        Args:
            state: A previously snapshotted partial state, or None to start empty.
        """
        self._running = state
