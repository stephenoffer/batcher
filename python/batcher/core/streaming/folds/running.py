"""`_AggFold` — the running (unwatermarked) streaming aggregate.

State grows with the group key and nothing evicts it, which is what makes it the operator a
changelog checkpoint mattered most for and the one a disk tier cannot yet help: it finalizes
every group on every micro-batch, so it has no cold end to shed.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import pyarrow as pa

from batcher._internal.native import engine
from batcher.config import active_config
from batcher.core.mergeable import RunningAggregate
from batcher.core.streaming.folds.shared import check_agg_state_bounded  # noqa: F401
from batcher.plan.logical import Aggregate
from batcher.plan.streaming import StateOperatorProgress

__all__ = ["_AggFold"]


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

    __slots__ = ("_cfg", "_delta", "_fold", "_input_ir", "_nat", "_updated")

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
        # The partial absorbed by the last `push`, held as this micro-batch's changelog
        # entry. See `delta`.
        self._delta: pa.RecordBatch | None = None

    def push(self, batch: pa.RecordBatch) -> int:
        """Fold one source batch into the running state; return rows consumed."""
        self._updated = 0
        self._delta = None
        if batch.num_rows == 0:
            return 0
        rows = self._nat.execute_plan(self._input_ir, [[batch]], self._cfg)
        if not rows or sum(b.num_rows for b in rows) == 0:
            return 0
        self._delta = self._fold.push(rows)
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

    def take_delta(self) -> pa.RecordBatch | None:
        """Take this micro-batch's changelog entry — the partial the last `push` absorbed.

        **Reading consumes it**, and that is the contract rather than an implementation
        detail. The engine commits epochs that never call `push`: the end-of-drain marker
        records a position under a fresh batch id so recovery resumes *after* the drain, and
        a trigger that read nothing commits too. A delta left in place is then written a
        second time under the new id, and recovery combines the same partial twice — the
        last micro-batch of every run counted double, silently, in the totals only.

        A checkpoint that persists this instead of `state()` costs bytes proportional to the
        *batch's* distinct group count rather than the *query's*. That is the whole
        difference between a checkpoint whose cost is flat and one that grows with the state
        it is protecting: an unwatermarked streaming aggregate never evicts, so its state
        only grows, and rewriting it whole every micro-batch means the per-epoch checkpoint
        cost rises for the entire life of the query.

        Replaying deltas is sound for exactly the reason the algebra exists. `combine` is
        associative and commutative (invariant #7), so combining a base snapshot with every
        partial recorded after it reconstructs the same state the full snapshot would have
        held — bit-for-bit on integers, and to the same last-bits tolerance on floats that
        any other change of combination order carries.

        **This is only valid for a fold that never removes rows.** `_WindowedAggFold` evicts
        closed windows, and a delta chain would resurrect them, so it does not offer one and
        keeps full snapshots. Nothing here can detect that mistake for a future fold: a
        resurrected window is a wrong number, not an error.

        Returns:
            The partial absorbed by the last `push`, or None when that push folded nothing
            in — an empty micro-batch has no changelog entry, and writing one would be a
            file per idle trigger forever.
        """
        delta, self._delta = self._delta, None
        return delta

    def restore(self, state: pa.RecordBatch) -> None:
        """Seed the running partial state from a checkpoint snapshot."""
        self._fold.restore(state)

    def restore_chain(self, states: Sequence[pa.RecordBatch]) -> None:
        """Rebuild the running state from a base snapshot and the deltas recorded after it.

        Args:
            states: The base snapshot followed by each delta, oldest first. Order is
                immaterial to the result — `combine` is commutative — but it is preserved
                so a reader of the checkpoint sees the sequence that produced the state.
        """
        self._fold.combine_all(list(states))
