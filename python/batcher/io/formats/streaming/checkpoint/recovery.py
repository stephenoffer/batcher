"""The recovery decision at streaming-query start.

`recover` is a pure function over the checkpoint logs: it returns the batch id to
resume from, the per-source positions to seek to, and the running-state snapshot to
restore — so the driver can replay exactly the in-flight (uncommitted) batch with
restored state and continue. Kept side-effect-free so it is unit-testable without a
live source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyarrow as pa

    from batcher.io.formats.streaming.checkpoint.store import CheckpointStore

__all__ = ["ResumePlan", "recover"]


@dataclass(frozen=True, slots=True)
class ResumePlan:
    """How a streaming query resumes: where to start, seek, and what state to restore.

    `state` is the base snapshot and `state_deltas` the changelog entries recorded after it.
    They are reported separately rather than pre-combined because combining partials is the
    *aggregate algebra*, which lives in `core`; this module is neutral and knows only that a
    checkpoint holds a sequence of batches. A driver with no way to combine them restores
    `state` alone, which is why a delta is only ever written for a fold that offered one.
    """

    start_batch: int = 0
    seek: dict[int, dict] = field(default_factory=dict)
    state: pa.RecordBatch | None = None
    state_deltas: tuple[pa.RecordBatch, ...] = ()


def recover(store: CheckpointStore) -> ResumePlan:
    """Decide the resume point from the committed/recorded logs.

    Fresh query (no offsets) → start at batch 0. Otherwise resume at the first
    *uncommitted* batch: seek each source to the position recorded at the last
    committed batch and restore that batch's running-state snapshot. A batch present
    in the offset log but not the commit log is re-run (the sink dedups it).
    """
    if store.offsets.latest_batch() is None:
        return ResumePlan()  # fresh query
    last_commit = store.commits.last_committed()
    if last_commit is None:
        return ResumePlan()  # nothing committed yet → reprocess from the start
    resume_batch = last_commit + 1
    seek = store.offsets.position_at(last_commit)
    chain = store.state.restore_chain(last_commit)
    if not chain:
        return ResumePlan(start_batch=resume_batch, seek=seek)
    return ResumePlan(
        start_batch=resume_batch, seek=seek, state=chain[0], state_deltas=tuple(chain[1:])
    )
