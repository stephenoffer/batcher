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

__all__ = ["FINALIZE_SOURCE_ID", "ResumePlan", "recover"]

#: The offset-log source id a *finalize* batch records alongside the real positions. A real
#: source id is a plan input's index, so it is never negative; the marker therefore cannot be
#: mistaken for one, and a reader that seeks only the sources it knows simply ignores it.
FINALIZE_SOURCE_ID = -1


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
    #: Whether `start_batch` is an end-of-stream flush that was write-ahead logged and never
    #: committed. Its output may or may not have reached the sink, so the driver re-runs the
    #: flush from the restored state under the same id (the sink absorbs a repeat) and
    #: commits it, rather than handing the id to fresh data the sink would then skip.
    pending_finalize: bool = False


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
        # Nothing committed yet → reprocess from the start, finishing a flush first if that
        # is what the first run died inside.
        return ResumePlan(pending_finalize=_is_finalize(store, 0))
    resume_batch = last_commit + 1
    seek = store.offsets.position_at(last_commit)
    pending = _is_finalize(store, resume_batch)
    chain = store.state.restore_chain(last_commit)
    if not chain:
        return ResumePlan(start_batch=resume_batch, seek=seek, pending_finalize=pending)
    return ResumePlan(
        start_batch=resume_batch,
        seek=seek,
        state=chain[0],
        state_deltas=tuple(chain[1:]),
        pending_finalize=pending,
    )


def _is_finalize(store: CheckpointStore, batch_id: int) -> bool:
    """Whether `batch_id` was write-ahead logged as an end-of-stream flush."""
    return FINALIZE_SOURCE_ID in store.offsets.position_at(batch_id)
