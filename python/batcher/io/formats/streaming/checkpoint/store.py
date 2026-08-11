"""`CheckpointStore` — the offset log, commit log, and state store under one dir.

A streaming query's ``checkpoint_location`` is a directory holding an offset log, a commit
log, and a ``state/`` subdirectory of Arrow-IPC running-state snapshots. The store ties them
to the micro-batch boundary so a crash is recoverable: the offset is recorded write-ahead,
the state snapshot and sink commit follow, and the commit-log entry is written last — so a
batch present in offsets but absent from commits is exactly the in-flight batch to replay.

**The location may be remote**, and until it could be, the engine's own spot-resilience
warning was advice it did not honor: it told the caller that a node-local checkpoint is lost
with the node and to use ``s3://``, then wrote to a local directory named ``s3:``. A local
location keeps the SQLite logs, which are the right tool for a lockable seekable file; a
remote one uses the file-per-batch logs in `fs_logs`, which are the right tool for a store
that has neither. The state store spans both itself.
"""

from __future__ import annotations

import os

import pyarrow as pa

from batcher.io.formats.streaming.checkpoint.location import is_local_location
from batcher.io.formats.streaming.checkpoint.state_store import StateStore

__all__ = ["CheckpointStore"]

#: Micro-batches between log sweeps. The logs are bounded by this many rows rather than by
#: one, which is the same guarantee at a sixty-fourth of the fsyncs.
_PRUNE_EVERY = 64


class CheckpointStore:
    """Bundles the three checkpoint logs for one streaming query."""

    __slots__ = ("_dir", "_pruned_through", "commits", "offsets", "state")

    def __init__(self, location: str) -> None:
        self._dir = location
        if is_local_location(location):
            from batcher.io.filesystem import local_path
            from batcher.io.formats.streaming.checkpoint.logs import CommitLog, OffsetLog

            root = local_path(location)
            os.makedirs(root, exist_ok=True)
            self.offsets = OffsetLog(os.path.join(root, "offsets.sqlite"))
            self.commits = CommitLog(os.path.join(root, "commits.sqlite"))
            self.state = StateStore(os.path.join(root, "state"))
        else:
            from batcher.io.formats.streaming.checkpoint.fs_logs import (
                FileCommitLog,
                FileOffsetLog,
            )

            root = location.rstrip("/")
            self.offsets = FileOffsetLog(f"{root}/offsets")
            self.commits = FileCommitLog(f"{root}/commits")
            self.state = StateStore(f"{root}/state")
        self._pruned_through = 0

    @property
    def location(self) -> str:
        return self._dir

    def record_offsets(self, batch_id: int, positions: dict[int, dict]) -> None:
        """Write-ahead: record each source's position for `batch_id`."""
        for source_id, position in positions.items():
            self.offsets.record(batch_id, source_id, position)

    def snapshot_state(self, batch_id: int, state: pa.RecordBatch | None) -> None:
        """Snapshot the running aggregation state for `batch_id` (if any)."""
        if state is not None:
            self.state.snapshot(batch_id, state)

    def commit(self, batch_id: int, sink_token: str | None = None) -> None:
        """Mark `batch_id` durably done (the last step of the micro-batch)."""
        self.commits.commit(batch_id, sink_token)

    def prune_state(self, keep_through: int) -> None:
        """Delete running-state snapshots older than `keep_through` (bounded state dir).

        Recovery restores from the *latest committed* snapshot only, so every earlier one
        is dead weight; pruning them after each commit keeps a long-running stateful query's
        ``state/`` directory bounded (one live snapshot) instead of growing without limit.
        """
        self.state.prune(keep_through)

    def prune_logs(self, keep_through: int) -> None:
        """Bound the offset and commit logs by dropping batches before `keep_through`.

        Recovery only ever consults the last committed batch's offsets and the commit-log
        maximum, so rows before `keep_through` (the last committed batch) are dead weight.

        Pruning is *amortized* rather than run on every commit. Running it per micro-batch
        cost two more `DELETE` statements and two more fsyncs per epoch — on top of the two
        the epoch already pays — almost always to remove a single row, so a low-latency
        trigger spent more of its budget deleting bookkeeping than writing it. Sweeping every
        `_PRUNE_EVERY` batches keeps the logs bounded by a small constant instead of by one
        row, which is the property that actually matters, at a fraction of the syncs. The
        counter is per-store rather than derived from `keep_through` so a restart cannot land
        on a stride boundary and skip the sweep forever.
        """
        if keep_through - self._pruned_through < _PRUNE_EVERY:
            return
        self._pruned_through = keep_through
        self.offsets.prune(keep_through)
        self.commits.prune(keep_through)

    def close(self) -> None:
        self.offsets.close()
        self.commits.close()
