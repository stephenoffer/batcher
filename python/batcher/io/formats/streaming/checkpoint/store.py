"""`CheckpointStore` — the offset log, commit log, and state store under one dir.

A streaming query's ``checkpoint_location`` is a directory holding ``offsets.sqlite``,
``commits.sqlite``, and a ``state/`` subdirectory of Arrow-IPC running-state
snapshots. The store ties them to the micro-batch boundary so a crash is recoverable:
the offset is recorded write-ahead, the state snapshot and sink commit follow, and
the commit-log entry is written last — so a batch present in offsets but absent from
commits is exactly the in-flight batch to replay.
"""

from __future__ import annotations

import os

import pyarrow as pa

from batcher.io.formats.streaming.checkpoint.logs import CommitLog, OffsetLog
from batcher.io.formats.streaming.checkpoint.state_store import StateStore

__all__ = ["CheckpointStore"]


class CheckpointStore:
    """Bundles the three checkpoint logs for one streaming query."""

    __slots__ = ("_dir", "commits", "offsets", "state")

    def __init__(self, location: str) -> None:
        os.makedirs(location, exist_ok=True)
        self._dir = location
        self.offsets = OffsetLog(os.path.join(location, "offsets.sqlite"))
        self.commits = CommitLog(os.path.join(location, "commits.sqlite"))
        self.state = StateStore(os.path.join(location, "state"))

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

    def close(self) -> None:
        self.offsets.close()
        self.commits.close()
