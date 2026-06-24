"""Streaming-query checkpointing — offset log, commit log, and state store.

A `CheckpointStore` over a ``checkpoint_location`` directory gives a streaming query
exactly-once recovery: source positions are recorded write-ahead, the running state
is snapshotted, the sink commits, and the commit log is written last. On restart,
`recover` replays the single uncommitted micro-batch with restored state. Reuses the
durable-SQLite pattern of `SeenStore` and Arrow IPC for state.
"""

from __future__ import annotations

from batcher.io.formats.streaming.checkpoint.logs import CommitLog, OffsetLog
from batcher.io.formats.streaming.checkpoint.recovery import ResumePlan, recover
from batcher.io.formats.streaming.checkpoint.state_store import StateStore
from batcher.io.formats.streaming.checkpoint.store import CheckpointStore

__all__ = [
    "CheckpointStore",
    "CommitLog",
    "OffsetLog",
    "ResumePlan",
    "StateStore",
    "recover",
]
