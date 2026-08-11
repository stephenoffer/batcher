"""Streaming-query checkpointing — offset log, commit log, and state store.

A `CheckpointStore` over a ``checkpoint_location`` gives a streaming query exactly-once
recovery: source positions are recorded write-ahead, the running state is snapshotted, the
sink commits, and the commit log is written last. On restart, `recover` replays the single
uncommitted micro-batch with restored state.

The location may be **local or remote**. Local keeps the durable-SQLite pattern of
`SeenStore`; a remote one (``s3://``, ``gs://``, ``hdfs://`` — the durable storage the
spot-resilience warning tells callers to use) keeps the same logs as one immutable file per
batch id (`fs_logs`), because SQLite needs a lockable seekable file an object store does not
have. State is Arrow IPC on both.
"""

from __future__ import annotations

from batcher.io.formats.streaming.checkpoint.fs_logs import FileCommitLog, FileOffsetLog
from batcher.io.formats.streaming.checkpoint.location import CheckpointDir, is_local_location
from batcher.io.formats.streaming.checkpoint.logs import CommitLog, OffsetLog
from batcher.io.formats.streaming.checkpoint.recovery import ResumePlan, recover
from batcher.io.formats.streaming.checkpoint.state_store import StateStore
from batcher.io.formats.streaming.checkpoint.store import CheckpointStore

__all__ = [
    "CheckpointDir",
    "CheckpointStore",
    "CommitLog",
    "FileCommitLog",
    "FileOffsetLog",
    "OffsetLog",
    "ResumePlan",
    "StateStore",
    "is_local_location",
    "recover",
]
