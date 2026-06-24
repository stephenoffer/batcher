"""Durable offset + commit logs for streaming-query checkpointing.

Both reuse the `SeenStore` pattern — a single small SQLite table with **eager
commit** so a crash never loses a recorded fact — over stdlib ``sqlite3`` (no extra
dependency). The shared `_LogTable` base owns the connection and schema; `OffsetLog`
records, per micro-batch, the source positions read (write-ahead, before
processing); `CommitLog` records which batches durably reached the sink. The
in-flight batch to replay on restart is the one present in the offset log but absent
from the commit log.
"""

from __future__ import annotations

import json
import sqlite3
from types import TracebackType
from typing import Any

__all__ = ["CommitLog", "OffsetLog"]


class _LogTable:
    """A SQLite-backed log table with eager commit (the `SeenStore` pattern)."""

    __slots__ = ("_conn",)

    def __init__(self, path: str, schema: str) -> None:
        # The streaming-query loop runs on a background thread, so the connection is
        # used from a different thread than it was opened on. Access is serialized
        # (recovery on the main thread before the loop starts, then only the loop
        # thread), so disabling the same-thread check is safe here.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(schema)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> _LogTable:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


_OFFSET_SCHEMA = """
CREATE TABLE IF NOT EXISTS offsets (
    batch_id  INTEGER NOT NULL,
    source_id INTEGER NOT NULL,
    position  TEXT    NOT NULL,
    PRIMARY KEY (batch_id, source_id)
)
"""


class OffsetLog(_LogTable):
    """Per-micro-batch source positions (write-ahead: recorded before processing).

    `position` is the opaque JSON a `Checkpointable` source returns from
    ``snapshot_position()`` — the log never interprets it.
    """

    def __init__(self, path: str) -> None:
        super().__init__(path, _OFFSET_SCHEMA)

    def record(self, batch_id: int, source_id: int, position: dict[str, Any]) -> None:
        """Record (idempotently) the position consumed for ``(batch_id, source_id)``."""
        self._conn.execute(
            "INSERT INTO offsets(batch_id, source_id, position) VALUES(?, ?, ?) "
            "ON CONFLICT(batch_id, source_id) DO UPDATE SET position = excluded.position",
            (batch_id, source_id, json.dumps(position)),
        )
        self._conn.commit()

    def latest_batch(self) -> int | None:
        """The highest batch id recorded, or ``None`` if the log is empty."""
        row = self._conn.execute("SELECT MAX(batch_id) FROM offsets").fetchone()
        return row[0] if row is not None else None

    def position_at(self, batch_id: int) -> dict[int, dict[str, Any]]:
        """The ``{source_id: position}`` recorded for ``batch_id`` (empty if none)."""
        cur = self._conn.execute(
            "SELECT source_id, position FROM offsets WHERE batch_id = ?", (batch_id,)
        )
        return {sid: json.loads(pos) for sid, pos in cur.fetchall()}


_COMMIT_SCHEMA = """
CREATE TABLE IF NOT EXISTS commits (
    batch_id     INTEGER PRIMARY KEY,
    sink_token   TEXT
)
"""


class CommitLog(_LogTable):
    """Which micro-batches completed and were durably written to the sink."""

    def __init__(self, path: str) -> None:
        super().__init__(path, _COMMIT_SCHEMA)

    def commit(self, batch_id: int, sink_token: str | None = None) -> None:
        """Mark ``batch_id`` durably committed (idempotent)."""
        self._conn.execute(
            "INSERT INTO commits(batch_id, sink_token) VALUES(?, ?) "
            "ON CONFLICT(batch_id) DO UPDATE SET sink_token = excluded.sink_token",
            (batch_id, sink_token),
        )
        self._conn.commit()

    def last_committed(self) -> int | None:
        """The highest committed batch id, or ``None`` if none committed yet."""
        row = self._conn.execute("SELECT MAX(batch_id) FROM commits").fetchone()
        return row[0] if row is not None else None

    def is_committed(self, batch_id: int) -> bool:
        """Whether ``batch_id`` has been committed."""
        cur = self._conn.execute("SELECT 1 FROM commits WHERE batch_id = ?", (batch_id,))
        return cur.fetchone() is not None
