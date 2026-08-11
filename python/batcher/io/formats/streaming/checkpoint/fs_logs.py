"""Offset and commit logs as one small file per micro-batch, on any filesystem.

`logs.py` keeps these in SQLite, which is the right answer on local disk and is not an
answer at all on an object store: SQLite needs a seekable, lockable local file, and there is
none behind an ``s3://`` URI. So a checkpoint on the durable storage the engine's own
spot-resilience warning recommends had nowhere to record its offsets.

The shape here is Spark's, for the reason Spark chose it: one immutable file per batch id,
written once and never updated, is atomic on an object store (a single PUT), atomic on
POSIX (temp file plus rename), and needs no lock on either. Recovery reads the highest id in
each directory and one file from each, which is a listing and two GETs — paid once per query
start, not once per micro-batch.

These implement the same methods `OffsetLog` and `CommitLog` do, and
`tests/unit/test_checkpoint_log_conformance.py` drives both through the same sequence and
asserts they agree. Two implementations of one contract is a drift risk, and a conformance
test is the thing that makes it a managed one rather than a latent one.
"""

from __future__ import annotations

import json
from typing import Any

from batcher.io.formats.streaming.checkpoint.location import CheckpointDir

__all__ = ["FileCommitLog", "FileOffsetLog"]

#: Batch ids are zero-padded so a lexical listing is already in numeric order, and wide
#: enough that a query running for years at a millisecond cadence cannot overflow the field
#: and start sorting wrongly.
_WIDTH = 20
_SUFFIX = ".json"


def _name(batch_id: int) -> str:
    return f"{batch_id:0{_WIDTH}d}{_SUFFIX}"


def _batch_ids(directory: CheckpointDir) -> list[int]:
    """Every batch id present, ascending; entries that are not one are ignored."""
    ids = []
    for name in directory.names(_SUFFIX):
        try:
            ids.append(int(name[: -len(_SUFFIX)]))
        except ValueError:
            continue
    return sorted(ids)


class _FileLog:
    """A directory of one JSON document per batch id."""

    __slots__ = ("_dir",)

    def __init__(self, root: str) -> None:
        self._dir = CheckpointDir(root)

    def _load(self, batch_id: int) -> dict[str, Any] | None:
        raw = self._dir.read(_name(batch_id))
        if raw is None:
            return None
        try:
            loaded = json.loads(raw)
        except ValueError:
            # A truncated document cannot happen through `write` (both backends land the
            # file atomically), so this is a foreign or corrupted file. Treating it as
            # absent replays the batch, which the sink's idempotency absorbs; treating it
            # as valid would seek a stream to a position nobody wrote.
            return None
        return loaded if isinstance(loaded, dict) else None

    def _store(self, batch_id: int, document: dict[str, Any]) -> None:
        self._dir.write(_name(batch_id), json.dumps(document).encode())

    def _latest(self) -> int | None:
        ids = _batch_ids(self._dir)
        return ids[-1] if ids else None

    def _prune(self, keep_through: int) -> None:
        for bid in _batch_ids(self._dir):
            if bid < keep_through:
                self._dir.remove(_name(bid))

    def close(self) -> None:
        """No connection to close — a file log holds nothing open between calls."""


class FileOffsetLog(_FileLog):
    """Per-micro-batch source positions (write-ahead: recorded before processing)."""

    def record(self, batch_id: int, source_id: int, position: dict[str, Any]) -> None:
        """Record (idempotently) the position consumed for ``(batch_id, source_id)``.

        Read-modify-write, because one batch's document holds every source's position and a
        multi-source plan records them one at a time. That is two round trips per source per
        epoch on an object store, against one for the single-source plans this path actually
        serves today — and correctness for the day it serves more.
        """
        document = self._load(batch_id) or {}
        document[str(source_id)] = position
        self._store(batch_id, document)

    def latest_batch(self) -> int | None:
        """The highest batch id recorded, or ``None`` if the log is empty."""
        return self._latest()

    def position_at(self, batch_id: int) -> dict[int, dict[str, Any]]:
        """The ``{source_id: position}`` recorded for ``batch_id`` (empty if none)."""
        document = self._load(batch_id)
        return {} if document is None else {int(k): v for k, v in document.items()}

    def prune(self, keep_through: int) -> None:
        """Delete offset documents for batches strictly before ``keep_through``."""
        self._prune(keep_through)


class FileCommitLog(_FileLog):
    """Which micro-batches completed and were durably written to the sink."""

    def commit(self, batch_id: int, sink_token: str | None = None) -> None:
        """Mark ``batch_id`` durably committed (idempotent)."""
        self._store(batch_id, {"sink_token": sink_token})

    def last_committed(self) -> int | None:
        """The highest committed batch id, or ``None`` if none committed yet."""
        return self._latest()

    def sink_token(self, batch_id: int) -> str | None:
        """What the sink reported writing for `batch_id`, or `None`."""
        document = self._load(batch_id)
        return None if document is None else document.get("sink_token")

    def is_committed(self, batch_id: int) -> bool:
        """Whether ``batch_id`` has been committed."""
        return self._dir.exists(_name(batch_id))

    def prune(self, keep_through: int) -> None:
        """Delete commit documents for batches strictly before ``keep_through``."""
        self._prune(keep_through)
