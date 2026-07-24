"""Durable per-pipeline notes — the one thing about a pipeline that outlives the process.

Everything else the dashboard shows is a *measurement*: it belongs to a run and is
recomputed from the event stream. A pipeline's **name** is different. It is a fact about
the pipeline as a thing you return to — "nightly rollup", "the report that keeps spilling"
— and it has to survive a restart, or naming it was pointless.

So this is the dashboard's one piece of writable, persistent state. It maps a pipeline id
(the plan signature Kyber already keys learned stats on) to a small record: a human name, a
free-text note, and when the pipeline was first seen. It is a JSON file under
`$BATCHER_HOME`, written atomically, and it is deliberately tiny — a pipeline is a *shape*,
so even a busy engine accrues only a handful.

It is not an archive of runs. Runs age out of the in-memory ring buffer as designed; this
remembers the identity, not the history.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from batcher._internal.logging import get_logger
from batcher._internal.paths import batcher_home

__all__ = ["PipelineMeta", "PipelineRegistry"]

#: The longest a name or note may be. A generous ceiling that still stops the file from
#: growing without bound if the write endpoint is ever handed something pathological.
_MAX_NAME = 200
_MAX_NOTE = 2000


@dataclass(slots=True)
class PipelineMeta:
    """The durable facts about one pipeline: what a person called it, and when it appeared."""

    name: str = ""
    note: str = ""
    first_seen_wall: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "note": self.note, "first_seen_wall": self.first_seen_wall}


class PipelineRegistry:
    """A JSON-backed map from pipeline id to its durable metadata.

    Reads are served from an in-memory copy; writes update it and re-serialize the whole
    file, which is cheap because the file is small and writes are rare (a person renaming a
    pipeline, not the engine running one). Every mutation takes `_lock`, so the dashboard's
    write thread and its read thread never see a half-written map.

    Construct with an explicit `path` in a test; the default resolves under `$BATCHER_HOME`.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._path = path if path is not None else batcher_home() / "pipelines.json"
        self._entries: dict[str, PipelineMeta] = {}
        self._load()

    # --- persistence --------------------------------------------------------

    def _load(self) -> None:
        """Read the file into memory, tolerating its absence or corruption.

        A registry that raised on a malformed file would take the whole dashboard down for a
        stray edit to a convenience file. A bad file is logged and treated as empty; the
        next write replaces it with something valid.
        """
        try:
            raw = json.loads(self._path.read_text())
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            get_logger("observe").warning(
                "pipeline registry at %s is unreadable; starting empty", self._path
            )
            return
        if not isinstance(raw, dict):
            return
        for pipeline_id, record in raw.items():
            if isinstance(record, dict):
                self._entries[str(pipeline_id)] = PipelineMeta(
                    name=str(record.get("name", ""))[:_MAX_NAME],
                    note=str(record.get("note", ""))[:_MAX_NOTE],
                    first_seen_wall=float(record.get("first_seen_wall", 0.0) or 0.0),
                )

    def _flush(self) -> None:
        """Write the whole map back, atomically. Assumes `_lock` is held.

        Atomic via write-to-temp-then-rename, so a crash mid-write can never leave a
        truncated file that `_load` would then discard — the previous good file stays until
        the new one is complete. A failure to write is logged, not raised: losing a rename
        must not fail a query or a dashboard poll.
        """
        payload = json.dumps({k: v.to_dict() for k, v in self._entries.items()}, indent=2)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".json.tmp")
            tmp.write_text(payload)
            os.replace(tmp, self._path)
        except OSError:
            get_logger("observe").warning(
                "could not persist the pipeline registry to %s", self._path, exc_info=True
            )

    # --- read ---------------------------------------------------------------

    def get(self, pipeline_id: str) -> PipelineMeta | None:
        """The stored metadata for a pipeline, or `None` if it has none."""
        with self._lock:
            return self._entries.get(pipeline_id)

    def all(self) -> dict[str, PipelineMeta]:
        """A copy of the whole map, for merging names into a pipeline listing in one pass."""
        with self._lock:
            return dict(self._entries)

    # --- write --------------------------------------------------------------

    def seen(self, pipeline_id: str, wall: float) -> None:
        """Record that a pipeline exists, stamping its first-seen time once.

        Called as pipelines are listed, so a pipeline a person later names already has a
        creation time attached rather than acquiring one only when renamed. Writes nothing
        and re-serializes nothing when the id is already known — the common case on every
        poll must be a dict lookup, not a file write.
        """
        if not pipeline_id:
            return
        with self._lock:
            entry = self._entries.get(pipeline_id)
            if entry is not None and entry.first_seen_wall > 0:
                return
            if entry is None:
                self._entries[pipeline_id] = PipelineMeta(first_seen_wall=wall)
            else:
                # Named before it was ever listed (a direct API rename); backfill the time
                # rather than leave it at zero forever.
                entry.first_seen_wall = wall
            self._flush()

    def set_meta(
        self, pipeline_id: str, *, name: str | None = None, note: str | None = None
    ) -> PipelineMeta:
        """Set a pipeline's name and/or note, persisting immediately.

        Only the fields that are not `None` change, so setting a note leaves a name intact
        and vice versa. Both are clamped to a sane length. Returns the resulting record.

        Args:
            pipeline_id: The pipeline's stable id (its plan signature).
            name: The new human name, or None to leave it unchanged. `""` clears it.
            note: The new free-text note, or None to leave it unchanged.

        Returns:
            The updated `PipelineMeta`.
        """
        with self._lock:
            entry = self._entries.get(pipeline_id) or PipelineMeta()
            if name is not None:
                entry.name = name.strip()[:_MAX_NAME]
            if note is not None:
                entry.note = note.strip()[:_MAX_NOTE]
            self._entries[pipeline_id] = entry
            self._flush()
            return entry
