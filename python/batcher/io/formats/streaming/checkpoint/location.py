"""Where a checkpoint lives — local disk, or the object store the durability advice names.

A streaming query's `checkpoint_location` is the whole of its exactly-once recovery, and
under `resilience="spot"` the engine already *warns* that a node-local one is lost with the
node and tells the caller to use `s3://`, `gs://`, or `hdfs://`. It then could not write to
one: the store called `os.makedirs` and `sqlite3.connect` on the string, so `s3://bucket/ckpt`
created a local directory literally named ``s3:`` and the query checkpointed into it happily.
The advice and the implementation disagreed, and the disagreement was silent — the failure
surfaced only when a reclaimed node took a checkpoint nobody realized was on it.

This module is the seam that closes that. It answers one question — is this location on the
local filesystem? — and provides the small directory operations both the state store and the
file-based logs need on the answer that is not.

Local stays on POSIX rather than routing through the same façade, and that is a durability
decision rather than an oversight. The state snapshot is written, fsynced, renamed, and the
*directory* fsynced, because the engine snapshots state and then records the commit: without
those syncs a machine crash can leave the commit durable and the state not, and recovery
resumes past data it already consumed with an empty running aggregate. `pyarrow.fs` offers no
fsync, so putting local writes through it would quietly drop a property this store argues for
at length. On an object store the equivalent property is the storage's: a PUT is durable when
it returns, and there is no rename to make durable separately.
"""

from __future__ import annotations

import contextlib
from typing import Any

__all__ = ["CheckpointDir", "is_local_location"]


def is_local_location(location: str) -> bool:
    """Whether `location` names a path on this machine's filesystem.

    A bare path and a ``file://`` URI are local; every other scheme is not. A shared mount
    is local by this test and by every test available — it is an ordinary path, which is
    exactly why `_warn_if_checkpoint_not_durable` warns rather than refuses.

    Args:
        location: The checkpoint location.

    Returns:
        True for a bare path or a ``file:`` URI.
    """
    from batcher.io._backend import _scheme

    return _scheme(location) in ("", "file")


class CheckpointDir:
    """The handful of directory operations a checkpoint needs, on any filesystem.

    Deliberately not a general filesystem wrapper: read a small file, write a small file
    atomically, list what is there, delete one. That is the whole vocabulary of an offset
    log, a commit log, and a state directory, and keeping it that small is what lets the
    remote path be reviewed against the local one at a glance.
    """

    __slots__ = ("_fs", "_root")

    def __init__(self, root: str) -> None:
        from batcher.io.filesystem import resolve_filesystem

        self._root = root.rstrip("/")
        self._fs = resolve_filesystem(root)
        self._fs.mkdirs(self._root)

    def path(self, name: str) -> str:
        """The full path of `name` inside this directory."""
        return f"{self._root}/{name}"

    def write(self, name: str, payload: bytes) -> None:
        """Write `payload` to `name`, atomically.

        The façade writes an object-store destination as a single PUT (already atomic) and
        a local/HDFS one as a temp sibling it renames, so a reader never sees a partial
        file on either.
        """
        with self._fs.atomic_writer(self.path(name)) as fh:
            fh.write(payload)

    def read(self, name: str) -> bytes | None:
        """`name`'s bytes, or None when it is not there."""
        target = self.path(name)
        if not self._fs.exists(target):
            return None
        with self._fs.open(target, "rb") as fh:
            return fh.read()

    def exists(self, name: str) -> bool:
        """Whether `name` is present."""
        return self._fs.exists(self.path(name))

    def names(self, suffix: str) -> list[str]:
        """The basenames in this directory ending in `suffix`, or an empty list.

        `expand` raises for a directory that is empty or absent, which for a log that has
        not been written yet is the ordinary state rather than an error.
        """
        listed: list[str] = []
        with contextlib.suppress(Exception):
            listed = self._fs.expand(self._root, suffix=suffix)
        return [entry.rstrip("/").rsplit("/", 1)[-1] for entry in listed]

    def remove(self, name: str) -> None:
        """Delete `name`, tolerating its absence."""
        with contextlib.suppress(Exception):
            self._fs.remove(self.path(name))

    def open_reader(self, name: str) -> Any:
        """A binary file handle on `name`, for a reader that streams rather than slurps."""
        return self._fs.open(self.path(name), "rb")
