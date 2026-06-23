"""A durable "seen-file" key-value store, backed by stdlib ``sqlite3``.

The incremental file source (the Auto Loader analog in
``io/formats/streaming/autoloader.py``) must discover *new* files exactly once
across repeated discovery passes — including across process restarts. This store
persists, for every file it has handed out, the file path plus its size and
modification time, so a later pass can ask "which of these candidates have I not
seen?" and get a stable answer.

It uses only the Python standard library (``sqlite3``) — no extra dependency —
and a single small table keyed by path. Writes are committed eagerly so a crash
mid-pass never re-emits already-processed files (exactly-once semantics).
"""

from __future__ import annotations

import sqlite3
from types import TracebackType

__all__ = ["SeenStore"]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS seen_files (
    path  TEXT PRIMARY KEY,
    size  INTEGER NOT NULL,
    mtime REAL    NOT NULL
)
"""


class SeenStore:
    """A persistent set of already-processed files, keyed by path.

    Backed by a single SQLite database at ``path``. Safe to open repeatedly
    against the same database file; the schema is created on first use. Each
    record carries the file's ``size`` and ``mtime`` so callers can detect a file
    that was rewritten in place if they choose to (the store itself dedups purely
    by path).

    Example:
        >>> store = SeenStore(":memory:")
        >>> store.seen("a.parquet")
        False
        >>> store.mark("a.parquet", size=10, mtime=1.0)
        >>> store.seen("a.parquet")
        True
        >>> store.unseen(["a.parquet", "b.parquet"])
        ['b.parquet']
    """

    __slots__ = ("_conn",)

    def __init__(self, path: str) -> None:
        """Open (creating if needed) the SQLite-backed store at ``path``.

        ``path`` may be ``":memory:"`` for an ephemeral in-process store (useful
        for tests). The parent directory of an on-disk path must already exist.
        """
        self._conn = sqlite3.connect(path)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def seen(self, file: str) -> bool:
        """Return whether ``file`` has already been recorded as processed."""
        cur = self._conn.execute("SELECT 1 FROM seen_files WHERE path = ?", (file,))
        return cur.fetchone() is not None

    def mark(self, file: str, size: int, mtime: float) -> None:
        """Record ``file`` (with its ``size``/``mtime``) as processed.

        Idempotent: re-marking an existing path updates its size/mtime. Committed
        immediately so the record survives a crash before the next pass.
        """
        self._conn.execute(
            "INSERT INTO seen_files(path, size, mtime) VALUES(?, ?, ?) "
            "ON CONFLICT(path) DO UPDATE SET size = excluded.size, mtime = excluded.mtime",
            (file, size, mtime),
        )
        self._conn.commit()

    def unseen(self, candidates: list[str]) -> list[str]:
        """Return the subset of ``candidates`` not yet recorded, order preserved.

        Computed with a single set-membership query rather than per-candidate
        lookups, so a large candidate list costs one round trip.
        """
        if not candidates:
            return []
        cur = self._conn.execute("SELECT path FROM seen_files")
        known = {row[0] for row in cur.fetchall()}
        return [c for c in candidates if c not in known]

    def max_seen(self) -> str | None:
        """The lexicographically greatest seen path, or ``None`` if empty.

        Enables the lexical fast-path in the file lister: candidates ordering
        after this name cannot have been seen yet.
        """
        cur = self._conn.execute("SELECT MAX(path) FROM seen_files")
        row = cur.fetchone()
        return row[0] if row is not None else None

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    def __enter__(self) -> SeenStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
