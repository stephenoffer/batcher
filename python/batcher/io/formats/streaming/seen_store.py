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

# Candidates per `unseen` probe. SQLite's default SQLITE_MAX_VARIABLE_NUMBER is 999.
_PROBE_CHUNK = 500


_SCHEMA = """
CREATE TABLE IF NOT EXISTS seen_files (
    path  TEXT PRIMARY KEY,
    size  INTEGER NOT NULL,
    mtime REAL    NOT NULL
)
"""


def _tune(conn: sqlite3.Connection) -> None:
    """Put the store in write-ahead-log mode, which is what this access pattern wants.

    SQLite's default rollback journal rewrites and fsyncs a journal file around every
    commit, and this store commits once per discovery pass — so at a 200ms trigger the
    engine was paying two full journal cycles a second to remember a handful of filenames.
    WAL appends instead, and lets a reader run while a writer commits.

    ``synchronous=NORMAL`` is correct here rather than merely faster. Under WAL it still
    survives a *process* crash; what it gives up is durability across an OS crash, where the
    last few commits may be lost. Losing them re-offers those files on the next pass, and a
    re-offered file is exactly the case this design already handles — the epoch's transaction
    makes the replay idempotent. The failure this store must never have is the opposite one,
    remembering a file whose rows were never published, and NORMAL cannot produce it.

    Both pragmas are best-effort: a filesystem that refuses WAL (some network mounts) keeps
    the default journal and the store still works.
    """
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.Error:  # pragma: no cover - filesystem-dependent
        pass


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

        ``check_same_thread=False``, for the same reason the checkpoint logs disable it
        (`streaming/checkpoint/logs.py`): a streaming query's loop runs on a background
        thread, but **recovery runs on the main one** before that thread starts. Recovery
        calls `seek` → `confirm`, which opens this connection — so a restore carrying
        pending files opens the store on the main thread and the first discovery pass then
        raises ``SQLite objects created in a thread can only be used in that same thread``.

        That sequence is reachable directly through the `Checkpointable` surface and is
        pinned by a test. It is **not** currently reached by the engine's own recovery,
        because the last committed offset happens to be the drain marker, recorded after
        `confirm()` has cleared the pending list. That is an accident of ordering rather
        than a property anything enforces, which is exactly why the store should not
        depend on it. Access is serialized either way — recovery completes before the loop
        begins, then only the loop touches it — so the check is redundant, not protective.
        """
        self._conn = sqlite3.connect(path, check_same_thread=False)
        _tune(self._conn)
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

    def mark_many(self, records: list[tuple[str, int, float]]) -> None:
        """Record many ``(path, size, mtime)`` files as processed in one transaction.

        Equivalent to calling `mark` for each record, but a single `executemany` and one
        `commit` — so confirming a discovery pass of N files costs one fsync, not N. A
        crash mid-transaction leaves the store exactly as it was (SQLite atomicity), which
        is the same all-or-nothing guarantee the per-epoch caller already relies on: the
        files are re-offered on restart rather than half-remembered.
        """
        if not records:
            return
        self._conn.executemany(
            "INSERT INTO seen_files(path, size, mtime) VALUES(?, ?, ?) "
            "ON CONFLICT(path) DO UPDATE SET size = excluded.size, mtime = excluded.mtime",
            records,
        )
        self._conn.commit()

    def unseen(self, candidates: list[str]) -> list[str]:
        """Return the subset of ``candidates`` not yet recorded, order preserved.

        Probes the `path` PRIMARY KEY index for just the candidates, so the cost is
        O(candidates) — not O(every file ever seen).

        It used to be ``SELECT path FROM seen_files`` — the whole table — loaded into a Python
        set on every discovery pass, to answer a question about a handful of new files. That is
        the classic unbounded-state wall, and it is quadratic in the lifetime of the stream:
        with 1,000,000 files already seen, asking about 10 new ones took **2.6 s and 185 MB**,
        and it grows forever. The index was there the whole time.
        """
        if not candidates:
            return []
        known: set[str] = set()
        # SQLite's default parameter ceiling (SQLITE_MAX_VARIABLE_NUMBER) is 999, so probe in
        # chunks. Each chunk is one round trip and one index seek per candidate.
        for start in range(0, len(candidates), _PROBE_CHUNK):
            chunk = candidates[start : start + _PROBE_CHUNK]
            placeholders = ",".join("?" * len(chunk))
            cur = self._conn.execute(
                f"SELECT path FROM seen_files WHERE path IN ({placeholders})", chunk
            )
            known.update(row[0] for row in cur)
        return [c for c in candidates if c not in known]

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
