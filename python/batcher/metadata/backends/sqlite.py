"""SQLite backend — the local durable default.

Keys (tuples) are encoded to a stable string so they can index a single
(tbl, key) → value table. Good enough for single-node persistence; Redis / cloud
object storage take over for shared clusters behind the same protocol.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator

from batcher._internal.errors import ConfigError
from batcher.metadata.store import Key

__all__ = ["SQLiteBackend"]


def _encode_key(key: Key) -> str:
    # JSON with a fixed separator gives a deterministic, prefix-comparable string.
    return json.dumps(list(key), separators=(",", ":"))


class SQLiteBackend:
    """A `MetadataBackend` backed by a SQLite database (file path or ``:memory:``)."""

    def __init__(self, uri: str = ":memory:") -> None:
        """Open (or create) the learned-stats database.

        Args:
            uri: A filesystem path, or ``":memory:"`` for an ephemeral store.

        Raises:
            ConfigError: If `uri` is not a string, or the database cannot be opened.
                SQLite answers a missing parent directory, a read-only volume, and a
                path that is actually a directory with the same five words — "unable to
                open database file" — so the path itself has to be in the message.
        """
        if not isinstance(uri, str):
            raise ConfigError(
                f"The sqlite metadata backend needs a path string, but got "
                f"{type(uri).__name__} {uri!r}.",
                hint="Pass a filesystem path, or ':memory:' for an ephemeral store.",
            )
        self._uri = uri
        try:
            self._conn = sqlite3.connect(uri)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS kv ("
                "  tbl TEXT NOT NULL, key TEXT NOT NULL, value BLOB NOT NULL,"
                "  PRIMARY KEY (tbl, key))"
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise ConfigError(
                f"Cannot open the sqlite metadata database at {uri!r}: {exc}.",
                hint=(
                    "Check that the parent directory exists and is writable, or use "
                    "':memory:' to keep learned stats for this process only."
                ),
            ) from exc

    def __repr__(self) -> str:
        """Name the database file, so two hubs on different stores are distinguishable."""
        return f"SQLiteBackend(uri={self._uri!r})"

    def get(self, table: str, key: Key) -> bytes | None:
        row = self._conn.execute(
            "SELECT value FROM kv WHERE tbl = ? AND key = ?", (table, _encode_key(key))
        ).fetchone()
        return row[0] if row else None

    def put(self, table: str, key: Key, value: bytes) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO kv (tbl, key, value) VALUES (?, ?, ?)",
            (table, _encode_key(key), value),
        )
        self._conn.commit()

    def scan(self, table: str, prefix: Key = ()) -> Iterator[tuple[Key, bytes]]:
        rows = self._conn.execute(
            "SELECT key, value FROM kv WHERE tbl = ? ORDER BY key", (table,)
        ).fetchall()
        plen = len(prefix)
        for enc_key, value in rows:
            key = tuple(json.loads(enc_key))
            if key[:plen] == prefix:
                yield key, value

    def batch_put(self, table: str, items: list[tuple[Key, bytes]]) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO kv (tbl, key, value) VALUES (?, ?, ?)",
            [(table, _encode_key(k), v) for k, v in items],
        )
        self._conn.commit()
