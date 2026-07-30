"""SQLite backend — the local durable default.

Keys (tuples) are encoded to a stable string so they can index a single
(tbl, key) → value table. Good enough for single-node persistence; Redis / cloud
object storage take over for shared clusters behind the same protocol.

Three properties of that encoding are load-bearing here, and none is accidental:

* It is **deterministic**, so the same tuple always produces the same string and the
  primary key `(tbl, key)` actually collides on re-writes instead of accumulating.
* It is **prefix-preserving**: `encode_key(("ns", "k"))` starts with `encode_key(("ns",))`
  minus its closing bracket. That is what lets `scan` push a prefix filter into a range
  query over the primary-key index rather than reading and decoding the whole table.
* It is **ordered** the same way SQLite orders text, so the range has two plain bounds.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
import threading
from collections.abc import Iterator

from batcher._internal.errors import ConfigError
from batcher._internal.logging import note_suppressed
from batcher._internal.paths import private_dir
from batcher.metadata.store import Key, decode_key, encode_key

__all__ = ["SQLiteBackend"]


def _prefix_bound(prefix: Key) -> str | None:
    """The inclusive lower bound of the encoded-key range covering `prefix`, or `None`.

    `None` for the empty prefix, which covers the whole table and needs no range at all.
    Otherwise it is `encode_key(prefix)` with the closing `]` removed, so that
    `encode_key(prefix + rest)` — which continues with `,` — sorts inside the range, and so
    does `encode_key(prefix)` itself, which continues with `]`.
    """
    if not prefix:
        return None
    return encode_key(prefix)[:-1]


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
            # The learned-stats database holds persisted column statistics, and those
            # include `min`/`max` — real values out of real columns. Owner-only before the
            # first connect, since sqlite creates the file under the default umask.
            #
            # Tightens an existing directory; deliberately does NOT create a missing one.
            # Creating it would swallow the "unable to open database file" error that names
            # the bad path — turning a clear misconfiguration into a database silently
            # appearing somewhere the operator did not intend. Hardening must not move a
            # failure.
            if uri != ":memory:":
                parent = os.path.dirname(os.path.abspath(uri))
                if parent and os.path.isdir(parent):
                    private_dir(parent)
            # `check_same_thread=False` plus `self._lock`, because the hub is a process
            # singleton (`core.default_hub`) and its writers are worker threads. SQLite's
            # default refuses a connection used off its creating thread with
            # `ProgrammingError: SQLite objects created in a thread can only be used in that
            # same thread` — which the hub's `record` catches and logs, so the symptom is not
            # an error but a durable store that silently persists **nothing** from any
            # pipeline that ran off the main thread. Reads have no such catch and raise into
            # planning. Disabling the check hands the serialization duty to this class, and
            # the lock discharges it.
            self._conn = sqlite3.connect(uri, check_same_thread=False)
            self._lock = threading.Lock()
            if uri != ":memory:":
                with contextlib.suppress(OSError):
                    os.chmod(uri, 0o600)
            self._tune()
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

    def _tune(self) -> None:
        """Apply the durability/throughput settings this store's access pattern wants.

        The write pattern is many small independent `put`s — one per operator per query —
        each of which commits. Under the rollback journal at `synchronous=FULL` that is a
        journal file created, an fsync, and the file deleted, *per operator*, which turns
        learning from execution into a measurable tax on every query and is the reason a
        durable store looked slower than learning nothing.

        WAL plus `synchronous=NORMAL` keeps the writes append-only and lets the OS batch the
        flushes. The trade is precise and acceptable here: a power loss or OS crash can lose
        the last few commits. These rows are *learned statistics* — losing the newest few
        means the next run's plan is fitted on slightly less history, and the loop re-converges
        within a handful of queries. Nothing a user asked to be stored is at stake, so paying
        a per-operator fsync to protect it is the wrong trade.

        Best-effort: `journal_mode=WAL` is refused on some network filesystems and on a
        read-only database, and neither is a reason to fail construction — the store simply
        keeps the stricter default.
        """
        for pragma in ("journal_mode=WAL", "synchronous=NORMAL"):
            try:
                self._conn.execute(f"PRAGMA {pragma}")
            except sqlite3.Error as exc:  # a filesystem or mode that will not take it
                note_suppressed("metadata", f"set sqlite {pragma}", exc)

    def __repr__(self) -> str:
        """Name the database file, so two hubs on different stores are distinguishable."""
        return f"SQLiteBackend(uri={self._uri!r})"

    def get(self, table: str, key: Key) -> bytes | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM kv WHERE tbl = ? AND key = ?", (table, encode_key(key))
            ).fetchone()
        return row[0] if row else None

    def put(self, table: str, key: Key, value: bytes) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (tbl, key, value) VALUES (?, ?, ?)",
                (table, encode_key(key), value),
            )
            self._conn.commit()

    def scan(self, table: str, prefix: Key = ()) -> Iterator[tuple[Key, bytes]]:
        """Every `(key, value)` under `prefix`, newest-agnostic, ordered by encoded key.

        The prefix is pushed into SQL as a range over the `(tbl, key)` primary key rather
        than applied in Python after reading the table. That matters because of what shares a
        table: `learned_params` holds every namespace at once, and source statistics take one
        namespace *per source path*, so a session that has read a few thousand files has a
        few thousand namespaces in there — each with a blob carrying bounds, blooms, and
        quantile grids. Reading and `decode_key`-ing all of them to answer
        `load_keyed_params("kyber.calibration")`, on every query, was the dominant cost of a
        durable store.

        The range never *excludes* a matching key, which is the property that has to hold.
        `encode_key` is JSON with compact separators, so every key extending `prefix` begins
        with `encode_key(prefix)` minus its closing `]`, and incrementing that string's last
        character gives an upper bound above all of them. For a string-valued prefix the range
        is also exact, because the prefix ends at the closing quote of its last element and no
        longer element can share it. For a *numeric* prefix it is a superset — the range for
        `(5,)` also spans `[50]` — so the tuple comparison below stays as the authority rather
        than an assertion. A no-op for the empty prefix, which scans the table as before.
        """
        sql = "SELECT key, value FROM kv WHERE tbl = ?"
        params: list[object] = [table]
        low = _prefix_bound(prefix)
        if low is not None:
            sql += " AND key >= ? AND key < ?"
            params += [low, low[:-1] + chr(ord(low[-1]) + 1)]
        with self._lock:
            rows = self._conn.execute(sql + " ORDER BY key", params).fetchall()
        plen = len(prefix)
        for enc_key, value in rows:
            key = decode_key(enc_key)
            # The range is exact for well-formed keys; the tuple check stays as the authority
            # so a key written by another encoding can never be mis-attributed to a namespace.
            if key[:plen] == prefix:
                yield key, value

    def batch_put(self, table: str, items: list[tuple[Key, bytes]]) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO kv (tbl, key, value) VALUES (?, ?, ?)",
                [(table, encode_key(k), v) for k, v in items],
            )
            self._conn.commit()

    def delete(self, table: str, keys: list[Key]) -> None:
        """Drop `keys` from `table`; absent keys are ignored.

        Optional beyond the four `MetadataBackend` methods, and offered here because
        "durable" is not the same as "unbounded". The hub appends one `op_stats` row per
        operator per query and prunes to its newest window through this when a backend has
        it; without it, a store that a served workload writes to for months grows a row per
        operator per query forever, and every process that opens it pays to scan the lot on
        its first view load. The window the hub keeps is far larger than anything the fitted
        models read, so pruning is invisible to them.
        """
        if not keys:
            return
        with self._lock:
            self._conn.executemany(
                "DELETE FROM kv WHERE tbl = ? AND key = ?",
                [(table, encode_key(k)) for k in keys],
            )
            self._conn.commit()
