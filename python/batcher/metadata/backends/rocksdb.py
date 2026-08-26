"""RocksDB backend — an embedded durable store for a write-heavy learned-stats loop.

The same contract `SQLiteBackend` implements, on a different storage engine, for the
case SQLite is worst at: many small writes. Core records execution feedback after every
query and Kyber reads it back before every plan, so the learned-stats store sees a
sustained write stream on a single node. SQLite answers that with a B-tree update and a
journal write per transaction; RocksDB's log-structured merge tree turns it into an
append to a memtable, which is what an LSM is for.

The encoding is `metadata.store`'s, unchanged and deliberately so. `encode_key` is
prefix-preserving and orders the same way a byte comparison does, which is exactly the
property RocksDB's ordered iteration needs — so `scan(prefix)` seeks straight to the
first matching key and stops at the first non-matching one, rather than reading the
table. The physical key is ``table\\x00encoded``, so one column family holds every table
and a table's keys are contiguous.

``rocksdict`` is an optional dependency (the ``rocksdb`` extra); importing this module
without it raises a clear `MissingDependencyError` rather than an `ImportError` naming a
package the user never typed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import ConfigError
from batcher._internal.optional import require
from batcher._internal.paths import private_dir
from batcher.metadata.store import Key, decode_key, encode_key, require_uri

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["RocksDBBackend"]

#: Separates the table name from the encoded key in the physical key. NUL cannot appear in
#: either half — a table name is an identifier and `encode_key` emits JSON — so the split is
#: unambiguous, and it sorts below every printable byte, which keeps each table's keys
#: contiguous and in the encoded order `scan` relies on.
_SEP = b"\x00"


def _physical(table: str, key: Key) -> bytes:
    """The stored key for `(table, key)`."""
    return table.encode("utf-8") + _SEP + encode_key(key).encode("utf-8")


def _seek_prefix(table: str, prefix: Key) -> bytes:
    """The byte prefix every key under `(table, prefix)` starts with.

    For the empty prefix this is the table's own prefix, so a full scan is still a seek
    plus a walk rather than a read of the whole database. For a non-empty one it is
    `encode_key(prefix)` with its closing `]` dropped — the same trick `SQLiteBackend` and
    `RedisBackend` use, and for the same reason: `encode_key(prefix)` continues with `]`
    and `encode_key(prefix + rest)` continues with `,`, so both sort inside the range.
    """
    head = table.encode("utf-8") + _SEP
    if not prefix:
        return head
    return head + encode_key(prefix)[:-1].encode("utf-8")


def _split(physical: bytes, table: str) -> Key | None:
    """The logical `Key` a physical key encodes, or `None` if it belongs to another table.

    The `None` case is what terminates a scan: iteration is ordered, so the first key that
    does not carry this table's prefix is past every key that does.
    """
    head = table.encode("utf-8") + _SEP
    if not physical.startswith(head):
        return None
    return decode_key(physical[len(head) :].decode("utf-8"))


class RocksDBBackend:
    """A `MetadataBackend` backed by an embedded RocksDB database directory."""

    __slots__ = ("_db", "_path")

    def __init__(self, uri: str | None) -> None:
        """Open (or create) the RocksDB database at `uri`.

        Args:
            uri: A filesystem *directory* path. RocksDB owns the whole directory, unlike
                SQLite's single file, so pointing this at an existing non-RocksDB
                directory is an error the driver reports rather than a store it adopts.

        Raises:
            ConfigError: If `uri` is missing, is not a string, or cannot be opened.
            MissingDependencyError: If the ``rocksdict`` package is not installed.
        """
        path = require_uri("rocksdb", uri, example="/var/lib/batcher/stats.rocksdb")
        rocksdict = require(
            "rocksdict",
            feature="the rocksdb metadata backend",
            provides="RocksDB",
            extra="rocksdb",
        )
        # The learned-stats store holds real column `min`/`max` values out of the user's
        # data, so the directory is owner-only — the same reason `SQLiteBackend` chmods
        # its file, and it has to happen before RocksDB creates anything inside.
        private_dir(path)
        try:
            self._db: Any = rocksdict.Rdict(path)
        except Exception as exc:
            raise ConfigError(
                f"Cannot open the RocksDB metadata store at {path!r}: {exc}.",
                hint=(
                    "Point metadata.uri at a writable directory this process owns. A "
                    "RocksDB database owns its whole directory, and only one process may "
                    "hold it open at a time — use the redis or object_storage backend to "
                    "share statistics between drivers."
                ),
            ) from exc
        self._path = path

    def get(self, table: str, key: Key) -> bytes | None:
        """Return the stored value for `(table, key)`, or `None`.

        Args:
            table: The logical table name.
            key: The key tuple.

        Returns:
            The stored bytes, or `None` if the key is absent.
        """
        value = self._db.get(_physical(table, key))
        return bytes(value) if value is not None else None

    def put(self, table: str, key: Key, value: bytes) -> None:
        """Store `value` under `(table, key)`, replacing any previous value.

        Args:
            table: The logical table name.
            key: The key tuple.
            value: The opaque bytes to store.
        """
        self._db[_physical(table, key)] = value

    def batch_put(self, table: str, items: list[tuple[Key, bytes]]) -> None:
        """Store many `(key, value)` pairs in one atomic write.

        One `WriteBatch` rather than a loop of writes: the loop pays a WAL append and an
        fsync decision per item, which is the whole cost when the values are the few dozen
        bytes a statistic occupies.

        Args:
            table: The logical table name.
            items: The pairs to write. An empty list is a no-op.
        """
        if not items:
            return
        rocksdict = require(
            "rocksdict",
            feature="the rocksdb metadata backend",
            provides="RocksDB",
            extra="rocksdb",
        )
        batch = rocksdict.WriteBatch()
        for key, value in items:
            batch.put(_physical(table, key), value)
        self._db.write(batch)

    def scan(self, table: str, prefix: Key = ()) -> Iterator[tuple[Key, bytes]]:
        """Yield every `(key, value)` in `table` whose key extends `prefix`.

        Seeks to the prefix and walks forward, stopping at the first key outside it, so
        the cost is proportional to what the prefix matches rather than to the table.

        Args:
            table: The logical table name.
            prefix: A key prefix; the empty tuple scans the whole table.

        Yields:
            Each matching key tuple and its stored bytes, in encoded-key order.
        """
        seek = _seek_prefix(table, prefix)
        iterator = self._db.iter()
        iterator.seek(seek)
        while iterator.valid():
            physical = bytes(iterator.key())
            if not physical.startswith(seek):
                return
            logical = _split(physical, table)
            if logical is not None:
                yield logical, bytes(iterator.value())
            iterator.next()

    def close(self) -> None:
        """Flush and close the database, releasing the directory lock.

        RocksDB holds an exclusive lock on its directory for as long as it is open, so a
        process that means to hand the store to another one has to say when it is done.
        Idempotent.
        """
        db, self._db = self._db, None
        if db is not None:
            db.close()
