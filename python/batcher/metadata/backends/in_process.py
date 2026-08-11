"""In-process dict backend — for tests and single-process runs."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from batcher.metadata.store import Key

__all__ = ["InProcessBackend"]


class InProcessBackend:
    """A `MetadataBackend` backed by nested dicts. Not durable across processes."""

    def __init__(self) -> None:
        # A slot holds either the encoded `bytes` a `put` supplied, or the raw row dict a
        # `put_row` deferred. `_encoded` collapses the difference at read time.
        self._tables: dict[str, dict[Key, bytes | dict[str, Any]]] = {}

    def __repr__(self) -> str:
        """Summarize what is stored, per table — the question asked when learning looks cold."""
        held = ", ".join(f"{t}={len(rows)}" for t, rows in sorted(self._tables.items()))
        return f"InProcessBackend({held or 'empty'})"

    def get(self, table: str, key: Key) -> bytes | None:
        rows = self._tables.get(table)
        if rows is None or key not in rows:
            return None
        return self._encoded(rows, key)

    def put(self, table: str, key: Key, value: bytes) -> None:
        self._tables.setdefault(table, {})[key] = value

    def put_row(self, table: str, key: Key, row: dict[str, Any]) -> None:
        """Store an already-structured row, deferring the JSON encoding to a read.

        The optional half of the backend protocol, and the reason it is worth having: this
        backend is a dict in the running process, so `put(json.dumps(row).encode())`
        serializes a flat dict of scalars purely so that a later read can parse it straight
        back. Every terminal op records one row per operator, and nothing reads them until
        a view loads (once per process), so the encoding was paid on the hot path for a
        result almost always thrown away — 30-40 us of a small query's ~1 ms.

        A backend that must actually transmit or persist bytes does not implement this, and
        `MetadataHub` falls back to `put` for it, so durability is unaffected.

        Args:
            table: The logical table to write into.
            key: The row's key.
            row: A JSON-serializable flat mapping. Held by reference, so the caller must
                not mutate it after handing it over.
        """
        self._tables.setdefault(table, {})[key] = row

    @staticmethod
    def _encoded(rows: dict[Key, bytes | dict[str, Any]], key: Key) -> bytes:
        """The slot's bytes, encoding and caching a deferred row in place on first read."""
        value = rows[key]
        if type(value) is bytes:
            return value
        encoded = json.dumps(value).encode()
        rows[key] = encoded
        return encoded

    def scan(self, table: str, prefix: Key = ()) -> Iterator[tuple[Key, bytes]]:
        """Every `(key, value)` under `prefix`, from a snapshot taken when the scan starts.

        The snapshot is the point. This is a generator over a live dict, and the hub's callers
        consume it lazily — a view load parses each row as it arrives — while `record` is
        writing to the same table from another pipeline and `_prune_op_stats` is deleting from
        it. Yielding straight out of `dict.items()` therefore raises `RuntimeError: dictionary
        changed size during iteration` in whichever query happened to be reading, which is not
        a metadata failure the caller is prepared for: the hub's `record` swallows exceptions,
        but a *read* raises into planning.

        Copying the item list is O(entries) in pointers and happens at most once per view per
        process, against a table whose reads are rare and whose writes are frequent.
        """
        rows = self._tables.get(table, {})
        for key, _value in list(rows.items()):
            if key[: len(prefix)] == prefix and key in rows:
                yield key, self._encoded(rows, key)

    def batch_put(self, table: str, items: list[tuple[Key, bytes]]) -> None:
        dst = self._tables.setdefault(table, {})
        for key, value in items:
            dst[key] = value

    def delete(self, table: str, keys: list[Key]) -> None:
        """Drop `keys` from `table`; absent keys are ignored.

        Beyond the four methods `MetadataBackend` requires, and deliberately optional: a
        *durable* store accumulating history across runs is the point of it, but this
        backend is a dict in the running process, so without a way to forget, the operator
        feedback every terminal op records would grow the process for its whole life. The
        hub prunes through this when a backend offers it (see `MetadataHub._prune_op_stats`)
        and leaves a backend without it untouched.
        """
        rows = self._tables.get(table)
        if rows is None:
            return
        for key in keys:
            rows.pop(key, None)

    def clear(self) -> None:
        """Drop everything stored, keeping this object's identity.

        What `LayeredBackend.refresh` needs. Rebinding to a fresh backend would work for
        the default cache and quietly discard a *caller-supplied* one, replacing it with a
        plain dict — so a cache with a size bound, an eviction policy, or shared state
        silently stops being used after the first refresh.
        """
        self._tables.clear()
