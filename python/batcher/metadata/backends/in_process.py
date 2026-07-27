"""In-process dict backend — for tests and single-process runs."""

from __future__ import annotations

from collections.abc import Iterator

from batcher.metadata.store import Key

__all__ = ["InProcessBackend"]


class InProcessBackend:
    """A `MetadataBackend` backed by nested dicts. Not durable across processes."""

    def __init__(self) -> None:
        self._tables: dict[str, dict[Key, bytes]] = {}

    def __repr__(self) -> str:
        """Summarize what is stored, per table — the question asked when learning looks cold."""
        held = ", ".join(f"{t}={len(rows)}" for t, rows in sorted(self._tables.items()))
        return f"InProcessBackend({held or 'empty'})"

    def get(self, table: str, key: Key) -> bytes | None:
        return self._tables.get(table, {}).get(key)

    def put(self, table: str, key: Key, value: bytes) -> None:
        self._tables.setdefault(table, {})[key] = value

    def scan(self, table: str, prefix: Key = ()) -> Iterator[tuple[Key, bytes]]:
        for key, value in self._tables.get(table, {}).items():
            if key[: len(prefix)] == prefix:
                yield key, value

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
