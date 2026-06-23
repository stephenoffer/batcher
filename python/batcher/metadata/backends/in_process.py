"""In-process dict backend — for tests and single-process runs."""

from __future__ import annotations

from collections.abc import Iterator

from batcher.metadata.store import Key

__all__ = ["InProcessBackend"]


class InProcessBackend:
    """A `MetadataBackend` backed by nested dicts. Not durable across processes."""

    def __init__(self) -> None:
        self._tables: dict[str, dict[Key, bytes]] = {}

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
