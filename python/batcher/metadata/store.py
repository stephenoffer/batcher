"""The pluggable persistence abstraction behind the MetadataHub.

A `MetadataBackend` is a simple keyed blob store partitioned into logical
"tables" (query_trace, op_stats, column_stats, learned_params, ...). Keeping the
surface this small lets every backend — an in-process dict for tests, SQLite for
single-node durability, Redis or cloud object storage for a shared cluster —
implement it trivially, and lets a `LayeredBackend` compose a fast local cache
over a durable shared store. None of them depends on Ray.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

# Keys are tuples of scalars (e.g. (table_id, column, version)); backends decide
# how to encode them. Values are opaque bytes (callers serialize their own rows).
Key = tuple[object, ...]

__all__ = ["Key", "MetadataBackend"]


@runtime_checkable
class MetadataBackend(Protocol):
    """A keyed blob store partitioned into named tables."""

    def get(self, table: str, key: Key) -> bytes | None: ...

    def put(self, table: str, key: Key, value: bytes) -> None: ...

    def scan(self, table: str, prefix: Key = ()) -> Iterator[tuple[Key, bytes]]: ...

    def batch_put(self, table: str, items: list[tuple[Key, bytes]]) -> None: ...
