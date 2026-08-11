"""MetadataHub persistence backends.

`InProcessBackend` (tests / single-process) and `SQLiteBackend` (local durable
default) are built in; `ObjectStorageBackend` / `RedisBackend` share statistics
across a cluster, and `LayeredBackend` caches one of those behind a local dict — all
behind the same `MetadataBackend` protocol, so the Hub never changes.
"""

from __future__ import annotations

from batcher.metadata.backends.factory import (
    BACKEND_NAMES,
    default_sqlite_uri,
    make_backend,
)
from batcher.metadata.backends.in_process import InProcessBackend
from batcher.metadata.backends.sqlite import SQLiteBackend

__all__ = [
    "BACKEND_NAMES",
    "InProcessBackend",
    "SQLiteBackend",
    "default_sqlite_uri",
    "make_backend",
]
