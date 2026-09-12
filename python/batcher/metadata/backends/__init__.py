"""MetadataHub persistence backends.

`InProcessBackend` (tests / single-process, and the configured default) and `SQLiteBackend`
(local durable) are built in; `RocksDBBackend` is the embedded alternative for a write-heavy
single-node loop, `ObjectStorageBackend` / `RedisBackend` share statistics across a cluster,
and `LayeredBackend` caches one of those behind a local dict — all behind the same
`MetadataBackend` protocol, so the Hub never changes.

`SQLiteBackend` is resolved on first access rather than at import, like every other backend
but `InProcessBackend` already was. Importing it eagerly made the stdlib `sqlite3` module a
precondition of *every* query — the executor imports this package to build its default
in-process store — and `sqlite3` is not always loadable: on a common Anaconda install with a
pip-installed `pyarrow`, pyarrow binds the system `libstdc++` first and Anaconda's `_sqlite3`
then fails to load its ICU dependency (`CXXABI_1.3.15 not found`). Every query failed there,
including every example, on a deployment that had never asked for SQLite.
"""

from __future__ import annotations

from typing import Any

from batcher.metadata.backends.factory import (
    BACKEND_NAMES,
    default_sqlite_uri,
    make_backend,
)
from batcher.metadata.backends.in_process import InProcessBackend

__all__ = [
    "BACKEND_NAMES",
    "InProcessBackend",
    "SQLiteBackend",
    "default_sqlite_uri",
    "make_backend",
]


def __getattr__(name: str) -> Any:
    """Import `SQLiteBackend` on first access (PEP 562), so `sqlite3` loads only when used."""
    if name != "SQLiteBackend":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from batcher.metadata.backends.sqlite import SQLiteBackend

    globals()[name] = SQLiteBackend  # cache, so the indirection is paid once
    return SQLiteBackend


def __dir__() -> list[str]:
    return sorted(__all__)
