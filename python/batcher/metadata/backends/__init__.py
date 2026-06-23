"""MetadataHub persistence backends.

`InProcessBackend` (tests / single-process) and `SQLiteBackend` (local durable
default) are built in; Redis and cloud-object-storage backends slot in behind the
same `MetadataBackend` protocol without touching the Hub.
"""

from __future__ import annotations

from batcher.metadata.backends.in_process import InProcessBackend
from batcher.metadata.backends.sqlite import SQLiteBackend

__all__ = ["InProcessBackend", "SQLiteBackend", "make_backend"]


def make_backend(name: str, uri: str | None = None):
    """Construct a backend by config name."""
    if name == "in_process":
        return InProcessBackend()
    if name == "sqlite":
        return SQLiteBackend(uri or ":memory:")
    raise ValueError(f"unknown metadata backend: {name!r}")
