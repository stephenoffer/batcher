"""MetadataHub persistence backends.

`InProcessBackend` (tests / single-process) and `SQLiteBackend` (local durable
default) are built in; `ObjectStorageBackend` / `RedisBackend` share statistics
across a cluster, and `LayeredBackend` caches one of those behind a local dict — all
behind the same `MetadataBackend` protocol, so the Hub never changes.
"""

from __future__ import annotations

import os

from batcher.metadata.backends.in_process import InProcessBackend
from batcher.metadata.backends.sqlite import SQLiteBackend

__all__ = ["InProcessBackend", "SQLiteBackend", "default_sqlite_uri", "make_backend"]


def default_sqlite_uri() -> str:
    """A stable on-disk location for the learned-stats store.

    So that `backend="sqlite"` *persists across restarts* with no path to manage —
    the one-liner that turns on cross-run learning (plans keep improving every time a
    query runs, even after a restart). Honors ``$BATCHER_HOME``, else a per-user
    ``~/.batcher`` directory; the directory is created if absent. Pass an explicit
    ``uri`` (including ``":memory:"`` for an ephemeral store) to override.
    """
    base = os.environ.get("BATCHER_HOME") or os.path.join(os.path.expanduser("~"), ".batcher")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "metadata.db")


def make_backend(name: str, uri: str | None = None):
    """Construct a backend by config name (``in_process``/``sqlite``/``object_storage``/
    ``redis``/``layered``)."""
    if name == "in_process":
        return InProcessBackend()
    if name == "sqlite":
        # No URI → a persistent per-user file (not an ephemeral `:memory:` store, which
        # would silently defeat the point of choosing the durable backend).
        return SQLiteBackend(uri if uri is not None else default_sqlite_uri())
    if name == "object_storage":
        from batcher.metadata.backends.object_storage import ObjectStorageBackend

        return ObjectStorageBackend(uri)
    if name == "redis":
        from batcher.metadata.backends.redis import RedisBackend

        return RedisBackend(uri)
    if name == "layered":
        from batcher.metadata.backends.layered import LayeredBackend

        return LayeredBackend.from_uri(uri)
    raise ValueError(f"unknown metadata backend: {name!r}")
