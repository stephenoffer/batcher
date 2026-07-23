"""MetadataHub persistence backends.

`InProcessBackend` (tests / single-process) and `SQLiteBackend` (local durable
default) are built in; `ObjectStorageBackend` / `RedisBackend` share statistics
across a cluster, and `LayeredBackend` caches one of those behind a local dict — all
behind the same `MetadataBackend` protocol, so the Hub never changes.
"""

from __future__ import annotations

import os

from batcher._internal.errors import ConfigError, unknown_value
from batcher.metadata.backends.in_process import InProcessBackend
from batcher.metadata.backends.sqlite import SQLiteBackend
from batcher.metadata.store import MetadataBackend

__all__ = [
    "BACKEND_NAMES",
    "InProcessBackend",
    "SQLiteBackend",
    "default_sqlite_uri",
    "make_backend",
]

#: Every name `make_backend` accepts. The one source of truth for the set, so the
#: "unknown backend" error can never offer a name the factory does not build.
BACKEND_NAMES: tuple[str, ...] = ("in_process", "sqlite", "object_storage", "redis", "layered")


def default_sqlite_uri() -> str:
    """A stable on-disk location for the learned-stats store.

    So that `backend="sqlite"` *persists across restarts* with no path to manage —
    the one-liner that turns on cross-run learning (plans keep improving every time a
    query runs, even after a restart). Honors ``$BATCHER_HOME``, else a per-user
    ``~/.batcher`` directory; the directory is created if absent. Pass an explicit
    ``uri`` (including ``":memory:"`` for an ephemeral store) to override.

    Returns:
        The path to the learned-stats database file.

    Raises:
        ConfigError: If the directory cannot be created. A bare `FileNotFoundError`
            naming a parent path the user never typed is not a usable answer here.
    """
    from_env = os.environ.get("BATCHER_HOME")
    base = from_env or os.path.join(os.path.expanduser("~"), ".batcher")
    try:
        os.makedirs(base, exist_ok=True)
    except OSError as exc:
        chose = "$BATCHER_HOME" if from_env else "the default learned-stats directory"
        raise ConfigError(
            f"Cannot create {chose} at {base!r}: {exc.strerror or exc}.",
            hint=(
                "Point $BATCHER_HOME at a writable directory, or set metadata.uri to "
                "an explicit path (':memory:' for an ephemeral store)."
            ),
        ) from exc
    return os.path.join(base, "metadata.db")


def make_backend(name: str, uri: str | None = None) -> MetadataBackend:
    """Construct a metadata backend by config name.

    Args:
        name: One of `BACKEND_NAMES`.
        uri: Where the backend stores its data. Required by ``object_storage`` and
            ``redis``; optional for ``sqlite``, which defaults to a per-user file.

    Returns:
        A `MetadataBackend` ready to hand to a `MetadataHub`.

    Raises:
        ConfigError: If `name` is not a known backend. The error suggests the closest
            match and lists every accepted name.
    """
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
    raise unknown_value(
        ConfigError,
        "metadata backend",
        name,
        BACKEND_NAMES,
        label="Available backends",
        hint="It is the metadata.backend setting in your Batcher config.",
    )
