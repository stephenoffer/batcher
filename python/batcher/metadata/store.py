"""The pluggable persistence abstraction behind the MetadataHub.

A `MetadataBackend` is a simple keyed blob store partitioned into logical
"tables" (query_trace, op_stats, column_stats, learned_params, ...). Keeping the
surface this small lets every backend — an in-process dict for tests, SQLite for
single-node durability, Redis or cloud object storage for a shared cluster —
implement it trivially, and lets a `LayeredBackend` compose a fast local cache
over a durable shared store. None of them depends on Ray.

The protocol is structural, which is what makes a custom backend a twenty-line class —
and also what makes a *near*-miss (a `put` spelled `set`, a backend passed where a URI
belongs) surface as `AttributeError: 'X' object has no attribute 'get'` from deep inside
the hub, on whichever query happened to read first. `check_backend` and `require_uri`
turn both into a typed error at the point of construction, naming what is missing.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from batcher._internal.errors import ConfigError

# Keys are tuples of scalars (e.g. (table_id, column, version)); backends decide
# how to encode them. Values are opaque bytes (callers serialize their own rows).
Key = tuple[object, ...]

__all__ = ["Key", "MetadataBackend", "check_backend", "require_uri"]

#: The methods a `MetadataBackend` must provide, in the order they are documented.
BACKEND_METHODS: tuple[str, ...] = ("get", "put", "scan", "batch_put")


@runtime_checkable
class MetadataBackend(Protocol):
    """A keyed blob store partitioned into named tables."""

    def get(self, table: str, key: Key) -> bytes | None: ...

    def put(self, table: str, key: Key, value: bytes) -> None: ...

    def scan(self, table: str, prefix: Key = ()) -> Iterator[tuple[Key, bytes]]: ...

    def batch_put(self, table: str, items: list[tuple[Key, bytes]]) -> None: ...


def check_backend(backend: object, *, role: str = "metadata backend") -> MetadataBackend:
    """Verify `backend` implements the `MetadataBackend` protocol, or raise.

    Checked once at construction rather than discovered on the first read, so a typo in
    a custom backend is reported against the line that built it instead of against an
    unrelated query minutes later.

    Args:
        backend: The candidate backend.
        role: How to name it in the error, e.g. ``"shared store"``.

    Returns:
        `backend` unchanged, so this can wrap an assignment.

    Raises:
        ConfigError: If any required method is missing or not callable, naming exactly
            which ones.
    """
    missing = [m for m in BACKEND_METHODS if not callable(getattr(backend, m, None))]
    if missing:
        raise ConfigError(
            f"{type(backend).__name__} cannot be used as a {role}: it is missing "
            f"{', '.join(repr(m) for m in missing)}.",
            hint=(
                "A metadata backend must define "
                f"{', '.join(repr(m) for m in BACKEND_METHODS)} — see "
                "batcher.metadata.store.MetadataBackend."
            ),
        )
    return backend  # type: ignore[return-value]


def require_uri(backend: str, uri: str | None, *, example: str) -> str:
    """The backend's URI, or a typed error showing what one looks like.

    Args:
        backend: The backend name, as it is written in config.
        uri: The URI the user supplied, if any.
        example: A concrete, copyable example URI for this backend.

    Returns:
        `uri`, guaranteed to be a non-empty string.

    Raises:
        ConfigError: If `uri` is missing, empty, or not a string.
    """
    if not uri or not isinstance(uri, str):
        raise ConfigError(
            f"The {backend} metadata backend needs a uri, but got {uri!r}.",
            hint=f"Set metadata.uri, for example {example!r}.",
        )
    return uri
