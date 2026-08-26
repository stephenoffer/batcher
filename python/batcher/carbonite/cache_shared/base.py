"""What a shared result cache is, and the one rule that makes sharing one sound.

The process cache (`carbonite.cache`) dies with the process, and its key uses each input
source's *object identity* — which is the right key inside one process and meaningless
across two. A shared cache is the other half: a store outside the process that a second
driver, a Ray worker, or tomorrow's run can read. That buys the case a process cache
cannot touch at all, which is the dashboard or scheduled job that re-issues the same
query from a fresh interpreter every time.

It also introduces the one way a cache can be *wrong* rather than merely cold, so the
rule is worth stating plainly:

    **A result may be shared only when every input has a durable identity and a content
    version.** A file path says which relation this is; a version token says which state
    of it. Without the second, a rewritten file serves the previous run's rows, and no
    error is raised because a wrong answer that arrives quickly looks exactly like a
    right one.

`shareable_key` enforces that and returns `None` when it cannot be met — for in-memory
data (no cross-run identity at all), for a source that cannot version itself, and for a
source that declines to say. Declining costs a recompute. Not declining costs correctness,
so nothing here guesses.

The backends are keyed blob stores and nothing more: `get`, `put`, `delete`. Entry
serialization is `plan.types.ipc`, and eviction belongs to the store — Redis has
`maxmemory-policy`, RocksDB has compaction and a TTL — because a shared store is shared
with other tenants and a client that evicted on their behalf would be wrong about the
budget it was enforcing.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

__all__ = ["SharedCache", "SharedCacheError", "entry_digest", "shareable_key"]


class SharedCacheError(Exception):
    """Raised inside a backend when the store is unreachable or answers unusably.

    Never propagated to a query: every call site treats a shared-cache failure as a miss,
    because a cache that is down must slow a query down rather than fail it. The type
    exists so that treatment can be *deliberate* — a bare `except Exception` around a
    store call swallows the bugs in this package along with the network.
    """


@runtime_checkable
class SharedCache(Protocol):
    """A keyed blob store outside this process, holding serialized query results."""

    def get(self, key: str) -> bytes | None:
        """Return the bytes stored under `key`, or `None` on a miss."""
        ...

    def put(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None:
        """Store `value` under `key`, expiring after `ttl_seconds` when the store can."""
        ...

    def delete(self, key: str) -> None:
        """Remove `key` if present."""
        ...

    def close(self) -> None:
        """Release the connection or database handle."""
        ...


def entry_digest(*parts: str) -> str:
    """A short, stable, filesystem- and Redis-safe key from arbitrary text parts.

    Hashed rather than concatenated for two reasons that both matter at this boundary: a
    plan signature carries user column names and literal values, which should not be
    written verbatim into a store shared with other tenants; and Redis, RocksDB and every
    object store have their own opinions about separators and metacharacters that a raw
    signature would trip over.

    SHA-256 truncated to 32 hex characters (128 bits). A collision needs ~2^64 distinct
    cached queries before it is even likely, which is far past any real cache, and the
    consequence bounds itself anyway — the two would have to be different queries over
    the same versioned inputs in the same tenant.

    Args:
        parts: The components of the identity, joined with a separator that cannot appear
            in a digest.

    Returns:
        The 32-character hex digest.
    """
    joined = "\x1f".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:32]


def shareable_key(signature: str, sources: list[object], scope: str) -> str | None:
    """A cross-process cache key for this query, or `None` if it must not be shared.

    `None` is the common answer and the safe one. It means at least one input cannot say
    *which state of itself* the result was computed from, so a cached result could not be
    invalidated when that input changed.

    Args:
        signature: The plan signature — what was computed.
        sources: The bound input sources, in scan order — what it was computed from.
        scope: The tenant and viewer the result belongs to, so two tenants issuing the
            same query over the same path cannot read each other's rows. A shared store
            makes that failure cross-*process*, which is the reason this is not optional.

    Returns:
        The key, or `None` when any source lacks a durable identity or a content version.
    """
    from batcher.plan.source_stats import content_version, stable_source_key

    versions: list[str] = []
    for source in sources:
        identity = stable_source_key(source)
        if not identity:
            return None  # in-memory data, or a source with no cross-run identity
        version = content_version(source)
        if version is None:
            return None  # cannot notice a later change, so must not be shared
        versions.append(f"{identity}@{version}")
    return entry_digest(signature, "|".join(versions), scope)
