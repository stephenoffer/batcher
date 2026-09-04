"""Building the shared result cache from its config URI, once per process.

The URI scheme picks the backend, the way `metadata.backends.factory` does for the
learned-stats store and for the same reason: one setting, no second "which backend"
option to keep consistent with it.

    redis:// · rediss:// · unix://   -> RedisSharedCache   (shared across processes/nodes)
    rocksdb:// or a bare path        -> RocksDBSharedCache (one node, across runs)

A construction failure disables the shared cache rather than failing the process. That is
the same judgement the rest of this package makes: the cache is an optimization, and a
misconfigured or unreachable one should cost recomputes, not a startup crash. It is logged
once, which is what keeps "disabled" from being silent.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from batcher._internal.logging import note_suppressed
from batcher.carbonite.cache_shared.store import SharedResultCache
from batcher.config import active_config

if TYPE_CHECKING:
    from batcher.carbonite.cache_shared.base import SharedCache

__all__ = ["current_shared_cache", "reset_shared_cache", "shared_cache"]

#: URI prefixes routed to Redis. The same three `redis.Redis.from_url` accepts, so the
#: check here and the one in the backend cannot disagree about what a Redis URI is.
_REDIS_SCHEMES = ("redis://", "rediss://", "unix://")
#: The explicit scheme for the embedded backend. A bare path also routes there, because
#: that is what a user types, but the scheme exists so the intent can be written down.
_ROCKSDB_SCHEME = "rocksdb://"

_cache: SharedResultCache | None = None
_cache_uri: str | None = None
_lock = threading.Lock()


def _backend(uri: str) -> SharedCache:
    """The backend `uri` names."""
    if uri.startswith(_REDIS_SCHEMES):
        from batcher.carbonite.cache_shared.redis import RedisSharedCache

        return RedisSharedCache(uri)
    from batcher.carbonite.cache_shared.rocksdb import RocksDBSharedCache

    path = uri.removeprefix(_ROCKSDB_SCHEME)
    return RocksDBSharedCache(path)


def shared_cache() -> SharedResultCache | None:
    """The process's shared result cache, or `None` when none is configured or reachable.

    Built once from `MemoryConfig.shared_cache_uri` and rebuilt if that setting changes,
    so a `config_context` that points at a different store is honored rather than served
    by the previous one. `None` is the default: sharing results between processes is
    opt-in, because it needs a store to point at and a store is a deployment decision.

    Returns:
        The shared cache, or `None`.
    """
    global _cache, _cache_uri
    mem = active_config().memory
    uri = mem.shared_cache_uri
    if not uri:
        return None
    if _cache is not None and _cache_uri == uri:
        return _cache
    with _lock:
        if _cache is not None and _cache_uri == uri:
            return _cache
        if _cache is not None:
            _cache.close()
            _cache = None
        try:
            _cache = SharedResultCache(_backend(uri), mem.shared_cache_ttl_seconds or None)
        except Exception as exc:
            # Logged once, then remembered as "this URI does not work" so a broken store
            # is not re-dialled on every query.
            note_suppressed("carbonite", f"open the shared result cache at {uri!r}", exc)
            _cache = None
        _cache_uri = uri
        return _cache


def current_shared_cache() -> SharedResultCache | None:
    """The shared result cache if one has been built, without building one.

    Returns:
        The cache, or `None`.
    """
    return _cache


def reset_shared_cache() -> None:
    """Close and forget the shared cache so the next call builds a fresh one.

    For tests, and for a process that has changed which store it should be talking to.
    """
    global _cache, _cache_uri
    with _lock:
        stale, _cache, _cache_uri = _cache, None, None
    if stale is not None:
        stale.close()
