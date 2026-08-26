"""The shared (cross-process, cross-node) result cache.

`carbonite.cache` holds results for the life of one process; this holds them past it, in
Redis or in an embedded RocksDB database. The two are separate stores rather than tiers of
one, because they cannot share a key: the process cache keys on each input's object
identity, which is exact inside a process and meaningless outside it, while a shared entry
must be keyed on *content* — an input's durable identity plus its version token — so a
rewritten table cannot serve the previous run's rows.

`shareable_key` is where that rule is enforced, and it returns `None` for anything it
cannot prove. Re-exports only; the logic lives in the sibling modules.
"""

from __future__ import annotations

from batcher.carbonite.cache_shared.base import (
    SharedCache,
    SharedCacheError,
    entry_digest,
    shareable_key,
)
from batcher.carbonite.cache_shared.factory import (
    current_shared_cache,
    reset_shared_cache,
    shared_cache,
)
from batcher.carbonite.cache_shared.store import SharedResultCache

__all__ = [
    "SharedCache",
    "SharedCacheError",
    "SharedResultCache",
    "current_shared_cache",
    "entry_digest",
    "reset_shared_cache",
    "shareable_key",
    "shared_cache",
]
