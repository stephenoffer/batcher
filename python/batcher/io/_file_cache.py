"""Local-SSD read-through file cache (the Disk-Cache analog) for remote reads.

A remote object-store read may be served from a local-SSD copy: the first read of a
remote file streams it here; later reads of the same file hit local disk, sparing the
object-store round-trip. It lives with the filesystem layer that opens files (not in the
``carbonite`` subsystem) because ``core``/``kyber`` depend on ``io``, so an io→carbonite
edge would transitively break their independence; the byte budget comes from config. The
cache is transparent and ephemeral — a miss just re-fetches, never a wrong result.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable

from batcher._internal.paths import private_dir

__all__ = ["FileBytesCache", "get_file_cache"]


class FileBytesCache:
    """A byte-bounded, LRU local-disk cache of whole remote files.

    Keyed by the remote path; the cached copy lives at ``<cache_dir>/<sha256(path)>``.
    Thread-safe. Fetching happens outside the lock (it is slow I/O) into a unique temp
    file that is atomically renamed into place, so concurrent readers never observe a
    half-written file.
    """

    __slots__ = ("_dir", "_entries", "_hits", "_lock", "_max_bytes", "_misses", "_used")

    def __init__(self, cache_dir: str, max_bytes: int) -> None:
        """Create the cache rooted at `cache_dir`, bounded to `max_bytes` on disk."""
        self._dir = cache_dir
        self._max_bytes = max(0, int(max_bytes))
        self._lock = threading.Lock()
        # key → on-disk size; insertion/most-recent order drives LRU eviction.
        self._entries: OrderedDict[str, int] = OrderedDict()
        self._used = 0
        # Hit/miss counters: the warm-vs-cold read win is invisible without them, so a slow
        # scan can't be told from a slow *uncached* scan. Measurement only — never behavior.
        self._hits = 0
        self._misses = 0
        # `private_dir`, not `os.makedirs`: an entry here is a byte-for-byte copy of one of
        # the user's data files, and the cache root is Batcher's own subdirectory of a node
        # volume other tenants also mount. The *file* mode is not ours to set — the caller's
        # `fetch` writes the bytes — so the directory is what protects them.
        private_dir(cache_dir)

    def get_or_fetch(self, remote_path: str, fetch: Callable[[str], None]) -> str:
        """Return the local path of the cached copy of `remote_path`.

        On a miss, `fetch(local_tmp_path)` is called to materialize the bytes (it must
        write the full file to the given path); the result is then admitted under the
        byte budget, evicting the least-recently-used entries if needed.
        """
        key = hashlib.sha256(remote_path.encode("utf-8")).hexdigest()
        local = os.path.join(self._dir, key)
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)  # mark most-recently-used
                self._hits += 1
                return local
            self._misses += 1

        # Miss: fetch outside the lock (slow remote I/O) to a unique temp, then rename.
        tmp = f"{local}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        try:
            fetch(tmp)
            size = os.path.getsize(tmp)
            os.replace(tmp, local)
        except BaseException:
            with contextlib.suppress(OSError):
                os.remove(tmp)
            raise

        with self._lock:
            # A racing thread may have admitted the same key first; only one accounts
            # for the bytes (the file content is identical, so the rename is harmless).
            if key not in self._entries:
                self._entries[key] = size
                self._used += size
                self._evict_locked()
            else:
                self._entries.move_to_end(key)
        return local

    def _evict_locked(self) -> None:
        """Drop least-recently-used entries until within budget (caller holds lock)."""
        while self._used > self._max_bytes and self._entries:
            old_key, old_size = self._entries.popitem(last=False)
            self._used -= old_size
            with contextlib.suppress(OSError):
                os.remove(os.path.join(self._dir, old_key))

    @property
    def used_bytes(self) -> int:
        """Total bytes currently held on disk by the cache."""
        with self._lock:
            return self._used

    def stats(self) -> dict[str, int | float]:
        """Cache effectiveness: hits, misses, hit-rate, and bytes held.

        The measured signal that turns a slow scan into a *diagnosable* one — a low hit-rate
        over a repeated read means the byte budget is too small for the working set, not that
        the storage is slow. Hit-rate is `0.0` before any lookup."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": (self._hits / total) if total else 0.0,
                "used_bytes": self._used,
            }


_CACHES: dict[str, FileBytesCache] = {}
_CACHES_LOCK = threading.Lock()


def get_file_cache() -> FileBytesCache | None:
    """The process-wide file cache for the active config, or `None` when disabled.

    Memoized per *resolved* cache directory, so `config_context` overriding `file_cache_dir`
    (e.g. in a test) yields a distinct cache without disturbing the default one. Keyed on the
    resolved path rather than the configured one, so `"auto"` and the path it resolves to are
    one cache rather than two views of the same files.
    """
    from batcher.config import active_config

    mem = active_config().memory
    directory = resolve_cache_dir(mem.file_cache_dir)
    if not directory:
        return None
    with _CACHES_LOCK:
        cache = _CACHES.get(directory)
        if cache is None:
            cache = FileBytesCache(directory, mem.file_cache_max_bytes)
            _CACHES[directory] = cache
        return cache


#: The sentinel that means "put the cache on whatever fast local disk this node has".
#:
#: A cache directory is a per-node fact — `/ephemeral` on one provider, `/mnt/local_disk` on
#: the next, a small container overlay on a laptop — so naming one in the config means naming
#: the wrong one everywhere but the machine it was written for. The sentinel lets a fleet
#: enable the cache once and have each node resolve its own volume, which is the same shape
#: `AUTOSCALE_WAIT_AUTO` uses for a figure only the node can know.
FILE_CACHE_AUTO = "auto"

#: The subdirectory Batcher takes under a node's scratch volume. The volume belongs to the
#: node — Ray's object spill is on the same mount — so the cache lives in a directory of its
#: own rather than scattering hashed filenames across a shared one.
_CACHE_SUBDIR = "batcher_file_cache"


def resolve_cache_dir(configured: str | None) -> str | None:
    """The directory the file cache should use, or `None` when it stays disabled.

    Args:
        configured: `MemoryConfig.file_cache_dir` — a path, the `"auto"` sentinel, or `None`.

    Returns:
        The path to cache into. An explicit path is used as given. `"auto"` resolves to a
        subdirectory of the node's measured local scratch volume, and to `None` on a node
        with no fast local disk — where a cache would be competing for the container overlay
        that the read it is caching would otherwise not touch.
    """
    if not configured:
        return None
    if configured != FILE_CACHE_AUTO:
        return configured
    from batcher._internal.site import local_scratch_root

    root = local_scratch_root()
    return os.path.join(root, _CACHE_SUBDIR) if root else None
