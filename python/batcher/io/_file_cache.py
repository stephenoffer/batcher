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
        os.makedirs(cache_dir, exist_ok=True)

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

    Memoized per cache directory, so `config_context` overriding `file_cache_dir`
    (e.g. in a test) yields a distinct cache without disturbing the default one.
    """
    from batcher.config import active_config

    mem = active_config().memory
    if not mem.file_cache_dir:
        return None
    with _CACHES_LOCK:
        cache = _CACHES.get(mem.file_cache_dir)
        if cache is None:
            cache = FileBytesCache(mem.file_cache_dir, mem.file_cache_max_bytes)
            _CACHES[mem.file_cache_dir] = cache
        return cache
