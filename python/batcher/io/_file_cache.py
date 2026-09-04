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


#: How long a coalesced reader waits for the thread already fetching the same file.
#:
#: Bounded rather than indefinite: a leader that wedges on a hung socket must cost the
#: waiters a redundant download, never the query. Generous enough that a genuinely large
#: file over a slow link is still shared rather than re-fetched by everyone at once, since
#: exceeding it is strictly worse than the duplication it exists to prevent.
_INFLIGHT_WAIT_S = 120.0


class FileBytesCache:
    """A byte-bounded, LRU local-disk cache of whole remote files.

    Keyed by the remote path; the cached copy lives at ``<cache_dir>/<sha256(path)>``.
    Thread-safe. Fetching happens outside the lock (it is slow I/O) into a unique temp
    file that is atomically renamed into place, so concurrent readers never observe a
    half-written file.
    """

    __slots__ = (
        "_coalesced",
        "_declined",
        "_dir",
        "_entries",
        "_hits",
        "_inflight",
        "_lock",
        "_max_bytes",
        "_misses",
        "_stale",
        "_used",
    )

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
        # Files declined because they alone exceed the budget. Not a failure: the read
        # still happens, straight from the object store. A workload where this dominates
        # has a budget too small for its files, which reads as a zero hit-rate otherwise.
        self._declined = 0
        # Hits whose file had been deleted underneath this ledger. Non-zero means something
        # else is managing the same directory -- another worker on the node, or the node's
        # scratch cleaner -- which is worth knowing, because each one is a re-fetch that the
        # hit rate alone would report as a hit.
        self._stale = 0
        # Fetches that waited on another thread's fetch of the same file instead of
        # issuing their own. This is the figure the cache saves *network* on, as distinct
        # from the hits it saves round trips on.
        self._coalesced = 0
        # key -> an Event set once the thread that claimed that key has finished fetching
        # it. Without this every reader that misses the same file at once downloads the
        # whole of it: a scan whose workers all open one dimension table pays the transfer
        # once per worker thread, and the copies overwrite each other byte for byte.
        self._inflight: dict[str, threading.Event] = {}
        # `private_dir`, not `os.makedirs`: an entry here is a byte-for-byte copy of one of
        # the user's data files, and the cache root is Batcher's own subdirectory of a node
        # volume other tenants also mount. The *file* mode is not ours to set — the caller's
        # `fetch` writes the bytes — so the directory is what protects them.
        private_dir(cache_dir)

    def get_or_fetch(
        self, remote_path: str, fetch: Callable[[str], None], size_hint: int | None = None
    ) -> str | None:
        """Return the local path of the cached copy of `remote_path`, or `None`.

        On a miss, `fetch(local_tmp_path)` is called to materialize the bytes (it must
        write the full file to the given path); the result is then admitted under the
        byte budget, evicting the least-recently-used entries if needed.

        `None` means the file will not be cached and the caller should read it remotely.
        That is the answer for a file bigger than the entire budget: admitting one used to
        evict the whole cache and then, having nothing else left to drop, the entry
        itself — deleting the file whose path was about to be returned, so the caller
        opened a path that no longer existed. Declining up front on `size_hint` also
        spares the download, which was being paid in full for bytes that were deleted
        before anything read them.

        Concurrent misses on the same key are coalesced: the first caller fetches and the
        rest wait for it, so a file opened by many reader threads at once crosses the
        network once rather than once per thread. The wait is bounded, and a waiter whose
        leader fails or is evicted falls through and fetches for itself, so no caller can
        be blocked by another's failure.

        Args:
            remote_path: The cache key, identifying the remote file and its version.
            fetch: Materializes the bytes to the local path it is given.
            size_hint: The file's size in bytes when the caller already knows it, used to
                decline an oversized file without downloading it first.

        Returns:
            The local path to read, or `None` when this file is not to be cached.
        """
        if size_hint is not None and size_hint > self._max_bytes:
            with self._lock:
                self._declined += 1
            return None

        key = hashlib.sha256(remote_path.encode("utf-8")).hexdigest()
        local = os.path.join(self._dir, key)
        with self._lock:
            if self._resident_locked(key, local):
                self._hits += 1
                return local
            self._misses += 1
            leader = self._inflight.get(key)
            mine = leader is None
            if mine:
                leader = threading.Event()
                self._inflight[key] = leader
            else:
                self._coalesced += 1

        if not mine:
            # Someone else is already pulling these bytes. Wait for them rather than
            # duplicating the transfer. Bounded, so a leader that wedges costs this
            # caller a redundant fetch instead of the query.
            assert leader is not None
            leader.wait(_INFLIGHT_WAIT_S)
            with self._lock:
                # Validated, not merely present: the leader's file can be evicted by a
                # worker sharing this volume between its admission and this wake, and
                # returning the path unchecked would reintroduce the stale hit one branch
                # over from where it is handled.
                if self._resident_locked(key, local):
                    return local
            # The leader failed, timed out, or its entry was evicted before we woke.
            # Fetch it ourselves; we do not own `_inflight[key]`, so we must not clear it.

        try:
            return self._fetch_and_admit(key, local, fetch)
        finally:
            if mine:
                with self._lock:
                    self._inflight.pop(key, None)
                assert leader is not None
                leader.set()

    def _fetch_and_admit(self, key: str, local: str, fetch: Callable[[str], None]) -> str | None:
        """Materialize `key`'s bytes and account them, returning the path or `None`.

        The fetch runs with the lock released, because it is slow remote I/O, and lands on
        a unique temp that is atomically renamed into place — so a concurrent reader never
        observes a half-written file.

        Args:
            key: The hashed cache key.
            local: The path the key resolves to.
            fetch: Writes the full file to the path it is given.

        Returns:
            The local path, or `None` when the file turned out to be too large to keep.
        """
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
            if size > self._max_bytes:
                # No `size_hint` was given and the file turns out not to fit. Admitting it
                # would evict the cache and then itself; keeping it unaccounted would leak
                # it. Drop it and let the caller read remotely.
                self._declined += 1
                with contextlib.suppress(OSError):
                    os.remove(local)
                return None
            # A racing thread may have admitted the same key first; only one accounts for
            # the bytes (the file content is identical, so the rename is harmless).
            if key not in self._entries:
                self._entries[key] = size
                self._used += size
                self._evict_locked(protect=key)
            else:
                self._entries.move_to_end(key)
        return local

    def _resident_locked(self, key: str, local: str) -> bool:
        """Whether `key`'s file is both in the ledger and still on disk. Caller holds the lock.

        The two can disagree, and the ledger is not the authority. The cache directory is a
        *node* volume rather than a private one: `file_cache_dir="auto"` puts it on shared
        scratch, several workers on one node keep separate ledgers over the same files, and
        the node's own cleaner may sweep it. So another process evicting -- or anything at
        all deleting -- leaves this ledger reporting a hit for a path that no longer exists,
        and the caller opens it and raises.

        A missing file is dropped here and reported as a miss, which re-fetches. The `stat`
        is paid on every hit, against a local-disk read the caller is about to do anyway.

        Args:
            key: The hashed cache key.
            local: The path the key resolves to.

        Returns:
            True when the entry may be served, having marked it most-recently-used.
        """
        if key not in self._entries:
            return False
        if not os.path.exists(local):
            self._stale += 1
            self._used -= self._entries.pop(key)
            return False
        self._entries.move_to_end(key)  # mark most-recently-used
        return True

    def _evict_locked(self, protect: str | None = None) -> None:
        """Drop least-recently-used entries until within budget (caller holds lock).

        `protect` names an entry that must survive, because its path is about to be
        returned to a caller that will open it. Eviction deletes the file, so evicting the
        entry being admitted hands back a path to something that no longer exists. The
        oversized case is refused before it gets here; this covers the concurrent one,
        where another thread's admission pushes this one out between the two lines.
        """
        while self._used > self._max_bytes and self._entries:
            old_key, old_size = self._entries.popitem(last=False)
            if old_key == protect:
                self._entries[old_key] = old_size  # re-admit at the most-recent end
                self._entries.move_to_end(old_key)
                break
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
        the storage is slow. Hit-rate is `0.0` before any lookup.

        `coalesced` counts the fetches that waited on another thread rather than issuing
        their own, which is where the cache saves *bandwidth* rather than round trips.
        `declined` counts files bigger than the whole budget, which are read remotely every
        time: a non-zero figure with a zero hit-rate says the budget is too small for a
        single file, not that the working set is too big."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": (self._hits / total) if total else 0.0,
                "coalesced": self._coalesced,
                "declined": self._declined,
                "stale": self._stale,
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
