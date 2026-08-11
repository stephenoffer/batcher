"""A cheap identity token for a file, so a metadata cache cannot serve a stale answer.

Every metadata cache in the IO layer — Parquet footers, row counts, fragment indexes —
exists because re-reading a footer is an object-store round trip and one file is read by
many splits. They were all keyed on the **path alone**, justified by "Parquet is
write-once".

That is true of an immutable data lake and false of the thing pipelines actually do:
`FileSink` writes deterministic names (`part-00000.parquet`), so re-running a job
overwrites its output in place. Read it back in the same process and the cache answers
from the previous version. It is not a stale *file* — the data read is new — it is stale
*metadata about* new data, which is worse: a `count()` returns the old row count while
`collect()` returns the new rows, and a row-group offset from the old footer points into
the middle of the new file.

Keying on `(path, size, mtime_ns)` closes it for the cost of a stat — microseconds
locally, one HEAD on an object store, against the footer fetch and parse it still saves.
`filesystem.py::_cached_local` already keys its download cache exactly this way; this is
that idea, extracted so every cache can share one definition of "the same file".
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

from batcher.io._concurrent import is_local_path

__all__ = ["FileMetaCache", "file_identity", "files_version"]


class FileMetaCache:
    """An LRU of per-file metadata, keyed by `file_identity` and bounded by **weight**.

    The three caches in this layer that memoize a file's metadata — Parquet footers, ORC
    footers, Parquet row counts — were each bounded differently and each bounded wrongly, so
    they share this one implementation rather than three near-copies that can drift again:

    - The Parquet footer cache was `lru_cache(maxsize=1024)` against a planner that reads up
      to 10,000 footers in a single pass and then hands the *same* files to workers that read
      them again. Nine of every ten entries were evicted before the pass that fetched them
      had finished, so the cache returned almost nothing on the one workload guaranteed to
      fill it.
    - The row-count cache was a plain `dict` with no bound at all: a leak proportional to a
      long-lived worker's entire scan history rather than to its working set.
    - The ORC footer cache carried the same 1,024 as Parquet's, for entries three orders of
      magnitude smaller.

    Hence `weight`. Counting *entries* bounds the wrong quantity whenever entries differ in
    size, and a `FileMetaData` is the extreme case: one row group of four columns against two
    thousand row groups of three hundred is four orders of magnitude of resident memory under
    an identical entry count. A caller that stores fixed-size records leaves the weight at 1
    and gets a plain entry-bounded LRU; the footer cache weighs an entry by its row-group
    count, which is what lets a million single-row-group files stay resident for what a few
    thousand wide ones cost — exactly the shape a small-file corpus has.

    Thread-safe: workers read splits of the same files concurrently.
    """

    __slots__ = ("_budget", "_entries", "_lock", "_used")

    def __init__(self, budget: int) -> None:
        """Create a cache holding up to `budget` total weight."""
        self._entries: OrderedDict[Any, tuple[Any, int]] = OrderedDict()
        self._budget = max(1, budget)
        self._used = 0
        self._lock = threading.Lock()

    def get(self, key: Any) -> Any | None:
        """The value cached under `key`, marked most-recently-used, or None on a miss."""
        with self._lock:
            hit = self._entries.get(key)
            if hit is None:
                return None
            self._entries.move_to_end(key)
            return hit[0]

    def put(self, key: Any, value: Any, weight: int = 1) -> None:
        """Admit `value` under `key`, evicting least-recently-used entries to stay in budget.

        The most-recently-admitted entry is never evicted, so a single item heavier than the
        whole budget is still returned to the caller that just asked for it rather than being
        dropped on the way in.
        """
        weight = max(1, weight)
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return
            self._entries[key] = (value, weight)
            self._used += weight
            while self._used > self._budget and len(self._entries) > 1:
                _key, (_value, evicted) = self._entries.popitem(last=False)
                self._used -= evicted

    def clear(self) -> None:
        """Drop every entry, for a test that needs a cold cache."""
        with self._lock:
            self._entries.clear()
            self._used = 0

    def __len__(self) -> int:
        """How many entries are resident — the *count*, not the weight they occupy."""
        with self._lock:
            return len(self._entries)


# What a cache stores when the file's identity cannot be established. Distinct from any
# real token, so an unstattable file is never mistaken for a cache hit against another
# unstattable one — it simply never hits.
_UNKNOWN = object()


def file_identity(path: str, fs: Any | None = None) -> tuple[str, int, int] | None:
    """A token that changes whenever `path`'s content could have changed.

    Args:
        path: The file to identify.
        fs: The filesystem to stat through; resolved from `path` when omitted.

    Returns:
        `(path, size, mtime_ns)`, or None when the file cannot be stat-ed — in which case
        the caller must **not** cache, since it has no way to detect a later change.

    Examples:
        .. doctest::

            >>> from batcher.io.stats.file_identity import file_identity
            >>> file_identity("/nonexistent/path/x.parquet") is None
            True
    """
    try:
        if fs is None:
            from batcher.io.filesystem import resolve_filesystem

            fs = resolve_filesystem(path)
        info = _stat(fs, path)
    except Exception:
        return None
    return None if info is None else (path, info[0], info[1])


def _stat(fs: Any, path: str) -> tuple[int, int] | None:
    """`(size, mtime_ns)` for `path`, through whichever stat surface `fs` offers.

    The directory listing is consulted first: `expand` already fetched every file's size
    and mtime, so asking it again costs one HEAD per file on an object store for
    information the caller has already paid for.
    """
    from_listing = getattr(fs, "listing_info", lambda _p: None)(path)
    if from_listing is not None:
        return from_listing
    # The pyarrow-backed filesystems expose `get_file_info`, which carries both fields in
    # one round trip; the abstract `FileSystem` only promises `size()`.
    target = getattr(fs, "native_read_target", lambda _p: None)(path)
    if target is not None:
        native_fs, native_path = target
        info = native_fs.get_file_info(native_path)
        if info.size is None:
            return None
        return int(info.size), int(info.mtime_ns or 0)
    size = fs.size(path)
    return (int(size), 0) if size is not None else None


def files_version(files: list[str], fs: Any) -> str | None:
    """A single token summarizing the version of every file in `files`.

    This is what lets a statistics memo be keyed on *which version* of a relation it
    describes rather than merely on its path. A path-keyed memo survives the path being
    rewritten — by a pipeline re-run under the same deterministic filename, by the
    upstream job, by a compaction — and it holds zone maps, so a stale entry prunes
    against bounds the data no longer has and the scan returns rows that do not exist or
    skips rows that do.

    The common path costs no I/O: the directory listing that produced `files` already
    reported each one's size and mtime, so this is a dict lookup per file. Only a pinned
    subset or a glob that bypassed the listing falls back to stat-ing, and then
    concurrently, since each is an object-store round trip.

    Args:
        files: The paths making up the relation, in any order (the digest is order-
            sensitive, and `files` is already sorted by the listing).
        fs: The filesystem to consult.

    Returns:
        A hex digest, or None when any file cannot be identified — the caller must then
        not cache, having no way to detect a later change.
    """
    import hashlib

    if not files:
        return None
    listing_info = getattr(fs, "listing_info", None)
    from_listing = (
        [listing_info(f) for f in files] if listing_info is not None else [None] * len(files)
    )
    if all(info is not None for info in from_listing):
        tokens: list[tuple[str, int, int] | None] = [
            (f, i[0], i[1]) for f, i in zip(files, from_listing, strict=True)
        ]
    elif len(files) == 1 or is_local_path(files[0]):
        # Serial for a LOCAL filesystem. A local stat is ~6 us, so fanning 2,000 of them
        # across a thread pool costs far more in dispatch than it saves in latency:
        # measured on 2,000 files, serial 15.4 ms vs pooled 194.1 ms — a 12.6x penalty for
        # "parallelism". The pool only pays when a stat is a network round trip, which is
        # the object-store case below. Identical work, identical tokens, identical digest —
        # this is purely where the work runs.
        tokens = [file_identity(f, fs) for f in files]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(64, len(files))) as pool:
            tokens = list(pool.map(lambda f: file_identity(f, fs), files))
    if any(t is None for t in tokens):
        return None
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(f"{token}".encode())
    return digest.hexdigest()[:32]
