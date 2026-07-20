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

from typing import Any

from batcher.io._concurrent import is_local_path

__all__ = ["file_identity", "files_version"]

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
