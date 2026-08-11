"""Concurrent per-file reads — the shared fan-out for footer/header stats and file bytes.

Reading one file's footer, header, or whole bytes is a single object-store round trip whose
work releases the GIL, so a serial loop over a many-file dataset dominates a scan's wall
clock (100 files ≈ several seconds of otherwise-idle time). Every metadata extractor
(``io.stats``) and every multi-file connector reader (binary/text/embeddings) goes through
:func:`read_each_file` so the fan-out — and its concurrency cap — lives in one place,
matching the ``base.FileSource`` footer/read pool.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

__all__ = ["is_local_path", "listed_sizes", "read_each_file", "total_file_bytes"]

T = TypeVar("T")

# Metadata reads are latency-bound (network round trips), not CPU-bound, so the pool can
# exceed the core count; shared with `base.FileSource` via the same env override.
_CONCURRENCY = max(8, int(os.environ.get("BATCHER_FOOTER_CONCURRENCY", "64")))


def is_local_path(path: str) -> bool:
    """Whether `path` names a local file, where a read is a syscall not a round trip.

    Drives the serial-vs-pooled choice in :func:`read_each_file` and in
    `io.stats.file_identity.files_version` — never correctness. One definition, because the
    two were making the same decision from the same reasoning and only one of them had
    measured it.
    """
    idx = path.find("://")
    return idx <= 0 or path[:idx].lower() == "file"


def read_each_file(fs: Any, files: list[str], read_one: Callable[[Any, str], T]) -> list[T]:
    """Apply ``read_one(fs, path)`` to every file, preserving file order.

    Concurrent for a **remote** filesystem, serial for a local one. That split is measured,
    not stylistic, and it goes the opposite way from intuition: a local footer read is a
    syscall on page cache, so fanning 1,000 of them across a pool costs more in dispatch
    than it saves in latency (1,000 local Parquet footers: 387 ms serial against 613 ms
    pooled). A remote read is a ~40 ms round trip, where the same pool is the difference
    between seconds and tens of seconds. Same work, same order, same results — this is
    purely *where* it runs.

    A single file (the common small case) skips the pool either way. Exceptions propagate to
    the caller, which decides all-or-nothing (an exact ``count()`` is void if any footer
    fails) versus skip (best-effort pruning bounds) — this helper only owns the concurrency.
    """
    if len(files) <= 1 or is_local_path(files[0]):
        return [read_one(fs, f) for f in files]
    workers = min(len(files), _CONCURRENCY)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda path: read_one(fs, path), files))  # order preserved


def listed_sizes(fs: Any, files: list[str]) -> list[int] | None:
    """Every file's on-disk size, from the directory listing, at **no I/O cost** — or None.

    `expand` already received each file's size and mtime from the very `get_file_info` /
    `ls` call that produced `files`, and the backend keeps them (`listing_info`). So the
    sizes are a dict lookup per file, where `fs.size(path)` is one HEAD per file — an
    O(files) round-trip sweep for a number already paid for.

    That distinction is what makes size-based split planning affordable at scale: packing a
    million files into byte-sized tasks needs a million sizes, and a million HEADs on the
    driver is exactly the cost the packing exists to avoid.

    Returns None when *any* file's size is not in the listing (a pinned subset, a glob that
    bypassed it, a backend that keeps none). Partial sizes are refused rather than mixed
    with stats, so a caller either plans on complete information or falls back knowingly.

    Args:
        fs: The filesystem the files were listed through.
        files: The paths to size, in order.

    Returns:
        One size per file, in `files` order, or None when the listing cannot answer for all.
    """
    info = getattr(fs, "listing_info", None)
    if info is None:
        return None
    sizes: list[int] = []
    for path in files:
        entry = info(path)
        if entry is None:
            return None
        sizes.append(entry[0])
    return sizes


def total_file_bytes(fs: Any, files: list[str]) -> int | None:
    """The summed on-disk byte size of every file, or None if it can't be stated whole.

    A *partial* total is worse than none: read-cost prediction and `total_bytes` sum across
    sources, so one unreadable size would silently under-report the whole query. Any file
    whose size cannot be read voids the figure rather than skewing it.

    The directory listing is consulted first (:func:`listed_sizes`), which costs nothing;
    only a source whose files did not come from one falls back to stat-ing them, and then
    concurrently for a remote store via :func:`read_each_file`.

    One home for the "sum sizes, void on any gap" rule, shared by `FileSource.statistics`, the
    delimited row estimator, and the standalone text source — it was pasted into all three.
    """
    if not files:
        return None
    listed = listed_sizes(fs, files)
    if listed is not None:
        return sum(listed) or None
    try:
        sizes = read_each_file(fs, files, lambda filesystem, path: filesystem.size(path))
    except Exception:
        return None
    if any(s is None for s in sizes):
        return None
    return sum(sizes) or None
