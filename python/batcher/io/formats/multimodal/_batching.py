"""How a media read is cut into file-batches — bounded by file count *and* bytes.

A fixed file count is the wrong unit for media. Media file sizes span orders of magnitude
within one directory, so batching 64-at-a-time gives batches whose weight varies by the
same orders of magnitude: 64 thumbnails is 256 KB, 64 videos is 12.8 GB. The large case is
an OOM rather than a slow query, and the small case leaves its worker idle — the classic
skew shape, arriving from the *storage layout* rather than from a join key.

Bounding both dimensions turns that into evenly-weighted work. This module is separate
from `media.py` because the packing is the part worth testing on its own: it is pure
arithmetic over (paths, sizes), needs no filesystem, and must produce an exact cover —
`splits()` and `iter_batches()` both derive their chunking from it, and a disagreement
between them would make a distributed read return a different set of batches from the
single-node one.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

__all__ = ["pack_by_count_and_bytes", "probe_sizes"]

# Size probes are object-store *latency*, not bandwidth, so fan out wide — the same
# reasoning (and the same width) as the Parquet footer reads.
_SIZE_PROBE_CONCURRENCY = 64


def pack_by_count_and_bytes(
    files: list[str], sizes: list[int], max_files: int, max_bytes: int
) -> list[list[str]]:
    """Group `files` into runs bounded by both a file count and a total byte size.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.multimodal._batching import pack_by_count_and_bytes
            >>> pack_by_count_and_bytes(["a", "b", "c"], [10, 10, 10], 8, 25)
            [['a', 'b'], ['c']]

            >>> # A file bigger than the whole budget still forms a group of its own.
            >>> pack_by_count_and_bytes(["big", "x"], [500, 1], 8, 100)
            [['big'], ['x']]

    Args:
        files: Paths, in the order they must be read.
        sizes: Each path's size in bytes, positionally aligned with `files`.
        max_files: Most files one group may contain.
        max_bytes: Most bytes one group may total, except that a single file larger than
            this still forms a group of its own rather than being dropped.

    Returns:
        The groups, covering `files` exactly once and in order.
    """
    groups: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for path, size in zip(files, sizes, strict=True):
        if current and (len(current) >= max_files or current_bytes + size > max_bytes):
            groups.append(current)
            current, current_bytes = [], 0
        current.append(path)
        current_bytes += size
    if current:
        groups.append(current)
    return groups


def probe_sizes(files: list[str], size_of: Callable[[str], int]) -> list[int]:
    """Every file's size, probed concurrently.

    One stat per file is negligible next to what a media source already does per file — it
    reads either the whole payload or a 64 KiB header. A file whose size cannot be
    determined counts as zero, which merely leaves it to the count bound rather than
    failing a read over a metadata hiccup.

    Args:
        files: The paths to size.
        size_of: Returns one path's size in bytes; may raise.

    Returns:
        The sizes, positionally aligned with `files`.
    """

    def _size(path: str) -> int:
        try:
            return size_of(path)
        except Exception:
            return 0

    if len(files) <= 1:
        return [_size(f) for f in files]
    with ThreadPoolExecutor(max_workers=min(_SIZE_PROBE_CONCURRENCY, len(files))) as pool:
        return list(pool.map(_size, files))
