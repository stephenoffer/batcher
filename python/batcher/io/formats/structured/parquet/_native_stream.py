"""Native-reader streaming for `ParquetSource._iter_file`, and the rule for when to use it.

The native Rust reader fetches a window of row-groups concurrently, which makes it a large
win over pyarrow's row-group-at-a-time iterator — *as long as nothing above it is already
fanning out*. `FileSource.iter_batches` reads `depth` files concurrently, and each native
call also submits its own row-group tasks to the shared tokio runtime, so the two compose
multiplicatively: `depth x rg_concurrency` in-flight decodes against a fixed core count,
with the read-ahead threads parked in `block_on` waiting for an oversubscribed runtime.

Measured, local NVMe, streaming every row through `ordered_readahead` at a fixed depth
(min of 3 runs; "before" is pyarrow's own iterator, the code this replaced):

    200 files x 20 row-groups x 10k rows, 1 column        3 columns
    depth   pyarrow    native   ratio                     pyarrow   native   ratio
      1     1321 ms    657 ms   2.01x                     1201 ms   780 ms   1.54x
      2      787 ms    389 ms   2.02x                      676 ms   495 ms   1.37x
      4      395 ms    574 ms   0.69x  <-- inverts         387 ms   633 ms   0.61x
     16      232 ms    710 ms   0.33x                      279 ms   742 ms   0.38x

    1 file x 200k rows/row-group (depth is necessarily 1)
      1      491 ms     72 ms   6.83x                      507 ms   127 ms   3.99x

Two regimes, one mechanism: row-group concurrency and file read-ahead are two routes to the
same parallelism, and using both at once oversubscribes. So the native path is taken only
where the outer loop is NOT already providing the fan-out. That is the `_LOCAL_MAX_DEPTH`
rule below, and it is why this decision needs the read-ahead depth rather than a file count.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pyarrow as pa

from batcher.io.formats.structured import _parquet_native

__all__ = ["iter_windows", "row_group_windows", "use_native_stream"]

# Ceiling on the decoded data one native window may materialize, plus a hard cap on the
# row-groups it may span. The window needs a bound because the native FFI returns a
# *materialized* batch list: the whole window is resident before its first batch is yielded.
# Bounding it by bytes rather than by a row-group count is what makes the bound mean
# something — row-groups run from a few MB to several hundred, so a fixed count turns that
# spread straight into peak memory. Sizing reads the footer's uncompressed `total_byte_size`,
# which counts every column and so over-estimates a projected read: it errs small.
#
# Against read-ahead: the worst case adds `depth x _NATIVE_WINDOW_BYTES` of transient decode
# footprint — one window per in-flight file, not a pile, because `ordered_readahead` parks a
# producer on its byte credit before it can start a second. With the depth rule below,
# `depth` is at most `_LOCAL_MAX_DEPTH` on local disk, so that is ~64 MiB.
_NATIVE_WINDOW_BYTES = max(
    1 << 20, int(os.environ.get("BATCHER_NATIVE_WINDOW_BYTES", str(32 << 20)))
)
_NATIVE_WINDOW_MAX_GROUPS = 8

# The measured crossover (see the module docstring): at read-ahead depth 1-2 the native
# reader wins 1.4-2.0x, at depth 4 it has already lost, and at the default depth of 16 it is
# 3x slower. Local disk is bandwidth- and CPU-bound, so the parallelism has to come from
# exactly one place.
_LOCAL_MAX_DEPTH = max(1, int(os.environ.get("BATCHER_NATIVE_STREAM_MAX_DEPTH", "2")))


def use_native_stream(depth: int, remote: bool) -> bool:
    """Whether `_iter_file` should stream natively at this read-ahead `depth`.

    Local: only when the read-ahead is not already fanning out (`_LOCAL_MAX_DEPTH`), because
    there the contended resource is cores and the two fan-outs multiply.

    Remote: always. The contended resource there is in-flight requests, not cores — the
    native reader's row-group concurrency is what hides a ~tens-of-ms GET, and the tasks it
    submits are awaiting I/O rather than burning CPU, so they do not compete the way the
    local decodes above do. This branch is reasoned from that mechanism and from the
    distributed scan's measured `_SCAN_PREFETCH` (8 -> 32 cut a TPC-H sf100 agg ~53s ->
    ~31s); it is **not** measured here, because this machine has no object store. If a
    cluster measurement disagrees, `BATCHER_NATIVE_STREAM_MAX_DEPTH=2` makes remote follow
    the same rule as local without a code change.

    Args:
        depth: How many files the caller's read-ahead decodes concurrently.
        remote: Whether the source's files sit behind a network round trip.

    Returns:
        True to stream through the native reader, False to use pyarrow's own iterator.
    """
    return remote or depth <= _LOCAL_MAX_DEPTH


def row_group_windows(metadata: Any) -> Iterator[list[int]]:
    """Contiguous row-group index runs, each under `_NATIVE_WINDOW_BYTES` and never empty.

    Walks the footer's per-row-group sizes — O(row-groups), which is file structure and not
    row count, so this is metadata work and not a hot-path tuple touch. A row-group larger
    than the whole budget still forms a window of one: it has to be readable, and one is the
    smallest window there is.

    Args:
        metadata: The file's `pyarrow.parquet.FileMetaData`.

    Returns:
        An iterator of row-group index lists, together covering every row-group once, in order.
    """
    window: list[int] = []
    nbytes = 0
    for i in range(metadata.num_row_groups):
        size = metadata.row_group(i).total_byte_size
        if window and (
            nbytes + size > _NATIVE_WINDOW_BYTES or len(window) >= _NATIVE_WINDOW_MAX_GROUPS
        ):
            yield window
            window, nbytes = [], 0
        window.append(i)
        nbytes += size
    if window:
        yield window


def iter_windows(path: str, pf: Any, projection: list[str] | None) -> Iterator[pa.RecordBatch]:
    """Stream `path`'s row-groups through the native reader in byte-bounded windows.

    A window whose native read fails is re-read from `pf` (pyarrow) rather than aborting.
    Per-window granularity is what makes that possible: the native call returns a
    materialized list, so a failure is seen *before* any of its batches are yielded. A
    whole-stream fallback would be unsound — batches already delivered cannot be taken back —
    and raising would fail a working read purely for want of a fast path.

    Args:
        path: The file's full path/URI, as the native reader addresses it.
        projection: Columns to read, or None for all of them.
        pf: An open `pyarrow.parquet.ParquetFile` for the same file, supplying both the
            footer that plans the windows and the fallback reader.

    Returns:
        An iterator over the file's batches, in row-group order.
    """
    for window in row_group_windows(pf.metadata):
        batches = _parquet_native.read_row_groups_filtered(path, window, projection, None)
        if batches is None:
            yield from pf.read_row_groups(window, columns=projection).to_batches()
        else:
            yield from batches
