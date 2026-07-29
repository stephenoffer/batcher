"""The exact output-granularity contract for `iter_batches(batch_size=N)`.

`batch_size` is a promise about the *output*: every emitted batch holds exactly N rows
except the last. None of the per-path chunkers can keep it — they slice each engine batch
independently, so a stream of unevenly-sized batches leaks a short one at every boundary
rather than at the end only. This module is the one place that buffers across boundaries
and cuts on the exact row count, and it lives apart from the router because the router
chooses *which* strategy runs while this shapes what any of them emitted.
"""

from __future__ import annotations

from collections.abc import Iterator

import pyarrow as pa

__all__ = ["_rebatch_exact"]


def _rebatch_exact(batches: Iterator[pa.RecordBatch], batch_size: int) -> Iterator[pa.RecordBatch]:
    """Re-chunk a batch stream so every emitted batch holds exactly `batch_size` rows.

    `pyarrow`'s per-batch slicing / ``to_batches(max_chunksize=…)`` chunks each input
    batch independently, so a stream of unevenly-sized engine batches leaks a short
    batch at every boundary rather than at the end only. Buffering across boundaries and
    cutting on the exact row count restores the "N rows per batch" contract: only the
    final remainder is smaller. An empty input yields nothing (matching the unbatched
    path, which emits no batch for an empty result).

    Buffered as a **list of batches**, not as a growing table. `concat_tables` copies the
    accumulator's whole chunk list on every call, so appending each arriving batch to a
    table made the buffering quadratic in the batches buffered — and the case that buffers
    the most is precisely the one this exists for: a large `batch_size` over a source that
    yields small batches, where a broker poll of a few rows against a 100k-row request
    buffered tens of thousands of chunks and re-copied the list for every one of them. The
    list appends in constant time and is turned into a table once per *emit*, not once per
    arrival.
    """
    buf: list[pa.RecordBatch] = []
    rows = 0
    for b in batches:
        if b.num_rows == 0:
            continue
        buf.append(b)
        rows += b.num_rows
        if rows < batch_size:
            continue
        table = pa.Table.from_batches(buf)
        offset = 0
        while table.num_rows - offset >= batch_size:
            yield table.slice(offset, batch_size).combine_chunks().to_batches()[0]
            offset += batch_size
        rest = table.slice(offset)
        # Compacted, so the carried remainder is one small batch rather than a view that
        # pins the whole emitted round's buffers until the next flush.
        buf = rest.combine_chunks().to_batches() if rest.num_rows else []
        rows = rest.num_rows
    if buf:
        yield from pa.Table.from_batches(buf).combine_chunks().to_batches()
