"""Splits — independently-readable, picklable slices of a source.

A `Split` is the unit of distributed read parallelism. A source advertises its
splits via `Source.splits()`; each split carries only *locators* (a format name +
path, a set of row-group ids, …) — never data — so it serializes cheaply to a
remote worker that then reads just its slice directly from storage. The default
for a source that cannot subdivide is a single `WholeSourceSplit`, which
reproduces today's whole-source read.

Splits intentionally mirror the `Source` read surface (`schema`/`read`/
`iter_batches`/`row_count`/`identity`) so a worker treats a split exactly like a
source. The contract and the whole-source fallback live in `base`; the file-locator
splits in `file`; the Parquet row-group split, its footer cache, and the shared
dataset fragment index in `parquet`.
"""

from __future__ import annotations

from batcher.io.splits.base import Split, WholeSourceSplit
from batcher.io.splits.file import (
    FileSplit,
    IpcFileSplit,
    LineRangeSplit,
    NormalizedFileSplit,
    read_aligned_range,
)
from batcher.io.splits.parquet import (
    RowGroupSplit,
    fragment_index,
    pack_row_groups,
    parquet_row_group_splits,
)

__all__ = [
    "FileSplit",
    "IpcFileSplit",
    "LineRangeSplit",
    "NormalizedFileSplit",
    "RowGroupSplit",
    "Split",
    "WholeSourceSplit",
    "fragment_index",
    "pack_row_groups",
    "parquet_row_group_splits",
    "read_aligned_range",
]
