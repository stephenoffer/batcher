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
splits in `file`; the line-delimited byte range in `text`; the Parquet row-group
split, its footer cache, and the shared dataset fragment index in `parquet`.
`clustering` holds the optional guarantee a split set can make about *which* rows
it groups together -- one split per Hive partition directory -- which is what lets a
consumer grouping on those columns skip its shuffle entirely.
"""

from __future__ import annotations

from batcher.io.splits.base import Split, WholeSourceSplit
from batcher.io.splits.clustering import (
    clustering_of,
    declared_clustering,
    group_by_clustering,
)
from batcher.io.splits.file import (
    FileSplit,
    IpcFileSplit,
    LineRangeSplit,
    MultiFileSplit,
    NormalizedFileSplit,
    pack_files,
    read_aligned_range,
)
from batcher.io.splits.parquet import (
    RowGroupSplit,
    fragment_index,
    pack_row_groups,
    parquet_row_group_splits,
)
from batcher.io.splits.text import TextRangeSplit, line_range_splits

__all__ = [
    "FileSplit",
    "IpcFileSplit",
    "LineRangeSplit",
    "MultiFileSplit",
    "NormalizedFileSplit",
    "RowGroupSplit",
    "Split",
    "TextRangeSplit",
    "WholeSourceSplit",
    "clustering_of",
    "declared_clustering",
    "fragment_index",
    "group_by_clustering",
    "line_range_splits",
    "pack_files",
    "pack_row_groups",
    "parquet_row_group_splits",
    "read_aligned_range",
]
