"""Parquet — lazy projection/predicate read + write, plus the dataset reader.

`ParquetSource` (in `source`) reads one or more Parquet files with row-group-level
splits. `ParquetDatasetSource` (in `dataset`) is the PB-scale workhorse: a
`pyarrow.dataset` over a (possibly Hive-partitioned) directory tree, recovering
partition columns from the path and supporting partition + row-group pruning, with
distributed listing so the driver never walks the whole tree. `ParquetSink` (in
`sink`) writes.
"""

from __future__ import annotations

from batcher.io.formats.structured.parquet.dataset import (
    ParquetDatasetSource,
    ParquetFragmentSplit,
    PartitionDirSplit,
)
from batcher.io.formats.structured.parquet.sink import ParquetSink
from batcher.io.formats.structured.parquet.source import ParquetSource

__all__ = [
    "ParquetDatasetSource",
    "ParquetFragmentSplit",
    "ParquetSink",
    "ParquetSource",
    "PartitionDirSplit",
]
