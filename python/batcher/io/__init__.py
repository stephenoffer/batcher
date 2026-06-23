"""`io` — data sources and sinks.

Sources are *lazy*: a `Dataset` holds `Source` handles, and bytes are read only
when a terminal operation runs, so the optimizer can push projections (and later
predicates) down into the scan. Sources advertise `splits()` — independently
readable, picklable slices — so distributed reads parallelize without the driver
materializing the whole source.

New formats subclass the Template-Method bases (`FileSource` / `FileSink`) and
register into the `SOURCES` / `SINKS` registries from a single `io/formats/<fmt>.py`
module; nothing else needs editing.
"""

from __future__ import annotations

from batcher.io.base import FileSink, FileSource
from batcher.io.formats import (
    SINKS,
    SOURCES,
    CSVSink,
    CSVSource,
    JSONSink,
    JSONSource,
    ParquetSink,
    ParquetSource,
)
from batcher.io.formats.multimodal.media import read_blob_bytes
from batcher.io.manifest import WriteManifest, WrittenFile
from batcher.io.sink import Sink
from batcher.io.source import InMemorySource, IteratorSource, Source
from batcher.io.splits import FileSplit, RowGroupSplit, Split, WholeSourceSplit

__all__ = [
    "SINKS",
    "SOURCES",
    "CSVSink",
    "CSVSource",
    "FileSink",
    "FileSource",
    "FileSplit",
    "InMemorySource",
    "IteratorSource",
    "JSONSink",
    "JSONSource",
    "ParquetSink",
    "ParquetSource",
    "RowGroupSplit",
    "Sink",
    "Source",
    "Split",
    "WholeSourceSplit",
    "WriteManifest",
    "WrittenFile",
    "read_blob_bytes",
]
