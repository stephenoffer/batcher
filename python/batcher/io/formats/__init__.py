"""`io.formats` — formats grouped by family, behind a registry.

Each `formats/<category>/<fmt>.py` module registers its concrete source/sink
classes into the `SOURCES` / `SINKS` registries in `base`, as a side effect of being
imported, so ``read(format="delta")`` (and friends) work without an explicit import.
Optional backends are deferred-imported inside methods, so importing a connector
module never requires its third-party dependency. A new format is one new file in the
right category that registers itself — `source.py` / `sink.py` need not change.

The file-shaped categories are imported here. The warehouse, NoSQL, broker, lakehouse,
and ML-shard categories are imported on the first registry lookup instead
(`base._DEFERRED_FAMILIES`), which keeps the widest and least-used part of the IO
surface off the cost of ``import batcher`` without changing how a format registers or
how it is reached.
"""

from __future__ import annotations

# Importing each category subpackage triggers its modules' registry side effects.
import batcher.io.formats.genomics
import batcher.io.formats.multimodal
import batcher.io.formats.robotics  # noqa: F401
from batcher.io.formats.base import SINKS, SOURCES, SinkFormat, SourceFormat
from batcher.io.formats.genomics import (
    BedSource,
    FastaSink,
    FastaSource,
    FastqSink,
    FastqSource,
    GffSource,
    VcfSource,
)
from batcher.io.formats.robotics import MCAP_SCHEMA, MDF_SCHEMA, MCAPSource, MDFSource
from batcher.io.formats.semistructured import JSONSink, JSONSource
from batcher.io.formats.structured import (
    CSVSink,
    CSVSource,
    ParquetDatasetSource,
    ParquetSink,
    ParquetSource,
)
from batcher.io.formats.unstructured import BinarySource, TextSource

__all__ = [
    "MCAP_SCHEMA",
    "MDF_SCHEMA",
    "SINKS",
    "SOURCES",
    "BedSource",
    "BinarySource",
    "CSVSink",
    "CSVSource",
    "FastaSink",
    "FastaSource",
    "FastqSink",
    "FastqSource",
    "GffSource",
    "JSONSink",
    "JSONSource",
    "MCAPSource",
    "MDFSource",
    "ParquetDatasetSource",
    "ParquetSink",
    "ParquetSource",
    "SinkFormat",
    "SourceFormat",
    "TextSource",
    "VcfSource",
]
