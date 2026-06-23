"""`io.formats` — formats grouped by family, behind a registry.

Each `formats/<category>/<fmt>.py` module registers its concrete source/sink
classes into the `SOURCES` / `SINKS` registries in `base`. Importing this package
imports every category subpackage, populating the registries as a side effect, so
``read(format="delta")`` (and friends) work without an explicit import. Optional
backends are deferred-imported inside methods, so importing a connector module
never requires its third-party dependency. A new format is one new file in the
right category that registers itself — `source.py` / `sink.py` need not change.
"""

from __future__ import annotations

# Importing each category subpackage triggers its modules' registry side effects.
import batcher.io.formats.lakehouse
import batcher.io.formats.ml
import batcher.io.formats.multimodal
import batcher.io.formats.nosql
import batcher.io.formats.sql
import batcher.io.formats.streaming  # noqa: F401
from batcher.io.formats.base import SINKS, SOURCES, SinkFormat, SourceFormat
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
    "SINKS",
    "SOURCES",
    "BinarySource",
    "CSVSink",
    "CSVSource",
    "JSONSink",
    "JSONSource",
    "ParquetDatasetSource",
    "ParquetSink",
    "ParquetSource",
    "SinkFormat",
    "SourceFormat",
    "TextSource",
]
