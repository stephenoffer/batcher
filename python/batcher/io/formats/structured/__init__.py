"""`io.formats.structured` — formats that already carry a schema, behind the registry.

The family a format belongs to is decided by what it declares about itself, and
these are the ones that arrive with a schema the reader can trust: columnar
(Parquet, ORC, Arrow IPC, Lance) and self-describing row formats (CSV headers,
Avro, Excel). That is what separates them from `semistructured` (a schema must be
inferred per record) and `unstructured` (there is no schema beyond bytes or text).

Each module subclasses `io.base.FileSource` / `FileSink` — the Template-Method
spine that owns path/glob resolution, schema caching, multi-file concatenation,
projection, and split generation — so a concrete format overrides only its per-file
read/write primitives, and registers itself into `SOURCES` / `SINKS` on import.
Because these formats carry a schema, this is also where projection and predicate
pushdown actually pay off; `parquet/` is a subpackage rather than a module for
exactly that reason (lazy column/row-group pruning plus the dataset reader).

A new schema-carrying format is one new module here that registers itself.
"""

from __future__ import annotations

from batcher.io.formats.structured.arrow_ipc import ArrowIPCSink, ArrowIPCSource
from batcher.io.formats.structured.avro import AvroSink, AvroSource
from batcher.io.formats.structured.csv import CSVSink, CSVSource
from batcher.io.formats.structured.excel import ExcelSource
from batcher.io.formats.structured.lance import LanceSink, LanceSource
from batcher.io.formats.structured.orc import ORCSink, ORCSource
from batcher.io.formats.structured.parquet import (
    ParquetDatasetSource,
    ParquetSink,
    ParquetSource,
)

__all__ = [
    "ArrowIPCSink",
    "ArrowIPCSource",
    "AvroSink",
    "AvroSource",
    "CSVSink",
    "CSVSource",
    "ExcelSource",
    "LanceSink",
    "LanceSource",
    "ORCSink",
    "ORCSource",
    "ParquetDatasetSource",
    "ParquetSink",
    "ParquetSource",
]
