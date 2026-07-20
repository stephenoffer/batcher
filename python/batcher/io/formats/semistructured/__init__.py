"""`io.formats.semistructured` — record-shaped data whose schema must be inferred.

JSON/NDJSON, XML, log lines, Protobuf, and MessagePack all have records and fields,
but no schema the reader can take on faith: it is inferred from the data (or from a
supplied descriptor) and may drift between files. That inference is what separates
this family from `structured`, where the schema arrives with the data. When files
disagree, reconciliation is `io.schema.evolution`'s job, not each format's.

Modules here subclass `io.base.FileSource` / `FileSink` where the data is
file-backed and register themselves into `SOURCES` / `SINKS` on import. Optional
third-party backends stay deferred-imported inside methods, so importing a
connector never requires its dependency.

A new record-shaped format whose schema is inferred is one new module here.
"""

from __future__ import annotations

from batcher.io.formats.semistructured.json import JSONSink, JSONSource
from batcher.io.formats.semistructured.logs import LogSource
from batcher.io.formats.semistructured.msgpack import MsgpackSink, MsgpackSource
from batcher.io.formats.semistructured.protobuf import ProtobufSource
from batcher.io.formats.semistructured.xml import XMLSource

__all__ = [
    "JSONSink",
    "JSONSource",
    "LogSource",
    "MsgpackSink",
    "MsgpackSource",
    "ProtobufSource",
    "XMLSource",
]
