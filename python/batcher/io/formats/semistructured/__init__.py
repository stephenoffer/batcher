"""Semistructured formats (JSON/NDJSON, XML, logs, Protobuf, MessagePack)."""

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
