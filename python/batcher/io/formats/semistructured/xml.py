"""XML format — Arrow-native nested read via `xml2arrow`, and a row-element writer.

`xml2arrow` parses XML directly into Arrow (preserving nested structure) without
materializing Python objects per row, so `XMLSource` reads to Arrow without a
row-oriented hop. One whole file is one `Split`.

`XMLSink` writes the layout Spark's XML data source writes: one root element wrapping one
element per row, a struct as nested elements, a list as the element repeated, and a field
whose name carries the attribute prefix as an attribute. XML has no columnar encoding, so
the writer serializes at batch granularity over each batch's Python values, the same
unavoidable hop the JSON and MessagePack writers make. It needs only the standard library.

All `xml2arrow` imports are deferred — importing this module never requires the
optional dependency. A missing dependency raises `BackendError` with a
``pip install 'batcher-engine[xml]'`` hint.
"""

from __future__ import annotations

import base64
import re
from typing import IO, Any
from xml.sax.saxutils import escape, quoteattr

import pyarrow as pa

from batcher._internal.errors import BackendError, FormatError, SchemaError
from batcher._internal.optional import require
from batcher.io.base import FileSink, FileSource
from batcher.io.formats.base import SINKS, SOURCES

__all__ = ["XMLSink", "XMLSource"]

#: An XML element or attribute name, restricted to the ASCII subset. A name outside it is
#: refused rather than written, because a malformed name makes the whole document unreadable.
_XML_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9._-]*$")


def _require_xml2arrow() -> Any:
    """Import and return the `xml2arrow` module or raise `BackendError`."""
    return require("xml2arrow", feature="XML", provides="xml2arrow", extra="xml")


def _to_table(fh: IO[Any]) -> pa.Table:
    """Parse one XML file handle into an Arrow table via xml2arrow."""
    xml2arrow = _require_xml2arrow()
    data = fh.read()
    if isinstance(data, str):
        data = data.encode("utf-8")
    try:
        result = xml2arrow.parse(data)
    except Exception as exc:
        raise BackendError(f"failed to parse XML: {exc}") from exc
    return result if isinstance(result, pa.Table) else pa.table(result)


@SOURCES.register("xml")
class XMLSource(FileSource):
    """One or more XML files read to nested Arrow (single file, directory, or glob)."""

    suffix = ".xml"
    format_name = "xml"

    __slots__ = ()

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:
        return _to_table(fh).schema

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        table = _to_table(fh)
        if projection is not None:
            table = table.select(projection)
        return table.to_batches()


def _check_name(name: str, what: str) -> str:
    if not _XML_NAME.match(name):
        raise FormatError(f"write.xml: {what} {name!r} is not a valid XML element name")
    return name


def _text(value: Any) -> str:
    """One leaf value as XML character data, in the spelling Spark's XML writer uses."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _leaf_type(dtype: pa.DataType) -> pa.DataType:
    """The item type under any number of list layers, which XML writes as repetition."""
    while (
        pa.types.is_list(dtype)
        or pa.types.is_large_list(dtype)
        or pa.types.is_fixed_size_list(dtype)
    ):
        dtype = dtype.value_type
    return dtype


@SINKS.register("xml")
class XMLSink(FileSink):
    """Write rows as XML elements under one root element, in Spark's XML layout.

    A row becomes ``<row_tag>`` holding one child element per column. A null is omitted
    rather than written as an empty element, so a reader sees the field as absent, which is
    how Spark writes it unless a ``nullValue`` is set. A list column repeats its element
    once per item, a struct nests, and a struct field whose name starts with
    `attribute_prefix` becomes an attribute of the struct's element (a top-level column
    with the prefix becomes an attribute of the row). A struct's `value_tag` field becomes
    its element text. Binary values are base64 and temporal values ISO 8601.

    The document streams: the declaration and root open when the file opens, each batch is
    appended as it arrives, and the root closes at the end, so a streaming write holds one
    batch.

    Args:
        row_tag: The element each row is written as (Spark's ``rowTag``).
        root_tag: The element wrapping every row (Spark's ``rootTag``).
        attribute_prefix: The field-name prefix that marks an attribute (Spark's
            ``attributePrefix``).
        value_tag: The struct field written as the element's text (Spark's ``valueTag``).
        null_value: Written as the text of a null field instead of omitting it (Spark's
            ``nullValue``). ``None`` omits the element.
        declaration: The XML declaration written first, without the ``<?xml`` and ``?>``
            delimiters; ``""`` writes none.
    """

    suffix = ".xml"
    format_name = "xml"

    __slots__ = (
        "_attribute_prefix",
        "_declaration",
        "_null_value",
        "_root_tag",
        "_row_tag",
        "_value_tag",
    )

    def __init__(
        self,
        *,
        row_tag: str = "ROW",
        root_tag: str = "ROWS",
        attribute_prefix: str = "_",
        value_tag: str = "_VALUE",
        null_value: str | None = None,
        declaration: str = 'version="1.0" encoding="UTF-8" standalone="yes"',
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)  # carries filesystem= / storage_options=
        self._row_tag = _check_name(row_tag, "row_tag")
        self._root_tag = _check_name(root_tag, "root_tag")
        if not attribute_prefix:
            raise FormatError("write.xml attribute_prefix must be a non-empty string")
        self._attribute_prefix = attribute_prefix
        self._value_tag = value_tag
        self._null_value = null_value
        self._declaration = declaration

    def _check_schema(self, fields: pa.Schema | pa.StructType, parent: str = "") -> None:
        """Refuse a name or a type XML cannot carry before a byte is written."""
        for field in fields:
            path = f"{parent}{field.name}"
            dtype = _leaf_type(field.type)
            if pa.types.is_map(dtype) or pa.types.is_union(dtype):
                raise SchemaError(
                    f"write.xml cannot write column {path!r} of type {field.type}: XML has no "
                    "element layout for a map or a union. Convert it to a struct first."
                )
            if field.name != self._value_tag:
                name = field.name
                if name.startswith(self._attribute_prefix):
                    name = name[len(self._attribute_prefix) :]
                _check_name(name, f"column {path!r} gives the name")
            if pa.types.is_struct(dtype):
                self._check_schema(dtype, f"{path}.")

    def _element(self, name: str, value: Any, out: list[str]) -> None:
        """Append `name` holding `value` to `out`: omitted, repeated, nested, or a leaf."""
        if value is None:
            if self._null_value is not None:
                out.append(f"<{name}>{escape(self._null_value)}</{name}>")
            return
        if isinstance(value, list):
            for item in value:
                self._element(name, item, out)
            return
        if isinstance(value, dict):
            out.append(self._open(name, value))
            text = value.get(self._value_tag)
            if text is not None:
                out.append(escape(_text(text)))
            for key, child in value.items():
                if key != self._value_tag and not key.startswith(self._attribute_prefix):
                    self._element(key, child, out)
            out.append(f"</{name}>")
            return
        out.append(f"<{name}>{escape(_text(value))}</{name}>")

    def _open(self, name: str, fields: dict[str, Any]) -> str:
        attrs = "".join(
            f" {key[len(self._attribute_prefix) :]}={quoteattr(_text(value))}"
            for key, value in fields.items()
            if key.startswith(self._attribute_prefix) and value is not None
        )
        return f"<{name}{attrs}>"

    def _encode(self, batch: pa.RecordBatch) -> bytes:
        out: list[str] = []
        for row in batch.to_pylist():
            self._element(self._row_tag, row, out)
            out.append("\n")
        return "".join(out).encode("utf-8")

    def _header(self) -> bytes:
        declaration = f"<?xml {self._declaration}?>\n" if self._declaration else ""
        return f"{declaration}<{self._root_tag}>\n".encode()

    def _write_file(self, table: pa.Table, fh: IO[Any]) -> None:
        self._check_schema(table.schema)
        fh.write(self._header())
        for batch in table.to_batches():
            fh.write(self._encode(batch))
        fh.write(f"</{self._root_tag}>\n".encode())

    def _open_stream_writer(self, fh: IO[Any], schema: pa.Schema) -> Any:
        self._check_schema(schema)
        fh.write(self._header())
        return fh

    def _write_batch(self, writer: Any, batch: pa.RecordBatch) -> None:
        writer.write(self._encode(batch))

    def _close_stream_writer(self, writer: Any) -> None:
        writer.write(f"</{self._root_tag}>\n".encode())
