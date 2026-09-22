"""XML format — a flat Arrow read via `xml2arrow`, and a row-element writer.

`xml2arrow` parses XML into Arrow without materializing Python objects per row, so the
values never take a row-oriented hop. It is configuration-driven, though: it has no schema
inference, and its parser takes a YAML file naming the record container, each leaf's
absolute path and each leaf's type. `_infer_config` writes that file by scanning a bounded
prefix of the document, which is the only Python-side pass and reads no values into the
result. One whole file is one `Split`.

The read is **flat**, not nested: a nested leaf becomes one column named for its path
(`<x><y>` is `x_y`), a record's attributes become columns, and a leaf repeated within one
record keeps its last value, because xml2arrow's table has nowhere to put a list. The type
vocabulary is Boolean/Int/Float/Utf8, so a date column reads back as the text it was
written as. Two shapes are refused outright rather than guessed at: a document with no
repeated record element under its root, and one using XML namespaces, whose expanded
`{uri}tag` names do not address anything in an xml2arrow path.

An earlier version of this module called `xml2arrow.parse(data)`, a module-level function
no release has ever provided (0.3.0 through 0.19.0 export only `XmlToArrowParser`). Every
read raised, and `tests/io/test_optional_formats.py::test_xml_read` hid it by skipping
whenever the optional dependency was absent, which in CI is always.

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
import io
import json
import os
import re
import tempfile
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


#: How many records the schema scan reads before it stops. The scan exists to name the
#: columns and their types, not to read the data -- xml2arrow re-reads the document for the
#: values -- so it is bounded rather than proportional to the file.
_INFER_RECORDS = 200

#: The Arrow types xml2arrow can be asked for, widest last. A column whose samples all parse
#: as the earlier type gets it; anything else falls back to `Utf8`. There is deliberately no
#: temporal entry: xml2arrow's vocabulary is Boolean/Int/Float/Utf8 only, so a date column
#: reads back as the text it was written as rather than as `date32`.
_TRUTHY = frozenset({"true", "false"})

#: `data_type` -> the Arrow type it produces, for typing an empty result. A document whose
#: record element never appears yields no batch at all, and an empty table still owes the
#: schema the non-empty one would have had.
_ARROW_OF: dict[str, Any] = {
    "Boolean": pa.bool_(),
    "Int32": pa.int32(),
    "Int64": pa.int64(),
    "Float32": pa.float32(),
    "Float64": pa.float64(),
    "Utf8": pa.string(),
}


def _sampled_type(samples: list[str]) -> str:
    """The narrowest xml2arrow `data_type` that holds every sample.

    Named apart from `_leaf_type` below, which is the writer's and takes an Arrow type: the
    two are unrelated and a shared name meant the later definition silently won.
    """
    if not samples:
        return "Utf8"
    if all(v.lower() in _TRUTHY for v in samples):
        return "Boolean"
    try:
        for v in samples:
            int(v)
        return "Int64"
    except ValueError:
        pass
    try:
        for v in samples:
            float(v)
        return "Float64"
    except ValueError:
        return "Utf8"


def _infer_config(data: bytes) -> dict[str, Any]:
    """The xml2arrow table config this document implies, discovered by reading it.

    xml2arrow is configuration-driven: it has no schema inference of its own, and its
    `XmlToArrowParser` takes a YAML file naming the record container, every leaf's absolute
    path, and every leaf's type. Nothing in the engine wrote that file, so this builds one.

    The shape assumed is the one `XMLSink` writes and the one Spark's XML source reads: a
    root element wrapping one element per record. The record element is the root's repeated
    child; the columns are the scalar leaves beneath it (an element with no children of its
    own) plus the record's own attributes, named by their path under the record and joined
    with ``_`` when nested. A leaf that repeats within one record is read as its last value,
    because xml2arrow's flat table has nowhere to put a list.

    Args:
        data: The whole document.

    Returns:
        The config, as the mapping that is dumped to YAML for the parser.

    Raises:
        BackendError: If the document has no root, no repeated record element, or uses XML
            namespaces, whose expanded `{uri}tag` names do not address anything in an
            xml2arrow path.
    """
    import xml.etree.ElementTree as ET

    root_tag: str | None = None
    record_tag: str | None = None
    stack: list[str] = []
    has_child: set[tuple[str, ...]] = set()
    samples: dict[tuple[str, ...], list[str]] = {}
    seen = 0

    for event, elem in ET.iterparse(io.BytesIO(data), events=("start", "end")):
        if event == "start":
            if "}" in elem.tag or elem.tag.startswith("{"):
                raise BackendError(
                    f"XML namespaces are not supported by the reader: element {elem.tag!r}"
                )
            if stack:
                has_child.add(tuple(stack))
            stack.append(elem.tag)
            if root_tag is None:
                root_tag = elem.tag
            elif record_tag is None and len(stack) == 2:
                record_tag = elem.tag
            if len(stack) > 2:
                for name, value in elem.attrib.items():
                    samples.setdefault((*stack[2:], f"@{name}"), []).append(value)
            elif len(stack) == 2:
                for name, value in elem.attrib.items():
                    samples.setdefault((f"@{name}",), []).append(value)
        else:
            path = tuple(stack)
            if len(path) > 2 and path not in has_child:
                samples.setdefault(path[2:], []).append((elem.text or "").strip())
            stack.pop()
            if len(stack) == 1 and elem.tag == record_tag:
                seen += 1
                elem.clear()
                if seen >= _INFER_RECORDS:
                    break

    if root_tag is None or record_tag is None:
        raise BackendError("XML has no root element wrapping a repeated record element")

    fields = []
    for path, values in samples.items():
        present = [v for v in values if v != ""]
        fields.append(
            {
                "name": "_".join(part.lstrip("@") for part in path),
                "xml_path": "/" + "/".join((root_tag, record_tag, *path)),
                "data_type": _sampled_type(present),
                "nullable": True,
            }
        )
    if not fields:
        raise BackendError(f"XML record element <{record_tag}> has no readable fields")
    return {"tables": [{"name": "t", "xml_path": f"/{root_tag}", "levels": [], "fields": fields}]}


def _to_table(fh: IO[Any]) -> pa.Table:
    """Parse one XML file handle into an Arrow table via xml2arrow."""
    xml2arrow = _require_xml2arrow()
    data = fh.read()
    if isinstance(data, str):
        data = data.encode("utf-8")
    config = _infer_config(data)
    # `XmlToArrowParser` takes a config *path*, not a config object, so the inferred config
    # is written out rather than passed. Written as JSON, which YAML is a superset of, so the
    # engine gains no YAML dependency for a document it generates itself. Deleted on the way
    # out; the parser reads it once, in its constructor, and holds the compiled trie.
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        json.dump(config, handle)
        config_path = handle.name
    try:
        batches = xml2arrow.XmlToArrowParser(config_path).parse(data)
    except Exception as exc:
        raise BackendError(f"failed to parse XML: {exc}") from exc
    finally:
        os.unlink(config_path)
    tables = [pa.Table.from_batches([batch]) for batch in batches.values() if batch.num_rows]
    if not tables:
        names = [f["name"] for f in config["tables"][0]["fields"]]
        types = [f["data_type"] for f in config["tables"][0]["fields"]]
        return pa.table({n: pa.array([], _ARROW_OF[t]) for n, t in zip(names, types, strict=True)})
    return tables[0]


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
