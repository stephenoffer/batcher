"""XML format — Arrow-native nested read via `xml2arrow`.

`xml2arrow` parses XML directly into Arrow (preserving nested structure) without
materializing Python objects per row, so `XMLSource` reads to Arrow without a
row-oriented hop. XML is read-only here (there is no `XMLSink`); persist results
as Parquet/Arrow. One whole file is one `Split`.

All `xml2arrow` imports are deferred — importing this module never requires the
optional dependency. A missing dependency raises `BackendError` with a
``pip install 'batcher[xml]'`` hint.
"""

from __future__ import annotations

from typing import IO, Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.io.base import FileSource
from batcher.io.formats.base import SOURCES

__all__ = ["XMLSource"]


def _require_xml2arrow() -> Any:
    """Import and return the `xml2arrow` module or raise `BackendError`."""
    try:
        import xml2arrow
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise BackendError("XML support requires xml2arrow: pip install 'batcher[xml]'") from exc
    return xml2arrow


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
