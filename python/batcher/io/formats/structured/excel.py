"""Excel format — read-only sheet ingestion via `python-calamine`, to Arrow.

calamine is a fast, pure-Rust spreadsheet reader (no Excel/LibreOffice needed).
`ExcelSource` reads one worksheet, taking the first row as the header, and
assembles the rows into Arrow at *batch* granularity — the unavoidable
deserialization for a row-oriented, non-Arrow source. Excel is read-only here
(there is no `ExcelSink`); persist results as Parquet/Arrow instead.

All `python_calamine` imports are deferred — importing this module never requires
the optional dependency. A missing dependency raises `BackendError` with a
``pip install 'batcher-engine[excel]'`` hint.
"""

from __future__ import annotations

from typing import IO, Any

import pyarrow as pa

from batcher._internal.optional import require
from batcher.config import active_config
from batcher.io.base import FileSource
from batcher.io.formats.base import SOURCES

__all__ = ["ExcelSource"]


def _require_calamine() -> Any:
    """Import and return the `python_calamine` module or raise `BackendError`."""
    return require(
        "python_calamine", feature="Excel support", provides="python-calamine", extra="excel"
    )


def _rows(fh: IO[Any], sheet: str | int) -> list[list[Any]]:
    """Read a worksheet into a list of cell rows via calamine."""
    calamine = _require_calamine()
    workbook = calamine.load_workbook(fh)
    return workbook.get_sheet_by_name(_sheet_name(workbook.sheet_names, sheet)).to_python()


def _sheet_name(names: list[str], sheet: str | int) -> str:
    """The worksheet `sheet` names, or a `FormatError` listing the ones that exist.

    The refusal is the discovery mechanism: a workbook's sheet names are otherwise only
    visible by opening it elsewhere, and calamine's own error for a bad name or index says
    neither which sheets there are nor which one was meant.

    Args:
        names: The workbook's sheet names, in order.
        sheet: A sheet name, or a zero-based index.

    Returns:
        The sheet's name.

    Raises:
        FormatError: When no sheet has that name or index.
    """
    from batcher._internal.errors import FormatError, unknown_value

    if isinstance(sheet, int):
        if -len(names) <= sheet < len(names):
            return names[sheet]
        raise FormatError(
            f"excel: sheet={sheet} is out of range; the workbook has {len(names)} sheet(s): {names}"
        )
    if sheet in names:
        return sheet
    raise unknown_value(
        FormatError, "sheet", sheet, names, hint="pass sheet= as one of these names or an index."
    )


def _to_columns(rows: list[list[Any]]) -> tuple[list[str], list[list[Any]]]:
    """Split a header row + data rows into (column names, column-major data)."""
    if not rows:
        return [], []
    header = [str(c) for c in rows[0]]
    columns: list[list[Any]] = [[] for _ in header]
    for row in rows[1:]:
        for i in range(len(header)):
            columns[i].append(row[i] if i < len(row) else None)
    return header, columns


@SOURCES.register("excel")
class ExcelSource(FileSource):
    """One worksheet of an Excel/ODS workbook, read to Arrow (read-only).

    Args:
        path: The workbook file (single file, directory, or glob).
        sheet: The worksheet to read — name (str) or zero-based index (int);
            defaults to the first sheet.
    """

    suffix = ".xlsx"
    format_name = "excel"

    __slots__ = ("_sheet",)

    def __init__(self, path: str, *, sheet: str | int = 0, **kwargs: Any) -> None:
        # Forward the base options: without this, `on_error="skip"` and `schema_mode`
        # were accepted by the reader and silently did nothing.
        super().__init__(path, **kwargs)
        self._sheet = sheet

    def _reader_kwargs(self) -> dict[str, object]:
        # Without the `sheet`, a worker rebuilding the reader falls back to sheet 0 and silently
        # reads a different worksheet than single-node requested. Carry it to the worker.
        return {**super()._reader_kwargs(), "sheet": self._sheet}

    def _schema_cache_token(self) -> object:
        # Two sheets of one workbook are one file and two schemas. Keyed on the file alone,
        # the second sheet read was served the first sheet's columns.
        return (self._sheet,)

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:
        header, columns = _to_columns(_rows(fh, self._sheet))
        return self._batches(header, columns)[0].schema if header else pa.schema([])

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        header, columns = _to_columns(_rows(fh, self._sheet))
        if not header:
            return []
        batches = self._batches(header, columns)
        if projection is not None:
            batches = [b.select(projection) for b in batches]
        return batches

    @staticmethod
    def _batches(header: list[str], columns: list[list[Any]]) -> list[pa.RecordBatch]:
        nrows = len(columns[0]) if columns else 0
        batch_rows = active_config().execution.morsel_rows
        out: list[pa.RecordBatch] = []
        for start in range(0, max(nrows, 1), batch_rows):
            stop = min(start + batch_rows, nrows)
            data = {name: columns[i][start:stop] for i, name in enumerate(header)}
            out.append(pa.RecordBatch.from_pydict(data))
        return out
