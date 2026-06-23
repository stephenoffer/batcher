"""Excel format — read-only sheet ingestion via `python-calamine`, to Arrow.

calamine is a fast, pure-Rust spreadsheet reader (no Excel/LibreOffice needed).
`ExcelSource` reads one worksheet, taking the first row as the header, and
assembles the rows into Arrow at *batch* granularity — the unavoidable
deserialization for a row-oriented, non-Arrow source. Excel is read-only here
(there is no `ExcelSink`); persist results as Parquet/Arrow instead.

All `python_calamine` imports are deferred — importing this module never requires
the optional dependency. A missing dependency raises `BackendError` with a
``pip install 'batcher[excel]'`` hint.
"""

from __future__ import annotations

from typing import IO, Any

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.config import active_config
from batcher.io.base import FileSource
from batcher.io.formats.base import SOURCES

__all__ = ["ExcelSource"]


def _require_calamine() -> Any:
    """Import and return the `python_calamine` module or raise `BackendError`."""
    try:
        import python_calamine
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise BackendError(
            "Excel support requires python-calamine: pip install 'batcher[excel]'"
        ) from exc
    return python_calamine


def _rows(fh: IO[Any], sheet: str | int) -> list[list[Any]]:
    """Read a worksheet into a list of cell rows via calamine."""
    calamine = _require_calamine()
    workbook = calamine.load_workbook(fh)
    name = workbook.sheet_names[sheet] if isinstance(sheet, int) else sheet
    return workbook.get_sheet_by_name(name).to_python()


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

    def __init__(self, path: str, *, sheet: str | int = 0) -> None:
        super().__init__(path)
        self._sheet = sheet

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
