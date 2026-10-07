"""`google_sheets`: read a range of a Google Sheet, and write a table into one.

Both directions use the Sheets API v4 ``spreadsheets.values`` resource over the shared
`http.transport.HttpClient`:

* **Read** is one ``GET .../values/{range}`` with ``majorDimension=ROWS``, the requested
  ``valueRenderOption`` and ``dateTimeRenderOption=FORMATTED_STRING``. The default
  ``UNFORMATTED_VALUE`` returns numbers as numbers and booleans as booleans, so a column
  comes back typed; ``FORMATTED_VALUE`` returns the text a user sees. The API omits
  trailing empty cells, so short rows are padded, and an empty cell (``""``) is null.
* **Write** is bounded and batched: rows go out ``batch_rows`` at a time through
  ``values:append``, so no single request carries an unbounded payload. The overwrite scope
  is the range the caller names and nothing else: ``mode="overwrite"`` first calls
  ``values:clear`` on exactly that range, then writes the header and rows into it with
  ``insertDataOption=OVERWRITE``, which never shifts cells outside it.
  ``mode="append"`` adds rows after the table the range holds (``INSERT_ROWS``), with no
  header.

**Auth.** A `BearerToken` (an access token by reference -- ``"cmd:..."`` can run
``gcloud auth print-access-token``), or, by default, Google Application Default Credentials
through `google-auth` (the ``gsheets`` extra), scoped to the Sheets API.

A column whose cells mix types (a number in one row, text in the next) is an error naming
the column, not a column silently coerced to text: declare ``schema=``, or read with
``value_render="FORMATTED_VALUE"`` to take every cell as text.
"""

from __future__ import annotations

import threading
from typing import Any
from urllib.parse import quote

import pyarrow as pa
import pyarrow.compute as pc

from batcher._internal.errors import BackendError, FormatError, PlanError
from batcher.io.formats.base import SINKS, SOURCES
from batcher.io.formats.http.auth import AuthProvider
from batcher.io.formats.http.options import RetryPolicy
from batcher.io.formats.http.transport import HttpClient
from batcher.io.manifest import WrittenFile

__all__ = ["GoogleSheetsSink", "GoogleSheetsSource"]

_API = "https://sheets.googleapis.com"
_SCOPES = ("https://www.googleapis.com/auth/spreadsheets",)
_RENDER = ("UNFORMATTED_VALUE", "FORMATTED_VALUE", "FORMULA")


class _GoogleDefaultCredentials:
    """Application Default Credentials as an auth provider (needs the ``gsheets`` extra)."""

    _lock = threading.Lock()
    _credentials: Any = None

    def headers(self) -> dict[str, str]:
        from batcher._internal.optional import require

        google_auth = require(
            "google.auth", feature="Google Sheets", provides="google-auth", extra="gsheets"
        )
        transport = require(
            "google.auth.transport.requests",
            feature="Google Sheets",
            provides="google-auth[requests]",
            extra="gsheets",
        )
        cls = type(self)
        with cls._lock:
            if cls._credentials is None:
                cls._credentials, _ = google_auth.default(scopes=list(_SCOPES))
            if not cls._credentials.valid:
                cls._credentials.refresh(transport.Request())
            return {"Authorization": f"Bearer {cls._credentials.token}"}

    def invalidate(self) -> None:
        with type(self)._lock:
            type(self)._credentials = None


def _values_url(base: str, spreadsheet_id: str, cell_range: str, verb: str = "") -> str:
    return (
        f"{base.rstrip('/')}/v4/spreadsheets/{quote(spreadsheet_id, safe='')}/values/"
        f"{quote(cell_range, safe='')}{verb}"
    )


def _client(auth: AuthProvider | None, retry: RetryPolicy | None) -> HttpClient:
    return HttpClient(auth=auth or _GoogleDefaultCredentials(), retry=retry)


def _column_names(header: bool | list[str], rows: list[list[Any]], width: int) -> list[str]:
    if isinstance(header, list):
        if len(header) < width:
            raise FormatError(f"header= names {len(header)} columns but the range has {width}")
        return list(header[:width])
    if not header:
        return [f"c{i}" for i in range(width)]
    first = rows[0] if rows else []
    names = [str(first[i]) if i < len(first) and first[i] != "" else f"c{i}" for i in range(width)]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise FormatError(
            f"the header row repeats column name(s) {duplicates}; pass header=[names] to "
            "name the columns yourself"
        )
    return names


def _column(name: str, values: list[Any], dtype: pa.DataType | None) -> pa.Array:
    if dtype is None:
        try:
            return pa.array(values)
        except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
            raise FormatError(
                f"column {name!r} mixes cell types ({exc}); declare schema=, or read with "
                "value_render='FORMATTED_VALUE' to take every cell as text"
            ) from exc
    try:
        return pa.array(values, type=dtype)
    except (pa.ArrowInvalid, pa.ArrowTypeError):
        pass
    try:
        if pa.types.is_string(dtype):
            return pa.array([None if v is None else str(v) for v in values], pa.string())
        return pa.array(values).cast(dtype)
    except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError) as exc:
        raise FormatError(f"column {name!r} does not fit {dtype}: {exc}") from exc


def values_to_table(
    rows: list[list[Any]], *, header: bool | list[str], schema: pa.Schema | None
) -> pa.Table:
    """The table a ``values`` response's rows hold, under the header policy.

    Args:
        rows: The response's ``values``: a list of rows, trailing empties omitted.
        header: True takes names from the first row, False names columns ``c0``...,
            a list names them.
        schema: Declared column types (only its columns are kept), or None to infer.

    Returns:
        The table.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.saas.sheets import values_to_table
            >>> values_to_table([["a", "b"], [1, "x"], [2]], header=True, schema=None).to_pydict()
            {'a': [1, 2], 'b': ['x', None]}
    """
    width = max((len(r) for r in rows), default=0)
    if isinstance(header, list):
        width = max(width, len(header))
    names = _column_names(header, rows, width)
    body = rows[1:] if header is True else rows
    cells = [[None if v == "" else v for v in r] + [None] * (width - len(r)) for r in body]
    by_name = {name: [row[i] for row in cells] for i, name in enumerate(names)}
    if schema is None:
        return pa.table({n: _column(n, vals, None) for n, vals in by_name.items()})
    missing = [f.name for f in schema if f.name not in by_name]
    if missing:
        raise FormatError(f"declared column(s) {missing} are not in the sheet's header {names}")
    return pa.Table.from_arrays(
        [_column(f.name, by_name[f.name], f.type) for f in schema], schema=schema
    )


@SOURCES.register("google_sheets")
class GoogleSheetsSource:
    """A range of a Google Sheet, read as one table.

    Args:
        spreadsheet_id: The spreadsheet's id.
        range: An A1 range, such as ``"Sheet1!A1:D"``.
        header: True (the first row names the columns), False, or a list of names.
        value_render: ``"UNFORMATTED_VALUE"``, ``"FORMATTED_VALUE"`` or ``"FORMULA"``.
        schema: Declared column types, or None to infer them.
        auth: A `BearerToken`; Application Default Credentials when None.
        base_url: The API root (for a test double).
        retry: A `RetryPolicy`.
    """

    format_name = "google_sheets"

    def __init__(
        self,
        spreadsheet_id: str,
        range: str,
        *,
        header: bool | list[str] = True,
        value_render: str = "UNFORMATTED_VALUE",
        schema: pa.Schema | None = None,
        auth: AuthProvider | None = None,
        base_url: str = _API,
        retry: RetryPolicy | None = None,
    ) -> None:
        if value_render not in _RENDER:
            raise PlanError(f"value_render must be one of {_RENDER}, got {value_render!r}")
        self._id = spreadsheet_id
        self._range = range
        self._header = header
        self._render = value_render
        self._declared = schema
        self._auth = auth
        self._base = base_url
        self._retry = retry
        self._schema: pa.Schema | None = schema

    def _fetch(self) -> pa.Table:
        response = _client(self._auth, self._retry).get(
            _values_url(self._base, self._id, self._range),
            {
                "majorDimension": "ROWS",
                "valueRenderOption": self._render,
                "dateTimeRenderOption": "FORMATTED_STRING",
            },
        )
        rows = response.json().get("values") or []
        return values_to_table(rows, header=self._header, schema=self._declared)

    def schema(self) -> pa.Schema:
        """The declared schema, or the one the range's cells imply."""
        if self._schema is None:
            self._schema = self._fetch().schema
        return self._schema

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        """The range as batches."""
        return list(self.iter_batches(projection))

    def iter_batches(self, projection: list[str] | None = None) -> Any:
        """The range as one batch (a sheet is read in a single request)."""
        table = self._fetch()
        if self._schema is not None and table.schema != self._schema:
            raise FormatError(
                f"sheet {self._range!r} changed shape since its schema was read: "
                f"{table.schema} vs {self._schema}"
            )
        if projection is not None:
            table = table.select(projection)
        yield from table.combine_chunks().to_batches()

    def row_count(self) -> int | None:
        """Unknown without reading the range."""
        return None

    def identity(self) -> str:
        """The spreadsheet and range."""
        return f"google_sheets:{self._id}:{self._range}"

    def splits(self, target_size: int | None = None) -> list[Any]:  # noqa: ARG002
        """One split: the range is one request."""
        from batcher.io.splits import WholeSourceSplit

        return [WholeSourceSplit(self)]


def _cells(table: pa.Table) -> list[list[Any]]:
    """`table` as JSON-ready rows: temporal and decimal columns as text, floats finite."""
    columns: list[list[Any]] = []
    for name, column in zip(table.column_names, table.columns, strict=True):
        dtype = column.type
        if pa.types.is_floating(dtype):
            finite = pc.fill_null(pc.is_finite(column), True)
            if not pc.all(finite).as_py():
                raise FormatError(
                    f"column {name!r} holds NaN or infinity, which Google Sheets cannot "
                    "store; fill or drop them first"
                )
        elif (
            pa.types.is_binary(dtype)
            or pa.types.is_large_binary(dtype)
            or pa.types.is_nested(dtype)
        ):
            raise FormatError(f"column {name!r} of type {dtype} has no Google Sheets cell form")
        elif not (
            pa.types.is_integer(dtype)
            or pa.types.is_boolean(dtype)
            or pa.types.is_string(dtype)
            or pa.types.is_large_string(dtype)
            or pa.types.is_null(dtype)
        ):
            column = column.cast(pa.string())
        columns.append(column.to_pylist())
    return [list(row) for row in zip(*columns, strict=True)] if columns else []


@SINKS.register("google_sheets")
class GoogleSheetsSink:
    """Write a table into a named range of a Google Sheet, in bounded batches.

    Args:
        range: The A1 range written into; under ``"overwrite"`` it is cleared first,
            and nothing outside it is touched.
        mode: ``"overwrite"`` or ``"append"``.
        header: Write the column names as the first row (``"overwrite"`` only).
        batch_rows: Rows per request.
        value_input: ``"RAW"`` (cells stored as given) or ``"USER_ENTERED"`` (parsed as
            if typed into the UI).
        auth: A `BearerToken`; Application Default Credentials when None.
        base_url: The API root (for a test double).
        retry: A `RetryPolicy`.
    """

    format_name = "google_sheets"
    dml_modes: tuple[str, ...] = ("overwrite", "append")

    def __init__(
        self,
        *,
        range: str,
        mode: str = "overwrite",
        header: bool = True,
        batch_rows: int = 500,
        value_input: str = "RAW",
        auth: AuthProvider | None = None,
        base_url: str = _API,
        retry: RetryPolicy | None = None,
    ) -> None:
        if mode not in self.dml_modes:
            raise BackendError(f"google_sheets write mode must be one of {self.dml_modes}")
        if batch_rows < 1:
            raise PlanError(f"batch_rows must be >= 1, got {batch_rows}")
        if value_input not in ("RAW", "USER_ENTERED"):
            raise PlanError(f"value_input must be 'RAW' or 'USER_ENTERED', got {value_input!r}")
        self._range = range
        self.mode = mode
        self._header = header
        self._batch = batch_rows
        self._input = value_input
        self._auth = auth
        self._base = base_url
        self._retry = retry

    def write(self, table: pa.Table, path: str) -> WrittenFile:
        """Write `table` into the range of spreadsheet `path`.

        Args:
            table: The rows to write.
            path: The spreadsheet id.

        Returns:
            A `WrittenFile` naming ``<id>!<range>`` and the rows written.
        """
        from batcher.plan.types import logical_bytes

        client = _client(self._auth, self._retry)
        rows = _cells(table)
        if self.mode == "overwrite":
            client.request(
                "POST", _values_url(self._base, path, self._range, ":clear"), json_body={}
            )
            if self._header:
                rows = [list(table.column_names), *rows]
        insert = "OVERWRITE" if self.mode == "overwrite" else "INSERT_ROWS"
        for start in range(0, len(rows), self._batch):
            client.request(
                "POST",
                _values_url(self._base, path, self._range, ":append"),
                params={"valueInputOption": self._input, "insertDataOption": insert},
                json_body={"majorDimension": "ROWS", "values": rows[start : start + self._batch]},
            )
        return WrittenFile(
            path=f"{path}!{self._range}", rows=table.num_rows, bytes=logical_bytes(table)
        )

    def write_partitioned(
        self,
        table: pa.Table,
        path: str,
        *,
        partition_by: list[str] | None = None,  # noqa: ARG002 - a sheet has no Hive layout
        file_index: int = 0,
    ) -> list[WrittenFile]:
        """Write one shard; an overwrite must stay on one shard, or each clears the last."""
        if file_index > 0 and self.mode == "overwrite":
            raise BackendError(
                "mode='overwrite' cannot be used for a distributed write to a Google Sheet: "
                "every shard would clear the range the shard before it wrote. Write from "
                "one worker (distributed=False), or use mode='append'."
            )
        return [self.write(table, path)]

    def commit(self, manifest: Any, path: str) -> None:
        """No commit phase: values are visible as soon as each request returns."""
