"""From a decoded API page to one Arrow batch, holding every page to one schema.

A JSON API hands back records as objects, so the bridge to Arrow is the one every
row-shaped connector in `io/formats/nosql` uses: `pa.RecordBatch.from_pylist` over the
page, at page granularity, at the IO boundary. No per-row query logic runs here.

**The schema is fixed before the first row is read, and every page is held to it.** A plan
is built against `Source.schema()`, so a page that disagrees cannot be allowed to change the
columns underneath it. When the caller declares ``schema=``, that is the contract: fields
the API sends beyond it are not read, and a value that does not fit its declared type is an
error naming the field. When the schema is *inferred* (from the first page), a later page
carrying a field the first page did not is an error rather than a silent drop, and so is a
value arriving in a field the first page held only nulls in: both say "declare ``schema=``",
which is the only way to read such an API without losing data.

Temporal columns are the one conversion done here rather than by `from_pylist`, because a
JSON API sends a timestamp as an ISO-8601 string and `from_pylist` will not parse one. A
top-level timestamp, date or time field in the declared schema is read as a string and
cast to its declared type with Arrow's own (vectorized) parser.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from batcher._internal.errors import FormatError
from batcher.io.formats.http.options import FieldPath, dig

__all__ = ["PageBuilder", "infer_schema", "page_records"]


def page_records(document: Any, records_path: FieldPath | None, *, where: str) -> list[dict]:
    """The list of record objects at `records_path` in a decoded page.

    Args:
        document: The decoded page.
        records_path: Where the records sit; None when the page *is* the list.
        where: The page's (redacted) URL, for the error message.

    Returns:
        The records; an empty list when the path is absent or null.

    Raises:
        FormatError: When the value there is not a list of objects.
    """
    records = dig(document, records_path)
    if records is None:
        return []
    if isinstance(records, dict) and not records_path:
        records = [records]
    if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
        found = type(records).__name__
        raise FormatError(
            f"{where}: expected a list of JSON objects at records_path={records_path!r}, "
            f"found {found}. Point records_path at the array holding the records."
        )
    return records


def _is_temporal(dtype: pa.DataType) -> bool:
    return pa.types.is_timestamp(dtype) or pa.types.is_date(dtype) or pa.types.is_time(dtype)


class PageBuilder:
    """Convert pages of records to batches of one fixed schema.

    Args:
        schema: The schema every batch has.
        declared: Whether the caller declared `schema` (extra fields are then ignored)
            or it was inferred from the first page (extra fields are then an error).
    """

    __slots__ = ("_declared", "_names", "_schema", "_temporal", "_wire")

    def __init__(self, schema: pa.Schema, *, declared: bool) -> None:
        self._schema = schema
        self._declared = declared
        self._names = frozenset(schema.names)
        self._temporal = [i for i, f in enumerate(schema) if _is_temporal(f.type)]
        wire = schema
        for i in self._temporal:
            wire = wire.set(i, pa.field(schema.field(i).name, pa.string()))
        self._wire = wire

    @property
    def schema(self) -> pa.Schema:
        """The schema every batch this builder returns has."""
        return self._schema

    def batch(self, records: list[dict], *, where: str) -> pa.RecordBatch:
        """One batch holding `records`, in the builder's schema.

        Args:
            records: The page's record objects.
            where: The page's (redacted) URL, for an error message.

        Returns:
            The batch.

        Raises:
            FormatError: When a record does not fit the schema.
        """
        if not self._declared:
            self._check_fields(records, where)
        try:
            batch = pa.RecordBatch.from_pylist(records, schema=self._wire)
        except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
            raise FormatError(self._mismatch(records, exc, where)) from exc
        if not self._temporal:
            return batch
        columns = list(batch.columns)
        for i in self._temporal:
            field = self._schema.field(i)
            try:
                columns[i] = columns[i].cast(field.type)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
                raise FormatError(
                    f"{where}: field {field.name!r} holds a value that does not parse as "
                    f"{field.type}: {exc}. Declare it as pa.string() in schema= to read it "
                    "as text."
                ) from exc
        return pa.RecordBatch.from_arrays(columns, schema=self._schema)

    def _check_fields(self, records: list[dict], where: str) -> None:
        extra = set().union(*records) - self._names if records else set()
        if extra:
            raise FormatError(
                f"{where}: records carry field(s) {sorted(extra)} that the first page did "
                "not, so the schema inferred from it cannot hold them. Declare schema= "
                "with every field you want to read (a declared schema ignores the rest)."
            )

    def _mismatch(self, records: list[dict], exc: Exception, where: str) -> str:
        nulls = [f.name for f in self._schema if pa.types.is_null(f.type)]
        hint = (
            f" Field(s) {nulls} were null throughout the first page, so their type could "
            "not be inferred; declare them in schema=."
            if nulls and any(r.get(n) is not None for r in records for n in nulls)
            else " Declare schema= with the type the field actually carries."
        )
        return f"{where}: a record does not fit the schema: {exc}.{hint}"


def infer_schema(records: list[dict]) -> pa.Schema:
    """The schema a sample page implies, over every record rather than the first.

    Args:
        records: The first page's records.

    Returns:
        The inferred schema; empty when the page has no records.
    """
    from batcher.io.formats.nosql.base import schema_from_rows

    return schema_from_rows(records)
