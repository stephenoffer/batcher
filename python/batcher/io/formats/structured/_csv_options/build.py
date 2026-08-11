"""Resolved CSV options, and the pyarrow option objects they build.

`CSVReadOptions` is the single carrier of "how is this file parsed": one source builds it
once, every read path asks it for the pyarrow objects, and `as_kwargs()`/`range_kwargs()`
are how it survives to a worker. `CSVWriteOptions` is its writer counterpart.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import pyarrow as pa

from batcher._internal.errors import FormatError, SchemaError
from batcher.io.base._bad_rows import bad_row_handler
from batcher.io.formats.structured._csv_options.dtypes import (
    DATE_FORMATS,
    arrow_type,
    header_and_skip,
)
from batcher.io.formats.structured._csv_options.spec import NO_INDEX, READ_SPEC, WRITE_SPEC

__all__ = ["CSVReadOptions", "CSVWriteOptions", "resolve_read_options", "resolve_write_options"]


@dataclass(frozen=True, slots=True)
class CSVReadOptions:
    """Resolved, canonical CSV reader options, ready to build pyarrow option objects."""

    delimiter: str | None = None
    quote_char: str | None = None
    escape_char: str | None = None
    has_header: bool = True
    column_names: tuple[str, ...] | None = None
    null_values: tuple[str, ...] | None = None
    skip_rows: int = 0
    skip_rows_after_header: int = 0
    encoding: str = "utf8"
    true_values: tuple[str, ...] | None = None
    false_values: tuple[str, ...] | None = None
    decimal_point: str | None = None
    try_parse_dates: bool = False
    date_columns: tuple[str, ...] = ()
    declared_schema: pa.Schema | None = None
    schema_overrides: tuple[tuple[str, pa.DataType], ...] = ()
    #: What to do with a line whose field count disagrees with the header. Defaults to
    #: aborting the read, which is pyarrow's behavior and the safe answer for a file you
    #: just wrote; ``"skip"``/``"warn"`` drop the line instead, which is what a corpus at
    #: scale needs. See `_csv_diagnostics.BadRowPolicy`.
    on_bad_lines: str = "error"

    @property
    def range_safe(self) -> bool:
        """Whether a byte-range split can re-parse this file with these options.

        A range is parsed from raw bytes with the header line prepended, so an option that
        reframes rows (`skip_rows`, `column_names`, a headerless file) or bytes (a non-UTF-8
        `encoding`) would be applied once per range instead of once per file. That is silent
        row loss on the distributed path alone, so a source reads such a file whole instead.

        Returns:
            True when every option is safe to re-apply to an arbitrary byte range.
        """
        return (
            self.has_header
            and not self.skip_rows
            and not self.skip_rows_after_header
            and self.column_names is None
            and self.encoding.lower().replace("-", "") in ("utf8", "ascii")
        )

    def read_options(self) -> Any:
        """The pyarrow `ReadOptions` for this configuration.

        Returns:
            The configured `pyarrow.csv.ReadOptions`.
        """
        import pyarrow.csv as pacsv

        return pacsv.ReadOptions(
            column_names=list(self.column_names) if self.column_names is not None else None,
            # Names are auto-generated only when the file has no header AND the caller gave
            # none; passing both makes pyarrow raise.
            autogenerate_column_names=not self.has_header and self.column_names is None,
            skip_rows=self.skip_rows,
            skip_rows_after_names=self.skip_rows_after_header,
            encoding=self.encoding,
        )

    def parse_options(self, path: str = "", *, observe: bool = True) -> Any:
        """The pyarrow `ParseOptions` for this configuration.

        Args:
            path: The file about to be parsed, used only in the malformed-row warning so a
                corpus read says which member the dropped line came from.
            observe: Whether dropped rows are counted and announced. False for the
                schema-inference pass, which meets the same rows the read will.

        Returns:
            The configured `pyarrow.csv.ParseOptions`.
        """
        import pyarrow.csv as pacsv

        kwargs: dict[str, Any] = {}
        if self.delimiter is not None:
            kwargs["delimiter"] = self.delimiter
        if self.quote_char is not None:
            kwargs["quote_char"] = self.quote_char
        if self.escape_char is not None:
            kwargs["escape_char"] = self.escape_char
        # Built fresh per call rather than cached on the (frozen) options: the handler keeps
        # a drop count, and one shared across every file of a corpus would report the corpus
        # total against whichever file happened to trip the warning limit first.
        handler = bad_row_handler(self.on_bad_lines, path, format_name="csv", observe=observe)
        if handler is not None:
            kwargs["invalid_row_handler"] = handler
        return pacsv.ParseOptions(**kwargs)

    def convert_options(
        self,
        *,
        column_types: dict[str, pa.DataType] | None = None,
        include_columns: list[str] | None = None,
    ) -> Any:
        """The pyarrow `ConvertOptions`, pinning `column_types` where the caller supplies them.

        Args:
            column_types: The types every read path must produce, so they cannot disagree.
            include_columns: The projected column subset, pushed into the parse.

        Returns:
            The configured `pyarrow.csv.ConvertOptions`.
        """
        import pyarrow.csv as pacsv

        kwargs: dict[str, Any] = {"include_columns": include_columns}
        if column_types is not None:
            kwargs["column_types"] = column_types
        if self.null_values is not None:
            # Extend rather than replace Arrow's defaults: pandas' `na_values` is additive,
            # and naming one extra token does not mean "stop treating '' and 'NA' as null".
            defaults = tuple(pacsv.ConvertOptions().null_values or ())
            kwargs["null_values"] = list(dict.fromkeys((*defaults, *self.null_values)))
        if self.true_values is not None:
            kwargs["true_values"] = list(self.true_values)
        if self.false_values is not None:
            kwargs["false_values"] = list(self.false_values)
        if self.decimal_point is not None:
            kwargs["decimal_point"] = self.decimal_point
        if self.try_parse_dates:
            kwargs["timestamp_parsers"] = [pacsv.ISO8601, *DATE_FORMATS]
        return pacsv.ConvertOptions(**kwargs)

    def resolve_schema(self, inferred: pa.Schema) -> pa.Schema:
        """The schema every read path is pinned to, given what inference saw.

        A full `pa.Schema` declares the whole file and wins outright. A `dtype` dict is a
        per-column *override* laid over inference, which is what pandas' `dtype=` and
        Polars' `schema_overrides=` mean, and `parse_dates=[...]` is the same thing spelled
        as a column list.

        Args:
            inferred: The schema read from the file's first block.

        Returns:
            The schema `schema()`, `read()`, `iter_batches()`, and every split must agree on.
        """
        if self.declared_schema is not None:
            return self.declared_schema
        overrides = dict(self.schema_overrides)
        overrides.update({name: pa.timestamp("us") for name in self.date_columns})
        if not overrides:
            return inferred
        missing = [name for name in overrides if name not in inferred.names]
        if missing:
            raise SchemaError(
                f"csv: dtype/parse_dates names {missing!r} are not columns of the file "
                f"(it has {list(inferred.names)!r}). Check the spelling, or declare the whole "
                f"file with schema=pa.schema([...])."
            )
        return pa.schema(
            [pa.field(f.name, overrides.get(f.name, f.type), f.nullable) for f in inferred]
        )

    def as_kwargs(self) -> dict[str, Any]:
        """The non-default options, as constructor keywords a worker can rebuild from.

        Every behavior-changing option must survive to a worker, which reconstructs the
        reader as ``SOURCES.get("csv")(path, **kwargs)``. Anything omitted here silently
        reverts to its default, so a distributed read would parse with a different delimiter
        than the plan was built against.

        Returns:
            Canonical option names mapped to their values, defaults omitted.
        """
        default = CSVReadOptions()
        # The schema fields are excluded because the *source*, not this object, decides what
        # a worker is pinned to: it sends the resolved `schema()` so a one-file worker cannot
        # re-infer a different answer from the rows it happens to hold.
        skip = ("declared_schema", "schema_overrides", "date_columns")
        out: dict[str, Any] = {}
        for spec in fields(self):
            if spec.name in skip:
                continue
            value = getattr(self, spec.name)
            if value != getattr(default, spec.name):
                out[spec.name] = list(value) if isinstance(value, tuple) else value
        if self.date_columns:
            out["parse_dates"] = list(self.date_columns)
        return out

    def range_kwargs(self) -> dict[str, Any]:
        """The subset of `as_kwargs()` a byte-range split needs in order to parse identically.

        A range re-runs only the *parse* and *convert* stages — its column names come from
        the prepended header — so it needs the separator, quoting and value vocabulary, and
        nothing that reframes rows. `range_safe` already guarantees the rest are at their
        defaults; naming the subset here is what makes that guarantee explicit rather than
        incidental, so a future option is not silently assumed safe.

        Returns:
            Canonical option names mapped to their values, defaults omitted.
        """
        keep = (
            "delimiter",
            "quote_char",
            "escape_char",
            "null_values",
            "true_values",
            "false_values",
            "decimal_point",
            "try_parse_dates",
            # Range-safe by construction: a field-count mismatch is a property of one line,
            # and byte ranges are newline-aligned, so a line is whole inside exactly one
            # range. Omitting it would make a distributed read of a corpus with a stray line
            # fail where the single-node read of the same file succeeded.
            "on_bad_lines",
        )
        return {k: v for k, v in self.as_kwargs().items() if k in keep}


def resolve_read_options(opts: dict[str, Any]) -> CSVReadOptions:
    """Fold reader keywords into canonical options, rejecting the unsupported ones.

    Args:
        opts: The CSV-specific keywords the caller passed, in any accepted spelling.

    Returns:
        The resolved options.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.structured._csv_options import resolve_read_options
            >>> resolve_read_options({"sep": ";", "skiprows": 2}).delimiter
            ';'
    """
    resolved = READ_SPEC.resolve(opts)
    out: dict[str, Any] = {}
    extra_skip = 0
    if "has_header" in resolved:
        out["has_header"], extra_skip = header_and_skip(resolved.pop("has_header"))
    if "skip_rows" in resolved or extra_skip:
        out["skip_rows"] = int(resolved.pop("skip_rows", 0)) + extra_skip
    if "try_parse_dates" in resolved:
        out.update(_dates(resolved.pop("try_parse_dates")))
    if "schema" in resolved:
        out.update(_schema(resolved.pop("schema")))
    for name in ("column_names", "null_values", "true_values", "false_values"):
        if name in resolved:
            out[name] = tuple(resolved.pop(name))
    out.update(resolved)
    options = CSVReadOptions(**out)
    # Validate here rather than at parse time. `on_bad_lines="Skip"` would otherwise build a
    # source, infer a schema and plan a query before the first byte-range worker raised, and
    # under `on_error="skip"` the raise would be swallowed as an unreadable file — a typo in
    # a tolerance flag turning into silent whole-corpus loss.
    bad_row_handler(options.on_bad_lines)
    return options


def _dates(value: Any) -> dict[str, Any]:
    """Split `parse_dates=` into its two distinct requests.

    pandas spells "these columns are dates" as a list; Polars spells "try harder
    everywhere" as a bool. One keyword, two meanings, so they map to two fields.
    """
    if isinstance(value, bool):
        return {"try_parse_dates": value}
    if isinstance(value, (list, tuple)):
        return {"date_columns": tuple(value)}
    raise FormatError(
        f"csv: parse_dates={value!r} is neither a flag nor a list of column names. Pass True "
        f"to widen date inference, or a list of columns to parse as dates."
    )


def _schema(value: Any) -> dict[str, Any]:
    """Split `schema=`/`dtype=` into a full declaration or a per-column override set."""
    if isinstance(value, pa.Schema):
        return {"declared_schema": value}
    if isinstance(value, dict):
        return {"schema_overrides": tuple((n, arrow_type(n, t)) for n, t in value.items())}
    raise SchemaError(
        f"csv: schema={value!r} is neither a pyarrow Schema nor a {{'column': type}} "
        f"mapping. Pass pa.schema([...]) to declare every column, or a dict to override the "
        f"inferred type of some of them."
    )


@dataclass(frozen=True, slots=True)
class CSVWriteOptions:
    """Resolved, canonical CSV writer options."""

    delimiter: str | None = None
    header: bool = True
    null_value: str | None = None

    def write_options(self, *, include_header: bool) -> Any:
        """The pyarrow `WriteOptions` for one encoded chunk.

        Args:
            include_header: Whether *this* chunk is the one whose turn it is to carry the
                header. A parallel or streaming write encodes many chunks and only the first
                may carry it, so the flag is per chunk and is ANDed with the caller's
                `header=` — which is what makes `header=False` suppress it everywhere.

        Returns:
            The configured `pyarrow.csv.WriteOptions`.
        """
        import pyarrow.csv as pacsv

        kwargs: dict[str, Any] = {"include_header": include_header and self.header}
        if self.delimiter is not None:
            kwargs["delimiter"] = self.delimiter
        return pacsv.WriteOptions(**kwargs)

    def apply_nulls(self, table: pa.Table) -> pa.Table:
        """Render nulls as `null_value` rather than as Arrow's empty field.

        Arrow's CSV writer has no null-representation option, so the substitution happens in
        the data: every column is cast to string and filled. The cast is a vectorized Arrow
        kernel, not a row loop, and it runs only when `null_value` is set, so an ordinary
        write costs nothing.

        Casting *every* column, rather than only those that currently hold a null, is
        deliberate: a streaming write encodes many batches against one schema, and a
        per-batch decision would give two batches of the same file different schemas.

        The visible consequence is that `null_value=` also quotes every field, because
        Arrow's writer quotes string values and there is no way to ask it not to without
        also disabling the quoting that keeps a value containing the delimiter readable.
        The output is still valid CSV that round-trips to the same values.

        Args:
            table: The rows about to be encoded.

        Returns:
            The table with nulls replaced, or the same table when there is nothing to do.
        """
        if self.null_value is None:
            return table
        import pyarrow.compute as pc

        return pa.table(
            [pc.fill_null(pc.cast(c, pa.string()), self.null_value) for c in table.columns],
            schema=self.null_schema(table.schema),
        )

    def null_schema(self, schema: pa.Schema) -> pa.Schema:
        """The schema `apply_nulls` produces, for an incremental writer opened ahead of time.

        Args:
            schema: The schema of the batches about to be written.

        Returns:
            The all-string schema when `null_value` is set, else `schema` unchanged.
        """
        if self.null_value is None:
            return schema
        return pa.schema([pa.field(f.name, pa.string(), f.nullable) for f in schema])


def resolve_write_options(opts: dict[str, Any]) -> CSVWriteOptions:
    """Fold writer keywords into canonical options, rejecting the unsupported ones.

    Args:
        opts: The CSV-specific keywords the caller passed, in any accepted spelling.

    Returns:
        The resolved options.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.structured._csv_options import resolve_write_options
            >>> resolve_write_options({"na_rep": "NA"}).null_value
            'NA'
    """
    resolved = WRITE_SPEC.resolve(opts)
    # `index=False` is noise in a ported pandas call and is accepted as such; `index=True`
    # asks for a column that does not exist, which has to be said out loud.
    if resolved.pop("index", False):
        raise FormatError(f"csv: 'index' is not a Batcher option. {NO_INDEX}")
    return CSVWriteOptions(**resolved)
