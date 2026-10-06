"""The CSV keyword tables — which spellings are accepted, refused, or a no-op.

One `OptionSpec` per direction. Nothing here builds a pyarrow object or parses a value;
this module is purely the vocabulary, so the answer to "does Batcher take `na_values`?"
lives in one readable place.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from batcher.io.base._options import (
    BASE_SINK_OPTIONS,
    BASE_SOURCE_OPTIONS,
    BaseReadOptions,
    OptionSpec,
)

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = ["NO_INDEX", "READ_SPEC", "WRITE_SPEC", "CsvReadOptions"]

NO_INDEX = (
    "Batcher has no row index; every column is a real column. Drop it, and select the "
    "column explicitly if you want it in the output."
)
_NO_COMMENT = (
    "pyarrow's CSV parser has no comment-line support, and ignoring this silently would "
    "parse the comment lines as data rows. Strip the comments before reading, or pass "
    "skip_rows= if they are a fixed leading block."
)
_SPARK_MODE = (
    "Spark's read mode has no single Batcher spelling because it conflates two independent "
    "decisions. Pass on_bad_lines='skip' for DROPMALFORMED and on_bad_lines='error' (the "
    "default) for FAILFAST. PERMISSIVE, which keeps a malformed row and parks its text in a "
    "corrupt-record column, has no equivalent."
)
_POLARS_IGNORE_ERRORS = (
    "Polars folds two behaviors into this flag. Pass on_bad_lines='skip' to drop rows with "
    "the wrong field count, and declare the column as a string with schema= if what you want "
    "is for an unconvertible value to survive the read."
)
_DEPRECATED_BAD_LINES = (
    "pandas retired these flags in 2.0. Use on_bad_lines='error', 'warn', or 'skip'."
)
_NO_QUOTE_CHAR = (
    "Arrow's CSV writer always quotes with '\"'. There is no way to change it, and "
    "ignoring this would emit output the option says it would not."
)

# `base=` is the keywords `FileSource` itself consumes. `CSVSource` splits them off by
# reading the base signature, so they never reach `resolve()` — but naming them here keeps
# the "accepted options" listing in an unknown-keyword error honest about them.
READ_SPEC = OptionSpec(
    "csv",
    base=BASE_SOURCE_OPTIONS,
    canonical=(
        "delimiter",
        "quote_char",
        "escape_char",
        "has_header",
        "column_names",
        "null_values",
        "skip_rows",
        "skip_rows_after_header",
        "encoding",
        "schema",
        "true_values",
        "false_values",
        "decimal_point",
        "try_parse_dates",
        "on_bad_lines",
    ),
    aliases={
        "on_bad_rows": "on_bad_lines",
        "sep": "delimiter",
        "separator": "delimiter",
        "quotechar": "quote_char",
        "escapechar": "escape_char",
        "header": "has_header",
        "names": "column_names",
        "new_columns": "column_names",
        "na_values": "null_values",
        "skiprows": "skip_rows",
        "skip_rows_after_names": "skip_rows_after_header",
        "dtype": "schema",
        "dtypes": "schema",
        "schema_overrides": "schema",
        "parse_dates": "try_parse_dates",
    },
    unsupported={
        "index_col": NO_INDEX,
        "mode": _SPARK_MODE,
        "ignore_errors": _POLARS_IGNORE_ERRORS,
        "error_bad_lines": _DEPRECATED_BAD_LINES,
        "warn_bad_lines": _DEPRECATED_BAD_LINES,
        "comment": _NO_COMMENT,
        "comment_prefix": _NO_COMMENT,
        "skipfooter": (
            "Trailing rows cannot be skipped without reading the file twice, which a lazy "
            "scan will not do. Read the file and drop the tail with a filter instead."
        ),
        "thousands": (
            "Arrow's CSV converter has no thousands separator. Read the column as a string "
            "and strip the separator with .str.replace() before casting."
        ),
        "chunksize": (
            "A Batcher read is already lazy and streams in batches. Use ds.iter_batches() "
            "to consume it a batch at a time."
        ),
        "iterator": (
            "A Batcher read is already lazy; nothing is materialized until a terminal "
            "operation. Use ds.iter_batches() to stream."
        ),
        "squeeze": (
            "A one-column result stays a Dataset. Call ds.to_pydict()[col] if you want the "
            "bare column."
        ),
    },
    ignored={
        "low_memory": "Batcher always streams the parse; there is no low-memory mode to select.",
        "memory_map": "Access is chosen by the filesystem backend, not per read.",
        "engine": "There is one CSV parser (Arrow's), so there is no engine to select.",
        "use_threads": "Parallelism is decided by the executor, not per source.",
        "rechunk": "Batch layout is owned by the engine's morsel sizing.",
        "cache": "Caching is a Dataset operation (ds.cache()), not a read option.",
    },
)

WRITE_SPEC = OptionSpec(
    "csv",
    base=BASE_SINK_OPTIONS,
    canonical=("delimiter", "header", "null_value", "index"),
    aliases={
        "sep": "delimiter",
        "separator": "delimiter",
        "include_header": "header",
        "with_header": "header",
        "na_rep": "null_value",
    },
    unsupported={
        "quote_char": _NO_QUOTE_CHAR,
        "quotechar": _NO_QUOTE_CHAR,
        "index_col": NO_INDEX,
    },
)


class CsvReadOptions(BaseReadOptions, total=False):
    """`bt.read.csv`'s keywords, typed: `READ_SPEC`'s vocabulary, aliases and no-ops included.

    Kept equal to `READ_SPEC.accepted` by `tests/unit/test_reader_option_types.py`, so the
    static view cannot drift from the runtime one. An alias carries its canonical option's
    type.
    """

    delimiter: str
    quote_char: str | Literal[False]
    escape_char: str
    has_header: bool | int | None
    column_names: list[str]
    null_values: str | list[str]
    skip_rows: int
    skip_rows_after_header: int
    encoding: str
    schema: pa.Schema | Mapping[str, Any]
    true_values: list[str]
    false_values: list[str]
    decimal_point: str
    try_parse_dates: bool | list[str]
    on_bad_lines: Literal["error", "warn", "skip"]
    on_bad_rows: Literal["error", "warn", "skip"]
    sep: str
    separator: str
    quotechar: str | Literal[False]
    escapechar: str
    header: bool | int | None
    names: list[str]
    new_columns: list[str]
    na_values: str | list[str]
    skiprows: int
    skip_rows_after_names: int
    dtype: pa.Schema | Mapping[str, Any]
    dtypes: pa.Schema | Mapping[str, Any]
    schema_overrides: pa.Schema | Mapping[str, Any]
    parse_dates: bool | list[str]
    low_memory: bool
    memory_map: bool
    engine: str
    use_threads: bool
    rechunk: bool
    cache: bool
