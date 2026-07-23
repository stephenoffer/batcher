"""Value coercion for the CSV options whose value is not already what pyarrow wants.

Two of them carry real semantics rather than a cast: a `dtype` entry may be a pyarrow
type, a Python type, or a NumPy/pandas dtype string, and pandas' `header=` overloads one
keyword with "is there a header" and "which row is it".
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from batcher._internal.errors import FormatError, SchemaError

__all__ = ["DATE_FORMATS", "arrow_type", "header_and_skip"]

# Formats tried, in order, when `try_parse_dates=True` widens timestamp inference beyond
# the ISO-8601 that Arrow accepts on its own. Month-first precedes day-first, matching
# pandas: the two are ambiguous for the first twelve days of any month, so the order is
# the whole answer and must not be sorted or deduplicated into something "tidier".
DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%Y%m%d",
)

# pandas/NumPy dtype spellings that are not Arrow type aliases.
_TYPE_ALIASES = {
    "str": pa.string(),
    "string": pa.string(),
    "object": pa.string(),
    "int": pa.int64(),
    "float": pa.float64(),
    "bool": pa.bool_(),
    "boolean": pa.bool_(),
    "datetime": pa.timestamp("us"),
    "datetime64[ns]": pa.timestamp("ns"),
    "datetime64[us]": pa.timestamp("us"),
    "category": pa.string(),
}
_PY_TYPES = {str: pa.string(), int: pa.int64(), float: pa.float64(), bool: pa.bool_()}


def arrow_type(name: str, value: Any) -> pa.DataType:
    """Coerce one `dtype`/`schema_overrides` entry into an Arrow type.

    Args:
        name: The column the entry is for, so a bad value names the column.
        value: A pyarrow type, a Python type, or a type-name string.

    Returns:
        The Arrow type to pin that column to.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.structured._csv_options.dtypes import arrow_type
            >>> arrow_type("v", "str")
            DataType(string)
    """
    if isinstance(value, pa.DataType):
        return value
    if isinstance(value, type) and value in _PY_TYPES:
        return _PY_TYPES[value]
    if isinstance(value, str):
        if value in _TYPE_ALIASES:
            return _TYPE_ALIASES[value]
        try:
            return pa.type_for_alias(value)
        except ValueError as exc:
            raise SchemaError(
                f"csv: {value!r} is not a type Batcher recognizes for column {name!r}. "
                f"Pass a pyarrow type (pa.int64()), a Python type (int), or an Arrow type "
                f"alias ('int64', 'string')."
            ) from exc
    raise SchemaError(
        f"csv: cannot read {value!r} as a type for column {name!r}. Pass a pyarrow type "
        f"(pa.int64()), a Python type (int), or a type name ('int64')."
    )


def header_and_skip(value: Any) -> tuple[bool, int]:
    """Split a pandas/Polars `header=`/`has_header=` value into (has_header, extra skip).

    pandas overloads one keyword with two meanings: ``header=None``/``False`` says the file
    has no header row, while ``header=2`` says row 2 *is* the header and rows 0-1 are
    preamble. Mapping the integer form to a plain boolean would read the preamble as the
    header, so the row offset is returned separately for the caller to fold into `skip_rows`.

    Args:
        value: The value the caller passed for `header=` or `has_header=`.

    Returns:
        Whether the file has a header row, and how many rows precede it.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.structured._csv_options.dtypes import header_and_skip
            >>> header_and_skip(None)
            (False, 0)
            >>> header_and_skip(2)
            (True, 2)
    """
    if value is None or value is False:
        return False, 0
    if value is True:
        return True, 0
    if isinstance(value, int):
        if value < 0:
            raise FormatError(f"csv: header={value!r} is negative; pass a row index or None.")
        return True, value
    raise FormatError(
        f"csv: header={value!r} is not a row index or a flag. Pass header=None for a file "
        f"with no header row, or header=<row index> for one whose header is not the first line."
    )
