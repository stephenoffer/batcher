"""The type gates every `ds.meta` accessor shares — refuse a question the column cannot answer.

Two paths answer each shortcut, a footer read and an executed query, and they must agree on
*whether* there is an answer as much as on what it is. A type mismatch is where they drifted:
the metadata path compared a string column's missing bounds with ``0`` and said "vacuously
true", while the engine refused the same comparison outright. Checking the declared type up
front, before either path runs, makes both give the same actionable `PlanError`.

The declared type comes from static analysis only (`_declared_schema`), never from a zero-row
execution: for a ``map_batches`` stage that execution builds the UDF, which can be a model
load on the driver. When the plan cannot state a column's type, the gate stands aside and the
executed values are checked instead (`require_numeric_values`).

Layer: `api`.
"""

from __future__ import annotations

import datetime as dt
import numbers
from collections.abc import Iterable
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from batcher._internal.errors import PlanError

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = [
    "declared_dtype",
    "is_numeric_type",
    "require_comparable",
    "require_numeric",
    "require_numeric_values",
    "require_rangeable",
]

#: Which kinds of Python value the engine will compare with a column of each type family.
#: Read off the engine rather than assumed: a string, date, or timestamp column compares with
#: a string or a date/datetime literal, a numeric or boolean column with any number, a time
#: column with a time. A family missing here is not validated, which leaves the engine as the
#: judge rather than inventing a rule it might not share.
_ACCEPTS: tuple[tuple[Any, frozenset[str]], ...] = (
    (lambda t: pa.types.is_integer(t) or pa.types.is_floating(t), frozenset({"number"})),
    (pa.types.is_decimal, frozenset({"number"})),
    (pa.types.is_boolean, frozenset({"number"})),
    (
        lambda t: (
            pa.types.is_string(t)
            or pa.types.is_large_string(t)
            or pa.types.is_date(t)
            or pa.types.is_timestamp(t)
        ),
        frozenset({"string", "date"}),
    ),
    (pa.types.is_time, frozenset({"time"})),
)

_KIND_NOUNS = {
    "date": "a date or datetime",
    "number": "a number",
    "string": "a string",
    "time": "a time",
}


def declared_dtype(ds: Dataset, column: str) -> pa.DataType | None:
    """`column`'s type from static analysis, or None when the plan cannot state it.

    A ``null`` type is the analysis saying "unknown" (an opaque stage erases it), not a
    column of nulls anyone should compare against, so it reads as None too.
    """
    from batcher.api.terminal.core import _declared_schema

    schema = _declared_schema(ds._plan, ds._sources)
    if schema is None or column not in schema.names:
        return None
    dtype = schema.field(column).type
    if pa.types.is_dictionary(dtype):
        dtype = dtype.value_type
    return None if pa.types.is_null(dtype) else dtype


def is_numeric_type(dtype: pa.DataType) -> bool:
    """Whether `dtype` is an integer, float, or decimal type — never a boolean."""
    return pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_decimal(dtype)


def require_numeric(ds: Dataset, column: str, op: str) -> None:
    """Raise `PlanError` unless `column` is numeric, when its type is declared."""
    dtype = declared_dtype(ds, column)
    if dtype is not None and not is_numeric_type(dtype):
        raise PlanError(
            f"meta.col({column!r}).{op}(): column {column!r} is {dtype}, not numeric; {op}() "
            f"needs an integer, float, or decimal column. Use .bounds() for the (min, max) "
            f"of any orderable column."
        )


def require_rangeable(ds: Dataset, column: str) -> None:
    """Raise `PlanError` unless `max - min` means something for `column`'s declared type.

    Numeric columns give a number and temporal ones a ``datetime.timedelta``. A string or a
    boolean has no difference to take.
    """
    dtype = declared_dtype(ds, column)
    if dtype is not None and not (is_numeric_type(dtype) or pa.types.is_temporal(dtype)):
        raise PlanError(
            f"meta.col({column!r}).range(): column {column!r} is {dtype}; range() needs a "
            f"numeric or temporal column. Use .bounds() for the (min, max) of {dtype} values."
        )


def require_numeric_values(column: str, op: str, *values: Any) -> None:
    """Raise `PlanError` if an executed bound is not a number — the undeclared-type fallback."""
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (numbers.Real, Decimal)):
            raise PlanError(
                f"meta.col({column!r}).{op}(): column {column!r} holds "
                f"{type(value).__name__} values, not numeric; {op}() needs an integer, float, "
                f"or decimal column. Use .bounds() for the (min, max) of any orderable column."
            )


def require_comparable(ds: Dataset, column: str, op: str, values: Iterable[Any]) -> None:
    """Raise `PlanError` if any of `values` cannot be compared with `column`'s declared type.

    ``None`` always passes: comparing with NULL is legal SQL, it only matches nothing.
    """
    dtype = declared_dtype(ds, column)
    accepted = _accepted_kinds(dtype) if dtype is not None else None
    if accepted is None:
        return
    for value in values:
        kind = _value_kind(value)
        rejected = kind is not None and kind not in accepted
        if rejected or (pa.types.is_decimal(dtype) and isinstance(value, bool)):
            wanted = " or ".join(_KIND_NOUNS[kind] for kind in sorted(accepted))
            raise PlanError(
                f"meta.col({column!r}).check.{op}(): cannot compare {dtype} column {column!r} "
                f"with {value!r} ({type(value).__name__}); pass {wanted}."
            )


def _accepted_kinds(dtype: pa.DataType) -> frozenset[str] | None:
    """The value kinds `dtype` compares with, or None for a family this module leaves alone."""
    for matches, kinds in _ACCEPTS:
        if matches(dtype):
            return kinds
    return None


def _value_kind(value: Any) -> str | None:
    """The comparison kind of a Python value, or None for one this module does not classify."""
    if value is None:
        return None
    if isinstance(value, str):
        return "string"
    if isinstance(value, numbers.Number):
        return "number"
    if isinstance(value, dt.date):  # a datetime is a date
        return "date"
    if isinstance(value, dt.time):
        return "time"
    if isinstance(value, dt.timedelta):
        return "duration"
    return None
