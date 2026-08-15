"""Literal unwrapping and the column-vs-literal shapes every translator needs.

Split out of the single `predicate.py` module so each backend translator can be read on
its own. Nothing here is backend-specific: it turns the IR's tagged literals into Python
and pyarrow values, and answers whether a column's type can be compared against one at
all — the check that keeps a pushed filter from raising inside a scanner task.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import pyarrow as pa


def _literal(ir: dict[str, Any]) -> Any:
    """Unwrap a literal IR ``{"e":"lit","value":{"int":5}}`` to its Python value.

    Temporal kinds (``date`` days, ``timestamp`` micros, ``time`` micros) unwrap to a
    plain Python ``date``/``datetime`` so a backend that types its own scalars (SQL,
    iceberg, mongo) gets a real temporal value, not a raw epoch offset.
    """
    ((kind, value),) = ir["value"].items()
    if kind == "date":
        return _dt.date(1970, 1, 1) + _dt.timedelta(days=value)
    if kind == "timestamp":
        return _dt.datetime(1970, 1, 1) + _dt.timedelta(microseconds=value)
    if kind == "time":
        return (_dt.datetime(1970, 1, 1) + _dt.timedelta(microseconds=value)).time()
    return value


def _pa_literal(ir: dict[str, Any], col_type: Any | None = None) -> Any:
    """A pyarrow scalar for a literal IR, typed for temporal kinds.

    A bare ``date``/``timestamp`` literal is an epoch offset (days / micros); handed to
    pyarrow as a Python ``int`` it infers ``int16``/``int64`` and the comparison kernel
    against a ``date32``/``timestamp`` column has no match (``greater_equal(date32,
    int16)``). Building an explicitly-typed ``date32``/``timestamp[us]`` scalar makes the
    column-vs-literal comparison type-check and enables row-group/page pruning on date
    columns (the common TPC-H shipdate/orderdate filters).

    When the column's own type is known (`col_type`), a ``timestamp`` literal is built to
    match it exactly. This is what a timezone-aware column needs: pyarrow refuses to
    compare a ``timestamp[us, tz=UTC]`` column against a tz-naive ``timestamp[us]`` scalar
    (``Cannot compare timestamp with timezone to timestamp without timezone``), which
    crashed a pushed filter on any UTC-normalized lakehouse timestamp column — the norm for
    event-time data. The literal's raw value is UTC micros, so the same instant is
    expressed in the column's unit and zone.
    """
    ((kind, value),) = ir["value"].items()
    if kind == "date":
        return pa.scalar(value, pa.date32())
    if kind == "timestamp":
        if col_type is not None and pa.types.is_timestamp(col_type):
            return _timestamp_scalar(value, col_type, pa)
        return pa.scalar(value, pa.timestamp("us"))
    if kind == "time":
        return pa.scalar(value, pa.time64("us"))
    return value


def _timestamp_scalar(micros: int, col_type: Any, pa: Any) -> Any:
    """A timestamp scalar for `micros` (UTC epoch micros) in `col_type`'s unit and zone.

    Building the scalar from a Python ``datetime`` lets pyarrow convert the unit; a tz-aware
    column gets a UTC-aware datetime (same instant), a tz-naive column a naive one, so the
    comparison type-checks either way.
    """
    moment = _dt.datetime(1970, 1, 1, tzinfo=_dt.UTC) + _dt.timedelta(microseconds=micros)
    if col_type.tz is None:
        moment = moment.replace(tzinfo=None)
    return pa.scalar(moment, col_type)


def _col_and_literal(left: dict[str, Any], right: dict[str, Any]) -> tuple[str, Any, bool] | None:
    """Return ``(column, value, flipped)`` for a column-vs-literal comparison."""
    if left.get("e") == "col" and right.get("e") == "lit":
        return left["name"], _literal(right), False
    if left.get("e") == "lit" and right.get("e") == "col":
        return right["name"], _literal(left), True
    return None


def _col_and_pa_literal(
    left: dict[str, Any], right: dict[str, Any], schema: Any | None = None
) -> tuple[str, Any, bool] | None:
    """Like :func:`_col_and_literal`, but the value is a typed pyarrow scalar.

    `schema` (when known) types a temporal literal to its column's own type, so a filter
    on a timezone-aware timestamp column type-checks instead of raising, and lets a
    literal the scanner could not compare at all be declined instead of pushed.
    """
    if left.get("e") == "col" and right.get("e") == "lit":
        col, lit = left["name"], right
    elif left.get("e") == "lit" and right.get("e") == "col":
        col, lit = right["name"], left
    else:
        return None
    col_type = _field_type(schema, col)
    if not _comparable(col_type, lit):
        return None
    return col, _pa_literal(lit, col_type), left.get("e") == "lit"


def _comparable(col_type: Any | None, lit: dict[str, Any]) -> bool:
    """Whether arrow has a comparison kernel for this column type against this literal.

    Arrow compares within a type family and promotes between numeric widths, but it has no
    ``greater_equal(date32, string)`` — and the scanner raises `ArrowNotImplementedError`
    rather than declining, from inside whatever task built it. SQL routinely writes exactly
    that: ``WHERE EventDate >= '2013-07-01'`` against a `date32` column is the ClickBench
    spelling, and it failed six of the 43 queries on the distributed path while running
    single-node, where the filter is the engine's and the engine coerces.

    So the mismatch is declined here instead. Pushdown is an optimization and the engine's
    own `Filter` re-checks every row, so dropping this term costs pruning and never a row.
    Coercing the string to the column's type would keep the pruning, but only if this
    module's parse agreed with the engine's cast on every input — and a pushdown that
    disagrees silently returns the wrong rows, which is the one outcome worth ruling out.

    An unknown column type (no schema) keeps the previous behavior: push and hope, which is
    what every caller without a schema has always done.
    """
    if col_type is None:
        return True
    ((kind, _),) = lit["value"].items()
    if pa.types.is_dictionary(col_type):
        col_type = col_type.value_type
    if pa.types.is_temporal(col_type):
        return kind in ("date", "timestamp", "time")
    if pa.types.is_string(col_type) or pa.types.is_large_string(col_type):
        return kind == "str"
    if pa.types.is_binary(col_type) or pa.types.is_large_binary(col_type):
        return kind in ("str", "bytes")
    if pa.types.is_boolean(col_type):
        return kind in ("bool", "int")
    if pa.types.is_decimal(col_type):
        # Arrow rescales the literal into the column's own precision and raises
        # `ArrowInvalid: Precision is not great enough` on an integer that does not fit.
        # ``WHERE price = 2`` against a DECIMAL(5,2) is ordinary SQL, and the scanner
        # raised there while the engine answered it. A `Decimal` lowers to a float
        # literal, so the float case covers both spellings a caller actually writes.
        return kind == "float"
    if pa.types.is_integer(col_type) or pa.types.is_floating(col_type):
        # Arrow promotes between numeric widths but has no `equal(int64, bool)`.
        return kind in ("int", "float")
    return True  # a type this does not model: unchanged, push it


def _field_type(schema: Any | None, name: str) -> Any | None:
    """The Arrow type of column `name` in `schema`, or None if unknown."""
    if schema is None:
        return None
    try:
        return schema.field(name).type
    except Exception:
        return None
