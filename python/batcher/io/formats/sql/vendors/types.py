"""Declared conversions and precise refusals for the vendor types Arrow cannot take as-is.

Three kinds of value reach this package from a real database and break the read in a way
that names nothing useful:

* **An unsigned 64-bit integer above 2^63-1** (MySQL ``BIGINT UNSIGNED``). Arrow carries it
  as ``uint64``, and the engine normalizes integers to ``int64`` at its boundary, so the
  value is refused there with a message about FFI rather than about the column. `conform`
  refuses it at the source instead, naming the column and the fix, or — when the caller
  opted in with ``unsigned="decimal"`` — reads the column as an exact ``decimal128(20, 0)``.
* **A value the driver handed over as the wrong Python type** — PyMySQL returns an
  illegal date such as ``'0000-00-00'`` as the *string* it read, python-oracledb returns a
  LOB handle, psycopg returns ``Decimal('NaN')`` for a NUMERIC NaN. Arrow raises
  ``ArrowTypeError: object of type str cannot be converted to int``, which names neither the
  column nor the cause. `explain_failure` turns that into a refusal that names both.
* **A zero date the caller would rather read as NULL** — `null_zero_dates`, opt-in through
  ``zero_dates="null"``.

Every check here is either an Arrow compute kernel over a whole column (`conform`) or runs
only on the failure path of a DB-API batch that is already a list of Python objects
(`explain_failure`, `null_zero_dates`), so the success path pays nothing per row.
"""

from __future__ import annotations

import decimal
from collections.abc import Sequence
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from batcher._internal.errors import BackendError

__all__ = [
    "ORACLE_NUMBER_POLICIES",
    "UNSIGNED_POLICIES",
    "ZERO_DATE_POLICIES",
    "check_policy",
    "conform",
    "conform_table",
    "explain_failure",
    "null_zero_dates",
]

#: ``unsigned=``: refuse a uint64 value the engine cannot hold, or read the column exactly.
UNSIGNED_POLICIES = ("refuse", "decimal")

#: ``zero_dates=``: refuse a ``'0000-00-00'`` date string, or read it as NULL.
ZERO_DATE_POLICIES = ("refuse", "null")

#: ``oracle_numbers=``: what python-oracledb returns, or every NUMBER as an exact Decimal.
ORACLE_NUMBER_POLICIES = ("native", "decimal")

_INT64_MAX = 2**63 - 1

#: MySQL's zero-date sentinels, as PyMySQL returns them verbatim for DATE and DATETIME.
_ZERO_DATE_PREFIX = "0000-00-00"


def check_policy(name: str, value: str, allowed: Sequence[str]) -> None:
    """Refuse a policy keyword outside its vocabulary, naming the alternatives.

    Args:
        name: The keyword, e.g. ``"unsigned"``.
        value: What the caller passed.
        allowed: The accepted values.

    Raises:
        BackendError: If `value` is not one of `allowed`.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.sql.vendors.types import check_policy
            >>> check_policy("unsigned", "decimal", ("refuse", "decimal"))
    """
    if value not in allowed:
        raise BackendError(f"{name}={value!r} is not supported; expected one of {list(allowed)}.")


def conform(batch: pa.RecordBatch, *, unsigned: str = "refuse") -> pa.RecordBatch:
    """Apply the unsigned-integer rule to one batch's top-level columns.

    Args:
        batch: A batch as the driver returned it.
        unsigned: ``"refuse"`` raises on a ``uint64`` value above 2^63-1; ``"decimal"``
            casts every ``uint64`` column to ``decimal128(20, 0)``, which holds the whole
            unsigned range exactly.

    Returns:
        The batch, unchanged unless a ``uint64`` column was converted.

    Raises:
        BackendError: If ``unsigned="refuse"`` meets a value the engine cannot represent.

    Examples:
        .. doctest::

            >>> import pyarrow as pa
            >>> from batcher.io.formats.sql.vendors.types import conform
            >>> batch = pa.record_batch({"id": pa.array([2**64 - 1], pa.uint64())})
            >>> conform(batch, unsigned="decimal").schema.field("id").type
            Decimal128Type(decimal128(20, 0))
    """
    if not any(pa.types.is_uint64(f.type) for f in batch.schema):
        return batch
    columns: list[pa.Array] = []
    fields: list[pa.Field] = []
    for field, column in zip(batch.schema, batch.columns, strict=True):
        if pa.types.is_uint64(field.type):
            if unsigned == "decimal":
                target = pa.decimal128(20, 0)
                column = column.cast(target)
                field = field.with_type(target)
            else:
                _refuse_overflow(field.name, column)
        columns.append(column)
        fields.append(field)
    return pa.RecordBatch.from_arrays(columns, schema=pa.schema(fields, batch.schema.metadata))


def conform_table(table: pa.Table, *, unsigned: str = "refuse") -> pa.Table:
    """`conform` over every batch of a table, including an empty one's schema.

    An empty result still has to report the converted type, or a zero-row schema probe
    and the real read would disagree about the column.

    Args:
        table: A table as the driver returned it.
        unsigned: As for `conform`.

    Returns:
        The conformed table.

    Examples:
        .. doctest::

            >>> import pyarrow as pa
            >>> from batcher.io.formats.sql.vendors.types import conform_table
            >>> empty = pa.table({"id": pa.array([], pa.uint64())})
            >>> conform_table(empty, unsigned="decimal").schema.field("id").type
            Decimal128Type(decimal128(20, 0))
    """
    batches = table.to_batches() or [
        pa.RecordBatch.from_arrays(
            [pa.array([], f.type) for f in table.schema], schema=table.schema
        )
    ]
    return pa.Table.from_batches([conform(b, unsigned=unsigned) for b in batches])


def _refuse_overflow(name: str, column: pa.Array) -> None:
    """Raise when an unsigned column holds a value `int64` cannot."""
    top = pc.max(column).as_py()
    if top is not None and top > _INT64_MAX:
        raise BackendError(
            f"column {name!r} is an unsigned 64-bit integer (MySQL BIGINT UNSIGNED) holding "
            f"{top}, above the engine's int64 range. Pass unsigned='decimal' to read it as "
            f"decimal128(20, 0), or CAST({name} AS DECIMAL(20, 0)) in the query."
        )


def null_zero_dates(values: list[Any]) -> list[Any]:
    """Replace MySQL zero-date strings with None, leaving every other value alone.

    Args:
        values: One column of a DB-API batch.

    Returns:
        The column with each ``'0000-00-00...'`` string replaced by None.

    Examples:
        .. doctest::

            >>> import datetime
            >>> from batcher.io.formats.sql.vendors.types import null_zero_dates
            >>> null_zero_dates([datetime.date(2024, 1, 2), "0000-00-00"])
            [datetime.date(2024, 1, 2), None]
    """
    return [None if _is_zero_date(v) else v for v in values]


def _is_zero_date(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(_ZERO_DATE_PREFIX)


def explain_failure(name: str, values: list[Any], exc: Exception) -> BackendError | None:
    """A refusal naming the column and the vendor value Arrow could not convert, if known.

    Called only after Arrow has already refused a column, so the scan below is on the
    failure path and costs nothing on a read that succeeds. A failure with no recognized
    vendor cause returns None, and the caller re-raises Arrow's own error unchanged — a
    genuinely mixed column keeps raising what it always raised.

    Args:
        name: The column name.
        values: The column's values for the failing batch.
        exc: Arrow's error.

    Returns:
        The error to raise, chained by the caller onto `exc`, or None.

    Examples:
        .. doctest::

            >>> import datetime
            >>> from batcher.io.formats.sql.vendors.types import explain_failure
            >>> err = explain_failure("d", [datetime.date(2024, 1, 2), "0000-00-00"], TypeError())
            >>> "zero_dates='null'" in str(err)
            True
    """
    if any(_is_zero_date(v) for v in values):
        return BackendError(
            f"column {name!r} holds the MySQL zero date '0000-00-00', which is not a date: "
            "the driver returned it as a string. Pass zero_dates='null' to read it as NULL, "
            f"or select NULLIF({name}, '0000-00-00') in the query."
        )
    special = next(
        (v for v in values if isinstance(v, decimal.Decimal) and not v.is_finite()), None
    )
    if special is not None:
        return BackendError(
            f"column {name!r} holds the NUMERIC value {special}, which an Arrow decimal cannot "
            f"represent. CAST({name} AS double precision) in the query to keep it as a float, "
            "or filter it out."
        )
    odd = next((v for v in values if v is not None and not _plain(v)), None)
    if odd is not None:
        kind = f"{type(odd).__module__}.{type(odd).__qualname__}"
        return BackendError(
            f"column {name!r} holds a {kind} value Arrow cannot convert ({exc}). If it is a "
            "LOB or another driver handle, configure the driver to fetch it as str or bytes, "
            "or CAST the column in the query."
        )
    return None


#: Python types `pa.array` converts natively. Anything else is a driver-specific object.
_PLAIN = (str, bytes, bytearray, int, float, bool, decimal.Decimal, list, tuple, dict)


def _plain(value: Any) -> bool:
    import datetime

    return isinstance(value, (*_PLAIN, datetime.date, datetime.time, datetime.timedelta))
