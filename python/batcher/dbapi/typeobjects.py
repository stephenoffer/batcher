"""PEP 249 type objects and constructors, over the Arrow types a cursor describes.

A cursor's `description` reports each column's type as its pyarrow `DataType`, the type
Batcher computed it in. PEP 249 asks for five *type objects* that compare equal to every
type code of their family, so ``description[i][1] == dbapi.NUMBER`` works without the
client knowing anything about Arrow. Each one here is a predicate over a `DataType`.

The constructors return the plain Python values `params=` already binds: a `datetime.date`
binds as ``DATE``, `bytes` as ``BLOB``, and so on (see `batcher.api.sql_session.params`).

This is the `dbapi` layer.
"""

from __future__ import annotations

import datetime as dt
import time
from collections.abc import Callable
from typing import Any

import pyarrow as pa

__all__ = [
    "BINARY",
    "DATETIME",
    "NUMBER",
    "ROWID",
    "STRING",
    "Binary",
    "Date",
    "DateFromTicks",
    "Time",
    "TimeFromTicks",
    "Timestamp",
    "TimestampFromTicks",
]


class _TypeObject:
    """A PEP 249 type object: equal to every Arrow type its predicate accepts."""

    __slots__ = ("_accepts", "_name")

    def __init__(self, name: str, accepts: Callable[[pa.DataType], bool]) -> None:
        self._name = name
        self._accepts = accepts

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _TypeObject):
            return other is self
        return isinstance(other, pa.DataType) and self._accepts(other)

    def __hash__(self) -> int:
        return hash(self._name)

    def __repr__(self) -> str:
        return f"dbapi.{self._name}"


def _is_string(t: pa.DataType) -> bool:
    return pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_string_view(t)


def _is_binary(t: pa.DataType) -> bool:
    return (
        pa.types.is_binary(t)
        or pa.types.is_large_binary(t)
        or pa.types.is_fixed_size_binary(t)
        or pa.types.is_binary_view(t)
    )


def _is_number(t: pa.DataType) -> bool:
    return pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_decimal(t)


def _is_datetime(t: pa.DataType) -> bool:
    return pa.types.is_temporal(t)


#: Text columns.
STRING = _TypeObject("STRING", _is_string)
#: Binary columns.
BINARY = _TypeObject("BINARY", _is_binary)
#: Integer, floating-point and decimal columns.
NUMBER = _TypeObject("NUMBER", _is_number)
#: Date, time, timestamp and duration columns.
DATETIME = _TypeObject("DATETIME", _is_datetime)
#: Row ids; Batcher has none, so no column type equals it.
ROWID = _TypeObject("ROWID", lambda _t: False)


def Date(year: int, month: int, day: int) -> dt.date:
    """A date value, for binding as a ``DATE`` parameter.

    Args:
        year: The year.
        month: The month, 1 to 12.
        day: The day of the month.

    Returns:
        The date.

    Examples:
        .. doctest::

            >>> from batcher import dbapi
            >>> dbapi.Date(2024, 1, 31)
            datetime.date(2024, 1, 31)
    """
    return dt.date(year, month, day)


def Time(hour: int, minute: int, second: int) -> dt.time:
    """A time-of-day value, for binding as a ``TIME`` parameter.

    Args:
        hour: The hour, 0 to 23.
        minute: The minute.
        second: The second.

    Returns:
        The time.

    Examples:
        .. doctest::

            >>> from batcher import dbapi
            >>> dbapi.Time(13, 5, 0)
            datetime.time(13, 5)
    """
    return dt.time(hour, minute, second)


def Timestamp(year: int, month: int, day: int, hour: int, minute: int, second: int) -> dt.datetime:
    """A naive timestamp value, for binding as a ``TIMESTAMP`` parameter.

    Args:
        year: The year.
        month: The month.
        day: The day.
        hour: The hour.
        minute: The minute.
        second: The second.

    Returns:
        The timestamp.

    Examples:
        .. doctest::

            >>> from batcher import dbapi
            >>> dbapi.Timestamp(2024, 1, 31, 9, 30, 0)
            datetime.datetime(2024, 1, 31, 9, 30)
    """
    return dt.datetime(year, month, day, hour, minute, second)


def DateFromTicks(ticks: float) -> dt.date:
    """The local date `ticks` seconds after the epoch, as `time.localtime` reads it.

    Args:
        ticks: Seconds since the epoch.

    Returns:
        The date.

    Examples:
        .. doctest::

            >>> from batcher import dbapi
            >>> isinstance(dbapi.DateFromTicks(0), __import__("datetime").date)
            True
    """
    return Date(*time.localtime(ticks)[:3])


def TimeFromTicks(ticks: float) -> dt.time:
    """The local time of day `ticks` seconds after the epoch.

    Args:
        ticks: Seconds since the epoch.

    Returns:
        The time.

    Examples:
        .. doctest::

            >>> from batcher import dbapi
            >>> isinstance(dbapi.TimeFromTicks(0), __import__("datetime").time)
            True
    """
    return Time(*time.localtime(ticks)[3:6])


def TimestampFromTicks(ticks: float) -> dt.datetime:
    """The naive local timestamp `ticks` seconds after the epoch.

    Args:
        ticks: Seconds since the epoch.

    Returns:
        The timestamp.

    Examples:
        .. doctest::

            >>> from batcher import dbapi
            >>> isinstance(dbapi.TimestampFromTicks(0), __import__("datetime").datetime)
            True
    """
    return Timestamp(*time.localtime(ticks)[:6])


def Binary(value: Any) -> bytes:
    """A binary value, for binding as a ``BLOB`` parameter.

    Args:
        value: Anything `bytes` accepts: bytes, a bytearray, a memoryview.

    Returns:
        The bytes.

    Examples:
        .. doctest::

            >>> from batcher import dbapi
            >>> dbapi.Binary(bytearray(b"ab"))
            b'ab'
    """
    return bytes(value)
