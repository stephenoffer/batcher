"""Row generators: `range` and `date_range`.

Synthetic keys and calendar dimensions, spelled the way ``builtins.range``,
``pandas.date_range``, and ``polars.date_range`` are, so the signatures a Python
user already knows carry over.
"""

from __future__ import annotations

import builtins
import re
from datetime import date, datetime, timedelta
from typing import Any

import pyarrow as pa

from batcher._internal.errors import PlanError, require_int
from batcher.api.dataset import Dataset
from batcher.api.session.frames import from_arrow

__all__ = ["date_range", "range"]

# ``<count><unit>`` intervals, in the vocabulary pandas and Polars share. ``m`` is
# minutes and ``mo``/``M`` months, which is the one place the two disagree; Polars'
# reading wins because it is the unambiguous one.
_UNITS = {
    "s": "seconds",
    "sec": "seconds",
    "second": "seconds",
    "seconds": "seconds",
    "m": "minutes",
    "min": "minutes",
    "minute": "minutes",
    "minutes": "minutes",
    "h": "hours",
    "hour": "hours",
    "hours": "hours",
    "d": "days",
    "day": "days",
    "days": "days",
    "w": "weeks",
    "wk": "weeks",
    "week": "weeks",
    "weeks": "weeks",
    "mo": "months",
    "month": "months",
    "months": "months",
    "y": "years",
    "year": "years",
    "years": "years",
}
_INTERVAL = re.compile(r"^\s*(\d*)\s*([A-Za-z]+)\s*$")
_SUB_DAY = frozenset({"seconds", "minutes", "hours"})
_MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def range(
    start: int,
    stop: int | None = None,
    step: int = 1,
    *,
    name: str = "value",
) -> Dataset:
    """A one-column `Dataset` of the integers ``[start, stop)`` stepped by `step`.

    Mirrors ``builtins.range``, single-argument form included: ``bt.range(5)`` is
    ``0, 1, 2, 3, 4``. The generator source for synthetic keys and joins; for date
    dimensions see `date_range`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.range(0, 5).to_pydict()
            {'value': [0, 1, 2, 3, 4]}

            >>> bt.range(5).count()
            5

            >>> bt.range(3, 0, -1).to_pydict()
            {'value': [3, 2, 1]}

    Args:
        start: The first integer (inclusive), or the exclusive end when `stop` is
            omitted (the ``builtins.range`` convention).
        stop: The end integer (exclusive).
        step: The stride between successive integers; may be negative.
        name: The output column name.

    Returns:
        A one-column lazy `Dataset` of the integer range.

    Raises:
        PlanError: If `step` is zero.
    """
    start = require_int(start, func="range", arg="start")
    step = require_int(step, func="range", arg="step")
    if stop is not None:
        stop = require_int(stop, func="range", arg="stop")
    if stop is None:
        start, stop = 0, start
    if step == 0:
        raise PlanError("range(): step must be non-zero")
    values = list(builtins.range(start, stop, step))
    return from_arrow(pa.table({name: pa.array(values, pa.int64())}))


def _parse_interval(text: str) -> tuple[int, str]:
    """Split ``"3d"`` / ``"1mo"`` / ``"D"`` into a ``(count, unit)`` pair."""
    match = _INTERVAL.match(text)
    unit = _UNITS.get(match.group(2).lower()) if match else None
    if unit is None:
        raise PlanError(
            f"date_range(): could not read the interval {text!r}. Use <count><unit>, "
            "e.g. '1d', '2w', '1mo', '6h' (units: s, m, h, d, w, mo, y)."
        )
    count = int(match.group(1)) if match.group(1) else 1
    if count < 1:
        raise PlanError(f"date_range(): the interval count in {text!r} must be >= 1")
    return count, unit


def _as_datetime(value: Any, arg: str) -> datetime:
    """Coerce an ISO string, `date`, or `datetime` bound to a `datetime`."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            raise PlanError(
                f"date_range(): {arg}={value!r} is not an ISO date or timestamp "
                "(e.g. '2024-01-01' or '2024-01-01T06:00:00')"
            ) from None
    raise PlanError(f"date_range(): {arg} must be an ISO string, date, or datetime")


def _advance(moment: datetime, count: int, unit: str) -> datetime:
    """`moment` moved forward by `count` `unit`s, with calendar-correct month/year steps."""
    if unit in {"months", "years"}:
        months = count * (12 if unit == "years" else 1)
        total = moment.month - 1 + months
        year, month = moment.year + total // 12, total % 12 + 1
        last_day = _MONTH_DAYS[month - 1]
        if month == 2 and (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
            last_day = 29
        return moment.replace(year=year, month=month, day=min(moment.day, last_day))
    return moment + timedelta(**{unit: count})


def date_range(
    start: Any,
    end: Any = None,
    *,
    periods: int | None = None,
    interval: str | None = None,
    freq: str | None = None,
    interval_days: int | None = None,
    closed: str = "both",
    name: str = "date",
) -> Dataset:
    """A one-column `Dataset` of dates or timestamps from `start`, the calendar dimension.

    Mirrors ``pandas.date_range`` and ``polars.date_range``: give `end`, or give
    `periods` to generate a fixed count. Bounds may be ISO strings, ``date``, or
    ``datetime`` objects. `interval` (Polars) and `freq` (pandas) both name the
    stride as ``<count><unit>`` — ``"1d"``, ``"2w"``, ``"1mo"``, ``"6h"`` — where
    ``m`` is minutes and ``mo`` months. A sub-day stride yields timestamps; anything
    coarser yields dates. `closed` drops an endpoint the way pandas' ``inclusive``
    does.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.date_range("2024-01-01", "2024-01-03").count()
            3

            >>> bt.date_range("2024-01-01", periods=3, interval="1mo").count()
            3

            >>> bt.date_range("2024-01-01", "2024-01-03", closed="left").count()
            2

    Args:
        start: The first date (inclusive), ISO string, ``date``, or ``datetime``.
        end: The last date (inclusive by default); omit it and pass `periods`.
        periods: Generate exactly this many points instead of stopping at `end`.
        interval: The stride, ``<count><unit>`` (Polars spelling); defaults to ``"1d"``.
        freq: The pandas spelling of `interval`; equivalent, and mutually exclusive.
        interval_days: A stride in whole days, kept for existing callers.
        closed: Which endpoints to keep — ``"both"``, ``"left"``, ``"right"``, ``"none"``.
        name: The output column name.

    Returns:
        A one-column lazy `Dataset` of the range.

    Raises:
        PlanError: If the bounds, stride, `periods`, or `closed` value is invalid, or
            more than one of `interval`/`freq`/`interval_days` is given.
    """
    given = [s for s in (interval, freq, None if interval_days is None else "d") if s is not None]
    if len(given) > 1:
        raise PlanError("date_range(): pass only one of interval=, freq=, or interval_days=")
    if closed not in {"both", "left", "right", "none"}:
        raise PlanError(
            f"date_range(): closed must be 'both', 'left', 'right', or 'none', got {closed!r}"
        )
    if end is None and periods is None:
        raise PlanError("date_range(): pass end= or periods= to bound the range")
    if periods is not None and periods < 0:
        raise PlanError(f"date_range(): periods must be >= 0, got {periods}")

    if interval_days is not None:
        count, unit = interval_days, "days"
        if interval_days < 1:
            raise PlanError("date_range(): interval_days must be >= 1")
    else:
        count, unit = _parse_interval(interval or freq or "1d")

    first = _as_datetime(start, "start")
    last = _as_datetime(end, "end") if end is not None else None
    if last is not None and last < first:
        raise PlanError(f"date_range(): end ({end}) is before start ({start})")

    points: list[datetime] = []
    moment = first
    while True:
        if last is not None and moment > last:
            break
        if periods is not None and len(points) >= periods:
            break
        points.append(moment)
        moment = _advance(moment, count, unit)

    if closed in {"none", "left"} and points and last is not None and points[-1] == last:
        points = points[:-1]
    if closed in {"none", "right"} and points:
        points = points[1:]

    if unit in _SUB_DAY:
        return from_arrow(pa.table({name: pa.array(points, pa.timestamp("us"))}))
    return from_arrow(pa.table({name: pa.array([p.date() for p in points], pa.date32())}))
