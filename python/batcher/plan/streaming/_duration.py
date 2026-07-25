"""Duration parsing for streaming intervals — the one gate every trigger/lateness flows through.

`Trigger` cadences and `Watermark` lateness arrive as a `timedelta`, a number of seconds, or a
duration string, and every one of them lands here. It is split out of `spec.py` because that
module holds the streaming *value types*, and parsing is a separate responsibility with its own
vocabulary table and its own edge cases (NaN, calendar units, two competing syntaxes).

Two duration syntaxes are accepted, deliberately: the spelled-out single-unit form Spark uses
(``"5 seconds"``, ``"500ms"``) and the compact combinable form window durations use (``"1d"``,
``"2h30m"``). A pipeline writes a window width and a watermark delay side by side, so both must
read both.
"""

from __future__ import annotations

import math
import re
from datetime import timedelta
from typing import Final

from batcher._internal.errors import PlanError, suggestion

__all__ = ["parse_interval_seconds"]

_UNIT_SECONDS: Final[dict[str, float]] = {
    "us": 1e-6,
    "microsecond": 1e-6,
    "microseconds": 1e-6,
    "ms": 1e-3,
    "millisecond": 1e-3,
    "milliseconds": 1e-3,
    "s": 1.0,
    "sec": 1.0,
    "secs": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "m": 60.0,
    "min": 60.0,
    "mins": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
    # `d`/`w` match `parse_offset`, so a watermark and the window it bounds speak one language.
    "d": 86_400.0,
    "day": 86_400.0,
    "days": 86_400.0,
    "w": 604_800.0,
    "week": 604_800.0,
    "weeks": 604_800.0,
}

_INTERVAL_RE: Final[re.Pattern[str]] = re.compile(r"\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]+)\s*")


def _compact_seconds(interval: str) -> float:
    """Seconds from a compact multi-unit duration such as ``"2h30m"`` or ``"1d"``.

    The fallback for anything `_INTERVAL_RE` cannot read, delegating to the parser window
    durations already use so one pipeline's window width and watermark delay accept the same
    spellings. They used to be disjoint: ``"1d"`` sized a window but could not delay a
    watermark, and ``"10 seconds"`` the reverse. Calendar units stay refused.
    """
    from batcher.plan.expr_ir.namespaces.temporal import parse_offset

    try:
        months, days, micros = parse_offset(interval)
    except ValueError as exc:
        raise PlanError(
            f"cannot parse interval {interval!r}; use a number of seconds, a string like "
            "'5 seconds', '1 minute' or '100ms', or a compact duration like '2h30m'"
        ) from exc
    if months:
        raise PlanError(
            f"interval {interval!r} uses a calendar unit (month/year) with no fixed length; "
            "use weeks/days/hours/minutes/seconds"
        )
    return days * 86_400.0 + micros / 1_000_000.0


def parse_interval_seconds(interval: float | int | str | timedelta) -> float:
    """Parse a trigger/lateness interval to seconds.

    Accepts a `datetime.timedelta`, a number already in seconds, or a Spark-style
    string such as ``"5 seconds"``, ``"1 minute"``, ``"500 milliseconds"``, or
    ``"100ms"``. Raises `PlanError` (a `ValueError`) for an unrecognized unit, an
    unparseable string, an unsupported type, or a negative duration.

    Examples:
        .. doctest::

            >>> import datetime
            >>> from batcher.plan.streaming import parse_interval_seconds
            >>> parse_interval_seconds("2 minutes")
            120.0
            >>> parse_interval_seconds(datetime.timedelta(seconds=5))
            5.0

    Args:
        interval: A `timedelta`, seconds as a number, or a Spark-style duration string.

    Returns:
        The duration in seconds as a float.

    Raises:
        PlanError: If the interval is not parseable, uses an unknown unit, is a type
            other than number/str/timedelta, or is negative.
    """
    if isinstance(interval, timedelta):
        seconds = interval.total_seconds()
    elif isinstance(interval, bool):
        # `bool` is an `int` subclass; a boolean interval is always a mistake.
        raise PlanError(
            f"interval must be a number, a string like '5 seconds', or a timedelta, "
            f"not a bool ({interval!r})"
        )
    elif isinstance(interval, (int, float)):
        seconds = float(interval)
    elif isinstance(interval, str):
        match = _INTERVAL_RE.fullmatch(interval)
        if match is None:
            seconds = _compact_seconds(interval)
        else:
            value, unit = match.group(1), match.group(2).lower()
            if unit not in _UNIT_SECONDS:
                hint = suggestion(unit, _UNIT_SECONDS) or "use 'ms', 's', 'm', 'h', or 'd'"
                raise PlanError(f"unknown interval unit {unit!r} in {interval!r}. {hint}")
            seconds = float(value) * _UNIT_SECONDS[unit]
    else:
        raise PlanError(
            f"interval must be a number, a string like '5 seconds', or a timedelta, "
            f"not {type(interval).__name__} ({interval!r})"
        )
    # A NaN or infinite interval slips past every check below (`nan < 0` is False, and
    # `inf` is "non-negative"), then poisons the loop that consumes it: a NaN trigger
    # cadence makes `remaining > 0` always False and busy-loops the micro-batch thread,
    # and an infinite lateness overflows the microsecond literal it lowers to. Reject it
    # here, at the one gate every duration flows through.
    if not math.isfinite(seconds):
        raise PlanError(f"interval must be a finite duration, got {seconds} (from {interval!r})")
    if seconds < 0:
        raise PlanError(f"interval must be non-negative, got {seconds}")
    return seconds
