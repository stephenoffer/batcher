"""The `.dt` accessor namespace plus the Polars-style offset-string parser.

`col("d").dt.year()`, `.dt.truncate("month")`, `.dt.offset_by("1mo15d")`, … — each
builds a `bc-expr` date node. The parameterless field extractions are generated
from `_DT_FIELDS` (data, not code).
"""

from __future__ import annotations

import re

from batcher.plan.expr_ir.core import Expr
from batcher.plan.expr_ir.func_nodes import (
    ConvertTimezone,
    DateFunc,
    DateOffset,
    DateTrunc,
    Strftime,
)
from batcher.plan.expr_ir.namespaces._bind import _bind_accessors

# Offset-string units → (months, days, micros) contribution per unit count. `mo`
# must precede `m` in the regex so "mo" parses as months, not minutes.
_OFFSET_UNITS = {
    "y": (12, 0, 0),
    "mo": (1, 0, 0),
    "w": (0, 7, 0),
    "d": (0, 1, 0),
    "h": (0, 0, 3_600_000_000),
    "m": (0, 0, 60_000_000),
    "s": (0, 0, 1_000_000),
}
_OFFSET_RE = re.compile(r"(-?\d+)(mo|[ymwdhs])")


def parse_offset(by: str) -> tuple[int, int, int]:
    """Parse a Polars-style offset string into ``(months, days, micros)`` components.

    Months, days, and microseconds are kept separate because months are calendar
    arithmetic (variable length) while days/micros are fixed. Units accumulate, so
    ``"1y"`` contributes 12 months and ``"1w"`` contributes 7 days.

    Args:
        by: Signed counts with units ``y``/``mo``/``w``/``d``/``h``/``m``/``s``,
            combinable, e.g. ``"1mo15d"`` or ``"-3d"``. ``mo`` is months, ``m`` minutes.

    Returns:
        A ``(months, days, micros)`` triple.

    Raises:
        ValueError: If ``by`` is empty or contains an unrecognized token.
    """
    pos = 0
    months = days = micros = 0
    for match in _OFFSET_RE.finditer(by):
        if match.start() != pos:
            break
        pos = match.end()
        n = int(match.group(1))
        mo, d, us = _OFFSET_UNITS[match.group(2)]
        months += n * mo
        days += n * d
        micros += n * us
    if pos != len(by) or not by:
        raise ValueError(
            f"invalid offset {by!r}; use counts with units y/mo/w/d/h/m/s, e.g. '1mo15d'"
        )
    return months, days, micros


class _DtNamespace:
    """Date/time field extractions: ``col("d").dt.year()``, ``.dt.hour()``, …

    The available extractors are **data, not code**: each is one row in
    ``_DT_FIELDS`` (Python accessor name → ``bc-expr`` ``DateFunc`` wire tag) and
    the no-argument accessor is generated below. Adding a field extractor is a
    single table entry — the pattern that keeps the namespace maintainable as it
    grows to hundreds of functions.
    """

    __slots__ = ("_e",)

    def __init__(self, e: Expr) -> None:
        self._e = e

    def truncate(self, unit: str) -> DateTrunc:
        """Truncate each timestamp down to the start of ``unit``.

        Zeroes out every field finer than ``unit`` (the floor toward the epoch), e.g.
        truncating to ``"month"`` gives the first of the month at midnight.
        Type-preserving.

        Args:
            unit: One of ``year``/``month``/``day``/``hour``/``minute``/``second``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45)]})
                >>> ds.select(bt.col("d").dt.truncate("month").alias("r")).to_pydict()
                {'r': [datetime.datetime(2024, 2, 1, 0, 0)]}
        """
        return DateTrunc(self._e, unit)

    def is_leap_year(self) -> DateFunc:
        """Test whether each row's year is a leap year (→ Bool)."""
        return DateFunc("is_leap_year", self._e)

    def days_in_month(self) -> DateFunc:
        """Return the number of days in each row's month, 28 to 31 (→ Int64)."""
        return DateFunc("days_in_month", self._e)

    def iso_year(self) -> DateFunc:
        """Return the ISO 8601 week-numbering year (→ Int64).

        May differ from the calendar year for dates in the first or last days of a
        year (e.g. 2021-01-01 can belong to ISO year 2020).
        """
        return DateFunc("iso_year", self._e)

    def strftime(self, format: str) -> Strftime:
        """Format each date/time as text with a chrono/strftime pattern (→ Utf8).

        DuckDB ``strftime`` / Polars ``dt.strftime``.

        Args:
            format: A strftime pattern, e.g. ``"%Y-%m-%d"`` or ``"%H:%M:%S"``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45)]})
                >>> ds.select(bt.col("d").dt.strftime("%Y-%m-%d").alias("r")).to_pydict()
                {'r': ['2024-02-15']}
        """
        return Strftime(self._e, format)

    def offset_by(self, by: str) -> DateOffset:
        """Shift each date/time by a Polars-style offset string. Type-preserving.

        Calendar units are calendar-correct: month/year arithmetic clamps to the end
        of the target month (e.g. Jan 31 + ``"1mo"`` → the last valid February day).
        A sub-day offset applied to a (date, not timestamp) column raises ``ValueError``.

        Args:
            by: Signed counts with units ``y``/``mo``/``w``/``d``/``h``/``m``/``s``,
                combinable, e.g. ``"1mo15d"``, ``"-3d"``, ``"1h30m"``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45, 30)]})
                >>> ds.select(bt.col("d").dt.offset_by("1mo15d").alias("r")).to_pydict()
                {'r': [datetime.datetime(2024, 3, 30, 13, 45, 30)]}
        """
        months, days, micros = parse_offset(by)
        return DateOffset(self._e, months, days, micros)

    def convert_timezone(self, from_tz: str, to_tz: str) -> ConvertTimezone:
        """Re-interpret each naive timestamp's wall-clock from one zone to another, DST-aware.

        DuckDB ``convert_timezone``. The instant is shifted so the wall-clock reads
        correctly in ``to_tz``. A local time that does not exist or is ambiguous under
        DST yields null. Type-preserving (Timestamp).

        Args:
            from_tz: IANA zone the naive timestamp is currently expressed in, e.g. ``"UTC"``.
            to_tz: IANA zone to convert the wall-clock to, e.g. ``"America/New_York"``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45, 30)]})
                >>> r = bt.col("d").dt.convert_timezone("UTC", "America/New_York")
                >>> ds.select(r.alias("r")).to_pydict()
                {'r': [datetime.datetime(2024, 2, 15, 8, 45, 30)]}
        """
        return ConvertTimezone(self._e, from_tz, to_tz)


# Python accessor name → engine `DateFunc` wire tag (serde snake_case). Each maps
# to one Arrow `DatePart` and matches the same-named DuckDB function.
_DT_FIELDS = {
    "year": "year",
    "month": "month",
    "day": "day",
    "hour": "hour",
    "minute": "minute",
    "second": "second",
    "quarter": "quarter",
    "week": "week",  # ISO week 1–53
    "dayofweek": "day_of_week",  # Sunday = 0
    "dayofyear": "day_of_year",  # 1–366
    "epoch": "epoch",  # seconds since the Unix epoch (→ Int64)
    "dayname": "dayname",  # full weekday name e.g. "Monday" (→ Utf8)
    "monthname": "monthname",  # full month name e.g. "January" (→ Utf8)
    "isodow": "isodow",  # ISO day of week: Monday = 1 … Sunday = 7 (→ Int64)
    "century": "century",  # the century, e.g. 2021 → 21 (→ Int64)
    "decade": "decade",  # the decade, e.g. 2021 → 202 (→ Int64)
    "millennium": "millennium",  # the millennium, e.g. 2021 → 3 (→ Int64)
    "last_day": "last_day",  # last day of the month at 00:00:00 (→ Timestamp(us))
}


_bind_accessors(
    _DtNamespace,
    _DT_FIELDS,
    lambda e, t: DateFunc(t, e),
    lambda n: f"Extract the {n} field of a date/time column (→ Int64).",
)
