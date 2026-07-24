"""The `.dt` accessor namespace plus the Polars-style offset-string parser.

`col("d").dt.year()`, `.dt.truncate("month")`, `.dt.offset_by("1mo15d")`, … — each
builds a `bc-expr` date node. The parameterless field extractions are generated
from `_DT_FIELDS` (data, not code).
"""

from __future__ import annotations

import re
from typing import Any

from batcher.plan.expr_ir.compat.guidance import DT_UNSUPPORTED, accessor_attribute_error
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
    """Date/time field extractions on a temporal column: ``col("d").dt.year()``, ``.dt.hour()``.

    The available extractors are **data, not code**: each is one row in
    ``_DT_FIELDS`` (Python accessor name → ``bc-expr`` ``DateFunc`` wire tag) and
    the no-argument accessor is generated below. Adding a field extractor is a
    single table entry — the pattern that keeps the namespace maintainable as it
    grows to hundreds of functions.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> import datetime as dt
            >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45, 30)]})
            >>> ds.select(bt.col("d").dt.year().alias("y")).to_pydict()
            {'y': [2024]}
    """

    __slots__ = ("_e",)

    def __init__(self, e: Expr) -> None:
        """Wrap the parent :class:`Expr` so its `.dt` methods can build on it."""
        self._e = e

    def __repr__(self) -> str:
        """Show the accessor and its parent, e.g. ``<.dt accessor of col('ts')>``."""
        return f"<.dt accessor of {self._e!r}>"

    def __getattr__(self, name: str) -> Any:
        """Point a pandas/Polars ``.dt`` idiom at its Batcher spelling.

        Only reached when normal lookup fails, so it never shadows a real ``.dt``
        method. ``.dt.tz_convert``, ``.dt.round``, ``.dt.to_period`` come back naming
        ``.dt.convert_timezone``, ``.dt.truncate``/``.dt.floor`` — see
        `batcher.plan.expr_ir.compat.guidance`.

        Args:
            name: The attribute name that was not found.

        Raises:
            AttributeError: Always, with guidance for `name`.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        raise accessor_attribute_error(self, "'.dt' accessor", name, DT_UNSUPPORTED)

    def truncate(self, unit: str) -> DateTrunc:
        """Truncate each timestamp down to the start of ``unit``.

        Zeroes out every field finer than ``unit`` (the floor toward the epoch), e.g.
        truncating to ``"month"`` gives the first of the month at midnight.
        Type-preserving.

        Args:
            unit: One of ``year``/``month``/``day``/``hour``/``minute``/``second``.

        Returns:
            A new Timestamp expression floored to ``unit``.

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
        """Test whether each row's year is a leap year (→ Bool).

        Follows the proleptic Gregorian rule: divisible by 4, except centuries that
        are not divisible by 400. Null → null.

        Returns:
            A new Boolean expression.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.date(2024, 1, 1), dt.date(2023, 1, 1)]})
                >>> ds.select(bt.col("d").dt.is_leap_year().alias("r")).to_pydict()
                {'r': [True, False]}
        """
        return DateFunc("is_leap_year", self._e)

    def days_in_month(self) -> DateFunc:
        """The number of days in each row's month, 28 to 31 (→ Int64).

        Accounts for leap years (February yields 29 in a leap year, else 28). Null →
        null.

        Returns:
            A new Int64 expression: the day count.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.date(2024, 2, 15), dt.date(2023, 2, 15)]})
                >>> ds.select(bt.col("d").dt.days_in_month().alias("r")).to_pydict()
                {'r': [29, 28]}
        """
        return DateFunc("days_in_month", self._e)

    def iso_year(self) -> DateFunc:
        """Return the ISO 8601 week-numbering year (→ Int64).

        May differ from the calendar year for dates in the first or last days of a
        year (e.g. 2021-01-01 can belong to ISO year 2020).

        Returns:
            A new Int64 expression: the ISO week-numbering year.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.date(2021, 1, 1)]})
                >>> ds.select(bt.col("d").dt.iso_year().alias("r")).to_pydict()
                {'r': [2020]}
        """
        return DateFunc("iso_year", self._e)

    def strftime(self, format: str) -> Strftime:
        """Format each date/time as text with a chrono/strftime pattern (→ Utf8).

        DuckDB ``strftime`` / Polars ``dt.strftime``.

        Args:
            format: A strftime pattern, e.g. ``"%Y-%m-%d"`` or ``"%H:%M:%S"``.

        Returns:
            A new Utf8 expression: the formatted text.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45)]})
                >>> ds.select(bt.col("d").dt.strftime("%Y-%m-%d").alias("r")).to_pydict()
                {'r': ['2024-02-15']}
        """
        return Strftime(self._e, format)

    def epoch_ms(self) -> Expr:
        """Milliseconds since the Unix epoch as an integer (DuckDB ``epoch_ms``, → Int64).

        The millisecond-resolution companion to the seconds-resolution ``.dt.epoch``;
        composed from the timestamp's underlying microseconds, so it carries no new IR.

        Returns:
            A new Int64 expression of milliseconds since 1970-01-01 UTC.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2021, 1, 1)]})
                >>> ds.select(r=bt.col("d").dt.epoch_ms()).to_pydict()
                {'r': [1609459200000]}
        """
        return (self._e.cast("int64") // 1000).cast("int64")

    def epoch_us(self) -> Expr:
        """Microseconds since the Unix epoch as an integer (DuckDB ``epoch_us``, → Int64).

        The microsecond-resolution epoch — the timestamp's own underlying value.

        Returns:
            A new Int64 expression of microseconds since 1970-01-01 UTC.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2021, 1, 1)]})
                >>> ds.select(r=bt.col("d").dt.epoch_us()).to_pydict()
                {'r': [1609459200000000]}
        """
        return self._e.cast("int64")

    def epoch_ns(self) -> Expr:
        """Nanoseconds since the Unix epoch as an integer (DuckDB ``epoch_ns``, → Int64).

        The nanosecond-resolution epoch; the stored microseconds scaled by 1000 (the
        sub-microsecond digits are always zero at microsecond storage resolution).

        Returns:
            A new Int64 expression of nanoseconds since 1970-01-01 UTC.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2021, 1, 1)]})
                >>> ds.select(r=bt.col("d").dt.epoch_ns()).to_pydict()
                {'r': [1609459200000000000]}
        """
        return self._e.cast("int64") * 1000

    def millisecond(self) -> Expr:
        """The millisecond-of-second component, 0-999 (Polars ``dt.millisecond``, → Int64).

        Returns:
            A new Int64 expression of the millisecond component.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 1, 1, 0, 0, 0, 123456)]})
                >>> ds.select(r=bt.col("d").dt.millisecond()).to_pydict()
                {'r': [123]}
        """
        return (self._e.cast("int64") % 1_000_000 // 1000).cast("int64")

    def microsecond(self) -> Expr:
        """The microsecond-of-second component, 0-999999 (Polars ``dt.microsecond``, → Int64).

        Returns:
            A new Int64 expression of the microsecond component.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 1, 1, 0, 0, 0, 123456)]})
                >>> ds.select(r=bt.col("d").dt.microsecond()).to_pydict()
                {'r': [123456]}
        """
        return self._e.cast("int64") % 1_000_000

    def nanosecond(self) -> Expr:
        """The nanosecond-of-second component, 0-999999000 (Polars ``dt.nanosecond``, → Int64).

        Microsecond-resolution storage means the last three digits are always zero.

        Returns:
            A new Int64 expression of the nanosecond component.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 1, 1, 0, 0, 0, 123456)]})
                >>> ds.select(r=bt.col("d").dt.nanosecond()).to_pydict()
                {'r': [123456000]}
        """
        return (self._e.cast("int64") % 1_000_000) * 1000

    # --- Polars-compatible spellings (delegate to the SQL-named accessors) ----------

    def weekday(self) -> Expr:
        """ISO weekday, Monday=1 … Sunday=7 — the Polars ``weekday`` spelling of ``isodow``.

        Returns:
            A new Int64 expression of the ISO weekday.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15)]})
                >>> ds.select(r=bt.col("d").dt.weekday()).to_pydict()
                {'r': [4]}
        """
        return self.isodow()

    def ordinal_day(self) -> Expr:
        """Day-of-year, 1-366 — the Polars ``ordinal_day`` spelling of ``dayofyear``.

        Returns:
            A new Int64 expression of the ordinal day.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15)]})
                >>> ds.select(r=bt.col("d").dt.ordinal_day()).to_pydict()
                {'r': [46]}
        """
        return self.dayofyear()

    def to_string(self, format: str) -> Expr:
        """Format as text — the Polars ``dt.to_string`` spelling of :meth:`strftime`.

        Args:
            format: A strftime pattern, e.g. ``"%Y-%m-%d"``.

        Returns:
            A new Utf8 expression: the formatted text.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15)]})
                >>> ds.select(r=bt.col("d").dt.to_string("%Y-%m-%d")).to_pydict()
                {'r': ['2024-02-15']}
        """
        return self.strftime(format)

    def date(self) -> Expr:
        """Truncate to the date (midnight) — the Polars ``dt.date`` spelling of ``truncate('day')``.

        Returns:
            A new Timestamp expression at 00:00:00 of the same day.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45)]})
                >>> ds.select(r=bt.col("d").dt.date()).to_pydict()
                {'r': [datetime.datetime(2024, 2, 15, 0, 0)]}
        """
        return self.truncate("day")

    def month_start(self) -> Expr:
        """First day of the month at midnight — ``truncate('month')`` (Polars ``month_start``).

        Returns:
            A new Timestamp expression at the start of the month.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45)]})
                >>> ds.select(r=bt.col("d").dt.month_start()).to_pydict()
                {'r': [datetime.datetime(2024, 2, 1, 0, 0)]}
        """
        return self.truncate("month")

    def month_end(self) -> Expr:
        """Last day of the month at midnight — the Polars ``month_end`` spelling of ``last_day``.

        Returns:
            A new Timestamp expression at the last day of the month.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15)]})
                >>> ds.select(r=bt.col("d").dt.month_end()).to_pydict()
                {'r': [datetime.datetime(2024, 2, 29, 0, 0)]}
        """
        return self.last_day()

    # --- time deltas between two timestamps -----------------------------------------

    def _delta_units(self, other: Expr, micros_per_unit: int) -> Expr:
        """Whole `micros_per_unit` units from `other` to this timestamp (truncated).

        Both sides are read as microseconds since the epoch and subtracted, so the
        difference is exact fixed-width arithmetic — no calendar ambiguity."""
        delta = self._e.cast("int64") - other.cast("int64")
        return (delta // micros_per_unit).cast("int64")

    def seconds_between(self, other: Expr) -> Expr:
        """Whole seconds from `other` to this timestamp (negative if `other` is later).

        Args:
            other: The earlier timestamp expression to measure from.

        Returns:
            An Int64 expression of the elapsed whole seconds.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict(
                ...     {"a": [dt.datetime(2024, 3, 1, 12)], "b": [dt.datetime(2024, 3, 1, 11)]}
                ... )
                >>> ds.select(r=bt.col("a").dt.seconds_between(bt.col("b"))).to_pydict()
                {'r': [3600]}
        """
        return self._delta_units(other, 1_000_000)

    def minutes_between(self, other: Expr) -> Expr:
        """Whole minutes from `other` to this timestamp (negative if `other` is later).

        Args:
            other: The earlier timestamp expression to measure from.

        Returns:
            An Int64 expression of the elapsed whole minutes.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict(
                ...     {"a": [dt.datetime(2024, 3, 1, 12)], "b": [dt.datetime(2024, 3, 1, 11)]}
                ... )
                >>> ds.select(r=bt.col("a").dt.minutes_between(bt.col("b"))).to_pydict()
                {'r': [60]}
        """
        return self._delta_units(other, 60_000_000)

    def hours_between(self, other: Expr) -> Expr:
        """Whole hours from `other` to this timestamp (negative if `other` is later).

        Args:
            other: The earlier timestamp expression to measure from.

        Returns:
            An Int64 expression of the elapsed whole hours.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict(
                ...     {"a": [dt.datetime(2024, 3, 1, 12)], "b": [dt.datetime(2024, 2, 28, 6)]}
                ... )
                >>> ds.select(r=bt.col("a").dt.hours_between(bt.col("b"))).to_pydict()
                {'r': [54]}
        """
        return self._delta_units(other, 3_600_000_000)

    def days_between(self, other: Expr) -> Expr:
        """Whole days from `other` to this timestamp — the elapsed-time feature.

        Counts fixed 24-hour days, so it is unaffected by calendar irregularities; a
        partial day truncates toward zero.

        Args:
            other: The earlier timestamp expression to measure from.

        Returns:
            An Int64 expression of the elapsed whole days.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict(
                ...     {"a": [dt.datetime(2024, 3, 1, 12)], "b": [dt.datetime(2024, 2, 28, 6)]}
                ... )
                >>> ds.select(r=bt.col("a").dt.days_between(bt.col("b"))).to_pydict()
                {'r': [2]}
        """
        return self._delta_units(other, 86_400_000_000)

    def weeks_between(self, other: Expr) -> Expr:
        """Whole 7-day weeks from `other` to this timestamp (negative if `other` is later).

        Args:
            other: The earlier timestamp expression to measure from.

        Returns:
            An Int64 expression of the elapsed whole weeks.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict(
                ...     {"a": [dt.datetime(2024, 3, 15)], "b": [dt.datetime(2024, 2, 1)]}
                ... )
                >>> ds.select(r=bt.col("a").dt.weeks_between(bt.col("b"))).to_pydict()
                {'r': [6]}
        """
        return self._delta_units(other, 7 * 86_400_000_000)

    def quarter_end(self) -> Expr:
        """Last day of the calendar quarter at midnight — the close of the quarter.

        Returns:
            A Timestamp expression at the quarter's final day.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 5, 15)]})
                >>> ds.select(r=bt.col("d").dt.quarter_end()).to_pydict()
                {'r': [datetime.datetime(2024, 6, 30, 0, 0)]}
        """
        return self.truncate("quarter").dt.offset_by("3mo").dt.offset_by("-1d")

    def year_end(self) -> Expr:
        """December 31st of this date's year at midnight.

        Returns:
            A Timestamp expression at the year's final day.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 5, 15)]})
                >>> ds.select(r=bt.col("d").dt.year_end()).to_pydict()
                {'r': [datetime.datetime(2024, 12, 31, 0, 0)]}
        """
        return self.truncate("year").dt.offset_by("1y").dt.offset_by("-1d")

    # --- pandas-compatible datetime spellings ---------------------------------------

    def day_name(self) -> Expr:
        """Full weekday name, e.g. ``"Monday"`` — the pandas ``dt.day_name``.

        Returns:
            A Utf8 expression of the weekday name.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15)]})
                >>> ds.select(r=bt.col("d").dt.day_name()).to_pydict()
                {'r': ['Thursday']}
        """
        return self.dayname()

    def month_name(self) -> Expr:
        """Full month name, e.g. ``"February"`` — the pandas ``dt.month_name``.

        Returns:
            A Utf8 expression of the month name.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15)]})
                >>> ds.select(r=bt.col("d").dt.month_name()).to_pydict()
                {'r': ['February']}
        """
        return self.monthname()

    def daysinmonth(self) -> Expr:
        """Days in this date's month — the pandas ``dt.daysinmonth`` spelling.

        Returns:
            An Int64 expression of the month's length in days.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15)]})
                >>> ds.select(r=bt.col("d").dt.daysinmonth()).to_pydict()
                {'r': [29]}
        """
        return self.days_in_month()

    def weekofyear(self) -> Expr:
        """ISO week number, 1-53 — the pandas ``dt.weekofyear`` spelling of ``week``.

        Returns:
            An Int64 expression of the ISO week number.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15)]})
                >>> ds.select(r=bt.col("d").dt.weekofyear()).to_pydict()
                {'r': [7]}
        """
        return self.week()

    def normalize(self) -> Expr:
        """Reset the time to midnight, keeping the date — the pandas ``dt.normalize``.

        Returns:
            A Timestamp expression at 00:00:00 of the same day.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45)]})
                >>> ds.select(r=bt.col("d").dt.normalize()).to_pydict()
                {'r': [datetime.datetime(2024, 2, 15, 0, 0)]}
        """
        return self.truncate("day")

    def floor(self, unit: str) -> Expr:
        """Round down to the start of `unit` — the pandas ``dt.floor`` spelling of ``truncate``.

        Args:
            unit: The granularity to floor to, e.g. ``"hour"``, ``"day"``, ``"month"``.

        Returns:
            A Timestamp expression floored to `unit`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45)]})
                >>> ds.select(r=bt.col("d").dt.floor("hour")).to_pydict()
                {'r': [datetime.datetime(2024, 2, 15, 13, 0)]}
        """
        return self.truncate(unit)

    # --- calendar feature flags (the date features a model actually consumes) -------

    def is_weekend(self) -> Expr:
        """True on Saturday or Sunday — the canonical calendar feature flag.

        Returns:
            A Boolean expression, true on weekend days.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 3), dt.datetime(2024, 2, 5)]})
                >>> ds.select(r=bt.col("d").dt.is_weekend()).to_pydict()
                {'r': [True, False]}
        """
        return self.isodow() >= 6

    def is_weekday(self) -> Expr:
        """True Monday through Friday — the complement of :meth:`is_weekend`.

        Returns:
            A Boolean expression, true on weekdays.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 3), dt.datetime(2024, 2, 5)]})
                >>> ds.select(r=bt.col("d").dt.is_weekday()).to_pydict()
                {'r': [False, True]}
        """
        return self.isodow() <= 5

    def is_month_start(self) -> Expr:
        """True on the first day of the month (pandas ``is_month_start``).

        Returns:
            A Boolean expression, true on the 1st.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 1), dt.datetime(2024, 2, 5)]})
                >>> ds.select(r=bt.col("d").dt.is_month_start()).to_pydict()
                {'r': [True, False]}
        """
        return self.day() == 1

    def is_month_end(self) -> Expr:
        """True on the last day of the month, leap years included (pandas ``is_month_end``).

        Returns:
            A Boolean expression, true on the month's final day.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 29), dt.datetime(2024, 2, 5)]})
                >>> ds.select(r=bt.col("d").dt.is_month_end()).to_pydict()
                {'r': [True, False]}
        """
        return self.day() == self.days_in_month()

    def is_quarter_start(self) -> Expr:
        """True on the first day of a calendar quarter (Jan/Apr/Jul/Oct 1).

        Returns:
            A Boolean expression, true on a quarter's first day.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 4, 1), dt.datetime(2024, 5, 1)]})
                >>> ds.select(r=bt.col("d").dt.is_quarter_start()).to_pydict()
                {'r': [True, False]}
        """
        return (self.month() % 3 == 1) & (self.day() == 1)

    def is_quarter_end(self) -> Expr:
        """True on the last day of a calendar quarter (Mar/Jun/Sep/Dec month-end).

        Returns:
            A Boolean expression, true on a quarter's final day.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 3, 31), dt.datetime(2024, 5, 15)]})
                >>> ds.select(r=bt.col("d").dt.is_quarter_end()).to_pydict()
                {'r': [True, False]}
        """
        return (self.month() % 3 == 0) & (self.day() == self.days_in_month())

    def is_year_start(self) -> Expr:
        """True on January 1st (pandas ``is_year_start``).

        Returns:
            A Boolean expression, true on the year's first day.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 1, 1), dt.datetime(2024, 2, 1)]})
                >>> ds.select(r=bt.col("d").dt.is_year_start()).to_pydict()
                {'r': [True, False]}
        """
        return (self.month() == 1) & (self.day() == 1)

    def is_year_end(self) -> Expr:
        """True on December 31st (pandas ``is_year_end``).

        Returns:
            A Boolean expression, true on the year's final day.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 12, 31), dt.datetime(2024, 2, 1)]})
                >>> ds.select(r=bt.col("d").dt.is_year_end()).to_pydict()
                {'r': [True, False]}
        """
        return (self.month() == 12) & (self.day() == 31)

    def quarter_start(self) -> Expr:
        """First day of the calendar quarter at midnight — ``truncate('quarter')``.

        Returns:
            A Timestamp expression at the start of the quarter.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 5, 15)]})
                >>> ds.select(r=bt.col("d").dt.quarter_start()).to_pydict()
                {'r': [datetime.datetime(2024, 4, 1, 0, 0)]}
        """
        return self.truncate("quarter")

    def year_start(self) -> Expr:
        """First day of the year at midnight — ``truncate('year')``.

        Returns:
            A Timestamp expression at the start of the year.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 5, 15)]})
                >>> ds.select(r=bt.col("d").dt.year_start()).to_pydict()
                {'r': [datetime.datetime(2024, 1, 1, 0, 0)]}
        """
        return self.truncate("year")

    def days_in_year(self) -> Expr:
        """Days in this date's year — 366 in a leap year, else 365 (→ Int64).

        Returns:
            An Int64 expression of the year's length in days.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 5, 1), dt.datetime(2023, 5, 1)]})
                >>> ds.select(r=bt.col("d").dt.days_in_year()).to_pydict()
                {'r': [366, 365]}
        """
        from batcher.plan.expr_ir.constructors import lit, when

        return when(self.is_leap_year()).then(lit(366)).otherwise(lit(365))

    def week_of_month(self) -> Expr:
        """Which week of the month the date falls in, 1-5 (→ Int64).

        Counted in whole 7-day blocks from the 1st, so days 1-7 are week 1.

        Returns:
            An Int64 expression of the 1-based week of the month.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 5, 3), dt.datetime(2024, 5, 15)]})
                >>> ds.select(r=bt.col("d").dt.week_of_month()).to_pydict()
                {'r': [1, 3]}
        """
        return ((self.day() - 1) // 7 + 1).cast("int64")

    def offset_by(self, by: str) -> DateOffset:
        """Shift each date/time by a Polars-style offset string. Type-preserving.

        Calendar units are calendar-correct: month/year arithmetic clamps to the end
        of the target month (e.g. Jan 31 + ``"1mo"`` → the last valid February day).
        A sub-day offset applied to a (date, not timestamp) column raises ``ValueError``.

        Args:
            by: Signed counts with units ``y``/``mo``/``w``/``d``/``h``/``m``/``s``,
                combinable, e.g. ``"1mo15d"``, ``"-3d"``, ``"1h30m"``.

        Returns:
            A new expression shifted by the offset, type-preserved.

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

        Returns:
            A new Timestamp expression, or null for invalid local times.

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
    "A new :class:`~batcher.Expr` carrying the extracted field.",
)
