"""The `.dt` accessor namespace plus the Polars-style offset-string parser.

`col("d").dt.year()`, `.dt.truncate("month")`, `.dt.offset_by("1mo15d")`, … — each
builds a `bc-expr` date node. The parameterless field extractions are generated
from `_DT_FIELDS` (data, not code).
"""

from __future__ import annotations

import re
from typing import Any

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.compat.guidance import DT_UNSUPPORTED, accessor_attribute_error
from batcher.plan.expr_ir.constructors import col, lit, when
from batcher.plan.expr_ir.core import Expr
from batcher.plan.expr_ir.func_nodes import (
    ConvertTimezone,
    DateFunc,
    DateOffset,
    DateTrunc,
    MakeTemporal,
    Strftime,
)
from batcher.plan.expr_ir.namespaces._bind import _bind_accessors
from batcher.plan.expr_ir.namespaces._temporal_units import trunc_unit
from batcher.plan.ir_tags import MICROS_PER_DAY

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

# `dt.next_day`: Spark's weekday spellings (full, three- and two-letter) as ISO day numbers.
_WEEKDAYS = {
    spelling: number
    for number, name in enumerate(
        ("MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"), start=1
    )
    for spelling in (name, name[:3], name[:2])
}


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


#: How far `ceil`/`round` advance to reach the next boundary of each unit. Only units with
#: a step `offset_by` can express appear: the calendar ones (`1mo`, `1y`) are exact because
#: `offset_by` does calendar arithmetic, so "the next month" is the next month rather than
#: thirty days. Sub-second units are absent because `offset_by` has no sub-second step —
#: `truncate` alone already reaches them.
_UNIT_STEP: dict[str, str] = {
    "second": "1s",
    "minute": "1m",
    "hour": "1h",
    "day": "1d",
    "week": "1w",
    "month": "1mo",
    "quarter": "3mo",
    "year": "1y",
}


def _step_offset(unit: str, func: str) -> str:
    """The `offset_by` step that reaches the next `unit` boundary, or raise.

    Normalizes through `_trunc_unit` first so `ceil`/`round` accept exactly the vocabulary
    `truncate` does -- they call `truncate` on the same string, so a spelling one of them
    understood and the other did not would fail halfway through building the expression.
    """
    step = _UNIT_STEP.get(trunc_unit(unit, func))
    if step is None:
        raise PlanError(
            f".dt.{func}({unit!r}) is not supported; use one of "
            f"{sorted(_UNIT_STEP)} (.dt.truncate reaches the finer units)"
        )
    return step


_CLOCK_RE = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2})(?:\.(\d{1,6}))?)?$")


def _clock_micros(value: str, arg: str) -> int:
    """A ``HH:MM``/``HH:MM:SS[.ffffff]`` clock time as microseconds since midnight."""
    match = _CLOCK_RE.match(value.strip()) if isinstance(value, str) else None
    if match is None:
        raise PlanError(
            f"is_between_time() {arg} must be a clock time like '09:30' or '09:30:00', "
            f"got {value!r}"
        )
    hh, mm, ss, frac = match.group(1), match.group(2), match.group(3) or "0", match.group(4) or ""
    if not (0 <= int(hh) <= 23 and 0 <= int(mm) <= 59 and 0 <= int(ss) <= 59):
        raise PlanError(f"is_between_time() {arg} is not a valid clock time: {value!r}")
    micros = (int(hh) * 3600 + int(mm) * 60 + int(ss)) * 1_000_000
    return micros + int(frac.ljust(6, "0") or 0)


def _wrap_temporal(other: Any) -> Expr:
    """The other side of a ``*_between`` difference, as an expression.

    A bare column name is the natural spelling (``a.dt.days_between("b")``) and used to
    reach ``other.cast(...)`` as a `str`, raising ``AttributeError: 'str' object has no
    attribute 'cast'`` -- an error naming an internal call rather than the argument.
    """
    if isinstance(other, Expr):
        return other
    if isinstance(other, str):
        return col(other)
    raise PlanError(
        f"dt.*_between(): other must be a timestamp column name or expression, got "
        f"{type(other).__name__} {other!r}"
    )


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

    def truncate(self, unit: str, preserve_type: bool = False) -> DateTrunc:
        """Truncate each date or timestamp down to the start of ``unit``.

        Zeroes out every field finer than ``unit`` (the floor toward the epoch), e.g.
        truncating to ``"month"`` gives the first of the month at midnight.

        The result is a Timestamp for a Date input too, as DuckDB's ``date_trunc`` is.
        Pass ``preserve_type=True`` for Polars' ``dt.truncate``, where a Date stays a
        Date. A timestamp input is a timestamp either way.

        Args:
            unit: One of ``millennium``/``century``/``decade``/``year``/``quarter``/
                ``month``/``week``/``day``/``hour``/``minute``/``second``/
                ``millisecond``/``microsecond``. The duration spellings ``offset_by``
                takes are accepted too (``"1mo"``/``"mo"``, ``"1d"``/``"d"``, ...),
                where ``mo`` is months and ``m`` is minutes.
            preserve_type: Return a Date for a Date input instead of a midnight Timestamp.

        Returns:
            A new expression floored to ``unit``: a Timestamp, or a Date for a Date input
            when ``preserve_type`` is set.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45)]})
                >>> ds.select(bt.col("d").dt.truncate("month").alias("r")).to_pydict()
                {'r': [datetime.datetime(2024, 2, 1, 0, 0)]}

                >>> ds.select(bt.col("d").dt.truncate("1mo").alias("r")).to_pydict()
                {'r': [datetime.datetime(2024, 2, 1, 0, 0)]}

                >>> days = bt.from_pydict({"d": [dt.date(2024, 2, 15)]})
                >>> days.select(r=bt.col("d").dt.truncate("month", preserve_type=True)).to_pydict()
                {'r': [datetime.date(2024, 2, 1)]}
        """
        return DateTrunc(self._e, trunc_unit(unit, "truncate"), preserve_type=bool(preserve_type))

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

    def _micros(self) -> Expr:
        """The microsecond epoch count, whatever temporal type the input is.

        The cast to timestamp is load-bearing. A `Date32`'s integer value is a **day**
        count, so reading it as an integer directly reported 19,787 microseconds for
        2024-03-05 (and 19 milliseconds) instead of the instant it denotes — a wrong
        answer with no error. On a timestamp column the cast is a no-op.

        Returns:
            An Int64 expression of microseconds since the Unix epoch.
        """
        return self._e.cast("timestamp").cast("int64")

    def epoch_us(self) -> Expr:
        """Microseconds since the Unix epoch as an integer (DuckDB ``epoch_us``, → Int64).

        The microsecond-resolution epoch: the timestamp's own underlying value. A ``Date``
        input reads as its midnight instant.

        Returns:
            A new Int64 expression of microseconds since 1970-01-01 UTC.

        Examples:
            .. doctest::

                >>> import datetime as dt
                >>> import batcher as bt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2021, 1, 1)]})
                >>> ds.select(r=bt.col("d").dt.epoch_us()).to_pydict()
                {'r': [1609459200000000]}
        """
        return self._micros()

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
        # Truncated toward zero, not floored: DuckDB's `epoch_ms` reports
        # `1969-12-31 23:59:59.999999` as 0 milliseconds, where `//` (floor division)
        # answered -1. The two agree on every instant at or after the epoch, which is why
        # the difference only ever showed up on historical data.
        micros = self._micros()
        return (micros // 1000 + (((micros % 1000) != 0) & (micros < 0)).cast("int64")).cast(
            "int64"
        )

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
        return self._micros() * 1000

    def _subsecond_micros(self) -> Expr:
        """Microseconds past the whole second, always in ``[0, 999999]``.

        Two corrections over the obvious ``self._e.cast("int64") % 1_000_000``, and each was
        wrong on its own axis. The engine's ``%`` is the *truncated* remainder, which takes
        the sign of the dividend — and a pre-1970 instant has a negative epoch, so
        ``1969-07-20 20:17:40.000001`` reported **-999999** microseconds past the second and
        ``.999999`` reported ``-1``, while `hour`/`minute`/`second` all read correctly. And
        the raw integer of a ``Date32`` is a *day* count, so a date column produced a
        six-digit number out of its day index; `_micros` is the accessor that normalizes
        that, and it exists for exactly this reason.

        Returns:
            An Int64 expression of the microseconds past the whole second, in [0, 999999].
        """
        micros = self._micros()
        return (micros % 1_000_000 + 1_000_000) % 1_000_000

    def microsecond(self) -> Expr:
        """The microsecond-of-second component, 0-999999 (Polars ``dt.microsecond``, → Int64).

        Returns:
            A new Int64 expression of the microseconds past the whole second.

        Examples:
            .. doctest::

                >>> import datetime as dt
                >>> import batcher as bt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 1, 1, 0, 0, 0, 123456)]})
                >>> ds.select(r=bt.col("d").dt.microsecond()).to_pydict()
                {'r': [123456]}
        """
        return self._subsecond_micros()

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
        return (self._subsecond_micros() // 1000).cast("int64")

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
        return self._subsecond_micros() * 1000

    # --- Polars-compatible spellings (delegate to the SQL-named accessors) ----------

    def date(self) -> Expr:
        """Extract the calendar date — the Polars ``dt.date`` spelling of ``CAST(ts AS DATE)``.

        The result is a ``date``, not a midnight timestamp. This method was previously an
        alias for ``truncate('day')``, which returns a ``timestamp`` at 00:00:00. Both Polars'
        ``dt.date`` and DuckDB's ``CAST(ts AS DATE)`` / ``date(ts)`` return a date, and a
        midnight timestamp does not compare equal to a date column, so the old spelling
        silently failed the join or filter a user reached for it to write. Use
        :meth:`truncate` when a timestamp at midnight is what you want.

        Returns:
            A new Date expression holding the calendar date.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45)]})
                >>> ds.select(r=bt.col("d").dt.date()).to_pydict()
                {'r': [datetime.date(2024, 2, 15)]}
        """
        return self._e.cast("date")

    def month_start(self, keep_time: bool = False) -> Expr:
        """First day of the month at midnight, which is ``truncate('month')``.

        By default the result is a midnight Timestamp for either input type, the DuckDB
        ``date_trunc('month', x)``. Pass ``keep_time=True`` for Polars' ``month_start``,
        which rolls only the date back: the input's type is kept (a Date stays a Date)
        and a timestamp keeps its time of day.

        Args:
            keep_time: Keep the input's type and time of day instead of a midnight Timestamp.

        Returns:
            A new expression at the start of the month.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45)]})
                >>> ds.select(r=bt.col("d").dt.month_start()).to_pydict()
                {'r': [datetime.datetime(2024, 2, 1, 0, 0)]}

                >>> ds.select(r=bt.col("d").dt.month_start(keep_time=True)).to_pydict()
                {'r': [datetime.datetime(2024, 2, 1, 13, 45)]}
        """
        if keep_time:
            return DateTrunc(self._e, "month", preserve_type=True, keep_time=True)
        return self.truncate("month")

    def last_day(self, keep_time: bool = False) -> Expr:
        """The last day of the month, as a Date (DuckDB and Spark ``last_day``).

        By default the result is a Date for either input type. Pass ``keep_time=True``
        for Polars' ``month_end``, which keeps the input's type and a timestamp's time of
        day. That form rolls to the start of the month, forward one month and back one
        day, so it is calendar-exact across leap years.

        Args:
            keep_time: Keep the input's type and time of day instead of returning a Date.

        Returns:
            A new expression holding the month's last day.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45, 30)]})
                >>> ds.select(r=bt.col("d").dt.last_day()).to_pydict()
                {'r': [datetime.date(2024, 2, 29)]}

                >>> ds.select(r=bt.col("d").dt.last_day(keep_time=True)).to_pydict()
                {'r': [datetime.datetime(2024, 2, 29, 13, 45, 30)]}
        """
        if keep_time:
            return self.month_start(keep_time=True).dt.offset_by("1mo").dt.offset_by("-1d")
        return DateFunc("last_day", self._e)

    def dayofweek(self, start: str = "sunday", base: int = 0) -> Expr:
        """The day of week as an integer, Sunday = 0 through Saturday = 6 by default.

        The default is DuckDB's ``dayofweek``. `start` names the first day of the week and
        `base` the number it gets, which covers every other convention in use:

        * ``start="sunday", base=1`` is Spark ``dayofweek`` and SQL ``DAYOFWEEK``.
        * ``start="monday", base=0`` is Spark ``weekday``, Daft ``day_of_week`` and
          pandas ``dayofweek``.
        * ``start="monday", base=1`` is ISO, the same as :meth:`weekday`.

        Args:
            start: The first day of the week, ``"sunday"`` or ``"monday"``.
            base: The number the first day gets, ``0`` or ``1``.

        Returns:
            A new Int64 expression of the day of week.

        Raises:
            PlanError: If `start` or `base` is not one of the accepted values.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.date(2024, 2, 18)]})  # a Sunday
                >>> ds.select(
                ...     duckdb=bt.col("d").dt.dayofweek(),
                ...     spark=bt.col("d").dt.dayofweek(base=1),
                ...     daft=bt.col("d").dt.dayofweek(start="monday"),
                ... ).to_pydict()
                {'duckdb': [0], 'spark': [1], 'daft': [6]}
        """
        if start not in ("sunday", "monday"):
            raise PlanError(f"dt.dayofweek(): start must be 'sunday' or 'monday', got {start!r}")
        if isinstance(base, bool) or base not in (0, 1):
            raise PlanError(f"dt.dayofweek(): base must be 0 or 1, got {base!r}")
        if start == "sunday":
            sunday0 = DateFunc("day_of_week", self._e)
            return sunday0 if base == 0 else sunday0 + 1
        iso = self.weekday()  # Monday = 1 ... Sunday = 7
        return iso if base == 1 else iso - 1

    def dayname(self, abbreviated: bool = False) -> Expr:
        """The English weekday name, e.g. ``"Monday"`` (→ Utf8).

        The full name is DuckDB's ``dayname``. Pass ``abbreviated=True`` for the
        three-letter form Spark's ``dayname`` returns, e.g. ``"Mon"``.

        Args:
            abbreviated: Return the three-letter abbreviation instead of the full name.

        Returns:
            A new Utf8 expression of the weekday name.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45, 30)]})
                >>> ds.select(
                ...     full=bt.col("d").dt.dayname(),
                ...     short=bt.col("d").dt.dayname(abbreviated=True),
                ... ).to_pydict()
                {'full': ['Thursday'], 'short': ['Thu']}
        """
        return Strftime(self._e, "%a") if abbreviated else DateFunc("dayname", self._e)

    def monthname(self, abbreviated: bool = False) -> Expr:
        """The English month name, e.g. ``"January"`` (→ Utf8).

        The full name is DuckDB's ``monthname``. Pass ``abbreviated=True`` for the
        three-letter form Spark's ``monthname`` returns, e.g. ``"Jan"``.

        Args:
            abbreviated: Return the three-letter abbreviation instead of the full name.

        Returns:
            A new Utf8 expression of the month name.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45, 30)]})
                >>> ds.select(
                ...     full=bt.col("d").dt.monthname(),
                ...     short=bt.col("d").dt.monthname(abbreviated=True),
                ... ).to_pydict()
                {'full': ['February'], 'short': ['Feb']}
        """
        return Strftime(self._e, "%b") if abbreviated else DateFunc("monthname", self._e)

    # --- time deltas between two timestamps -----------------------------------------

    def _delta_units(self, other: Expr, micros_per_unit: int) -> Expr:
        """Whole `micros_per_unit` units from `other` to this timestamp (truncated).

        Both sides are read as microseconds since the epoch and subtracted, so the
        difference is exact fixed-width arithmetic — no calendar ambiguity.

        The truncation is toward *zero*, not toward negative infinity, which is what DuckDB's
        `date_diff` does and what every docstring here promises. Plain `//` floors, so a
        backwards interval of 2.5 days returned -3 while the same interval measured forwards
        returned 2: the operation was not antisymmetric, and an SLA computed in the other
        direction silently gained a whole day. Dividing the magnitude and reapplying the sign
        keeps `a.days_between(b) == -b.days_between(a)` for every input.

        Both sides go through `_micros`. Casting straight to Int64 read a Date32's *day*
        count as microseconds, so ``date(2024, 1, 10).days_between(date(2024, 1, 1))``
        answered 0 where DuckDB's ``date_diff('day', ...)`` answers 9: nine microseconds
        is no whole day.
        """
        other = _wrap_temporal(other)
        delta = self._micros() - other.dt._micros()
        magnitude = delta.abs() // micros_per_unit
        return (delta.sign().cast("int64") * magnitude).cast("int64")

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
        """Whole days from `other` to this date or timestamp — the elapsed-time feature.

        Counts fixed 24-hour days, so it is unaffected by calendar irregularities; a
        partial day truncates toward zero. On two Date columns it is the day count
        DuckDB's ``date_diff('day', other, self)`` and Spark's ``datediff(self, other)``
        return.

        Args:
            other: The earlier date or timestamp expression to measure from.

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

                >>> days = bt.from_pydict({"a": [dt.date(2024, 1, 10)], "b": [dt.date(2024, 1, 1)]})
                >>> days.select(r=bt.col("a").dt.days_between("b")).to_pydict()
                {'r': [9]}
        """
        return self._delta_units(other, MICROS_PER_DAY)

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
        return self._delta_units(other, 7 * MICROS_PER_DAY)

    def months_between(self, other: Expr | str, *, round_off: bool = True) -> Expr:
        """Fractional months from `other` to this date or timestamp (Spark ``months_between``).

        Spark's definition. The whole-month count comes from the calendar fields. When both
        values fall on the same day of the month, or both on the last day of their months,
        the answer is that whole count. Otherwise the leftover days and time of day are
        added as a fraction of a fixed 31-day month, not of the month's real length.
        `round_off` rounds the result to 8 decimal places, as Spark does by default.
        Timestamps are read as naive wall-clock values, with no session time zone.

        DuckDB has no fractional form. Its ``date_diff('month', b, a)`` counts month
        boundaries crossed and answers an integer.

        Args:
            other: The earlier date or timestamp, as an expression or a column name.
            round_off: Round to 8 decimal places; ``False`` keeps full precision.

        Returns:
            A Float64 expression, negative when `other` is later.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict(
                ...     {"a": [dt.datetime(1997, 2, 28, 10, 30)], "b": [dt.datetime(1996, 10, 30)]}
                ... )
                >>> ds.select(r=bt.col("a").dt.months_between("b")).to_pydict()
                {'r': [3.94959677]}
        """
        other = _wrap_temporal(other)
        right = other.dt
        months = (self.year() - right.year()) * 12 + (self.month() - right.month())
        same_day = self.day() == right.day()
        both_month_ends = (self.day() == self.days_in_month()) & (
            right.day() == right.days_in_month()
        )
        micros = (self.day() - right.day()) * MICROS_PER_DAY + (
            self.time_of_day() - right.time_of_day()
        )
        fraction = months + micros.cast("float64") / lit(float(31 * MICROS_PER_DAY))
        exact = fraction.round(8) if round_off else fraction
        whole = months.cast("float64")
        return when(same_day | both_month_ends).then(whole).otherwise(exact)

    def next_day(self, day_of_week: str) -> Expr:
        """The first date strictly after this one that falls on `day_of_week` (→ Date).

        Spark ``next_day``. `day_of_week` is a day name, its three-letter abbreviation or
        its two-letter one (``"Sunday"``, ``"Sun"``, ``"SU"``), in any case. A date already
        on that weekday moves a full week ahead. A timestamp is read as its date.

        Args:
            day_of_week: The weekday to advance to.

        Returns:
            A Date expression.

        Raises:
            PlanError: If `day_of_week` does not name a weekday.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.date(2015, 7, 27)]})
                >>> ds.select(r=bt.col("d").dt.next_day("Sun")).to_pydict()
                {'r': [datetime.date(2015, 8, 2)]}
        """
        target = (
            _WEEKDAYS.get(day_of_week.strip().upper()) if isinstance(day_of_week, str) else None
        )
        if target is None:
            raise PlanError(f"dt.next_day(): {day_of_week!r} does not name a weekday such as 'Mon'")
        # ((target - today + 6) mod 7) + 1 is 7 when the two coincide and never 0. The shift
        # is per row and `offset_by` takes a constant, so it runs on the day count, which is
        # what a Date holds.
        shift = ((lit(target) - self.weekday() + 6) % 7) + 1
        return MakeTemporal("from_unix_date", [self._e.cast("date").cast("int64") + shift])

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

    def ceil(self, unit: str) -> Expr:
        """Round **up** to the start of the next `unit` — pandas ``dt.ceil``.

        The mirror of :meth:`floor`: an instant already on a boundary stays put, and any
        other advances to the next one. Use it to close a half-open bucket — the end of the
        hour a reading belongs to — where `floor` gives its start.

        Composed from `truncate` and `offset_by`, so it adds no engine surface and inherits
        their calendar correctness: rounding February 15th up to a month gives March 1st,
        not "thirty days later".

        Args:
            unit: The granularity to round up to. One of ``second``, ``minute``, ``hour``,
                ``day``, ``week``, ``month``, ``quarter``, ``year``.

        Returns:
            A Timestamp expression at the start of the next `unit`, or unchanged if it is
            already there.

        Raises:
            PlanError: If `unit` has no fixed step to advance by.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45)]})
                >>> ds.select(r=bt.col("d").dt.ceil("hour")).to_pydict()
                {'r': [datetime.datetime(2024, 2, 15, 14, 0)]}

                >>> on_the_hour = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 0)]})
                >>> on_the_hour.select(r=bt.col("d").dt.ceil("hour")).to_pydict()
                {'r': [datetime.datetime(2024, 2, 15, 13, 0)]}
        """
        floor = self.truncate(unit)
        step = _step_offset(unit, "ceil")
        # `truncate` returns a Timestamp while the input may be a Date or a text column, so
        # compare through the epoch rather than directly: `d == floor` would be a
        # cross-type comparison for exactly the inputs a user is most likely to pass.
        return (
            when(floor.dt.epoch_us() == self.epoch_us())
            .then(floor)
            .otherwise(floor.dt.offset_by(step))
        )

    def round(self, unit: str) -> Expr:
        """Round to the **nearest** `unit` boundary — pandas ``dt.round``.

        An instant exactly half way rounds **up**, the everyday reading of "round to the
        nearest hour". (pandas breaks that tie to the even boundary instead; the difference
        shows only for an instant landing precisely on a half-boundary.) Where :meth:`floor`
        and :meth:`ceil` bias every value one way, this is the one to bucket by when the bias
        would accumulate — plotting a downsampled series, or aligning two feeds sampled off
        each other's grid.

        Composed from `truncate` and `offset_by` over the microsecond epoch, so a calendar
        unit rounds by real elapsed time: a date in mid-February is nearer to March 1st than
        to February 1st, and rounds there.

        Args:
            unit: The granularity to round to. One of ``second``, ``minute``, ``hour``,
                ``day``, ``week``, ``month``, ``quarter``, ``year``.

        Returns:
            A Timestamp expression at the nearer `unit` boundary.

        Raises:
            PlanError: If `unit` has no fixed step to advance by.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 45)]})
                >>> ds.select(r=bt.col("d").dt.round("hour")).to_pydict()
                {'r': [datetime.datetime(2024, 2, 15, 14, 0)]}

                >>> early = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 13, 10)]})
                >>> early.select(r=bt.col("d").dt.round("hour")).to_pydict()
                {'r': [datetime.datetime(2024, 2, 15, 13, 0)]}
        """
        floor = self.truncate(unit)
        nxt = floor.dt.offset_by(_step_offset(unit, "round"))
        here, below, above = self.epoch_us(), floor.dt.epoch_us(), nxt.dt.epoch_us()
        # Strict `<`, so an exact half-way instant falls to the `otherwise` branch and
        # rounds up. Comparing elapsed microseconds is what makes a calendar unit round by
        # real distance: mid-February is nearer to March than to February.
        return when((here - below) < (above - here)).then(floor).otherwise(nxt)

    def time_of_day(self) -> Expr:
        """Microseconds since midnight — the clock time, with the date discarded.

        The handle for "when in the day did this happen": how far into the trading session a
        trade landed, whether a reading came from the night shift, how a weekday's load curve
        compares across weeks. Comparing timestamps directly cannot answer any of those,
        because the date dominates the ordering.

        Composed from `truncate` and the microsecond epoch, so it adds no engine surface and
        needs no timezone of its own: it is the clock time in whatever zone the column is
        already expressed in.

        Returns:
            An Int64 expression of microseconds since the day's midnight, always in
            ``[0, 86_400_000_000)``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 15, 9, 30)]})
                >>> ds.select(r=bt.col("d").dt.time_of_day() // 1_000_000).to_pydict()
                {'r': [34200]}
        """
        return self.epoch_us() - self.truncate("day").dt.epoch_us()

    def is_between_time(self, start: str, end: str) -> Expr:
        """True where the clock time falls in ``[start, end]`` — pandas ``between_time``.

        The filter a session-bounded query needs: market hours, a night shift, a maintenance
        window. `start` and `end` are ``"HH:MM"`` or ``"HH:MM:SS"`` clock times, and the date
        is ignored entirely.

        **A window that wraps past midnight is handled**, and that is the reason this exists
        rather than a bare comparison: ``is_between_time("22:00", "02:00")`` means the four
        hours around midnight, where ``hour() >= 22 and hour() <= 2`` is empty. Getting that
        wrong returns no rows rather than an error, which is why it is worth having in one
        tested place.

        Both endpoints are inclusive, matching pandas' default.

        Args:
            start: The first clock time in the window, ``"HH:MM"`` or ``"HH:MM:SS"``.
            end: The last clock time in the window; may be earlier than `start` to wrap
                past midnight.

        Returns:
            A Boolean expression, true inside the window.

        Raises:
            PlanError: If a bound is not a valid ``HH:MM``/``HH:MM:SS`` clock time.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict(
                ...     {"d": [dt.datetime(2024, 2, 15, 9, 30), dt.datetime(2024, 2, 15, 20, 0)]}
                ... )
                >>> ds.select(open=bt.col("d").dt.is_between_time("09:00", "17:00")).to_pydict()
                {'open': [True, False]}

                >>> # A window that wraps past midnight keeps the late evening.
                >>> ds.select(night=bt.col("d").dt.is_between_time("22:00", "10:00")).to_pydict()
                {'night': [True, False]}
        """
        lo, hi = _clock_micros(start, "start"), _clock_micros(end, "end")
        now = self.time_of_day()
        if lo <= hi:
            return (now >= lit(lo)) & (now <= lit(hi))
        # A wrapping window is the union of "after start today" and "before end today".
        return (now >= lit(lo)) | (now <= lit(hi))

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
        return self.weekday() >= 6

    def timestamp(self, unit: str = "us") -> Expr:
        """Epoch count at `unit` — the Polars ``dt.timestamp`` spelling (→ Int64).

        Args:
            unit: ``"s"``, ``"ms"``, ``"us"`` (the default, as in Polars) or ``"ns"``.
                Daft's ``to_unix_epoch`` defaults to seconds, so it is ``unit="s"``.

        Returns:
            A new Int64 expression: the epoch count at that resolution.

        Raises:
            PlanError: If `unit` is not one of the four.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(1970, 1, 1, 0, 0, 1)]})
                >>> ds.select(r=bt.col("d").dt.timestamp("ms")).to_pydict()
                {'r': [1000]}
        """
        readers = {
            "s": self.epoch,
            "ms": self.epoch_ms,
            "us": self.epoch_us,
            "ns": self.epoch_ns,
        }
        reader = readers.get(unit)
        if reader is None:
            raise PlanError(f"dt.timestamp(): unit must be one of {sorted(readers)}, got {unit!r}")
        return reader()

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
        """Days in this date's year — 366 in a leap year, else 365 (→ Int64). NULL in, NULL out.

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
        # Arithmetic rather than `when(is_leap_year()).then(366).otherwise(365)`: a NULL date
        # makes `is_leap_year()` NULL, which is not *true*, so the CASE fell through to its
        # ELSE and answered 365 for a row that has no year at all — a silent wrong answer
        # where the neighbouring `days_in_month` returns NULL. Adding to a NULL propagates,
        # and it is one kernel rather than a three-branch CASE.
        return self.is_leap_year().cast("int64") + 365

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

    def is_business_day(self) -> Expr:
        """True Monday through Friday — the complement of :meth:`is_weekend`.

        Returns:
            A Boolean expression, true on weekdays.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import datetime as dt
                >>> ds = bt.from_pydict({"d": [dt.datetime(2024, 2, 3), dt.datetime(2024, 2, 5)]})
                >>> ds.select(r=bt.col("d").dt.is_business_day()).to_pydict()
                {'r': [False, True]}
        """
        return self.weekday() <= 5


# Python accessor name → engine `DateFunc` wire tag (serde snake_case). Each maps
# to one Arrow `DatePart` and matches the same-named DuckDB function.
_DT_FIELDS = {
    "weekday": "isodow",  # ISO day of week: Monday = 1 … Sunday = 7 (→ Int64)
    "year": "year",
    "month": "month",
    "day": "day",
    "hour": "hour",
    "minute": "minute",
    "second": "second",
    "quarter": "quarter",
    "week": "week",  # ISO week 1–53
    "dayofyear": "day_of_year",  # 1–366
    "epoch": "epoch",  # seconds since the Unix epoch (→ Int64)
    "century": "century",  # the century, e.g. 2021 → 21 (→ Int64)
    "decade": "decade",  # the decade, e.g. 2021 → 202 (→ Int64)
    "millennium": "millennium",  # the millennium, e.g. 2021 → 3 (→ Int64)
}


_bind_accessors(
    _DtNamespace,
    _DT_FIELDS,
    lambda e, t: DateFunc(t, e),
    lambda n: f"Extract the {n} field of a date/time column (→ Int64).",
    "A new :class:`~batcher.Expr` carrying the extracted field.",
)
