"""Temporal free functions.

`current_timestamp`/`current_date` bind the wall-clock **once, at plan-build time**
to a literal (SQL statement-timestamp semantics) — so the value is fixed for the
query and stays identical single-node and distributed, never a per-row clock read.
`date_part`/`date_add`/`date_sub` dispatch onto the existing `.dt` accessor, so they
add no engine surface.
"""

from __future__ import annotations

import datetime as _dt

from batcher._internal.errors import PlanError, require_int
from batcher.plan.expr_ir.core import Expr, IntoExpr, Lit, _wrap
from batcher.plan.expr_ir.func_nodes import MakeTemporal, WindowBuckets, WindowStart
from batcher.plan.expr_ir.namespaces.temporal import parse_offset

_DAY_MICROS = 86_400_000_000


def _duration_micros(duration: str, *, arg: str) -> int:
    """Parse a fixed-length duration string to microseconds (no calendar units).

    Event-time windows must have a fixed width, so a calendar duration (months, years) is
    rejected — ``"1mo"`` has no constant microsecond length. Both duration syntaxes are
    accepted: the compact combinable form `parse_offset` reads (``"1d"``, ``"1h30m"``) and the
    spelled-out single-unit form the streaming module reads (``"1 hour"``, ``"500ms"``). The
    two used to be disjoint, so ``"1d"`` sized a window but could not delay a watermark and
    ``"10 seconds"`` did the reverse — a trap, because one pipeline writes both.

    Every rejection here raises `PlanError`, including an unparseable string: `parse_offset`
    raises a bare `ValueError` whose advice names the ``y``/``mo`` units *this* function goes
    on to refuse, so passing it through would both break the typed-error contract and point
    the caller at a unit that cannot work.
    """
    try:
        months, days, micros = parse_offset(duration)
    except ValueError as exc:
        from batcher.plan.streaming.spec import parse_interval_seconds

        try:
            return _positive_micros(
                round(parse_interval_seconds(duration) * 1_000_000), duration, arg
            )
        except PlanError:
            raise PlanError(
                f"cannot parse {arg} {duration!r}; a window needs a fixed-length duration: "
                "counts with units w/d/h/m/s, e.g. '1h', '30m', '1h30m' ('m' is minutes), or "
                "the spelled-out form '1 hour'. Calendar units (y/mo) are not fixed-length."
            ) from exc
    if months:
        raise PlanError(
            f"{arg} {duration!r} uses a calendar unit (month/year) with no fixed length; "
            "use fixed units (days/hours/minutes/seconds)"
        )
    return _positive_micros(days * _DAY_MICROS + micros, duration, arg)


def _positive_micros(total: int, duration: str, arg: str) -> int:
    """The parsed microseconds, rejecting a zero or negative window width."""
    if total <= 0:
        raise PlanError(f"{arg} must be a positive duration, got {duration!r}")
    return total


def window(time_col: IntoExpr, duration: str, slide: str | None = None) -> Expr:
    """Assign each row to an event-time window (Spark ``window``).

    Returns the window-**start** timestamp to group by:
    ``ds.group_by(w=window(col("ts"), "1 hour")).agg(...)`` buckets rows into hourly
    tumbling windows. With `slide`, it returns the *list* of overlapping sliding
    windows' starts (width `duration`, hop `slide`) — fan it out with ``unnest``
    before grouping. Durations are fixed-length (days/hours/minutes/seconds); a
    calendar unit (month/year) is rejected.

    Args:
        time_col: The event-time column to bucket.
        duration: Fixed-length window width, e.g. ``"1h"`` or ``"30m"``.
        slide: Optional hop for sliding windows; ``None`` gives tumbling windows.

    Returns:
        The window-start timestamp to group by (a list of starts when ``slide`` is set).

    Raises:
        PlanError: If a duration uses a calendar unit or is not positive.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> import datetime as dt
            >>> ds = bt.from_pydict(
            ...     {
            ...         "ts": [dt.datetime(2024, 1, 1, 10, 5), dt.datetime(2024, 1, 1, 11, 5)],
            ...         "v": [1, 3],
            ...     }
            ... )
            >>> agg = ds.group_by(w=bt.window(bt.col("ts"), "1h")).agg(s=bt.col("v").sum())
            >>> out = agg.sort("w").to_pydict()
            >>> out["w"]
            [datetime.datetime(2024, 1, 1, 10, 0), datetime.datetime(2024, 1, 1, 11, 0)]
            >>> out["s"]
            [1, 3]
    """
    width = _duration_micros(duration, arg="window duration")
    expr = _wrap(time_col)
    if slide is None:
        return WindowStart(expr, width)
    return WindowBuckets(expr, width, _duration_micros(slide, arg="window slide"))


# date_part unit (lowercased) → `.dt` accessor method name. Covers the DuckDB/SQL
# unit vocabulary; unknown units raise at plan-build time.
_PART_TO_DT = {
    "year": "year",
    "month": "month",
    "day": "day",
    "hour": "hour",
    "minute": "minute",
    "second": "second",
    "quarter": "quarter",
    "week": "week",
    "dow": "dayofweek",
    "dayofweek": "dayofweek",
    "doy": "dayofyear",
    "dayofyear": "dayofyear",
    "isodow": "isodow",
    "isoyear": "iso_year",
    "epoch": "epoch",
    "decade": "decade",
    "century": "century",
    "millennium": "millennium",
}


def current_timestamp() -> Lit:
    """Return the current timestamp as a literal, bound once at plan-build time.

    SQL ``CURRENT_TIMESTAMP``: the wall-clock is read once when the expression is
    constructed, so every row sees the same value and the result is identical
    single-node and distributed. It is never a per-row clock read.

    Returns:
        A timestamp literal expression.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1, 2]})
            >>> out = ds.with_columns(t=bt.current_timestamp()).to_pydict()
            >>> out["t"][0] == out["t"][1]  # same value for every row
            True
    """
    return Lit(_dt.datetime.now())


def current_date() -> Lit:
    """Return today's date as a literal, bound once at plan-build time.

    SQL ``CURRENT_DATE``: the date is captured once when the expression is built and
    is the same for every row and on every node.

    Returns:
        A date literal expression.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1, 2]})
            >>> out = ds.with_columns(d=bt.current_date()).to_pydict()
            >>> out["d"][0] == out["d"][1]  # same date for every row
            True
    """
    return Lit(_dt.date.today())


def date_part(part: str, expr: IntoExpr) -> Expr:
    """Extract a calendar field from a date/time column (SQL ``date_part``).

    ``date_part("year", col("d"))`` is equivalent to ``col("d").dt.year()``. Accepts
    the SQL unit vocabulary (``year``/``month``/``dow``/``doy``/``isodow``/``epoch``/
    …); an unknown unit raises ``PlanError``.

    Args:
        part: The calendar field name (case-insensitive), e.g. ``"year"`` or ``"dow"``.
        expr: The date/time column to read.

    Returns:
        An integer expression holding the requested calendar field.

    Raises:
        PlanError: If ``part`` is not a recognized unit.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> import datetime as dt
            >>> ds = bt.from_pydict({"d": [dt.date(2024, 3, 15)]})
            >>> y = bt.date_part("year", bt.col("d"))
            >>> m = bt.date_part("month", bt.col("d"))
            >>> ds.select(y=y, m=m).to_pydict()
            {'y': [2024], 'm': [3]}
    """
    method = _PART_TO_DT.get(part.lower())
    if method is None:
        raise PlanError(f"unknown date_part unit {part!r}; valid: {sorted(_PART_TO_DT)}")
    return getattr(_wrap(expr).dt, method)()


def date_add(expr: IntoExpr, days: int) -> Expr:
    """Add a whole number of days to a date/time column (Spark ``date_add``).

    ``days`` is a plain integer literal; for calendar units like months or years use
    ``.dt.offset_by``. Negative values subtract.

    Args:
        expr: The date/time column to shift.
        days: Number of days to add (may be negative).

    Returns:
        The date/time column shifted forward by ``days`` days.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> import datetime as dt
            >>> ds = bt.from_pydict({"d": [dt.date(2024, 1, 31)]})
            >>> ds.select(bt.date_add(bt.col("d"), 5).alias("r")).to_pydict()
            {'r': [datetime.date(2024, 2, 5)]}
    """
    days = require_int(days, func="date_add", arg="days")
    return _wrap(expr).dt.offset_by(f"{days}d")


def date_sub(expr: IntoExpr, days: int) -> Expr:
    """Subtract a whole number of days from a date/time column (Spark ``date_sub``).

    The mirror of :func:`date_add`; ``days`` is a plain integer literal.

    Args:
        expr: The date/time column to shift.
        days: Number of days to subtract (may be negative).

    Returns:
        The date/time column shifted back by ``days`` days.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> import datetime as dt
            >>> ds = bt.from_pydict({"d": [dt.date(2024, 3, 15)]})
            >>> ds.select(bt.date_sub(bt.col("d"), 5).alias("r")).to_pydict()
            {'r': [datetime.date(2024, 3, 10)]}
    """
    days = require_int(days, func="date_sub", arg="days")
    return _wrap(expr).dt.offset_by(f"{-days}d")


# Epoch unit → the engine `MakeTemporal` function that reads a count in that unit.
_EPOCH_UNITS = {
    "s": "from_unix_seconds",
    "ms": "from_unix_millis",
    "us": "from_unix_micros",
    "ns": "from_unix_nanos",
}


def make_date(year: IntoExpr, month: IntoExpr, day: IntoExpr) -> Expr:
    """Build a Date column from year, month, and day columns (Spark ``make_date``).

    The inverse of ``.dt.year()`` / ``.dt.month()`` / ``.dt.day()``. Each argument may
    be a column or an integer literal, and is read as an integer.

    A date that does not exist is null rather than an error, so one bad row in a scan
    of dirty upstream integers cannot abort the query. February 30, month 13, and day 0
    are all null; so is any row where any argument is null.

    Args:
        year: The year component.
        month: The month component, 1-12.
        day: The day-of-month component, 1-31.

    Returns:
        A Date32 expression.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [2024, 2023], "m": [2, 2], "d": [29, 29]})
            >>> ds.select(
            ...     r=bt.make_date(bt.col("y"), bt.col("m"), bt.col("d"))
            ... ).to_pydict()
            {'r': [datetime.date(2024, 2, 29), None]}
    """
    return MakeTemporal("make_date", [_wrap(year), _wrap(month), _wrap(day)])


def make_timestamp(
    year: IntoExpr,
    month: IntoExpr,
    day: IntoExpr,
    hour: IntoExpr = 0,
    minute: IntoExpr = 0,
    second: IntoExpr = 0,
) -> Expr:
    """Build a Timestamp column from calendar and clock components (Spark ``make_timestamp``).

    The clock components default to zero, so it doubles as midnight-of-a-date. An
    out-of-range component (hour 24, minute or second 60) yields null, as does a date
    that does not exist or a null in any argument. Second 60 is rejected rather than
    folded into the next minute: a leap second has no Arrow timestamp to land on.

    Args:
        year: The year component.
        month: The month component, 1-12.
        day: The day-of-month component, 1-31.
        hour: The hour component, 0-23.
        minute: The minute component, 0-59.
        second: The second component, 0-59.

    Returns:
        A Timestamp expression at microsecond resolution.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"y": [2024], "mo": [3], "d": [15]})
            >>> ds.select(
            ...     r=bt.make_timestamp(bt.col("y"), bt.col("mo"), bt.col("d"), 13, 45, 30)
            ... ).to_pydict()
            {'r': [datetime.datetime(2024, 3, 15, 13, 45, 30)]}
    """
    parts = [year, month, day, hour, minute, second]
    return MakeTemporal("make_timestamp", [_wrap(p) for p in parts])


def from_epoch(expr: IntoExpr, unit: str = "s") -> Expr:
    """Read an integer column of epoch counts as a Timestamp, at a stated `unit`.

    The unit has to be stated because the data cannot carry it: an Int64 column of
    epoch values looks identical whether it counts seconds or nanoseconds. A plain
    ``cast("timestamp")`` has to assume one — Arrow assumes microseconds — which turns
    a column of epoch *seconds* into January 1970 with no error at all. Naming the unit
    is the only way to be right.

    Nanoseconds are truncated toward negative infinity, so a pre-1970 sub-microsecond
    instant lands in the microsecond that contains it rather than one microsecond late.
    A value too large to scale into microseconds is null rather than a wrapped instant.

    Args:
        expr: An integer column of epoch counts.
        unit: The unit of those counts: ``"s"``, ``"ms"``, ``"us"``, or ``"ns"``.

    Returns:
        A Timestamp expression at microsecond resolution.

    Raises:
        PlanError: If `unit` is not one of the four recognized units.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"t": [1700000000]})
            >>> ds.select(r=bt.from_epoch(bt.col("t"))).to_pydict()
            {'r': [datetime.datetime(2023, 11, 14, 22, 13, 20)]}

            >>> ms = bt.from_pydict({"t": [1700000000123]})
            >>> ms.select(r=bt.from_epoch(bt.col("t"), "ms")).to_pydict()
            {'r': [datetime.datetime(2023, 11, 14, 22, 13, 20, 123000)]}
    """
    if unit not in _EPOCH_UNITS:
        raise PlanError(f"from_epoch(): unit must be one of {sorted(_EPOCH_UNITS)}, got {unit!r}")
    return MakeTemporal(_EPOCH_UNITS[unit], [_wrap(expr)])


def from_unix_date(expr: IntoExpr) -> Expr:
    """Read an integer column of days since 1970-01-01 as a Date (Spark ``date_from_unix_date``).

    The counterpart of :func:`from_epoch` for a column that counts whole days rather
    than sub-day units, which is how several warehouse exports encode a date.

    Args:
        expr: An integer column of days since the Unix epoch.

    Returns:
        A Date32 expression.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"d": [0, 19782]})
            >>> ds.select(r=bt.from_unix_date(bt.col("d"))).to_pydict()
            {'r': [datetime.date(1970, 1, 1), datetime.date(2024, 2, 29)]}
    """
    return MakeTemporal("from_unix_date", [_wrap(expr)])
