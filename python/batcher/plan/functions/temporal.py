"""Temporal free functions.

`now`/`current_timestamp`/`current_date`/`today` bind the wall-clock **once, at
plan-build time** to a literal (DuckDB statement-timestamp semantics) — so the value
is fixed for the query and stays identical single-node and distributed, never a
per-row clock read. `date_part`/`date_add`/`date_sub` dispatch onto the existing
`.dt` accessor, so they add no engine surface.
"""

from __future__ import annotations

import datetime as _dt

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.core import Expr, IntoExpr, Lit, _wrap
from batcher.plan.expr_ir.func_nodes import WindowBuckets, WindowStart
from batcher.plan.expr_ir.namespaces.temporal import parse_offset

_DAY_MICROS = 86_400_000_000


def _duration_micros(duration: str, *, arg: str) -> int:
    """Parse a fixed-length duration string to microseconds (no calendar units).

    Event-time windows must have a fixed width, so a calendar duration (months) is
    rejected — ``"1mo"`` has no constant microsecond length. ``"1d"``, ``"1h30m"``,
    ``"500ms"`` etc. are fine.
    """
    months, days, micros = parse_offset(duration)
    if months:
        raise PlanError(
            f"{arg} {duration!r} uses a calendar unit (month/year) with no fixed length; "
            "use fixed units (days/hours/minutes/seconds)"
        )
    total = days * _DAY_MICROS + micros
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

    Raises:
        PlanError: If a duration uses a calendar unit or is not positive.
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


def now() -> Lit:
    """Return the current timestamp as a literal, bound once at plan-build time.

    DuckDB ``now()`` / statement-timestamp semantics: the wall-clock is read once
    when the expression is constructed, so every row sees the same value and the
    result is identical single-node and distributed. It is never a per-row clock read.

    Returns:
        A timestamp literal expression.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1, 2]})
            >>> out = ds.with_columns(t=bt.now()).to_pydict()
            >>> out["t"][0] == out["t"][1]  # same value for every row
            True
    """
    return Lit(_dt.datetime.now())


def current_timestamp() -> Lit:
    """Return the current timestamp as a literal, bound once at plan-build time.

    SQL ``CURRENT_TIMESTAMP``; the spelling of :func:`now` with identical semantics
    (read once at plan build, constant across rows and across nodes).

    Returns:
        A timestamp literal expression.
    """
    return Lit(_dt.datetime.now())


def current_date() -> Lit:
    """Return today's date as a literal, bound once at plan-build time.

    SQL ``CURRENT_DATE``: the date is captured once when the expression is built and
    is the same for every row and on every node.

    Returns:
        A date literal expression.
    """
    return Lit(_dt.date.today())


def today() -> Lit:
    """Return today's date as a literal, bound once at plan-build time.

    DuckDB ``today()``; the spelling of :func:`current_date` with identical semantics.

    Returns:
        A date literal expression.
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

    Raises:
        PlanError: If ``part`` is not a recognized unit.
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

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> import datetime as dt
            >>> ds = bt.from_pydict({"d": [dt.date(2024, 1, 31)]})
            >>> ds.select(bt.date_add(bt.col("d"), 5).alias("r")).to_pydict()
            {'r': [datetime.date(2024, 2, 5)]}
    """
    return _wrap(expr).dt.offset_by(f"{int(days)}d")


def date_sub(expr: IntoExpr, days: int) -> Expr:
    """Subtract a whole number of days from a date/time column (Spark ``date_sub``).

    The mirror of :func:`date_add`; ``days`` is a plain integer literal.

    Args:
        expr: The date/time column to shift.
        days: Number of days to subtract (may be negative).
    """
    return _wrap(expr).dt.offset_by(f"{-int(days)}d")
