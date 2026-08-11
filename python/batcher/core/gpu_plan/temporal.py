"""The calendar half of the date vocabulary: `date_trunc`, `offset_by`, the year-derived
fields, the tumbling-window key, and the epoch constructors.

`scalar_fns` covers the date functions that are a `.dt` attribute or a fixed-duration floor.
Everything here needs *calendar* arithmetic instead, where the unit's length varies by month
and by year, and neither backend offers a construction for it that the other also has.

Two constructions carry all of it, and which one applies is the module's whole structure:

* a **distance**, for a target that is a whole number of days from the input's own midnight.
  The start of a month is the current day minus `day - 1` of them, the start of a year minus
  `day_of_year - 1`, the start of an ISO week minus the weekday index. These are computed from
  the `.dt` attributes the engine's own extraction path already agrees on, so they inherit that
  agreement rather than making a second claim about the calendar.
* a **construction**, `days_from_civil`, for a target no distance reaches — the January 1 of a
  floored decade, or a clamped day in a shifted month. It maps `(y, m, d)` to a day count in
  seven integer operations.

Doing all of it as integer microseconds rather than through a library's period type is what
keeps the two backends on one path. pandas would express most of this as
`to_period(...).to_timestamp()`; cuDF has no `to_period`. Translating through it would pass
every test here — the tests run on pandas — and raise on the device, where the failure reads as
an ordinary fallback and the whole accelerated path quietly disappears. A path only the
verification backend can run is a path nothing verifies.

What is declined is declined because the engine *validates* and this module maps:
`make_date` reports null for a February 30, where a day count computed for it would be a real
date in March. A mapping cannot answer that, so the plan goes to the CPU engine instead.
"""

from __future__ import annotations

from batcher.core.gpu_plan.backend import Unsupported
from batcher.plan.ir_tags import MICROS_PER_DAY

__all__ = [
    "DATE_FNS",
    "TRUNC_UNITS",
    "epoch_micros",
    "eval_calendar_date",
    "eval_calendar_trunc",
    "eval_date_offset",
]

#: The widest value `Date32` holds, so a day count outside it becomes null rather than a
#: wrapped date — the engine's `i32::try_from` in column form.
_I32_MAX = 2**31 - 1


#: `date_trunc` units this module truncates to, beyond the fixed-duration ones `scalar_fns`
#: floors directly. `week` is the ISO week (starting Monday), matching the engine.
TRUNC_UNITS = frozenset({"year", "quarter", "month", "week", "millisecond", "milliseconds",
                         "microsecond", "microseconds", "decade", "century", "millennium",
                         "millenium"})  # fmt: skip

#: Units that truncate to the January 1 of a *floored* year rather than to a day-count away
#: from the input. The engine floors the year itself (2024 -> 2000 for a century), which is a
#: different rule from the `century` **field**, where the first century ran from year 1.
#: `millenium` is the engine's accepted misspelling, carried so the two vocabularies match.
_YEAR_SPANS = {"decade": 10, "century": 100, "millennium": 1000, "millenium": 1000}

#: Date functions this module evaluates — the year-derived counters, the ISO weekday and year,
#: and `last_day`. The rest are `.dt` attributes and stay in `scalar_fns`.
DATE_FNS = frozenset({"isodow", "century", "decade", "millennium", "iso_year", "last_day"})


def _int64():
    import pyarrow as pa

    return pa.int64()


def _timestamp_us():
    import pyarrow as pa

    return pa.timestamp("us")


def _date32():
    import pyarrow as pa

    return pa.date32()


def _int32():
    import pyarrow as pa

    return pa.int32()


def epoch_micros(x, be):
    """`x` as int64 microseconds since the epoch, whatever temporal type it arrived as.

    The cast to `timestamp[us]` comes first and is not optional. A DATE column's own integer
    representation is a count of *days*, so reading its bits directly would scale every
    arithmetic below by 86.4 billion — a value so far out that it reads as a corrupt timestamp
    rather than as an off-by-a-unit, but it is still silent. The engine casts to
    `Timestamp(Microsecond)` on the way into `date_trunc` for the same reason.
    """
    return x.astype(be.dtype(_timestamp_us())).astype(be.dtype(_int64()))


def _floor_day(us):
    """The microsecond of the input's own midnight.

    Floor division rather than truncation, because the two differ for a timestamp before 1970:
    truncating toward zero would round such an instant *up* to the following midnight. Both
    backends' `//` on integers floors toward negative infinity, which is the engine's rule.
    """
    return (us // MICROS_PER_DAY) * MICROS_PER_DAY


def _as_int(series, be, fill):
    """`series` as a non-null int64 column, so an arithmetic chain over it cannot raise.

    Filling is safe here and only here: every value this feeds is subtracted from a microsecond
    column that is *already* null wherever the input was, so the result's null mask is decided
    before these values are consulted. Casting an Arrow null to an integer raises on one of the
    backends, so the fill is what keeps a null row from failing the whole column.
    """
    return series.fillna(fill).astype(be.dtype(_int64()))


def _days_back(x, unit: str, be):
    """How many whole days to step back from the input's midnight to reach the start of `unit`."""
    if unit == "month":
        return _as_int(x.dt.day, be, 1) - 1
    if unit == "year":
        return _as_int(x.dt.day_of_year, be, 1) - 1
    if unit == "week":
        # The ISO week starts on Monday and both backends number Monday as 0, so the weekday
        # index *is* the number of days back. This is the one place the two libraries' Monday-0
        # convention is the convenient one; `day_of_week` in `scalar_fns` has to undo it.
        return _as_int(x.dt.dayofweek, be, 0)
    # The first day of the quarter, as a day-of-year: 1, 91, 182, 274, each one later in a leap
    # year once February has been passed. Built by accumulating the quarter lengths rather than
    # by a lookup, because neither backend has a per-element table lookup that the other also
    # has, and a `map` over 100M rows would be a host-side Python call per row besides.
    quarter = _as_int(x.dt.quarter, be, 1)
    leap = _as_int(x.dt.is_leap_year, be, False)
    start = 1 + _ge(quarter, 2, be) * 90 + _ge(quarter, 3, be) * 91 + _ge(quarter, 4, be) * 92
    return _as_int(x.dt.day_of_year, be, 1) - (start + leap * _ge(quarter, 2, be))


def _ge(series, bound: int, be):
    """`series >= bound` as an int64 0/1 column."""
    return _as_int(series >= bound, be, False)


def _le(series, bound: int, be):
    """`series <= bound` as an int64 0/1 column."""
    return _as_int(series <= bound, be, False)


def days_from_civil(y, m, d, be):
    """Days since 1970-01-01 for a proleptic Gregorian `(y, m, d)`, as int64 columns.

    Howard Hinnant's `days_from_civil`, which is the standard closed form and is what makes
    the *calendar* units expressible here at all. Everything else in this module is a distance
    from the input's own midnight, which cannot reach a January 1 an arbitrary number of years
    away or a clamped day in a shifted month; this constructs the target date outright, in
    seven integer operations and no per-row call.

    Two deliberate spellings, each avoiding something one of the backends lacks:

    * the March-based month index is written as a comparison rather than `(m + 9) % 12`,
      because one backend does not implement `%` on an Arrow-typed column at all (the same
      reason `day_of_week` wraps with a `where`);
    * `//` is used where the reference uses C's truncating division, which is correct because
      the reference's `(y >= 0 ? y : y-399) / 400` is exactly floor division, and both backends
      floor toward negative infinity. That equivalence is what makes pre-1970 dates agree.

    Args:
        y: Proleptic Gregorian year.
        m: Month, 1-12.
        d: Day of month, 1-31.
        be: The dataframe backend to compute on.

    Returns:
        Days since the Unix epoch, as an int64 column.
    """
    # March-based year: January and February belong to the preceding one, which is what makes
    # the leap day the last day of the year and so removes it from every other case.
    before_march = _le(m, 2, be)
    shifted = y - before_march
    era = shifted // 400
    yoe = shifted - era * 400
    month_index = m - 3 + 12 * before_march
    doy = (153 * month_index + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def _ones(like):
    """A column of `1`s shaped like `like`, without materializing a Python list per row."""
    return like * 0 + 1


def eval_calendar_trunc(x, unit: str, be):
    """`date_trunc(unit, x)` for a calendar or sub-second unit, as `timestamp[us]`.

    Args:
        x: The timestamp column to truncate.
        unit: The unit to truncate to, one of `TRUNC_UNITS`.
        be: The dataframe backend to compute on.

    Returns:
        The truncated column, typed as the engine types it.

    Raises:
        Unsupported: For a unit this construction cannot express.
    """
    if unit not in TRUNC_UNITS:
        raise Unsupported(f"date_trunc to {unit}")
    us = epoch_micros(x, be)
    span = _YEAR_SPANS.get(unit)
    if unit.startswith("microsecond"):
        # The engine's finest unit is already the storage unit, so this is the identity — but
        # it still has to come back as a timestamp rather than as the integer it passed through.
        truncated = us
    elif unit.startswith("millisecond"):
        truncated = (us // 1_000) * 1_000
    elif span is not None:
        # A January 1 an arbitrary number of years back, which no day-offset from the input can
        # reach — so the date is constructed instead. The year is floored toward negative
        # infinity, which is the engine's `div_euclid` and keeps the pre-year-1 cases agreeing.
        year = _as_int(x.dt.year, be, 1970)
        floored = (year // span) * span
        truncated = days_from_civil(floored, _ones(floored), _ones(floored), be) * MICROS_PER_DAY
    else:
        truncated = _floor_day(us) - _days_back(x, unit, be) * MICROS_PER_DAY
    # The year-span units are computed from a *filled* year, so their nullness has to be put
    # back explicitly; the others inherit it from `us` and this is a no-op for them.
    return truncated.where(x.notna(), None).astype(be.dtype(_timestamp_us()))


def eval_date_offset(x, months: int, days: int, micros: int, be):
    """`offset_by` — shift a date or timestamp by exact days and microseconds.

    A **calendar month** shift is declined. Months clamp to the end of the target month, so
    they are a construction from (year, month, day) rather than a distance, and that is the
    one thing neither backend expresses the same way. Days and microseconds are exact
    distances, which is why they can be added to the instant directly.

    The type is preserved, which is the engine's contract and matters more than it looks: a
    DATE shifted by a week is still a DATE, and handing back a timestamp instead produces a
    column a fan-out cannot concatenate with a shard the CPU engine produced.

    Args:
        x: The date or timestamp column to shift.
        months: Calendar months to shift by; anything but zero is declined.
        days: Exact days to shift by.
        micros: Exact microseconds to shift by.
        be: The dataframe backend to compute on.

    Returns:
        The shifted column, in the input's own type.

    Raises:
        Unsupported: For a calendar-month shift, or a sub-day shift of a DATE.
    """
    is_date = be.is_date(x)
    if is_date and micros:
        # The engine errors here rather than inventing a time of day, so the fallback has to
        # reach it — swallowing this into a truncated result would answer a question the user
        # did not ask, and would do it only on the accelerated path.
        raise Unsupported("offset_by a sub-day amount on a DATE")
    us = epoch_micros(x, be)
    # Months first, then exact days, then microseconds — the engine's order, and it is not
    # commutative with clamping: shifting Jan 31 by a month and then a day gives Mar 1, while
    # a day and then a month gives Mar 1 too only by coincidence of February's length.
    base = _shifted_months(x, us, months, be) if months else us
    shifted = base + (days * MICROS_PER_DAY + micros)
    stepped = shifted.astype(be.dtype(_timestamp_us()))
    return stepped.astype(be.dtype(_date32())) if is_date else stepped


def _shifted_months(x, us, months: int, be):
    """`us` moved by `months` calendar months, keeping the time of day, clamping the day.

    A month is not a distance, so the target date is constructed: the month index moves, and
    the day is clamped to the target month's length. Clamping is the rule every calendar
    library follows and the one place this is easy to get wrong — January 31 plus one month is
    February 28 or 29, never March 3.

    The target month's length is taken as the gap between the first of it and the first of the
    following month, which reuses `days_from_civil` rather than restating a per-month table and
    a leap-year rule that would then have to agree with it.
    """
    index = _as_int(x.dt.year, be, 1970) * 12 + (_as_int(x.dt.month, be, 1) - 1) + months
    year, month = _year_month(index)
    first = days_from_civil(year, month, _ones(index), be)
    next_year, next_month = _year_month(index + 1)
    length = days_from_civil(next_year, next_month, _ones(index), be) - first
    day = _as_int(x.dt.day, be, 1)
    clamped = day.where(day <= length, length)
    # The time of day is carried across untouched: a month shift moves the date and nothing
    # else, so it is the input's own offset from its midnight.
    return days_from_civil(year, month, clamped, be) * MICROS_PER_DAY + (us - _floor_day(us))


def _year_month(index):
    """A months-since-year-0 index as `(year, month)` with `month` in 1-12."""
    year = index // 12
    return year, index - year * 12 + 1


def eval_calendar_date(x, fn: str, be):
    """One of the year-derived date functions, the ISO weekday/year, or `last_day`.

    Args:
        x: The date or timestamp column.
        fn: The function name, one of `DATE_FNS`.
        be: The dataframe backend to compute on.

    Returns:
        The extracted column, typed as the engine types it — int64, except `last_day`, which is
        a timestamp.

    Raises:
        Unsupported: For a name outside `DATE_FNS`.
    """
    if fn == "last_day":
        return _last_day(x, be)
    if fn == "iso_year":
        return _iso_year(x, be)
    if fn == "isodow":
        # ISO numbers Monday 1 through Sunday 7; both backends number Monday 0, so this is the
        # `+ 1` that `day_of_week` cannot use — that one counts from Sunday and has to wrap.
        return (x.dt.dayofweek + 1).astype(be.dtype(_int64())).where(x.notna(), None)
    if fn not in ("century", "decade", "millennium"):
        raise Unsupported(f"date fn {fn}")
    return _year_counter(x, fn, be)


def _year_counter(x, fn: str, be):
    """`century`/`decade`/`millennium`, derived from the year exactly as the engine derives them.

    The off-by-one on century and millennium is not a quirk to be tidied: the first century ran
    from year 1 to year 100, so 2000 is the 20th century and 2001 the 21st. DuckDB counts them
    this way, the engine follows DuckDB, and `decade` genuinely is the plain floor that the
    other two only look like.

    Both backends' `//` floors toward negative infinity, which is `div_euclid` and is what makes
    the pre-year-1 cases agree rather than splitting at zero.
    """
    year = x.dt.year.astype(be.dtype(_int64()))
    if fn == "decade":
        counted = year // 10
    else:
        span = 100 if fn == "century" else 1000
        counted = (year - 1) // span + 1
    return counted.where(x.notna(), None)


def isocalendar_field(x, be, field: str, label: str):
    """One field of `isocalendar()`, as an int64 column with the input's null mask restored.

    Shared by the two ISO-8601 fields the translator supports, which differ only in the
    attribute they read and the name they decline under. The part that is easy to miss:
    `isocalendar` fills a null timestamp with **0** rather than propagating it, and neither a
    week zero nor a year zero exists, so leaving it through puts a value where a hole belongs.

    Args:
        x: The timestamp column.
        be: The backend, for its dtype constructor.
        field: The `isocalendar()` attribute to read (`"week"` or `"year"`).
        label: What to call this function in an `Unsupported` message.

    Returns:
        An int64 column, null wherever the input was null.

    Raises:
        Unsupported: The backend has no `isocalendar`, or it declined this input.
    """
    iso = getattr(x.dt, "isocalendar", None)
    if iso is None:
        raise Unsupported(label)
    try:
        value = getattr(iso(), field)
    except (AttributeError, TypeError, NotImplementedError) as exc:
        raise Unsupported(f"{label}: {exc}") from exc
    return value.astype(be.dtype(_int64())).where(x.notna(), None)


def _iso_year(x, be):
    """The ISO-8601 week-numbering year, which differs from the calendar year at both ends of it.

    Late December can belong to the next ISO year and early January to the previous one, which
    is the entire reason the function exists — pairing an ISO week with a calendar year puts a
    week 1 and a week 53 in the same bucket.
    """
    return isocalendar_field(x, be, "year", "date fn iso_year")


def _last_day(x, be):
    """The last day of the input's month, as a **DATE**.

    Stepping *forward* from midnight by the days left in the month, rather than to the first of
    the next month and back one day: the forward step needs no month-end wraparound and no
    December special case, and `days_in_month` already knows about February.

    The result is a `date32`, not the `timestamp` every other construction here produces. That
    is the engine's contract, inherited from DuckDB and Spark, and it is the whole column's
    type rather than a detail of one value — a translation that handed back a timestamp would
    be a column a fan-out could not concatenate with a shard the CPU engine produced.
    """
    us = epoch_micros(x, be)
    remaining = _as_int(x.dt.days_in_month, be, 1) - _as_int(x.dt.day, be, 1)
    stepped = (_floor_day(us) + remaining * MICROS_PER_DAY).astype(be.dtype(_timestamp_us()))
    return stepped.astype(be.dtype(_date32()))
