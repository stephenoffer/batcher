"""SQL ``INTERVAL`` literals → the ``(months, days, microseconds)`` triple.

`DateOffset` already carries a shift as three independent components, because that is what
a calendar shift *is*: months are variable-length, days are variable-length under a time
zone, and microseconds are exact. So an interval literal has a natural lowering and needs
no new IR — only a parser for the text SQL admits, which is more than one number and one
unit:

* **Compound** — ``INTERVAL '1 day 3 hours'``, ``INTERVAL '2 years 3 months'``. sqlglot
  hands the whole string over with no ``unit``, and the previous lowering read it with
  ``int(...)``, so every compound literal raised a bare
  ``ValueError: invalid literal for int() with base 10: '1 day 3 hours'`` out of
  `Session.sql()` — an internal Python error, not a Batcher one, for standard SQL.
* **Fractional** — ``INTERVAL '1.5 hours'``, ``INTERVAL '2.5 weeks'``. Same `int(...)`,
  same raw `ValueError`, and this one bites the single-unit form the engine did support.
* **Abbreviated** — ``mon``, ``yr``, ``hrs``, ``mins``, ``secs``, ``ms``, ``us``. The unit
  table held only the long names, so ``INTERVAL '1 mon'`` was declined by name.
* **Time-only** — ``INTERVAL '04:05:06'``.

A fractional count spills into the next finer component the way DuckDB and PostgreSQL both
do it: a month is 30 days for that purpose and a day is 86,400 seconds, so ``1.5 months``
is one month and fifteen days (*not* 45 days, which would land on a different date), and
``0.25 days`` is six hours. Integer counts never spill, so the common case is exact.

The unit vocabulary is DuckDB's, read off DuckDB rather than assumed — including the two
that collide on their first letter: bare ``m`` is a **minute** and ``mon`` is a **month**.
"""

from __future__ import annotations

import re

__all__ = ["interval_parts"]

#: Months per calendar unit. These shift by *calendar* months, so they can never be
#: rewritten as a day count without changing the answer at a month boundary.
_MONTHS = {
    "mon": 1,
    "mons": 1,
    "month": 1,
    "months": 1,
    "quarter": 3,
    "quarters": 3,
    "y": 12,
    "yr": 12,
    "yrs": 12,
    "year": 12,
    "years": 12,
    "decade": 120,
    "decades": 120,
    "century": 1200,
    "centuries": 1200,
    "millennium": 12000,
    "millenniums": 12000,
    "millennia": 12000,
}

#: Days per whole-day unit. Exact days, not 24-hour spans: under a time zone a "day" is
#: what `DateOffset` resolves it to, which is why these stay in the day slot.
_DAYS = {"d": 1, "day": 1, "days": 1, "w": 7, "week": 7, "weeks": 7}

#: Microseconds per sub-day unit.
_MICROS = {
    "h": 3_600_000_000,
    "hr": 3_600_000_000,
    "hrs": 3_600_000_000,
    "hour": 3_600_000_000,
    "hours": 3_600_000_000,
    "m": 60_000_000,
    "min": 60_000_000,
    "mins": 60_000_000,
    "minute": 60_000_000,
    "minutes": 60_000_000,
    "s": 1_000_000,
    "sec": 1_000_000,
    "secs": 1_000_000,
    "second": 1_000_000,
    "seconds": 1_000_000,
    "ms": 1_000,
    "msec": 1_000,
    "msecs": 1_000,
    "millisecond": 1_000,
    "milliseconds": 1_000,
    "us": 1,
    "usec": 1,
    "usecs": 1,
    "microsecond": 1,
    "microseconds": 1,
}

#: Days a month is taken to be when a *fractional* month spills, and seconds in a day when
#: a fractional day does. Both match DuckDB and PostgreSQL.
_DAYS_PER_MONTH = 30
_MICROS_PER_DAY = 86_400_000_000

#: One ``<count> <unit>`` term. The count may be signed and fractional; the unit is
#: optional only for the bare-number form a caller supplies a default unit for.
_TERM = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*([A-Za-z]+)?")

#: A bare ``HH:MM:SS`` / ``HH:MM:SS.ffffff`` clock term, which carries no unit words.
_CLOCK = re.compile(r"^([+-]?)(\d+):(\d{1,2}):(\d{1,2}(?:\.\d+)?)$")


def interval_parts(text: str, default_unit: str = "") -> tuple[int, int, int]:
    """The ``(months, days, microseconds)`` an interval literal denotes.

    Args:
        text: The literal's text — one term (``"3"``, ``"1.5 hours"``), several
            (``"1 day 3 hours"``), or a clock (``"04:05:06"``).
        default_unit: The unit to apply to a term that names none, which is how the
            ``INTERVAL 3 DAY`` form arrives (sqlglot parks the unit beside the number).

    Returns:
        The shift as months, whole days, and microseconds — the three components
        `DateOffset` takes.

    Raises:
        ValueError: If `text` is not an interval, or names a unit that is not one. The
            caller turns this into the front end's typed error with the SQL in hand.
    """
    stripped = text.strip()
    clock = _CLOCK.match(stripped)
    if clock is not None:
        sign, hours, minutes, seconds = clock.groups()
        micros = (
            int(hours) * 3_600_000_000 + int(minutes) * 60_000_000 + round(float(seconds) * 1e6)
        )
        return (0, 0, -micros if sign == "-" else micros)

    months = days = micros = 0
    matched = 0
    for count_text, unit_text in _TERM.findall(stripped):
        matched += 1
        unit = (unit_text or default_unit).lower().strip()
        count = float(count_text)
        m, d, u = _term_parts(count, unit, text)
        months += m
        days += d
        micros += u
    if not matched:
        raise ValueError(f"{text!r} is not an interval")
    # A term is only rejected above; a leftover word (`INTERVAL 'day'`) matches no term at
    # all and must not read as a zero shift.
    if not _TERM.sub("", stripped).strip(" \t,").replace("and", "") == "":
        raise ValueError(f"{text!r} is not an interval")
    return (months, days, micros)


def _term_parts(count: float, unit: str, text: str) -> tuple[int, int, int]:
    """One ``<count> <unit>`` term as ``(months, days, microseconds)``.

    A fractional count spills into the next finer component rather than being rounded,
    which is what makes ``INTERVAL '1.5 months'`` land on "one month and fifteen days"
    instead of a day count that drifts across a month boundary.
    """
    if unit in _MONTHS:
        total = count * _MONTHS[unit]
        whole = int(total)
        spill_days = (total - whole) * _DAYS_PER_MONTH
        day_whole = int(spill_days)
        return (whole, day_whole, round((spill_days - day_whole) * _MICROS_PER_DAY))
    if unit in _DAYS:
        total = count * _DAYS[unit]
        whole = int(total)
        return (0, whole, round((total - whole) * _MICROS_PER_DAY))
    if unit in _MICROS:
        return (0, 0, round(count * _MICROS[unit]))
    raise ValueError(f"INTERVAL unit {unit or '(none)'!r} in {text!r} is not supported")
