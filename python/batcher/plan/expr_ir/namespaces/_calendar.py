"""Argument contracts shared by the zone and business-calendar parts of the `.dt` namespace.

Split out of `temporal.py` for size (`just lint-structure`). Each helper here is the single
place one vocabulary is decided, because two callers meet it: the DST policies are taken by
`convert_timezone` and `replace_timezone`, and the business calendar (holidays plus a
weekmask) by `.dt.add_business_days`, `.dt.is_business_day` and `bt.business_day_count`. A
calendar spelled two ways would let two of those disagree about whether a date is a
business day, which is the one thing the shared `BusinessDay` engine node exists to rule out.

It also holds the truncation-unit vocabulary shared by `.dt.truncate`/`floor`/`ceil`/`round`,
for the same reason: two spellings of one vocabulary met in this namespace and neither
understood the other, so what a unit *means* is decided in exactly one place.
"""

from __future__ import annotations

import datetime as _dt
import re
from collections.abc import Iterable, Sequence
from typing import Any

from batcher._internal.errors import PlanError, require_choice
from batcher.plan.expr_ir.core import Expr, _wrap
from batcher.plan.expr_ir.func_nodes import BusinessDay

__all__ = [
    "AMBIGUOUS",
    "NONEXISTENT",
    "TRUNC_UNITS",
    "business_day",
    "duration_total",
    "strptime_formats",
    "trunc_unit",
    "tz_policies",
]

#: How a wall clock a DST overlap makes ambiguous is resolved (the engine's `Ambiguous`).
AMBIGUOUS: tuple[str, ...] = ("raise", "earliest", "latest", "null")
#: How a wall clock a DST gap skips is resolved (the engine's `Nonexistent`).
NONEXISTENT: tuple[str, ...] = ("raise", "shift_forward", "null")

_EPOCH = _dt.date(1970, 1, 1)
_MONDAY_TO_FRIDAY = (True, True, True, True, True, False, False)

#: Microseconds per `dt.total` unit. ``m`` is minutes and ``mo`` is not here at all: a
#: duration is elapsed time, and a month has no fixed length.
_TOTAL_MICROS: dict[str, int] = {
    "d": 86_400_000_000,
    "h": 3_600_000_000,
    "m": 60_000_000,
    "s": 1_000_000,
    "ms": 1_000,
    "us": 1,
}


def tz_policies(func: str, ambiguous: str, nonexistent: str) -> tuple[str | None, str | None]:
    """Validate the two DST policies, mapping the engine default (``"null"``) to ``None``.

    ``None`` is left out of the IR, so a call that keeps the default serializes exactly as it
    did before the policies existed.

    Args:
        func: The calling method, for the error message.
        ambiguous: One of `AMBIGUOUS`.
        nonexistent: One of `NONEXISTENT`.

    Returns:
        The two policies, each ``None`` when it is the engine default.

    Raises:
        PlanError: If either policy is not one of its choices.
    """
    amb = require_choice(ambiguous, func=func, arg="ambiguous", choices=AMBIGUOUS)
    gap = require_choice(nonexistent, func=func, arg="nonexistent", choices=NONEXISTENT)
    return (None if amb == "null" else amb), (None if gap == "null" else gap)


def _holiday_days(func: str, holidays: Iterable[Any]) -> list[int]:
    """Holidays as sorted, unique days since the epoch."""
    if isinstance(holidays, (str, bytes)):
        raise PlanError(f"{func}(): holidays must be a collection of dates, not one string")
    days: set[int] = set()
    for value in holidays:
        day = value
        if isinstance(value, str):
            try:
                day = _dt.date.fromisoformat(value)
            except ValueError:
                raise PlanError(
                    f"{func}(): holiday {value!r} is not an ISO date like '2024-12-25'"
                ) from None
        if isinstance(day, _dt.datetime):
            day = day.date()
        if not isinstance(day, _dt.date):
            raise PlanError(
                f"{func}(): holidays must be dates (datetime.date or 'YYYY-MM-DD'), "
                f"got {value!r}. The list is a plan-time constant; it cannot be a column."
            )
        days.add((day - _EPOCH).days)
    return sorted(days)


def _weekmask(func: str, weekmask: str | Sequence[bool]) -> list[bool] | None:
    """Seven Monday-first flags, or ``None`` for the default Monday-to-Friday week."""
    if isinstance(weekmask, str):
        flags = [c == "1" for c in weekmask] if set(weekmask) <= {"0", "1"} else None
    else:
        flags = [bool(f) for f in weekmask] if all(f in (0, 1) for f in weekmask) else None
    if flags is None or len(flags) != 7:
        raise PlanError(
            f"{func}(): weekmask must be seven Monday-first flags, such as '1111100' (Monday "
            f"to Friday) or [True] * 5 + [False] * 2, got {weekmask!r}"
        )
    if not any(flags):
        raise PlanError(f"{func}(): weekmask admits no weekday, so no day is a business day")
    return None if tuple(flags) == _MONDAY_TO_FRIDAY else flags


def business_day(
    func: str,
    op: str,
    start: Expr,
    other: Expr | int | None,
    holidays: Iterable[Any],
    weekmask: str | Sequence[bool],
    roll: str = "raise",
) -> BusinessDay:
    """Build a `BusinessDay` node with a validated calendar.

    Args:
        func: The calling function, for error messages.
        op: ``"add"``, ``"count"`` or ``"is"``.
        start: The date or timestamp operand.
        other: The day count (``add``) or end date (``count``); ``None`` for ``is``.
        holidays: Dates that are never business days.
        weekmask: Seven Monday-first flags, as a ``"1111100"`` string or a sequence.
        roll: For ``add``, what to do with a start that is not a business day.

    Returns:
        The engine node.

    Raises:
        PlanError: If the calendar or `roll` is malformed.
    """
    roll = require_choice(roll, func=func, arg="roll", choices=("raise", "forward", "backward"))
    return BusinessDay(
        op,
        start,
        None if other is None else _wrap(other),
        holidays=_holiday_days(func, holidays),
        weekmask=_weekmask(func, weekmask),
        roll=None if roll == "raise" else roll,
    )


def duration_total(e: Expr, unit: str) -> Expr:
    """`e`, a duration, as a whole count of `unit`, truncated toward zero.

    Composed from the microsecond count: ``cast("duration")`` normalizes any duration
    resolution to microseconds, and the engine's ``%`` is the truncated remainder, so
    ``(m - m % per) // per`` is an exact division that truncates rather than floors -- a
    negative 90 minutes is -1 hour, as Polars' ``total_hours`` reads it.

    Args:
        e: A duration expression.
        unit: One of ``d``/``h``/``m``/``s``/``ms``/``us``.

    Returns:
        An Int64 expression.

    Raises:
        PlanError: If `unit` is not one of those.
    """
    per = _TOTAL_MICROS.get(unit)
    if per is None:
        raise PlanError(
            f"dt.total(): unit must be one of {sorted(_TOTAL_MICROS)} ('m' is minutes; a "
            f"month has no fixed length), got {unit!r}"
        )
    micros = e.cast("duration").cast("int64")
    if per == 1:
        return micros
    return ((micros - micros % per) // per).cast("int64")


def strptime_formats(format: str | Sequence[str]) -> str | list[str]:
    """One parse format, or a non-empty list of them tried in order, validated.

    A single format (or a one-element list) is returned as the plain string the IR always
    carried, so a single-format parse serializes byte-for-byte as before lists existed.

    Args:
        format: A chrono/strftime pattern, or a sequence of them.

    Returns:
        The format string, or the list of format strings.

    Raises:
        PlanError: If `format` is empty or holds a non-string.
    """
    if isinstance(format, str):
        return format
    formats = list(format) if isinstance(format, Sequence) else None
    if not formats or not all(isinstance(f, str) for f in formats):
        raise PlanError(
            "to_datetime(): format must be a strftime pattern or a non-empty list of them, "
            f"got {format!r}"
        )
    return formats[0] if len(formats) == 1 else formats


#: The truncation units `bc_expr`'s `date_trunc` accepts, in coarse-to-fine order. This
#: list mirrors the Rust match in `bc-expr/src/eval/temporal/date.rs` exactly; the two must
#: change together, because this side is what decides whether the plan is built at all.
TRUNC_UNITS: tuple[str, ...] = (
    "millennium",
    "century",
    "decade",
    "year",
    "quarter",
    "month",
    "week",
    "day",
    "hour",
    "minute",
    "second",
    "millisecond",
    "microsecond",
)

#: Spellings that mean one of `TRUNC_UNITS`. Two vocabularies met here and neither
#: understood the other: `truncate` took only the long names, while `offset_by`/`ceil`/
#: `round` take the duration spellings (`1mo`, `1d`), so `truncate("1mo")` failed on a
#: string the neighbouring method accepts and `offset_by("1month")` failed on the string
#: `truncate` requires. The duration spellings are also what Polars' `dt.truncate` takes,
#: so accepting both is the compatible reading rather than a new dialect.
#:
#: `mo` is months and `m` is minutes, matching `_OFFSET_UNITS` above -- the one ambiguity
#: in the vocabulary, resolved the same way in both places.
_ALIASES: dict[str, str] = {
    "millennia": "millennium",
    "millenium": "millennium",  # the misspelling DuckDB itself accepts
    "centuries": "century",
    "decades": "decade",
    "y": "year",
    "yr": "year",
    "yrs": "year",
    "years": "year",
    "q": "quarter",
    "qtr": "quarter",
    "quarters": "quarter",
    "mo": "month",
    "mon": "month",
    "mons": "month",
    "months": "month",
    "w": "week",
    "wk": "week",
    "wks": "week",
    "weeks": "week",
    "d": "day",
    "dy": "day",
    "days": "day",
    "h": "hour",
    "hr": "hour",
    "hrs": "hour",
    "hours": "hour",
    "m": "minute",
    "min": "minute",
    "mins": "minute",
    "minutes": "minute",
    "s": "second",
    "sec": "second",
    "secs": "second",
    "seconds": "second",
    "ms": "millisecond",
    "milli": "millisecond",
    "millis": "millisecond",
    "milliseconds": "millisecond",
    "us": "microsecond",
    "micro": "microsecond",
    "micros": "microsecond",
    "microseconds": "microsecond",
}

_COUNT_RE = re.compile(r"^(\d+)\s*([a-z]+)$")


def trunc_unit(unit: str, func: str) -> str:
    """Normalize a truncation unit to the one spelling `date_trunc` knows, or raise.

    Validation happens *here*, at the API edge, rather than in the engine. It used not to
    happen at all on this path: the string was handed straight to `bc-expr`, so a typo
    built a perfectly good plan and only failed once the query ran -- after the scan --
    with a bare `RuntimeError` from Rust rather than a typed `PlanError`. Every
    neighbouring method (`ceil`, `round`, `offset_by`) already rejected a bad unit at
    build time; this one was the outlier.

    Args:
        unit: The caller's spelling, e.g. ``"month"``, ``"1mo"``, or ``"mo"``.
        func: The method name to name in the error, e.g. ``"truncate"``.

    Returns:
        The canonical unit name from `TRUNC_UNITS`.

    Raises:
        PlanError: If `unit` is not a string, names no known unit, or carries a
            multiplier other than 1.
    """
    if not isinstance(unit, str):
        raise PlanError(f".dt.{func}() unit must be a string, got {type(unit).__name__}")
    key = unit.strip().lower()
    count_match = _COUNT_RE.match(key)
    if count_match is not None:
        count, key = int(count_match.group(1)), count_match.group(2)
        if count != 1:
            # "5d" is a *bucket width*, which flooring to a calendar boundary cannot
            # express. Silently dropping the multiplier would floor to 1 day and return
            # plausible, wrong timestamps -- so it is refused rather than approximated.
            raise PlanError(
                f".dt.{func}({unit!r}) is not supported: a truncation unit floors to a "
                f"calendar boundary, so it takes no multiplier other than 1. Use "
                f"{'1' + count_match.group(2)!r} for the boundary itself."
            )
    canonical = _ALIASES.get(key, key)
    if canonical not in TRUNC_UNITS:
        raise PlanError(
            f".dt.{func}({unit!r}) is not a known unit; use one of "
            f"{list(TRUNC_UNITS)} (the duration spellings '1mo'/'mo', '1d'/'d', ... are "
            f"accepted too, where 'mo' is months and 'm' is minutes)"
        )
    return canonical
