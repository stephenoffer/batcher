"""Time constraints: how old the newest row is, and whether any row is dated ahead.

Freshness is the check a batch pipeline needs most and expresses least well: nothing about
the *values* in a table is wrong when an upstream feed stops, so every value constraint
passes while the table quietly stops being true. It is a relation-level constraint — the
age of the newest row — not a row-level one, because no individual row is at fault.

The clock is read once, in Python, at the moment the constraint is built, and enters the
plan as a literal. That is the same statement-timestamp semantics `bt.current_timestamp`
uses, and it is what keeps the answer identical single-node and distributed.
"""

from __future__ import annotations

import datetime as dt

from batcher._internal.errors import PlanError
from batcher.api.dataset.dq.constraints import AggregateConstraint, RowConstraint
from batcher.plan.expr_ir import Col, lit
from batcher.plan.expr_ir.namespaces.temporal import parse_offset

__all__ = ["fresh_within", "not_in_future", "seconds_of"]

_DAY_SECONDS = 86_400


def seconds_of(age: str | dt.timedelta | int | float, *, arg: str) -> float:
    """A max-age argument as a number of seconds, whichever way it was spelled.

    Accepts a `datetime.timedelta`, a plain number of seconds, or the compact duration
    string the rest of Batcher reads (`"1d"`, `"6h"`, `"90m"`). Calendar units are rejected
    for the reason event-time windows reject them: a month has no fixed length, so an age
    measured in months is not a number.

    Args:
        age: The duration, as a timedelta, seconds, or a duration string.
        arg: The calling argument's name, used in the error message.

    Returns:
        The duration in seconds.
    """
    if isinstance(age, dt.timedelta):
        seconds = age.total_seconds()
    elif isinstance(age, bool):  # bool is an int; a True max-age is a mistake, not 1 second
        raise PlanError(f"{arg} must be a duration, not a boolean")
    elif isinstance(age, (int, float)):
        seconds = float(age)
    else:
        try:
            months, days, micros = parse_offset(age)
        except ValueError as exc:
            raise PlanError(
                f"cannot parse {arg} {age!r}; use a timedelta, a number of seconds, or a "
                "fixed-length duration string such as '1d', '6h', or '90m'."
            ) from exc
        if months:
            raise PlanError(
                f"{arg} {age!r} uses a calendar unit (month/year) with no fixed length; "
                "use days, hours, minutes, or seconds."
            )
        seconds = days * _DAY_SECONDS + micros / 1_000_000
    if seconds <= 0:
        raise PlanError(f"{arg} must be a positive duration, got {age!r}")
    return seconds


def _now_millis() -> int:
    """The wall clock in the frame a timestamp column is stored in, read once.

    Two conventions have to line up here, and reading the clock in UTC breaks both. Batcher's
    own `bt.current_timestamp` is ``datetime.now()`` — the **naive local** wall clock — and a
    naive Arrow timestamp carries no offset, so ``dt.epoch_ms()`` returns exactly the naive
    value counted from 1970. Reading the clock as UTC instead made every row appear one
    UTC-offset older than it is: seven hours, in the timezone this was found in, so a table
    written seconds ago failed a one-hour freshness bound.
    """
    return int(dt.datetime.now().replace(tzinfo=dt.UTC).timestamp() * 1000)


def fresh_within(
    column: str, max_age: str | dt.timedelta | int | float, *, now_ms: int | None = None
) -> AggregateConstraint:
    """The newest value of `column` must be no older than `max_age`.

    Args:
        column: The timestamp or date column carrying the row's event time.
        max_age: How stale the newest row may be.
        now_ms: The reference time in epoch milliseconds; defaults to the wall clock.

    Returns:
        The relation-level constraint, measuring the newest row's age in seconds.
    """
    seconds = seconds_of(max_age, arg="max_age")
    reference = _now_millis() if now_ms is None else now_ms
    age = (lit(reference) - Col(column).dt.epoch_ms().max()) / 1000.0
    label = max_age if isinstance(max_age, str) else f"{seconds:g}s"
    # No lower bound: data from the future is a different failure, and `not_in_future`
    # is the constraint that names it. Bounding it here would report "stale" for a row
    # whose clock ran ahead, which is the one diagnosis that sends you to the wrong system.
    return AggregateConstraint(f"fresh_within({column}, {label})", age, None, seconds)


def not_in_future(column: str, *, tolerance: str | dt.timedelta | int | float = 0) -> RowConstraint:
    """No row's `column` may be dated later than now (NULL passes).

    Clock skew between producers is real, so `tolerance` widens the bound rather than
    forcing a choice between a check that flaps and no check at all.

    Args:
        column: The timestamp or date column to test.
        tolerance: How far ahead of now a value may be before it counts as a violation.

    Returns:
        The row constraint.
    """
    slack = 0.0 if not tolerance else seconds_of(tolerance, arg="tolerance")
    cutoff = _now_millis() + int(slack * 1000)
    c = Col(column)
    label = "" if not slack else f", tolerance={slack:g}s"
    return RowConstraint(
        f"not_in_future({column}{label})", c.is_null() | (c.dt.epoch_ms() <= lit(cutoff))
    )
