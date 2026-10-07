"""`time_bucket` — snapping a timestamp to the start of the period that contains it.

Two mechanisms under one SQL name, which is why they live together and apart from the rest
of the temporal surface. A **fixed** width (DAY and below) is a number of microseconds, so
it is an epoch-anchored `WindowStart`; a **calendar** width (MONTH and above) is not a
number of microseconds at all, so it is bucketed on the *month index* instead. Neither can
express the other, and each has its own origin problem to answer to DuckDB.
"""

from __future__ import annotations

import datetime as _dt

from sqlglot import expressions as exp

from batcher.plan.expr_ir import Cast, Expr, Lit, lit
from batcher.plan.expr_ir.func_nodes import WindowStart
from batcher.plan.functions.temporal import make_date
from batcher.plan.ir_tags import MICROS_PER_DAY

__all__ = ["time_bucket"]


# `time_bucket` widths, in microseconds, for the fixed-length interval units. MONTH and
# larger are absent because they are not a number of microseconds: `_BUCKET_MONTHS` below
# carries them, on the month index.
_BUCKET_MICROS = {
    "DAY": MICROS_PER_DAY,
    "HOUR": 3_600_000_000,
    "MINUTE": 60_000_000,
    "SECOND": 1_000_000,
    "MILLISECOND": 1_000,
    "MICROSECOND": 1,
}

# DuckDB anchors `time_bucket` at 2000-01-03 00:00:00, not at the Unix epoch; that is
# 10,959 days later. `WindowStart` carries its origin, so the bucket grid is anchored there
# directly. It used to be epoch-anchored, which agreed with DuckDB only when the width
# divided the gap between the two origins (1 DAY, 2 HOUR, 5 MINUTE do; 2 DAY and 7 DAY do
# not), so those widths had to be refused rather than answered on a shifted grid.
_BUCKET_ORIGIN_MICROS = 10_959 * MICROS_PER_DAY


#: Calendar `time_bucket` units, as a count of months. A month is not a fixed number of
#: microseconds, so these cannot go through `WindowStart` at all — they are bucketed on the
#: *month index* instead, which is exact and needs no width that divides anything.
_BUCKET_MONTHS = {"MONTH": 1, "QUARTER": 3, "YEAR": 12, "DECADE": 120, "CENTURY": 1200}

#: DuckDB anchors a calendar `time_bucket` at 2000-01, which is month index 24000 counting
#: from year 0. Buckets run outward from there in both directions, so a pre-2000 timestamp
#: needs *floored* division — truncation would round a negative index toward zero and put
#: 1999-07 in the bucket starting 1999-09 under a 6-month width.
_BUCKET_MONTH_ORIGIN = 2000 * 12


def _month_bucket(value: Expr, months: int, to_date: bool) -> Expr:
    """`time_bucket(INTERVAL n MONTH/QUARTER/YEAR, ts)` by month-index arithmetic."""
    index = value.dt.year() * lit(12) + (value.dt.month() - lit(1))
    offset = index - lit(_BUCKET_MONTH_ORIGIN)
    start = lit(_BUCKET_MONTH_ORIGIN) + (offset // lit(months)) * lit(months)
    built = make_date(start // lit(12), start % lit(12) + lit(1), lit(1))
    return built if to_date else Cast(built, "timestamp")


def _interval_micros(interval: exp.Interval) -> tuple[str, int, int | None]:
    """`(unit, count, fixed width in micros or None)` of an ``INTERVAL n unit`` literal."""
    unit = (interval.text("unit") or "DAY").upper().removesuffix("S")
    count = int(interval.this.name)
    per = _BUCKET_MICROS.get(unit)
    return unit, count, None if per is None else count * per


def _bucket_origin(tr, third: exp.Expression | None) -> int | None:
    """The bucket grid's origin in epoch micros from `time_bucket`'s optional third argument.

    DuckDB takes either a TIMESTAMP origin the grid passes through, or an INTERVAL offset
    that slides its default origin. The argument used to be ignored, so a query asking for
    a custom origin got the default grid with no error. Anything else is refused (`None`).
    """
    if third is None:
        return _BUCKET_ORIGIN_MICROS
    if isinstance(third, exp.Interval):
        _unit, _count, offset = _interval_micros(third)
        return None if offset is None else _BUCKET_ORIGIN_MICROS + offset
    origin = tr._scalar(third)
    if isinstance(origin, Cast) and isinstance(origin.input, Lit):
        origin = origin.input
    if isinstance(origin, Lit) and isinstance(origin.value, str):
        try:
            value = _dt.datetime.fromisoformat(origin.value)
        except ValueError:
            return None
        epoch = _dt.datetime(1970, 1, 1, tzinfo=value.tzinfo)
        return (value - epoch) // _dt.timedelta(microseconds=1)
    if isinstance(origin, Lit) and isinstance(origin.value, _dt.datetime):
        return (origin.value - _dt.datetime(1970, 1, 1)) // _dt.timedelta(microseconds=1)
    return None


def time_bucket(tr, node) -> Expr | None:
    """`time_bucket(INTERVAL n unit, ts[, origin])` → the start of the bucket per row."""
    interval = node.this
    if not isinstance(interval, exp.Interval):
        return None
    unit, count, width = _interval_micros(interval)
    third = node.args.get("unit")
    value = tr._scalar(node.expression)
    # DuckDB gives back the type it was given: bucketing a DATE yields a DATE, not the
    # midnight timestamp `WindowStart` computes in. Reading the argument's type here is
    # what keeps a `GROUP BY time_bucket(...)` key joinable against the date column it
    # came from, instead of silently widening it.
    is_date = _is_date(tr, value)
    if unit in _BUCKET_MONTHS:
        if third is not None:
            raise NotImplementedError(
                f"time_bucket(INTERVAL {count} {unit}, ts, origin) is not supported: a custom "
                "origin is supported for fixed widths (DAY and below) only"
            )
        if count <= 0:
            raise NotImplementedError(
                f"time_bucket(INTERVAL {count} {unit}, ...) is not supported: a bucket "
                "width must be positive"
            )
        return _month_bucket(value, count * _BUCKET_MONTHS[unit], is_date)
    if width is None:
        raise NotImplementedError(
            f"time_bucket(INTERVAL {count} {unit}, ...) is not supported: {unit} is "
            "neither a fixed width nor a calendar unit the month index can express. Use "
            "DAY/HOUR/MINUTE/SECOND for fixed widths or MONTH/QUARTER/YEAR for calendar "
            "ones, or date_trunc for a single period"
        )
    if width <= 0:
        raise NotImplementedError(
            f"time_bucket(INTERVAL {count} {unit}, ...) is not supported: a bucket width "
            "must be positive"
        )
    origin = _bucket_origin(tr, third)
    if origin is None:
        raise NotImplementedError(
            "time_bucket(..., origin) is not supported for this third argument: pass a "
            "TIMESTAMP literal origin or a fixed INTERVAL offset"
        )
    # Any origin congruent modulo the width gives the same grid; the reduced one keeps the
    # epoch-anchored IR byte-identical for every width that divides the DuckDB gap.
    bucketed = WindowStart(value, width, origin % width)
    return Cast(bucketed, "date") if is_date else bucketed


def _is_date(tr, value: Expr) -> bool:
    """Whether the bucketed value is a DATE, so the bucket should come back as one.

    Asked of the *built expression*, not the AST node: `DATE '2024-01-31'` is a literal, and
    a column-only lookup answered None for it and widened the result to a timestamp.
    """
    import pyarrow as pa

    arrow_type = tr.expr_type(value)
    return arrow_type is not None and pa.types.is_date(arrow_type)
