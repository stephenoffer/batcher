"""Constant folding for the temporal expressions the engine's `ConstantFolding` skips.

`normalize.fold` folds a `Binary` over two literals but never looks inside a `DateFunc`,
a `DateOffset`, or a comparison between two temporal literals — so each of those survives
to the data plane as a per-row kernel over a constant. This module folds them at plan time.

Split out of `temporal_extra` on the seam between *rewriting a predicate into a range* and
*evaluating a constant*: the two share no helpers, and the file was over the size limit.
The rules register here in the order they registered there, and `extra/__init__` imports
this module immediately after `temporal_extra` so within-phase run order is unchanged.
"""

from __future__ import annotations

import calendar as _calendar
import datetime as _dt
import math as _math
import operator
from collections.abc import Callable

from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rule import Phase, plan_rule
from batcher.kyber.rules.leaf_rewrite import whole_plan_expr_rule
from batcher.plan.expr_ir import Binary, Expr, Lit
from batcher.plan.expr_ir.func_nodes import DateFunc, DateOffset

__all__ = ["fold_date_func", "fold_date_offset", "fold_temporal_comparison"]

# Comparisons foldable over two temporal literals.
_COMPARISONS = {
    "eq": operator.eq, "ne": operator.ne, "lt": operator.lt,
    "le": operator.le, "gt": operator.gt, "ge": operator.ge,
}  # fmt: skip


# --- constant folding the engine's `ConstantFolding` leaves alone -----------------

# `DateFunc`s whose value Python computes *identically* to the engine. Each mirrors an
# arrow `DatePart` (`date_part` is 1-based for quarter/day_of_year, as Python is) or the
# engine's explicit year formula (`bc-expr/src/eval/temporal/date.rs`). A fold that is not
# bit-identical is a wrong answer, so membership here is measured, never assumed.
#
# The four weekday/calendar entries were checked against the engine rather than reasoned
# about, and each is exhaustively covered by that check: a weekday extraction has exactly
# seven possible outputs and all seven agree (`day_of_week` is Sunday-0, which is
# `weekday() + 1 mod 7`, and `isodow` is `isoweekday()` outright), while `days_in_month`
# and `is_leap_year` follow the same Gregorian century rule Python's `calendar` does —
# including 2100, the year a naive "divisible by four" fold gets wrong.
#
# `week`/`iso_year` are Python's `isocalendar()` outright, and the engine agrees on every
# year-boundary case — 2021-01-01 is week 53 of ISO year 2020, 2024-12-30 is week 1 of 2025
# — which is exactly where a non-ISO implementation would diverge. `epoch` **floors**
# rather than truncating (an instant half a second before the epoch answers -1, measured),
# so its fold uses `math.floor` on a naive-datetime delta and exact day arithmetic on a
# date; `int()` would round the wrong way for every pre-1970 instant.
#
# Still excluded, on purpose: `dayname`/`monthname` are locale-sensitive in Python but
# chrono `%A`/`%B` in the engine, and `last_day` returns a temporal value rather than the
# integer this table folds to.
_FOLDABLE_DATE_FNS: dict[str, Callable[[_dt.date], int | bool]] = {
    "year": lambda v: v.year,
    "month": lambda v: v.month,
    "day": lambda v: v.day,
    "hour": lambda v: v.hour,  # datetime only (see `fold_date_func`)
    "minute": lambda v: v.minute,
    "second": lambda v: v.second,
    "quarter": lambda v: (v.month - 1) // 3 + 1,
    "day_of_year": lambda v: v.timetuple().tm_yday,
    "decade": lambda v: v.year // 10,
    "century": lambda v: (v.year - 1) // 100 + 1,
    "millennium": lambda v: (v.year - 1) // 1000 + 1,
    "day_of_week": lambda v: (v.weekday() + 1) % 7,  # Sunday-0, matching the engine
    "isodow": lambda v: v.isoweekday(),  # Monday-1 .. Sunday-7
    "days_in_month": lambda v: _calendar.monthrange(v.year, v.month)[1],
    "is_leap_year": lambda v: _calendar.isleap(v.year),
    "week": lambda v: v.isocalendar()[1],
    "iso_year": lambda v: v.isocalendar()[0],
    "epoch": lambda v: (
        _math.floor((v - _dt.datetime(1970, 1, 1)).total_seconds())
        if isinstance(v, _dt.datetime)
        else (v - _dt.date(1970, 1, 1)).days * 86_400
    ),
}
# Fields a `date` literal does not carry. The engine reads them off a Date32 array
# through arrow's `date_part`; rather than assume that yields 0, only fold them when the
# literal is a `datetime` and the answer is on the literal itself.
_TIME_FIELDS = frozenset({"hour", "minute", "second"})


def _naive_temporal(value: object) -> bool:
    """Whether `value` is a naive `date`/`datetime` literal (never a bool)."""
    if not isinstance(value, _dt.date) or isinstance(value, bool):
        return False
    return not (isinstance(value, _dt.datetime) and value.tzinfo is not None)


def fold_date_func(expr: Expr) -> Expr:
    if not (isinstance(expr, DateFunc) and isinstance(expr.input, Lit)):
        return expr
    value = expr.input.value
    if not _naive_temporal(value):
        return expr
    if expr.fn in _TIME_FIELDS and not isinstance(value, _dt.datetime):
        return expr
    fold = _FOLDABLE_DATE_FNS.get(expr.fn)
    return expr if fold is None else Lit(fold(value))


def fold_date_offset(expr: Expr) -> Expr:
    if not (isinstance(expr, DateOffset) and isinstance(expr.input, Lit)):
        return expr
    value = expr.input.value
    if not _naive_temporal(value) or expr.months:
        return expr  # calendar months clamp to the month end — the engine's rule, not ours
    if expr.micros and not isinstance(value, _dt.datetime):
        return expr  # the engine *errors* on a sub-day offset of a Date — preserve that
    try:
        return Lit(value + _dt.timedelta(days=expr.days, microseconds=expr.micros))
    except (OverflowError, ValueError):
        return expr  # off the end of the calendar — let the engine decide


def fold_temporal_comparison(expr: Expr) -> Expr:
    if not (isinstance(expr, Binary) and expr.op in _COMPARISONS):
        return expr
    left, right = expr.left, expr.right
    if not (isinstance(left, Lit) and isinstance(right, Lit)):
        return expr
    a, b = left.value, right.value
    if not (_naive_temporal(a) and _naive_temporal(b)):
        return expr
    if isinstance(a, _dt.datetime) != isinstance(b, _dt.datetime):
        return expr  # date vs datetime: the engine coerces; don't guess its rebasing
    return Lit(_COMPARISONS[expr.op](a, b))


for _name, _fn in (
    ("fold_date_func_of_literal", fold_date_func),
    ("fold_date_offset_of_literal", fold_date_offset),
    ("fold_temporal_literal_comparison", fold_temporal_comparison),
):
    DEFAULT_REGISTRY.add(plan_rule(_name, Phase.NORMALIZE, whole_plan_expr_rule(_fn)))
