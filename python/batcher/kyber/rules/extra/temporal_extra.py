"""NORMALIZE-phase temporal rewrites — the sargability gaps `temporal_sargable` leaves.

`temporal_sargable` turns ``year(col)``/``decade(col)`` comparisons into raw-column
ranges, and `normalize/ranges::date_trunc_to_range` does the same for
``date_trunc(u, col) = lit``. This module closes the rest of the *provably contiguous*
temporal surface, reusing those modules' helpers rather than restating them:

* **``date_trunc`` inequalities** (``<``/``<=``/``>``/``>=``) — the equality case is the
  existing rule; truncation is monotone, so for a unit-aligned literal each inequality
  is a single bound on the raw column.
* **``strftime`` equality + inequalities** for the three *fixed-width, zero-padded,
  big-endian* formats ``%Y`` / ``%Y-%m`` / ``%Y-%m-%d``. Those formats are order
  isomorphisms — string order **is** chronological order — so a comparison against a
  well-formed literal is exactly a range on the column.
* **``date_trunc`` collapsing** — a truncation of a truncation is one truncation (the
  coarser unit wins; the units nest exactly).
* **constant folding** the engine's `ConstantFolding` deliberately does not do: a
  `DateFunc` / `DateOffset` over a literal, and a comparison of two temporal literals
  (`_comparable` there admits only bool/str/number, so a date comparison never folds).

Deliberately **not** rewritten — each omission is a decision, with the reason:

* ``month`` / ``quarter`` / ``day`` / ``week`` / ``day_of_week`` / ``day_of_year``
  extraction predicates. **They recur.** ``month(col) = 3`` is one interval *per year in
  the data*, never a single contiguous range, so no bound on ``col`` is equivalent to it
  — a range rewrite would be a wrong answer, not a slower one. Only extractions that are
  *monotone over the column's whole domain* (year/decade, and the ``century``/
  ``millennium`` extension noted below) are range-sargable.
* ``century`` / ``millennium`` — these *are* monotone and contiguous
  (``century = (Y-1)//100 + 1``, ``millennium = (Y-1)//1000 + 1``; see
  ``bc-expr/src/eval/temporal/date.rs``), but their home is the ``_BUCKETS`` table in
  `temporal_sargable` — two entries, no new code. Left there deliberately rather than
  re-implemented here.
* ``convert_timezone`` elimination — **not** an identity, even from a zone to *itself*:
  the engine reads each naive timestamp as a local time in ``from_tz``
  (``from_local_datetime(..).single()``), so a DST-ambiguous or nonexistent local time
  yields **NULL**. Dropping the node would resurrect those rows' values. The
  ``A → B → A`` round trip is worse (it is not injective across a DST fold).
* folding ``strftime`` over a literal — Python's ``strftime`` and the engine's ``chrono``
  are different implementations (locale-dependent ``%A``/``%B``, platform-specific
  padding), so a plan-time fold cannot be proven bit-identical to the engine's.
* folding a ``DateOffset`` with a non-zero ``months`` — calendar month arithmetic clamps
  to the end of the target month (chrono ``checked_add_months``); that rule is the
  engine's, and re-deriving it in Python is exactly the kind of drift this codebase
  refuses. Zero-month (pure day/micro) offsets are exact duration arithmetic and *are*
  folded.

Nulls are preserved throughout: every rewritten form is null-in/null-out, and a filter
drops a NULL predicate either way.
"""

from __future__ import annotations

import datetime as _dt
import operator
import re
from collections.abc import Callable

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rule import Phase, node_rule, plan_rule
from batcher.kyber.rules.extra.temporal_sargable import _MIRROR, _OPS, _column_kind, _range_expr
from batcher.kyber.rules.leaf_rewrite import whole_plan_expr_rule
from batcher.kyber.rules.normalize.ranges import (
    _TRUNC_SUBDAY,
    _match_trunc_and_lit,
    _next_unit,
    _trunc_aligned,
)
from batcher.plan.expr_ir import Binary, Cast, Col, Expr, Lit
from batcher.plan.expr_ir.func_nodes import DateTrunc, Strftime
from batcher.plan.expr_rewrite import transform_expr_up
from batcher.plan.logical import Filter, LogicalPlan

__all__ = [
    "DATE_TRUNC_RANGE_RULES",
    "STRFTIME_RANGE_RULES",
    "rewrite_date_cast_filter",
    "rewrite_date_trunc_filter",
    "rewrite_strftime_filter",
]

# --- date_trunc inequalities → raw-column bounds ---------------------------------

# The eq case is `normalize/ranges::date_trunc_to_range`; these are the four it leaves.
_INEQUALITIES = ("lt", "le", "gt", "ge")


def rewrite_date_trunc_filter(node: Filter, op: str) -> Filter | None:
    """Rewrite ``date_trunc(u, col) <op> lit`` conjuncts to a bound on the raw column.

    `date_trunc` floors to the start of ``u`` and is monotone, so for a ``u``-aligned
    literal ``L``: ``trunc(col) >= L ⟺ col >= L``, ``trunc(col) < L ⟺ col < L``,
    ``trunc(col) <= L ⟺ col < L+1u``, and ``trunc(col) > L ⟺ col >= L+1u``. The bound is
    on the bare column, which zone-map pruning and source predicate pushdown can use;
    the truncation itself disappears.

    Guards (identical to the equality rule's, and for the same reasons): a bare-`Col`
    input, a `date`/`datetime` literal already aligned to the unit (an unaligned literal
    is left to the engine), and no sub-day unit against a plain `date` literal — `date`
    arithmetic silently drops a sub-day `timedelta`, so the upper bound would collapse
    onto the lower one.

    Args:
        node: The `Filter` whose predicate is scanned.
        op: The effective comparison operator to match (`lt`/`le`/`gt`/`ge`/`ne`).

    Returns:
        A new `Filter` when at least one comparison was rewritten, else None.
    """
    changed = False

    def rewrite(expr: Expr) -> Expr:
        nonlocal changed
        if not isinstance(expr, Binary) or expr.op not in _OPS:
            return expr
        trunc, lit = _match_trunc_and_lit(expr.left, expr.right)
        if trunc is None or not isinstance(trunc.input, Col):
            return expr
        # The literal on the left mirrors the operator (`L < trunc(c)` is `trunc(c) > L`).
        eff_op = expr.op if isinstance(expr.left, DateTrunc) else _MIRROR[expr.op]
        if eff_op != op:
            return expr
        bounds = _trunc_bounds(lit.value, trunc.unit)
        if bounds is None:
            return expr
        changed = True
        return _range_expr(trunc.input.name, op, *bounds)

    new_pred = transform_expr_up(node.predicate, rewrite)
    return Filter(node.input, new_pred) if changed else None


def _trunc_bounds(value: object, unit: str) -> tuple[_dt.date, _dt.date] | None:
    """The half-open ``[L, L+1unit)`` bounds of an aligned truncation literal, else None."""
    if not isinstance(value, _dt.date) or isinstance(value, bool):
        return None
    if isinstance(value, _dt.datetime) and value.tzinfo is not None:
        return None  # tz-aware: the engine compares naive instants — leave it
    is_datetime = isinstance(value, _dt.datetime)
    if not is_datetime and unit in _TRUNC_SUBDAY:
        return None  # a sub-day `timedelta` added to a `date` is dropped → wrong bound
    if not _trunc_aligned(value, unit, is_datetime):
        return None  # an unaligned literal matches nothing this range would
    try:
        upper = _next_unit(value, unit)
    except (ValueError, OverflowError):
        return None  # year 9999 + 1 is unrepresentable
    return None if upper is None else (value, upper)


# --- strftime comparisons → raw-column ranges ------------------------------------

# The formats whose text is a *fixed-width, zero-padded, big-endian* encoding of the
# instant: for these, lexicographic string order is exactly chronological order, and the
# set of instants printing a given well-formed literal is one contiguous half-open range
# (one calendar unit wide). chrono zero-pads `%Y` to four digits, `%m`/`%d` to two, so a
# literal outside the strict shape below is not producible and is left alone.
_STRFTIME_UNITS: dict[str, tuple[re.Pattern[str], str]] = {
    "%Y": (re.compile(r"^(\d{4})$"), "year"),
    "%Y-%m": (re.compile(r"^(\d{4})-(\d{2})$"), "month"),
    "%Y-%m-%d": (re.compile(r"^(\d{4})-(\d{2})-(\d{2})$"), "day"),
    # The sub-day formats are the same isomorphism carried one, two and three fields
    # further: chrono zero-pads `%H`/`%M`/`%S` to two digits and the separators are
    # constant, so the rendering stays fixed-width and most-significant-field-first. They
    # apply to a *timestamp* column only — a date has no sub-day component to bound, which
    # `_strftime_bounds` checks before building the range.
    "%Y-%m-%d %H": (re.compile(r"^(\d{4})-(\d{2})-(\d{2}) (\d{2})$"), "hour"),
    "%Y-%m-%d %H:%M": (re.compile(r"^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2})$"), "minute"),
    "%Y-%m-%d %H:%M:%S": (
        re.compile(r"^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})$"),
        "second",
    ),
    # The separator is not what makes the encoding an isomorphism — the fixed width and
    # the field order are — so the separator-free partition-key spellings and the ISO-8601
    # `T` form belong here on exactly the same argument. These are the shapes a Hive-style
    # `dt=20240105` path and a machine-generated ISO timestamp actually take.
    "%Y%m": (re.compile(r"^(\d{4})(\d{2})$"), "month"),
    "%Y%m%d": (re.compile(r"^(\d{4})(\d{2})(\d{2})$"), "day"),
    "%Y/%m/%d": (re.compile(r"^(\d{4})/(\d{2})/(\d{2})$"), "day"),
    "%Y-%m-%dT%H": (re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2})$"), "hour"),
    "%Y-%m-%dT%H:%M": (
        re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})$"),
        "minute",
    ),
    "%Y-%m-%dT%H:%M:%S": (
        re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})$"),
        "second",
    ),
}


def rewrite_strftime_filter(node: Filter, op: str) -> Filter | None:
    """Rewrite ``strftime(col, fmt) <op> '<text>'`` conjuncts to a range on the column.

    Only for the fixed-width formats in `_STRFTIME_UNITS` — the dash-separated calendar
    prefixes, their sub-day extensions, the separator-free partition-key spellings and the
    ISO-8601 ``T`` form — whose output is
    an order isomorphism (zero-padded, most-significant-field-first): the rows whose text
    equals ``'2021-03'`` are exactly ``[2021-03-01, 2021-04-01)``, and because the string
    order matches the chronological one, ``<``/``<=``/``>``/``>=`` map to the same
    single bounds the `date_trunc` family uses. The literal's type follows the *column's*
    (a `date` literal for a date column, `datetime` for a naive timestamp), so the
    emitted comparison is the one the engine already knows how to prune on.

    Skipped when the format is not one of the twelve, a sub-day format is applied to a date
    column, the literal does not match the
    format's exact shape (e.g. ``'2021-3'``), the date it names does not exist
    (``'2021-02-30'``), the column's type is unknown/tz-aware, or the bound would run off
    the representable calendar.

    Args:
        node: The `Filter` whose predicate is scanned.
        op: The effective comparison operator to match (`eq`/`lt`/`le`/`gt`/`ge`).

    Returns:
        A new `Filter` when at least one comparison was rewritten, else None.
    """
    changed = False

    def rewrite(expr: Expr) -> Expr:
        nonlocal changed
        if not isinstance(expr, Binary) or expr.op not in _OPS:
            return expr
        left, right = expr.left, expr.right
        if isinstance(left, Strftime) and isinstance(right, Lit):
            fmt_expr, lit, eff_op = left, right, expr.op
        elif isinstance(right, Strftime) and isinstance(left, Lit):
            fmt_expr, lit, eff_op = right, left, _MIRROR[expr.op]
        else:
            return expr
        if eff_op != op or not isinstance(fmt_expr.input, Col) or not isinstance(lit.value, str):
            return expr
        is_timestamp = _column_kind(node, fmt_expr.input.name)
        if is_timestamp is None:
            return expr
        bounds = _strftime_bounds(lit.value, fmt_expr.format, is_timestamp=is_timestamp)
        if bounds is None:
            return expr
        changed = True
        return _range_expr(fmt_expr.input.name, op, *bounds)

    new_pred = transform_expr_up(node.predicate, rewrite)
    return Filter(node.input, new_pred) if changed else None


def _strftime_bounds(
    text: str, fmt: str, *, is_timestamp: bool
) -> tuple[_dt.date, _dt.date] | None:
    """The half-open instant range whose `fmt` rendering is exactly `text`, else None."""
    spec = _STRFTIME_UNITS.get(fmt)
    matched = spec[0].match(text) if spec is not None else None
    if spec is None or matched is None:
        return None
    if spec[1] in _TRUNC_SUBDAY and not is_timestamp:
        return None  # a date column carries no sub-day component to bound
    fields = [int(group) for group in matched.groups()]
    year, month, day, hour, minute, second = (*fields, *(1, 1, 0, 0, 0)[len(fields) - 1 :])
    try:
        lower = (
            _dt.datetime(year, month, day, hour, minute, second)
            if is_timestamp
            else _dt.date(year, month, day)
        )
        upper = _next_unit(lower, spec[1])
    except (ValueError, OverflowError):
        return None  # a month/day the calendar has no room for, or year 9999 + 1
    return None if upper is None else (lower, upper)


# --- date_trunc(date_trunc(x)) → one truncation ----------------------------------

# The truncation units, finest → coarsest. They *nest exactly*: every boundary of a
# coarser unit is also a boundary of every finer one (a year starts at a month start,
# which starts at a day start, …). Hence `trunc(a, trunc(b, x)) == trunc(coarser, x)` for
# any pair — the coarser floor absorbs the finer one, in either nesting order. (`week`
# and `quarter` would break this — a week can straddle a month — and the engine does not
# accept them as units, so the set is closed.)
_TRUNC_ORDER = ("second", "minute", "hour", "day", "month", "year")


def _nested_trunc(expr: Expr) -> tuple[DateTrunc, DateTrunc] | None:
    """`(outer, inner)` when `expr` is a `DateTrunc` of a `DateTrunc` with known units."""
    if not (isinstance(expr, DateTrunc) and isinstance(expr.input, DateTrunc)):
        return None
    inner = expr.input
    if expr.unit not in _TRUNC_ORDER or inner.unit not in _TRUNC_ORDER:
        return None  # an unknown unit errors in the engine — do not "fix" it here
    return expr, inner


def _collapse_same_unit(expr: Expr) -> Expr:
    pair = _nested_trunc(expr)
    if pair is None or pair[0].unit != pair[1].unit:
        return expr
    return DateTrunc(pair[1].input, pair[0].unit)


def _collapse_to_coarser(expr: Expr) -> Expr:
    pair = _nested_trunc(expr)
    if pair is None or pair[0].unit == pair[1].unit:
        return expr
    outer, inner = pair
    coarser = max(outer.unit, inner.unit, key=_TRUNC_ORDER.index)
    return DateTrunc(inner.input, coarser)


# --- date → timestamp cast in a comparison ---------------------------------------

_TIMESTAMP_DTYPES = frozenset({"timestamp", "datetime"})
_COMPARISONS: dict[str, Callable[[object, object], bool]] = {
    "eq": operator.eq,
    "ne": operator.ne,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
}


def _date_cast_side(expr: Expr, node: Filter) -> str | None:
    """The column name if `expr` is ``cast(<date col>, timestamp)``, else None."""
    if not (isinstance(expr, Cast) and expr.dtype in _TIMESTAMP_DTYPES):
        return None
    if not isinstance(expr.input, Col):
        return None
    # False = a naive `date` column (True = timestamp, None = unknown/tz-aware).
    return expr.input.name if _column_kind(node, expr.input.name) is False else None


def _midnight_date(value: object) -> _dt.date | None:
    """The `date` a midnight-aligned naive `datetime` literal names, else None."""
    if not isinstance(value, _dt.datetime) or value.tzinfo is not None:
        return None
    if value.hour or value.minute or value.second or value.microsecond:
        return None
    return value.date()


def rewrite_date_cast_filter(node: Filter) -> Filter | None:
    """Drop a ``cast(<date col>, timestamp)`` under a comparison with a midnight literal.

    A Date → Timestamp cast maps each date to that date's midnight, which is an order
    isomorphism onto midnight-aligned instants. So for a *midnight* literal ``T``, every
    comparison ``cast(d, timestamp) <op> T`` holds exactly when ``d <op> T::date`` — the
    cast can go, taking the per-row conversion with it and leaving a bare-column
    comparison that zone maps and source predicate pushdown can prune on. (This is the
    widening-cast sibling of `drop_self_cast_in_filter`, which only removes a cast to the
    column's own type.)

    Restricted to a midnight literal: a non-aligned instant is *also* decidable, but only
    by rounding the bound in a direction that depends on the operator, and the shape SQL
    actually produces (``d::timestamp >= DATE '2021-01-01'``) is midnight-aligned.
    Returns None when nothing was rewritten, so the rule is idempotent (its output holds
    no `Cast` for it to match).

    Args:
        node: The `Filter` whose predicate is scanned.

    Returns:
        A new `Filter` when at least one cast was dropped, else None.
    """
    changed = False

    def rewrite(expr: Expr) -> Expr:
        nonlocal changed
        if not isinstance(expr, Binary) or expr.op not in _COMPARISONS:
            return expr
        left, right = expr.left, expr.right
        column = _date_cast_side(left, node)
        if column is not None and isinstance(right, Lit):
            bound = _midnight_date(right.value)
            if bound is None:
                return expr
            changed = True
            return Binary(expr.op, Col(column), Lit(bound))
        column = _date_cast_side(right, node)
        if column is not None and isinstance(left, Lit):
            bound = _midnight_date(left.value)
            if bound is None:
                return expr
            changed = True
            return Binary(expr.op, Lit(bound), Col(column))
        return expr

    new_pred = transform_expr_up(node.predicate, rewrite)
    return Filter(node.input, new_pred) if changed else None


# --- registration ----------------------------------------------------------------


def _trunc_rule(op: str) -> Callable[[Filter, OptimizerContext], LogicalPlan | None]:
    def apply(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
        return rewrite_date_trunc_filter(node, op)

    return apply


def _strftime_rule(op: str) -> Callable[[Filter, OptimizerContext], LogicalPlan | None]:
    def apply(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
        return rewrite_strftime_filter(node, op)

    return apply


#: One rule per (`date_trunc`, comparison) pair — the four inequalities the equality rule
#: leaves, plus `<>`. The inequality is the complement of the same `[L, L+1u)` band the
#: equality rule builds, so it needs no new bound arithmetic: `trunc(col) <> L` is
#: `col < L OR col >= L+1u`.
DATE_TRUNC_RANGE_RULES = [
    DEFAULT_REGISTRY.add(
        node_rule(
            f"date_trunc_{op}_to_range",
            Phase.NORMALIZE,
            _trunc_rule(op),
            matches=(Filter,),
            # The rewrite lands on a `Binary`, but it cannot fire without a `DateTrunc`
            # to lift the comparison off -- a far sharper gate than the comparison itself.
            expr_matches=(DateTrunc,),
        )
    )
    for op in (*_INEQUALITIES, "ne")
]

#: One rule per (`strftime`, comparison) pair, over the three order-isomorphic formats.
STRFTIME_RANGE_RULES = [
    DEFAULT_REGISTRY.add(
        node_rule(
            f"strftime_{op}_to_range",
            Phase.NORMALIZE,
            _strftime_rule(op),
            matches=(Filter,),
            expr_matches=(Strftime,),
        )
    )
    for op in _OPS
]

DEFAULT_REGISTRY.add(
    node_rule(
        "drop_date_to_timestamp_cast_in_comparison",
        Phase.NORMALIZE,
        lambda node, _ctx: rewrite_date_cast_filter(node),
        matches=(Filter,),
        # Needs a timestamp `Cast` to drop; a plan without one can never match.
        expr_matches=(Cast,),
    )
)

for _name, _fn in (
    ("date_trunc_idempotent", _collapse_same_unit),
    ("date_trunc_nested_to_coarser", _collapse_to_coarser),
):
    DEFAULT_REGISTRY.add(plan_rule(_name, Phase.NORMALIZE, whole_plan_expr_rule(_fn)))
