"""`offset_by(ts, …) OP instant` restated as `ts OP shifted_instant`.

A shifted timestamp on the column side of a comparison is the temporal twin of
`col + 100 = 500`: the shift is a constant, so it belongs on the literal. Moving it there
leaves a bare `ts OP <instant>`, which is the only shape partition pruning, zonemap
refutation, and Parquet/Iceberg predicate pushdown recognize. This is the shape a
service-level query takes — `WHERE event_time + INTERVAL 15 MINUTE < deadline` — and
today every row of every file is read to answer it.

The transform is a translation of the instant axis, so it preserves the order and is
exact for *every* comparison operator, not only equality. That is the opposite of the
integer case in `extra/sargable`, where wrapping arithmetic restricts the same rewrite to
`=` and `<>`, and the reason is that the days-and-microseconds shift here is a genuine
addition on a value the guard has already checked stays in range.

`offset_by` shifts a `Date` as well as a `Timestamp`, and the two need separate rules
because the shifted literal has a different *type*: a date column compared against a
timestamp literal is a different comparison. Both families are registered here, each
guarded by the schema and declining when it cannot answer.

**Calendar months are excluded from both.** A month is not a fixed duration: `ts + 1 month
> L` is not `ts > L - 1 month`, because shifting 31 January forward a month and back again
does not return 31 January. A `DateOffset` carrying a non-zero `months` declines outright,
and on the date side a non-zero `micros` declines too — a sub-day shift has nowhere to go
in a date literal.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable

from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rule import Phase, node_rule
from batcher.kyber.rules.exprs.guards import is_date, is_timestamp, schema_rule
from batcher.plan.expr_ir import Binary, Expr, Lit
from batcher.plan.expr_ir.func_nodes import DateOffset
from batcher.plan.ir_tags import COMPARISON_FLIP
from batcher.plan.logical import Filter, Project
from batcher.plan.schema import SchemaRef

__all__ = ["OFFSET_DATE_SHIFT_RULES", "OFFSET_SHIFT_RULES"]

_COMPARISONS = ("lt", "le", "gt", "ge", "eq", "ne")
_MICROS_PER_DAY = 86_400_000_000


def _instant_literal(expr: Expr) -> dt.datetime | None:
    """The instant a timestamp literal names, else ``None``.

    A `date` is rejected: `datetime` is a subclass of `date`, so the order of these two
    checks is what keeps a bare date out.
    """
    if isinstance(expr, Lit) and isinstance(expr.value, dt.datetime):
        return expr.value
    return None


def _offset_comparison(expr: Expr) -> tuple[str, DateOffset, dt.datetime] | None:
    """`(op, offset_call, instant)` for an `offset_by(ts, …) OP <timestamp literal>`."""
    if not isinstance(expr, Binary) or expr.op not in COMPARISON_FLIP:
        return None
    for computed, other, op in (
        (expr.left, expr.right, expr.op),
        (expr.right, expr.left, COMPARISON_FLIP[expr.op]),
    ):
        if isinstance(computed, DateOffset):
            instant = _instant_literal(other)
            if instant is not None:
                return op, computed, instant
    return None


def _shifted(instant: dt.datetime, offset: DateOffset) -> Lit | None:
    """The literal `instant` shifted *back* by the offset, or ``None`` when the shift is
    a calendar month or would leave the representable range."""
    if offset.months:
        return None
    micros = offset.days * _MICROS_PER_DAY + offset.micros
    try:
        return Lit(instant - dt.timedelta(microseconds=micros))
    except OverflowError:
        return None


def _offset_leaf(op_wanted: str) -> Callable[[Expr, SchemaRef | None], Expr]:
    def leaf(expr: Expr, schema: SchemaRef | None) -> Expr:
        parts = _offset_comparison(expr)
        if parts is None:
            return expr
        op, offset, instant = parts
        if op != op_wanted or not is_timestamp(offset.input, schema):
            return expr
        shifted = _shifted(instant, offset)
        if shifted is None:
            return expr
        return Binary(op, offset.input, shifted)

    return leaf


def _register(op: str):
    leaf = _offset_leaf(op)
    return DEFAULT_REGISTRY.add(
        node_rule(
            f"offset_by_{op}_to_shifted_instant",
            Phase.NORMALIZE,
            lambda node, _ctx, _leaf=leaf: schema_rule(node, _leaf, carries=(Binary,)),
            matches=(Filter, Project),
            expr_schema_fn=leaf,
            expr_matches=(Binary,),
            # Both `op` and its mirror: the comparison is normalized with the computed side
            # on the left, so a `lt` leaf is reached by a `gt` node with the literal on the left.
            expr_ops=(op, COMPARISON_FLIP[op]),
        )
    )


#: `offset_by(ts, days=d, micros=m) OP L` -> `ts OP L - (d days + m micros)`, one rule per
#: comparison operator. Order-preserving, so every operator carries over unchanged; a
#: calendar-month component or a non-`Timestamp` argument declines.
OFFSET_SHIFT_RULES = [_register(op) for op in _COMPARISONS]


# --- the same shift over a Date column ---------------------------------------


def _date_literal(expr: Expr) -> dt.date | None:
    """The date a date literal names, else ``None``. A `datetime` is rejected — it is a
    `date` subclass, and the order of the two checks is what keeps an instant out."""
    if (
        isinstance(expr, Lit)
        and isinstance(expr.value, dt.date)
        and not isinstance(expr.value, dt.datetime)
    ):
        return expr.value
    return None


def _date_offset_comparison(expr: Expr) -> tuple[str, DateOffset, dt.date] | None:
    if not isinstance(expr, Binary) or expr.op not in COMPARISON_FLIP:
        return None
    for computed, other, op in (
        (expr.left, expr.right, expr.op),
        (expr.right, expr.left, COMPARISON_FLIP[expr.op]),
    ):
        if isinstance(computed, DateOffset):
            day = _date_literal(other)
            if day is not None:
                return op, computed, day
    return None


def _offset_date_leaf(op_wanted: str) -> Callable[[Expr, SchemaRef | None], Expr]:
    def leaf(expr: Expr, schema: SchemaRef | None) -> Expr:
        parts = _date_offset_comparison(expr)
        if parts is None:
            return expr
        op, offset, day = parts
        if op != op_wanted or not is_date(offset.input, schema):
            return expr
        if offset.months or offset.micros:
            # A calendar month is not a fixed duration, and a sub-day shift on a date has
            # no representation in the shifted literal — decline rather than approximate.
            return expr
        try:
            shifted = day - dt.timedelta(days=offset.days)
        except OverflowError:
            return expr
        return Binary(op, offset.input, Lit(shifted))

    return leaf


def _register_date(op: str):
    leaf = _offset_date_leaf(op)
    return DEFAULT_REGISTRY.add(
        node_rule(
            f"offset_by_{op}_to_shifted_date",
            Phase.NORMALIZE,
            lambda node, _ctx, _leaf=leaf: schema_rule(node, _leaf, carries=(Binary,)),
            matches=(Filter, Project),
            expr_schema_fn=leaf,
            expr_matches=(Binary,),
            # Both `op` and its mirror: the comparison is normalized with the computed side
            # on the left, so a `lt` leaf is reached by a `gt` node with the literal on the left.
            expr_ops=(op, COMPARISON_FLIP[op]),
        )
    )


#: `offset_by(d, days=k) OP L` -> `d OP L - k days` over a **Date** column and a date
#: literal. Separate from the timestamp family because the shifted literal has a different
#: type, and a date column compared against a timestamp literal is a different comparison.
#: A microsecond component declines here as well as a calendar month: a sub-day shift has
#: nowhere to go in a date literal.
OFFSET_DATE_SHIFT_RULES = [_register_date(op) for op in _COMPARISONS]
