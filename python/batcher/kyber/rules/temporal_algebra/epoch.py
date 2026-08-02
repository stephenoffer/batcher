"""`epoch(ts) OP seconds` restated as a half-open interval on the timestamp itself.

`epoch` reports the **floored** Unix-second count of an instant: an instant half a second
before the epoch answers `-1`, not `0` (verified against the engine, and the reason this
family can be exact at all). Flooring makes the comparison a bucket test on a
one-second-wide bucket, so it carries the same interval algebra as integer `//`:

    epoch(ts) >= c   <=>   ts >= T(c)          epoch(ts) <  c   <=>   ts <  T(c)
    epoch(ts) >  c   <=>   ts >= T(c + 1)      epoch(ts) <= c   <=>   ts <  T(c + 1)
    epoch(ts) =  c   <=>   T(c) <= ts < T(c+1)

where `T(c)` is the instant `c` seconds after the epoch. Every bound is exact: a second
is exactly 1,000,000 microseconds in the engine's timestamp representation, so `T(c)` is
representable without rounding for any `c` the guard admits.

The type guard is load-bearing rather than defensive. `epoch` accepts a `Date` as well as
a `Timestamp`, and a `Date` column compared against a timestamp literal is a different
comparison with a different answer — so the rules fire only when the schema says the
argument is a `Timestamp`, and decline when it cannot say.

The inverse round trip, `epoch(from_unix_seconds(n)) -> n`, is deliberately **not** a rule
here even though it is the same fact read backwards. An epoch count that names no
representable instant makes the constructor answer null, and the floor then carries that
null out — while the bare `n` is not null. The two differ on exactly the inputs the
constructor was built to tolerate, so there is no rule to write.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable

from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rule import Phase, node_rule
from batcher.kyber.rules.exprs.guards import is_date, is_timestamp, schema_rule
from batcher.plan.expr_ir import Binary, Expr, Lit
from batcher.plan.expr_ir.func_nodes import DateFunc
from batcher.plan.ir_tags import COMPARISON_FLIP
from batcher.plan.logical import Filter, Project
from batcher.plan.schema import SchemaRef

__all__ = ["EPOCH_DATE_RANGE_RULES", "EPOCH_RANGE_RULES"]

_COMPARISONS = ("lt", "le", "gt", "ge", "eq", "ne")
#: The second counts `datetime` can name, so `T(c)` is always constructible inside them.
#: Outside, the rule declines rather than folding an instant the literal cannot hold.
_MIN_SECONDS = -62_135_596_800  # 0001-01-01T00:00:00
_MAX_SECONDS = 253_402_300_799  # 9999-12-31T23:59:59


def _seconds_literal(expr: Expr) -> int | None:
    if isinstance(expr, Lit) and isinstance(expr.value, int) and not isinstance(expr.value, bool):
        return expr.value
    return None


def _instant(seconds: int) -> Lit | None:
    """The `Lit` naming the instant `seconds` after the Unix epoch, or ``None`` when it
    falls outside the representable timestamp range."""
    if not _MIN_SECONDS <= seconds <= _MAX_SECONDS:
        return None
    return Lit(dt.datetime(1970, 1, 1) + dt.timedelta(seconds=seconds))


def _epoch_comparison(expr: Expr) -> tuple[str, Expr, int] | None:
    """`(op, timestamp_argument, seconds)` for an `epoch(ts) OP <int literal>`."""
    if not isinstance(expr, Binary) or expr.op not in COMPARISON_FLIP:
        return None
    for computed, other, op in (
        (expr.left, expr.right, expr.op),
        (expr.right, expr.left, COMPARISON_FLIP[expr.op]),
    ):
        if isinstance(computed, DateFunc) and computed.fn == "epoch":
            seconds = _seconds_literal(other)
            if seconds is not None:
                return op, computed.input, seconds
    return None


def _interval(op: str, ts: Expr, low: Lit, high: Lit) -> Expr | None:
    if op == "lt":
        return Binary("lt", ts, low)
    if op == "le":
        return Binary("lt", ts, high)
    if op == "gt":
        return Binary("ge", ts, high)
    if op == "ge":
        return Binary("ge", ts, low)
    if op == "eq":
        return Binary("and", Binary("ge", ts, low), Binary("lt", ts, high))
    if op == "ne":
        return Binary("or", Binary("lt", ts, low), Binary("ge", ts, high))
    return None


def _epoch_leaf(op_wanted: str) -> Callable[[Expr, SchemaRef | None], Expr]:
    def leaf(expr: Expr, schema: SchemaRef | None) -> Expr:
        parts = _epoch_comparison(expr)
        if parts is None:
            return expr
        op, ts, seconds = parts
        if op != op_wanted or not is_timestamp(ts, schema):
            return expr
        low, high = _instant(seconds), _instant(seconds + 1)
        if low is None or high is None:
            return expr
        rewritten = _interval(op, ts, low, high)
        return expr if rewritten is None else rewritten

    return leaf


def _register(op: str):
    leaf = _epoch_leaf(op)
    return DEFAULT_REGISTRY.add(
        node_rule(
            f"epoch_{op}_to_instant_range",
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


#: `epoch(ts) OP c` as a half-open instant interval on `ts`, one rule per comparison
#: operator. The rewritten form is a bare timestamp comparison, which is what partition
#: pruning, zonemap refutation, and source pushdown all match on.
EPOCH_RANGE_RULES = [_register(op) for op in _COMPARISONS]


# --- the same comparison over a Date column ----------------------------------

_SECONDS_PER_DAY = 86_400
_MIN_DAYS = (dt.date(1, 1, 1) - dt.date(1970, 1, 1)).days
_MAX_DAYS = (dt.date(9999, 12, 31) - dt.date(1970, 1, 1)).days


def _day(days: int) -> Lit | None:
    """The `Lit` naming the date `days` after the epoch, or ``None`` when out of range."""
    if not _MIN_DAYS <= days <= _MAX_DAYS:
        return None
    return Lit(dt.date(1970, 1, 1) + dt.timedelta(days=days))


def _date_bound(op: str, seconds: int) -> tuple[str, int] | None:
    """`(operator, day count)` for `epoch(d) OP seconds` restated on the date itself.

    A date's epoch is an exact multiple of 86,400, so the comparison lands *between* two
    days whenever `seconds` is not itself a multiple. The four ordered operators therefore
    round the boundary in whichever direction keeps the same set of days:
    `>= s` needs the first day at or after `s/86400`, `> s` the first strictly after it,
    and the two upper bounds mirror that. `=` and `<>` have a date restatement only when
    `s` names a midnight exactly; otherwise no date satisfies them and the honest answer
    is to decline rather than fold to a constant that would misreport a null row.
    """
    exact, remainder = divmod(seconds, _SECONDS_PER_DAY)  # floor division, remainder >= 0
    ceiling = exact if remainder == 0 else exact + 1
    if op == "ge":
        return "ge", ceiling
    if op == "gt":
        return "ge", exact + 1
    if op == "lt":
        return "le", ceiling - 1
    if op == "le":
        return "le", exact
    if remainder != 0:
        return None
    return ("eq" if op == "eq" else "ne"), exact


def _epoch_date_leaf(op_wanted: str) -> Callable[[Expr, SchemaRef | None], Expr]:
    def leaf(expr: Expr, schema: SchemaRef | None) -> Expr:
        parts = _epoch_comparison(expr)
        if parts is None:
            return expr
        op, date_expr, seconds = parts
        if op != op_wanted or not is_date(date_expr, schema):
            return expr
        bound = _date_bound(op, seconds)
        if bound is None:
            return expr
        day = _day(bound[1])
        return expr if day is None else Binary(bound[0], date_expr, day)

    return leaf


def _register_date(op: str):
    leaf = _epoch_date_leaf(op)
    return DEFAULT_REGISTRY.add(
        node_rule(
            f"epoch_{op}_to_date_range",
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


#: `epoch(d) OP c` over a **Date** column, restated as a comparison against a date literal.
#: A date's epoch is an exact multiple of 86,400 seconds, so a bound that falls inside a day
#: rounds to the day boundary that keeps the same set of dates; `=` and `<>` fold only when
#: the bound names a midnight exactly. Separate from the timestamp family because the two
#: produce different literal *types*, and comparing a date column against a timestamp
#: literal is a different comparison.
EPOCH_DATE_RANGE_RULES = [_register_date(op) for op in _COMPARISONS]
