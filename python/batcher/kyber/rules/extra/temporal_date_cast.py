"""``CAST(ts AS DATE) <op> DATE 'd'`` — the timestamp-to-date cast, turned into a range.

The one remaining opaque spelling of the most common date filter there is. Batcher already
rewrites `year(ts) = 2024` and `date_trunc('day', ts) = t` into raw-column ranges, and it
already drops the *widening* cast in `cast(date_col, timestamp) >= t`
(`temporal_extra.rewrite_date_cast_filter`). The narrowing direction — the one SQL spells
`WHERE order_ts::date = '2024-01-01'` or `WHERE CAST(order_ts AS DATE) >= '2024-01-01'` — was
opaque to every consumer: zone-map pruning, bloom skipping, and source predicate pushdown all
match on a comparison whose column side is a bare `Col`, so a query written that way read every
row group in a date-partitioned table.

The rewrite is exact because the cast is a **floor**, which was measured rather than assumed:
the engine casts `1969-12-31 12:00` to `1969-12-31` and `1969-12-30 23:59:59` to `1969-12-30`,
so it floors toward negative infinity on both sides of the epoch rather than truncating toward
zero. Flooring is monotone non-decreasing, and it maps exactly the instants in
`[d 00:00, d+1 00:00)` to the date `d` — so each comparison against `d` becomes the
corresponding bound on that half-open band, which is what `_range_expr` builds.

**A timezone-aware timestamp is excluded**, and that guard is the whole soundness argument for
the boundary. The cast uses the *local* date, so `2024-01-01 01:00Z` in `America/New_York` casts
to `2023-12-31` — the day boundary is a local midnight, which is not a fixed instant offset
across a DST transition, so no single pair of naive bounds names it. `_kind_from_schema` answers
`True` only for a naive timestamp, and the rule declines otherwise.

One rule per comparison, six in all, following the `temporal_sargable` family: they are schema
*leaves*, so the driver offers every expression to all six in the one traversal it already makes
and resolves the node's schema once, which is what makes the per-operator split free here.
"""

from __future__ import annotations

import datetime as _dt

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rule import Phase, node_rule

# `_MIRROR`/`_OPS` (the operator vocabulary), `_kind_from_schema`/`_column_kind` (the naive
# date-vs-timestamp guard) and `_range_expr` (the per-operator band predicate) are the sibling
# family's helpers, imported rather than re-implemented.
from batcher.kyber.rules.extra.temporal_sargable import (
    _MIRROR,
    _OPS,
    _column_kind,
    _kind_from_schema,
    _range_expr,
)
from batcher.plan.expr_ir import Binary, Cast, Col, Expr, Lit
from batcher.plan.expr_rewrite import transform_expr_up
from batcher.plan.logical import Filter, LogicalPlan

__all__ = ["DATE_CAST_RANGE_RULES", "rewrite_date_cast_range"]

#: The `Cast` dtype spellings that narrow a timestamp to a date. Both name the same Arrow
#: type; `plan.types.CAST_DTYPES` accepts either, so a rule matching only one would fire on
#: half the queries.
_DATE_DTYPES = frozenset({"date", "date32"})


def _match_date_cast(expr: Binary) -> tuple[str, _dt.date, str] | None:
    """`(column, date, effective_op)` for `cast(<col>, date) <op> DATE 'd'`, else ``None``.

    Accepts the literal on either side, mirroring the operator so a predicate written
    `DATE '2024-01-01' < ts::date` is analyzed as the `gt` case. The literal must be a plain
    `date`: a `datetime` compared against a date-typed cast is a different comparison (the
    engine coerces one side), and this rule claims nothing about it.
    """
    if expr.op not in _MIRROR:
        return None
    for cast, other, op in (
        (expr.left, expr.right, expr.op),
        (expr.right, expr.left, _MIRROR[expr.op]),
    ):
        if not (isinstance(cast, Cast) and cast.dtype in _DATE_DTYPES):
            continue
        if not isinstance(cast.input, Col) or not isinstance(other, Lit):
            continue
        value = other.value
        if isinstance(value, _dt.date) and not isinstance(value, _dt.datetime):
            return cast.input.name, value, op
    return None


def _band(day: _dt.date) -> tuple[_dt.datetime, _dt.datetime]:
    """The half-open instant band `[day 00:00, day+1 00:00)` the cast maps onto `day`."""
    lo = _dt.datetime.combine(day, _dt.time())
    return lo, lo + _dt.timedelta(days=1)


def _rewrite(expr: Expr, op: str, kind) -> Expr:
    """The shared body: rewrite `expr` when it is this rule's `(cast, op)` shape.

    `kind` resolves the column's temporal kind — the naive-timestamp guard — and is the only
    difference between the standalone and the fused form.
    """
    if not isinstance(expr, Binary):
        return expr
    matched = _match_date_cast(expr)
    if matched is None:
        return expr
    column, day, effective_op = matched
    if effective_op != op:
        return expr
    # `True` is a naive timestamp; `False` is a date column (where the cast is a no-op that
    # `drop_self_cast_in_filter` removes) and `None` covers an unknown schema and, critically,
    # a timezone-aware column, whose day boundary is a local midnight.
    if kind(column) is not True:
        return expr
    lo, hi = _band(day)
    return _range_expr(column, op, lo, hi)


def rewrite_date_cast_range(node: Filter, op: str) -> Filter | None:
    """Rewrite `cast(ts, date) <op> DATE 'd'` conjuncts in `node`'s predicate to a band on `ts`.

    The standalone whole-node form, equivalent to the fused leaf below and the entry point the
    unit tests drive.

    Args:
        node: The `Filter` whose predicate is scanned.
        op: The effective comparison to match, one of `eq`/`ne`/`lt`/`le`/`gt`/`ge`.

    Returns:
        A new `Filter` when at least one comparison was rewritten, else ``None`` — which is
        also what makes the rule idempotent, since the rewritten form holds no `Cast` to match.
    """
    changed = False

    def rewrite(expr: Expr) -> Expr:
        nonlocal changed
        out = _rewrite(expr, op, lambda name: _column_kind(node, name))
        changed = changed or out is not expr
        return out

    new_pred = transform_expr_up(node.predicate, rewrite)
    return Filter(node.input, new_pred) if changed else None


def _make_rule(op: str):
    def apply(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
        return rewrite_date_cast_range(node, op)

    return apply


def _make_leaf(op: str):
    def leaf(expr: Expr, schema) -> Expr:
        return _rewrite(expr, op, lambda name: _kind_from_schema(schema, name))

    return leaf


#: `cast(ts, date) <op> DATE 'd'` -> the instant band on `ts`, one rule per comparison:
#: `=` becomes `ts >= d AND ts < d+1`, `<>` its complement as a disjunction, and each ordered
#: comparison the single bound at whichever end of the band it names.
DATE_CAST_RANGE_RULES = [
    DEFAULT_REGISTRY.add(
        node_rule(
            f"cast_date_{op}_to_range",
            Phase.NORMALIZE,
            _make_rule(op),
            matches=(Filter,),
            expr_schema_fn=_make_leaf(op),
            expr_matches=(Binary,),
            expr_ops=(op, _MIRROR[op]),
        )
    )
    for op in _OPS
]
