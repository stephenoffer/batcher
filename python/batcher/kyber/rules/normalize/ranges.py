"""Predicate → sargable-range rewrites in the NORMALIZE phase.

A pure-prefix `LIKE 'abc%'`, a `date_trunc(unit, col) = lit`, and an `OR` chain of
equalities on one column are each *exactly* equivalent to plain comparisons or an
`IN` list. Zone-map pruning and source predicate pushdown are blind to the original
forms and can use the rewritten ones, so the rewrite turns an opaque scan into a
skippable one.
"""

from __future__ import annotations

import datetime as _dt

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import DEFAULT_REGISTRY, rule
from batcher.kyber.rule import Phase, plan_rule
from batcher.plan.expr_ir import Binary, Col, Expr, Lit
from batcher.plan.expr_ir.namespaces import DateTrunc, StrFunc
from batcher.plan.expr_rewrite import combine_conjuncts, map_node_expressions, split_conjuncts
from batcher.plan.logical import Filter, LogicalPlan
from batcher.plan.visitor import transform_up

__all__ = ["date_trunc_to_range", "like_prefix_to_range", "or_to_in_and_range"]


# --- LIKE-prefix → range ----------------------------------------------------

# Characters that make a LIKE pattern more than a plain prefix.
_PATTERN_SPECIAL = frozenset("%_\\")
# Largest last-prefix character we will increment: the increment must stay a single
# byte whose UTF-8 order matches its code point (true for ASCII below 0x7F).
_MAX_INCREMENTABLE = 0x7E


def like_prefix_to_range(plan: LogicalPlan) -> LogicalPlan:
    """Rewrite every pure-prefix `LIKE 'abc%'` to the exact range `col >= 'abc' AND
    col < 'abd'`.

    With no other wildcards the range is exact — a string matches `'abc%'` iff it is
    in `['abc', 'abd')` — so the opaque `LIKE` is replaced by plain comparisons that
    zone-map pruning and predicate pushdown can use (both are blind to `LIKE`). The
    classic prefix-search accelerant DuckDB/Spark apply, feeding Batcher's
    metadata-driven `zonemap_prune_filter`. Conservative: it fires only for a
    `<prefix>%` pattern whose prefix is non-empty, contains no further `%`/`_`/escape,
    and ends in a safely-incrementable ASCII character; `ILIKE` and mid-string
    wildcards are left untouched.
    """
    return transform_up(plan, lambda node: map_node_expressions(node, _rewrite_like))


def _rewrite_like(expr: Expr) -> Expr:
    if not (isinstance(expr, StrFunc) and expr.fn == "like" and isinstance(expr.pattern, str)):
        return expr
    upper = _prefix_upper_bound(expr.pattern)
    if upper is None:
        return expr
    prefix = expr.pattern[:-1]
    return Binary(
        "and",
        Binary("ge", expr.input, Lit(prefix)),
        Binary("lt", expr.input, Lit(upper)),
    )


def _prefix_upper_bound(pattern: str) -> str | None:
    """The exclusive upper bound for `<prefix>%`, or None if not a safe pure prefix."""
    if len(pattern) < 2 or not pattern.endswith("%"):
        return None
    prefix = pattern[:-1]
    if any(c in _PATTERN_SPECIAL for c in prefix):
        return None
    if ord(prefix[-1]) > _MAX_INCREMENTABLE:
        return None
    return prefix[:-1] + chr(ord(prefix[-1]) + 1)


DEFAULT_REGISTRY.add(
    plan_rule(
        "like_prefix_to_range",
        Phase.NORMALIZE,
        lambda plan, _ctx: like_prefix_to_range(plan),
    )
)


# Fixed-width truncation units: one unit is an exact micro/second interval, so the
# upper bound is plain `literal + timedelta` (no calendar math). `month`/`year` are
# calendar units, handled in `_next_unit`. `date_trunc` accepts exactly this set
# (`.dt.truncate` — year/month/day/hour/minute/second), matching DuckDB.
_TRUNC_FIXED_STEP = {
    "second": _dt.timedelta(seconds=1),
    "minute": _dt.timedelta(minutes=1),
    "hour": _dt.timedelta(hours=1),
    "day": _dt.timedelta(days=1),
}
_TRUNC_SUBDAY = frozenset({"second", "minute", "hour"})


def date_trunc_to_range(plan: LogicalPlan) -> LogicalPlan:
    """Rewrite `date_trunc(unit, col) = <aligned literal>` to the half-open range
    `col >= lit AND col < lit + one_unit`.

    `date_trunc` floors each value to the start of `unit` and is monotonic, so a row
    satisfies `date_trunc(unit, col) = lit` (for a unit-aligned `lit`) iff
    `lit <= col < lit + one_unit` — an exact rewrite. The opaque truncation is replaced
    by plain comparisons on the raw column, which `zonemap_prune_filter` and source
    predicate pushdown can use to skip whole row groups / files. This is the classic
    time-series sargability transform (the dominant lakehouse shape: partitioned by
    date), feeding Batcher's metadata-driven data skipping.

    Conservative — it fires only for an *equality* whose truncated side is a bare
    `Col` and whose literal is a `date`/`datetime` already aligned to `unit` (an
    unaligned literal is left untouched; it is still evaluated correctly, just
    unpruned). Sub-day units on a `date` literal are skipped (the truncation is a
    no-op there and a range would change the literal's type). Inequalities are left to
    the engine.
    """
    return transform_up(plan, lambda node: map_node_expressions(node, _rewrite_date_trunc_eq))


def _match_trunc_and_lit(a: Expr, b: Expr) -> tuple[DateTrunc, Lit] | tuple[None, None]:
    """`(DateTrunc, Lit)` from a comparison's two sides in either order, else `(None, None)`."""
    if isinstance(a, DateTrunc) and isinstance(b, Lit):
        return a, b
    if isinstance(b, DateTrunc) and isinstance(a, Lit):
        return b, a
    return None, None


def _rewrite_date_trunc_eq(expr: Expr) -> Expr:
    if not (isinstance(expr, Binary) and expr.op == "eq"):
        return expr
    trunc, lit = _match_trunc_and_lit(expr.left, expr.right)
    if trunc is None or not isinstance(trunc.input, Col):
        return expr
    value, unit = lit.value, trunc.unit
    # `datetime` is a subclass of `date`; both carry exact calendar arithmetic.
    if not isinstance(value, _dt.date) or isinstance(value, bool):
        return expr
    is_datetime = isinstance(value, _dt.datetime)
    if not is_datetime and unit in _TRUNC_SUBDAY:
        return expr  # truncating a date below a day is a no-op → don't change its type
    if not _trunc_aligned(value, unit, is_datetime):
        return expr  # an unaligned literal matches nothing this range would; leave it
    upper = _next_unit(value, unit)
    if upper is None:
        return expr  # an unknown unit (defensive; the accepted set is fixed above)
    return Binary(
        "and",
        Binary("ge", trunc.input, Lit(value)),
        Binary("lt", trunc.input, Lit(upper)),
    )


def _trunc_aligned(value, unit: str, is_datetime: bool) -> bool:
    """Whether `value` is already a valid `date_trunc(unit, ·)` output (every field
    finer than `unit` is zero, and day/month are at their unit start)."""
    if is_datetime:
        if unit == "second" and value.microsecond:
            return False
        if unit == "minute" and (value.second or value.microsecond):
            return False
        if unit == "hour" and (value.minute or value.second or value.microsecond):
            return False
        if unit in ("day", "month", "year") and (
            value.hour or value.minute or value.second or value.microsecond
        ):
            return False
    if unit == "month" and value.day != 1:
        return False
    return not (unit == "year" and (value.month != 1 or value.day != 1))


def _next_unit(value, unit: str):
    """`value` plus exactly one `unit`, preserving its `date`/`datetime` type. `value`
    is assumed unit-aligned, so the calendar cases never hit a missing day."""
    if unit in _TRUNC_FIXED_STEP:
        return value + _TRUNC_FIXED_STEP[unit]
    if unit == "month":
        year, month = (value.year + 1, 1) if value.month == 12 else (value.year, value.month + 1)
        return value.replace(year=year, month=month)
    if unit == "year":
        return value.replace(year=value.year + 1)
    return None


DEFAULT_REGISTRY.add(
    plan_rule(
        "date_trunc_to_range",
        Phase.NORMALIZE,
        lambda plan, _ctx: date_trunc_to_range(plan),
    )
)


def _flat_or_equalities(expr: Expr) -> tuple[str, list] | None:
    """If `expr` is `c == v1 OR c == v2 OR …` (≥2 disjuncts, one column, literal
    values), return `(column, [values])`; else None. The shape SQL `IN (...)` and
    chained `OR` equalities lower to."""
    if not (isinstance(expr, Binary) and expr.op == "or"):
        return None
    leaves: list[tuple[str, object]] = []

    def collect(e: Expr) -> bool:
        if isinstance(e, Binary) and e.op == "or":
            return collect(e.left) and collect(e.right)
        if isinstance(e, Binary) and e.op == "eq":
            left, right = e.left, e.right
            if isinstance(left, Col) and isinstance(right, Lit):
                leaves.append((left.name, right.value))
                return True
            if isinstance(right, Col) and isinstance(left, Lit):
                leaves.append((right.name, left.value))
                return True
        return False

    if not collect(expr) or len(leaves) < 2:
        return None
    cols = {name for name, _ in leaves}
    values = [v for _, v in leaves]
    if len(cols) != 1 or any(v is None or isinstance(v, bool) for v in values):
        return None
    return cols.pop(), values


@rule(name="or_to_in_and_range", phase=Phase.NORMALIZE, matches=(Filter,))
def or_to_in_and_range(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Add `c >= min AND c <= max` alongside a `c = v1 OR c = v2 OR …` conjunct.

    A disjunction of equalities (what `IN (...)` lowers to) is opaque to range-based
    zone-map pruning. Its values imply the bound `min(vs) ≤ c ≤ max(vs)`, a superset
    that — ANDed with the original disjunction — leaves the result unchanged but gives
    `zonemap_prune_filter` a range it can use to skip whole row groups (and each
    equality is still a bloom-index probe). Idempotent: the bounds are added only if
    not already present. Skipped when the literals aren't mutually comparable.
    """
    conjuncts = split_conjuncts(node.predicate)
    existing = [c.to_ir() for c in conjuncts]  # IR dicts are unhashable → list + `in`
    added: list[Expr] = []
    for conj in conjuncts:
        info = _flat_or_equalities(conj)
        if info is None:
            continue
        col_name, values = info
        try:
            lo, hi = min(values), max(values)
        except TypeError:
            continue  # values not mutually comparable (mixed types)
        for bound in (Binary("ge", Col(col_name), Lit(lo)), Binary("le", Col(col_name), Lit(hi))):
            if bound.to_ir() not in existing:
                added.append(bound)
                existing.append(bound.to_ir())
    if not added:
        return None
    return Filter(node.input, combine_conjuncts([*conjuncts, *added]))
