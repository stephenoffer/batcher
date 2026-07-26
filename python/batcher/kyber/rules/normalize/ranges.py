"""Predicate → sargable-range rewrites in the NORMALIZE phase.

A pure-prefix `LIKE 'abc%'`, a `date_trunc(unit, col) = lit`, and an `OR` chain of
equalities on one column are each *exactly* equivalent to plain comparisons or an
`IN` list. Zone-map pruning and source predicate pushdown are blind to the original
forms and can use the rewritten ones, so the rewrite turns an opaque scan into a
skippable one.
"""

from __future__ import annotations

import datetime as _dt

from batcher._internal.mathx import is_nan
from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import DEFAULT_REGISTRY, rule
from batcher.kyber.rule import Phase, plan_rule
from batcher.plan.expr_ir import Binary, Col, Expr, InList, Lit
from batcher.plan.expr_ir.namespaces import DateTrunc, StrFunc
from batcher.plan.expr_rewrite import (
    combine_conjuncts,
    expr_key,
    map_node_expressions,
    split_conjuncts,
)
from batcher.plan.logical import Filter, LogicalPlan
from batcher.plan.visitor import transform_up

__all__ = [
    "date_trunc_to_range",
    "len_zero_to_empty_string",
    "like_prefix_to_range",
    "or_equalities_to_in_list",
    "or_to_in_and_range",
    "prefix_predicates_to_range",
]


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


# --- starts_with / substr prefix → range ------------------------------------


def prefix_predicates_to_range(plan: LogicalPlan) -> LogicalPlan:
    """Rewrite `starts_with(col, 'abc')` and `substr(col, 1, 3) = 'abc'` to the same
    exact range `col >= 'abc' AND col < 'abd'` that `LIKE 'abc%'` already gets.

    `like_prefix_to_range` has handled the SQL spelling for a while; these are the two
    *DataFrame* spellings of the identical predicate, and neither was rewritten. So
    `ds.filter(col("s").str.starts_with("abc"))` — the idiom a Batcher user actually
    writes — scanned every row group while the SQL form skipped them. The asymmetry was
    invisible because both return the right rows.

    Both equivalences are exact:

    * a string starts with `p` iff it lies in `[p, next(p))`, which is the same argument
      `like_prefix_to_range` rests on;
    * `substr(s, 1, n) = lit` (with `len(lit) == n`) is the same statement — a shorter `s`
      yields a shorter substring that cannot equal `lit`, and is also below `lit` in the
      range, so both forms exclude it.

    NULL is preserved: `starts_with`/`substr`/`=` are all null-propagating, and so is the
    conjunction of two comparisons. Same conservative guard as the `LIKE` rule — the
    prefix must be non-empty and end in a safely incrementable ASCII character, since the
    upper bound is built by incrementing that character.
    """
    return transform_up(plan, lambda node: map_node_expressions(node, _rewrite_prefix))


def _rewrite_prefix(expr: Expr) -> Expr:
    prefix, column = _prefix_and_column(expr)
    if prefix is None or column is None:
        return expr
    upper = _increment_prefix(prefix)
    if upper is None:
        return expr
    return Binary("and", Binary("ge", column, Lit(prefix)), Binary("lt", column, Lit(upper)))


def _prefix_and_column(expr: Expr) -> tuple[str | None, Expr | None]:
    """`(prefix, column)` for the two prefix spellings, else `(None, None)`."""
    if isinstance(expr, StrFunc) and expr.fn == "starts_with" and isinstance(expr.pattern, str):
        return expr.pattern, expr.input
    # `substr(col, 1, n) = lit` — only from the first character, and only when the
    # literal is exactly `n` characters. A shorter literal makes the predicate
    # unsatisfiable and a longer one makes it constant-false; neither is a prefix test,
    # and folding them here would hide a user's mistake behind a range.
    if not (isinstance(expr, Binary) and expr.op == "eq"):
        return None, None
    for value, other in ((expr.left, expr.right), (expr.right, expr.left)):
        if (
            isinstance(value, StrFunc)
            and value.fn == "substr"
            and value.start == 1
            and isinstance(other, Lit)
            and isinstance(other.value, str)
            and value.length == len(other.value)
        ):
            return other.value, value.input
    return None, None


def _increment_prefix(prefix: str) -> str | None:
    """The exclusive upper bound for a literal prefix, or None if it is not safe.

    Shares `_MAX_INCREMENTABLE` with the `LIKE` rule so the two cannot disagree about
    which prefixes are incrementable.
    """
    if not prefix or ord(prefix[-1]) > _MAX_INCREMENTABLE:
        return None
    return prefix[:-1] + chr(ord(prefix[-1]) + 1)


DEFAULT_REGISTRY.add(
    plan_rule(
        "prefix_predicates_to_range",
        Phase.NORMALIZE,
        lambda plan, _ctx: prefix_predicates_to_range(plan),
    )
)


# --- len(col) = 0 → col = '' ------------------------------------------------


def len_zero_to_empty_string(plan: LogicalPlan) -> LogicalPlan:
    """Rewrite `len(col) = 0` to `col = ''` (and `len(col) <> 0` to `col <> ''`).

    A length test wraps the column in a function, which makes it opaque to zone-map
    pruning, bloom probing and source pushdown — every one of which needs a bare `Col` on
    one side. The equality is the same predicate in a form all three can use.

    Exact: a string has zero characters iff it is the empty string, and both spellings
    are null-propagating, so a null row is null either way.
    """
    return transform_up(plan, lambda node: map_node_expressions(node, _rewrite_len_zero))


def _rewrite_len_zero(expr: Expr) -> Expr:
    if not (isinstance(expr, Binary) and expr.op in ("eq", "ne")):
        return expr
    for value, other in ((expr.left, expr.right), (expr.right, expr.left)):
        if (
            isinstance(value, StrFunc)
            and value.fn == "len"
            and isinstance(other, Lit)
            # `type(...) is int` because `True` is an `int` subclass and `len(s) = True`
            # is not this predicate.
            and type(other.value) is int
            and other.value == 0
        ):
            return Binary(expr.op, value.input, Lit(""))
    return expr


DEFAULT_REGISTRY.add(
    plan_rule(
        "len_zero_to_empty_string",
        Phase.NORMALIZE,
        lambda plan, _ctx: len_zero_to_empty_string(plan),
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
    if len(cols) != 1 or any(_bad_range_literal(v) for v in values):
        return None
    return cols.pop(), values


def _in_list_values(expr: Expr) -> tuple[str, list] | None:
    """`(column, values)` for a `col IN (…)` over literals, else None.

    `or_to_in_and_range` must see this shape as well as the `OR` chain, because
    `or_equalities_to_in_list` folds the chain into an `InList` in the same phase — and
    whichever of the two runs first, the bounds this rule derives must still be derived.
    Reading only the `OR` form made the fold silently *remove* the zone-map bounds.
    """
    if not (isinstance(expr, InList) and isinstance(expr.input, Col)):
        return None
    values = list(expr.values)
    if len(values) < 2 or any(_bad_range_literal(v) for v in values):
        return None
    return expr.input.name, values


def _bad_range_literal(value: object) -> bool:
    """Whether a literal cannot participate in a min/max range bound.

    ``NULL`` and booleans are excluded (a range over them is meaningless), and so is
    **NaN**: the engine's ``=`` matches a NaN row (``col = NaN`` is TRUE where ``col`` is
    NaN), but ``NaN >= lo`` / ``NaN <= hi`` are both FALSE — so a range bound derived from a
    disjunction containing ``col = NaN`` would drop the very NaN rows that disjunct keeps,
    and Python's ``min``/``max`` over a list containing NaN is order-dependent garbage (a
    leading NaN yields ``col >= NaN``, which rejects every row). Such a disjunction gets no
    range bound at all — the original ``OR`` still selects exactly the right rows.
    """
    if value is None or isinstance(value, bool):
        return True
    return is_nan(value)


@rule(name="or_equalities_to_in_list", phase=Phase.NORMALIZE, matches=(Filter,))
def or_equalities_to_in_list(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Fold `c = v1 OR c = v2 OR …` into `c IN (v1, v2, …)` (DuckDB's
    `contains_to_in_clause`).

    This pays twice, and the second time is the larger one.

    At runtime the disjunction is *n* sequential comparisons over the whole column,
    while `InList` lowers to one hash-set probe per row (`bc_expr::eval::in_list`,
    which builds an `ahash` set once per operator). That is the direct win.

    The compounding one is that this repository already carries a family of rules keyed
    on `InList` — `prune_in_list_by_zonemap`, `prune_in_list_by_bloom`, `dedup_in_list`,
    `refine_in_list_by_equality`/`_comparison`/`_neq`, `intersect_in_lists`,
    `push_in_list_across_join_keys` — **none of which can fire on the `OR` form**. A user
    who writes the chain (or whose ORM does) got none of them. Producing the node they
    match is what turns eight existing rules on.

    Exactly equivalent, including NULL: `x = 1 OR x = 2` and `x IN (1, 2)` both yield
    NULL for a null `x` (verified against DuckDB, which agrees on both forms), so this is
    safe in a projection as well as under a filter — though it only matches `Filter`,
    where the pruning rules that consume it live.

    `or_to_in_and_range` is the sibling rewrite and is *not* superseded: it derives
    `min ≤ c ≤ max` bounds from the same shape, which zone maps use directly. Both fire;
    the bounds are added to the conjunction and the disjunct itself becomes the `IN`.
    """
    conjuncts = split_conjuncts(node.predicate)
    rewritten: list[Expr] = []
    changed = False
    for conj in conjuncts:
        info = _flat_or_equalities(conj)
        if info is None:
            rewritten.append(conj)
            continue
        col_name, values = info
        # First-occurrence order, deduplicated: the set semantics are the same and a
        # stable order keeps the rewrite idempotent (`dedup_in_list` would otherwise
        # rewrite it again on the next fixpoint iteration).
        deduped = tuple(dict.fromkeys(values))
        if len(deduped) < 2:
            # One distinct value is an equality, not a membership test; leave it for
            # the equality rules rather than wrapping a single value in a hash set.
            rewritten.append(conj)
            continue
        rewritten.append(InList(Col(col_name), deduped))
        changed = True
    if not changed:
        return None
    return Filter(node.input, combine_conjuncts(rewritten))


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
    # Keyed by `expr_key` (the canonical, memoized IR serialization) rather than by the
    # raw IR dict: dicts are unhashable, so the presence test used to be a linear scan of
    # a list with a full dict comparison at each step — quadratic in the conjunct count on
    # exactly the wide `OR` chains this rule exists to fold.
    existing = {expr_key(c) for c in conjuncts}
    added: list[Expr] = []
    for conj in conjuncts:
        info = _flat_or_equalities(conj) or _in_list_values(conj)
        if info is None:
            continue
        col_name, values = info
        try:
            lo, hi = min(values), max(values)
        except TypeError:
            continue  # values not mutually comparable (mixed types)
        for bound in (Binary("ge", Col(col_name), Lit(lo)), Binary("le", Col(col_name), Lit(hi))):
            key = expr_key(bound)
            if key not in existing:
                added.append(bound)
                existing.add(key)
    if not added:
        return None
    return Filter(node.input, combine_conjuncts([*conjuncts, *added]))
