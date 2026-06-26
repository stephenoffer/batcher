"""Join rewrites — change a join's type, push aggregates below it, and prune a side.

`outer_to_inner_join` strengthens an outer join to a less-preserving form (or to
an inner join) when a filter above it rejects the null-extended rows that the
outer join exists to keep. A `LEFT JOIN` keeps unmatched left rows by padding the
right columns with nulls; if a downstream predicate is *null-rejecting* on a right
column — guaranteed false or null whenever that column is null — those padded rows
cannot survive, so the outer join produces the same result as an inner join. The
rewrite matters because it unblocks the rest of the pipeline: an inner join can
have predicates pushed into either side and its build side swapped, neither of
which is safe across an outer join's preserved side.

The analysis is deliberately conservative: a column is treated as null-rejecting
only through the constructs that provably propagate a null up to a false/null
result (comparisons, `IS NOT NULL`, and the null-propagating scalar functions),
combined with `AND`/`OR` the standard way. Anything it cannot prove leaves the
join untouched — correctness over an extra rewrite.
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase, RuleCategory
from batcher.plan.expr_ir import (
    Binary,
    Cast,
    Col,
    Expr,
    IsNotNull,
    IsNull,
    Lit,
    Math2Expr,
    MathExpr,
    Not,
    referenced_columns,
    remap_columns,
)
from batcher.plan.logical import (
    Aggregate,
    Distinct,
    Filter,
    Join,
    LogicalPlan,
    Project,
    Projection,
)
from batcher.plan.stats import ColumnStat, Provenance

__all__ = [
    "eliminate_left_join",
    "join_to_semijoin",
    "outer_to_inner_join",
    "runtime_join_filter",
]


@rule(name="join_to_semijoin", phase=Phase.REWRITE, matches=(Distinct,))
def join_to_semijoin(node: Distinct, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Distinct(Project_left-only(Inner Join(L, R)))` → `Distinct(Project(SemiJoin(L, R)))`.

    When a deduplicated projection over an inner join reads only the left side, the
    right side is used purely to test existence — a semi-join. This is correct with no
    uniqueness precondition: an inner join repeats each left row once per matching
    right row, but the enclosing `Distinct` collapses those duplicates, so the result
    equals the deduplicated left rows that have a match — exactly the semi-join. The
    win is avoiding the inner join's fan-out (and materializing right columns). Returns
    None unless the projection touches no right-side output column.
    """
    proj = node.input
    if not isinstance(proj, Project):
        return None
    join = proj.input
    if not isinstance(join, Join) or join.join_type != "inner":
        return None
    left_aliases = {o.alias for o in join.output if o.side == "left"}
    right_aliases = {o.alias for o in join.output if o.side == "right"}
    used: set[str] = set()
    for item in proj.items:
        used |= referenced_columns(item.expr)
    if (used & right_aliases) or not (used <= left_aliases):
        return None
    # Semi-join output is the left-side columns the projection still needs (same aliases).
    new_output = tuple(o for o in join.output if o.side == "left")
    semi = Join(
        join.left,
        join.right,
        join.left_keys,
        join.right_keys,
        "semi",
        new_output,
        join.strategy,
    )
    return Distinct(Project(semi, proj.items))


# Comparisons and arithmetic propagate a null operand to a null result; boolean
# `and`/`or` do not (three-valued logic: `true OR null = true`).
_NULL_PROPAGATING_BINARY = frozenset(
    {"lt", "le", "gt", "ge", "eq", "ne", "add", "sub", "mul", "div", "mod"}
)
_COMPARISONS = frozenset({"lt", "le", "gt", "ge", "eq", "ne"})


@rule(name="outer_to_inner_join", phase=Phase.REWRITE, matches=(Filter,))
def outer_to_inner_join(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Filter(Join(L, R, outer), p)` → the same filter over a stronger join when
    `p` rejects the join's null-extended rows.

    A null-rejecting predicate on the right (null-supplied) side collapses a
    `left` join to `inner`; symmetrically on the left for a `right` join. A `full`
    join, null-extended on both sides, weakens one preserved side per rejecting
    side (and to `inner` when both reject). Returns None for inner/semi/anti joins
    and when nothing is provably rejected (so the rule is idempotent).
    """
    inner = node.input
    if not isinstance(inner, Join) or inner.join_type not in {"left", "right", "full"}:
        return None

    rejected = _null_rejecting_cols(node.predicate)
    if not rejected:
        return None
    left_aliases = {o.alias for o in inner.output if o.side == "left"}
    right_aliases = {o.alias for o in inner.output if o.side == "right"}
    rejects_left = bool(rejected & left_aliases)
    rejects_right = bool(rejected & right_aliases)

    new_type = _strengthened(inner.join_type, rejects_left, rejects_right)
    if new_type == inner.join_type:
        return None
    new_join = Join(
        inner.left,
        inner.right,
        inner.left_keys,
        inner.right_keys,
        new_type,
        inner.output,
        inner.strategy,
    )
    return Filter(new_join, node.predicate)


def _strengthened(join_type: str, rejects_left: bool, rejects_right: bool) -> str:
    """The strongest join type still producing the filtered result.

    For an outer join, rejecting nulls on one side removes the rows that side
    null-extends: a `left` join's null-extended rows live on the right, so a
    right-side rejection makes it inner. A `full` join null-extends left-only rows
    on the right and right-only rows on the left; a right-side rejection drops the
    left-only rows (leaving a `right` join), a left-side rejection drops the
    right-only rows (leaving a `left` join), and both drop everything unmatched
    (inner).
    """
    if join_type == "left":
        return "inner" if rejects_right else "left"
    if join_type == "right":
        return "inner" if rejects_left else "right"
    # full
    if rejects_left and rejects_right:
        return "inner"
    if rejects_right:
        return "right"
    if rejects_left:
        return "left"
    return "full"


def _null_rejecting_cols(expr: Expr) -> set[str]:
    """Columns whose nullity guarantees `expr` is not true (i.e. false or null).

    These are the columns an outer join can stop preserving: if the predicate is
    never true when such a column is null, the join's null-extended rows (which set
    that column to null) are filtered out anyway.
    """
    if isinstance(expr, Binary):
        if expr.op == "and":  # either conjunct rejecting → the conjunction rejects
            return _null_rejecting_cols(expr.left) | _null_rejecting_cols(expr.right)
        if expr.op == "or":  # both disjuncts must reject for the disjunction to
            return _null_rejecting_cols(expr.left) & _null_rejecting_cols(expr.right)
        if expr.op in _COMPARISONS:  # a null operand makes the comparison null
            return _null_propagating_cols(expr.left) | _null_propagating_cols(expr.right)
        return set()
    if isinstance(expr, IsNotNull):  # false when its argument is null
        return _null_propagating_cols(expr.input)
    # IS NULL is null-accepting; Not()/Case/Coalesce are not provably rejecting.
    return set()


@rule(name="eliminate_left_join", phase=Phase.PUSHDOWN, matches=(Project, Join))
def eliminate_left_join(node: LogicalPlan, ctx: OptimizerContext) -> LogicalPlan | None:
    """Drop a `LEFT JOIN` to a unique, unused right side.

    A left join keeps every L row; if R is provably unique on the join key, each L row
    matches at most one R row, so the output is exactly L's rows — and if no R column
    is read, R contributes nothing and the join is dead weight. The classic
    redundant-dimension-join elimination (common in generated / view SQL).

    Fires in two shapes: `Project(LEFT JOIN, <only L columns>)` (the projection reads
    no R column), and a bare `LEFT JOIN` whose output column-pruning has already
    dropped every R column (so its output is all-left). Both rewrite to read L
    directly. Uniqueness must be *proven*, never estimated — a duplicate R key would
    multiply L rows; the provable cases are a `GROUP BY`/`DISTINCT` over the join keys
    (structural) or an EXACT distinct count equal to the EXACT row count. Inner joins
    are excluded (they also drop unmatched L rows, needing an FK guarantee we lack).
    """
    if isinstance(node, Join):
        if node.join_type != "left" or any(o.side == "right" for o in node.output):
            return None
        if not _right_unique_on_keys(node, ctx):
            return None
        # Output is all-left → read those columns straight from L.
        items = tuple(Projection(o.alias, Col(o.name)) for o in node.output)
        return Project(node.left, items)

    join = node.input
    if not isinstance(join, Join) or join.join_type != "left":
        return None
    right_aliases = {o.alias for o in join.output if o.side == "right"}
    used: set[str] = set()
    for item in node.items:
        used |= referenced_columns(item.expr)
    if used & right_aliases:
        return None  # the projection reads an R column → R is needed
    if not _right_unique_on_keys(join, ctx):
        return None
    # Rewrite the projection to read L's source columns directly, dropping the join.
    left_src = {o.alias: o.name for o in join.output if o.side == "left"}
    items = tuple(Projection(it.alias, remap_columns(it.expr, left_src)) for it in node.items)
    return Project(join.left, items)


def _right_unique_on_keys(join: Join, ctx: OptimizerContext) -> bool:
    """Whether the join's right input is *provably* unique on the right keys."""
    right, keys = join.right, set(join.right_keys)
    # Structural: a GROUP BY whose keys are all join keys yields one row per key combo;
    # a DISTINCT over exactly the join key columns likewise.
    if isinstance(right, Aggregate) and {k.alias for k in right.group_keys} <= keys:
        return True
    if isinstance(right, Distinct) and set(right.available_columns()) == keys:
        return True
    # Metadata: an EXACT distinct count equal to the EXACT row count (rare but solid).
    if len(join.right_keys) == 1:
        stats = ctx.estimator.estimate(right)
        col = stats.column(join.right_keys[0])
        if (
            stats.rows_exact
            and col.provenance is Provenance.EXACT
            and col.ndv is not None
            and col.ndv >= stats.rows
        ):
            return True
    return False


def _null_propagating_cols(expr: Expr) -> set[str]:
    """Columns whose nullity forces `expr` itself to null.

    A bare column propagates its own null; comparisons, arithmetic, casts, and the
    unary/binary math functions propagate a null operand. Constructs that can turn
    a null input into a non-null result — `COALESCE`, `CASE`, `IS NULL`,
    `GREATEST`/`LEAST` — propagate nothing and are conservatively excluded.
    """
    if isinstance(expr, Col):
        return {expr.name}
    if isinstance(expr, Binary) and expr.op in _NULL_PROPAGATING_BINARY:
        return _null_propagating_cols(expr.left) | _null_propagating_cols(expr.right)
    if isinstance(expr, (Cast, MathExpr)):
        return _null_propagating_cols(expr.input)
    if isinstance(expr, Math2Expr):
        return _null_propagating_cols(expr.left) | _null_propagating_cols(expr.right)
    if isinstance(expr, (IsNull, IsNotNull, Not)):
        # These yield a non-null boolean from a null input — null does not propagate.
        return set()
    return set()


# --- Runtime join filters (sideways information passing) --------------------

# Which side(s) of each join type may be safely reduced by the *other* side's key
# range. "left"/"right" name the side that receives the filter. A side is filterable
# only when its unmatched rows are not required in the output: an outer join's
# preserved side and an anti join's left side must keep their unmatched rows.
_FILTERABLE_SIDES = {
    "inner": ("left", "right"),
    "semi": ("left", "right"),  # semi emits a left row only if it matches → both prunable
    "anti": ("right",),  # left rows without a match MUST survive; only prune the right
    "left": ("right",),  # left rows are preserved; only prune the right
    "right": ("left",),  # right rows are preserved; only prune the left
    # "full" preserves both sides → nothing is safely prunable.
}


@rule(
    name="runtime_join_filter",
    phase=Phase.ENFORCE,
    matches=(Join,),
    category=RuleCategory.ENFORCE,
)
def runtime_join_filter(node: Join, ctx: OptimizerContext) -> LogicalPlan | None:
    """Push a `key BETWEEN other_min AND other_max` filter onto a prunable join side.

    For an equi-join every matching row has equal keys, so a row whose key falls
    outside the *other* side's `[min, max]` range can never match. Pushing that range
    onto the opposite input is a superset filter — it drops only provably-non-matching
    rows, never a real match — the cheap form of the sideways-information-passing /
    bloom-filter join pruning DuckDB and Spark AQE rely on, available here purely from
    the `ColumnStat.min`/`max` Kyber already propagates. When the prunable side is a
    scan, the added `Filter` is captured by `required_predicates_per_source` at
    lowering and pushed to the source, so zonemaps prune whole row-groups / Hive
    partitions — dynamic partition pruning with no new IR node.

    Multi-key joins are handled per key: a matching row must fall inside the other
    side's range on **every** key, so each narrowing key contributes a `BETWEEN`
    conjunct (`k1 BETWEEN .. AND k2 BETWEEN ..`). Runs once in ENFORCE (after physical
    selection) so it never re-adds, and fires only when bounds are known *and*
    genuinely narrower (so the filter prunes rather than adds overhead), on a side the
    join does not preserve.
    """
    sides = _FILTERABLE_SIDES.get(node.join_type)
    if sides is None or not node.left_keys or len(node.left_keys) != len(node.right_keys):
        return None
    left_stats = ctx.estimator.estimate(node.left)
    right_stats = ctx.estimator.estimate(node.right)

    # Per side, collect a BETWEEN conjunct for every key the opposite range narrows.
    right_preds: list[Expr] = []
    left_preds: list[Expr] = []
    for lk, rk in zip(node.left_keys, node.right_keys, strict=True):
        left_col = left_stats.column(lk)
        right_col = right_stats.column(rk)
        if "right" in sides and _narrows(left_col, right_col):
            right_preds.append(_between(rk, left_col))
        if "left" in sides and _narrows(right_col, left_col):
            left_preds.append(_between(lk, right_col))

    new_left, new_right = node.left, node.right
    changed = False
    if right_preds:
        new_right = Filter(new_right, _conjoin(right_preds))
        changed = True
    if left_preds:
        new_left = Filter(new_left, _conjoin(left_preds))
        changed = True
    if not changed:
        return None
    ctx.notes.setdefault("runtime_join_filters", []).append(node.join_type)
    return Join(
        new_left,
        new_right,
        node.left_keys,
        node.right_keys,
        node.join_type,
        node.output,
        node.strategy,
    )


def _narrows(source: ColumnStat, target: ColumnStat) -> bool:
    """Whether `source`'s key range is known and strictly inside `target`'s — so a
    `target BETWEEN source.min AND source.max` filter would actually drop rows.

    Both ranges must be known: without the target's spread we cannot tell the filter
    is selective, and adding a non-selective filter is pure overhead.
    """
    if source.min is None or source.max is None or target.min is None or target.max is None:
        return False
    try:
        return source.min > target.min or source.max < target.max
    except TypeError:
        return False  # incomparable bound types → leave the join untouched


def _between(column: str, bounds: ColumnStat) -> Expr:
    """`column >= bounds.min AND column <= bounds.max`."""
    col = Col(column)
    return (col >= Lit(bounds.min)) & (col <= Lit(bounds.max))


def _conjoin(preds: list[Expr]) -> Expr:
    """AND a non-empty list of predicates (a single predicate is returned as-is)."""
    out = preds[0]
    for pred in preds[1:]:
        out = out & pred
    return out
