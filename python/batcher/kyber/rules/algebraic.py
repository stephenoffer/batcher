"""Algebraic relational rewrites — small, local, always-correct simplifications.

Each rule here is a node-local transformation registered with `@rule`: it returns
a rewritten node, or `None` to leave it unchanged. The driver supplies traversal
and fixpoint iteration, and pattern-indexes by the declared `matches` so a plan
that lacks the node type never pays for the rule. These are the bread-and-butter
rewrites every optimizer carries (filter merging, limit collapsing, distinct/limit
elimination); they shrink the plan and feed the cost-based phases cleaner input.

All are unconditionally semantics-preserving — they do not depend on cardinality or
cost — so they carry no risk of changing results, only of removing redundant work.
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase, RuleCategory
from batcher.kyber.stats.selectivity import comparison_col_side
from batcher.plan.expr_ir import Binary, Expr, Lit
from batcher.plan.expr_rewrite import combine_conjuncts, split_conjuncts, substitute_columns
from batcher.plan.logical import (
    Aggregate,
    Distinct,
    Filter,
    Limit,
    LogicalPlan,
    Project,
    Sort,
    Union,
)

__all__ = [
    "combine_limits",
    "constant_propagation",
    "eliminate_sort_before_aggregate",
    "merge_adjacent_filters",
    "prune_true_filter",
    "push_filter_into_union",
    "push_limit_into_union",
    "push_limit_through_project",
    "remove_redundant_distinct",
]


@rule(name="eliminate_sort_before_aggregate", phase=Phase.NORMALIZE, matches=(Aggregate,))
def eliminate_sort_before_aggregate(node: Aggregate, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Aggregate(Sort(x))` → `Aggregate(x)`. Grouping and the aggregate functions
    (sum/min/max/mean/median/var/std/count/n_unique) are all order-independent, so a
    sort feeding directly into a group-by is wasted work. Skipped when the sort
    carries a `limit` (a top-N changes *which* rows are aggregated)."""
    inner = node.input
    if isinstance(inner, Sort) and inner.limit is None:
        return Aggregate(inner.input, node.group_keys, node.aggregates)
    return None


@rule(name="constant_propagation", phase=Phase.NORMALIZE, matches=(Filter,))
def constant_propagation(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Propagate `col = literal` equalities into the rest of a filter conjunction.

    Within a top-level `AND`, a surviving row has `col == literal` (the equality is
    null-rejecting, so `col` is non-null and equal), which means every *other*
    conjunct can read `literal` wherever it reads `col`: `x = 5 AND y > x` →
    `x = 5 AND y > 5`. The defining equality is kept (it still does the filtering);
    the substitution exposes constants for folding and — crucially — turns
    `y > x` into a `col OP literal` shape that zone-map pruning and predicate
    pushdown can use. A conflicting equality (`x = 5 AND x = 6`) folds to an empty
    relation, which is correct.
    """
    conjuncts = split_conjuncts(node.predicate)
    consts: dict[str, Expr] = {}
    for conj in conjuncts:
        if isinstance(conj, Binary) and conj.op == "eq":
            cs = comparison_col_side(conj)
            if cs is not None:
                consts.setdefault(cs[0], Lit(cs[1]))
    if not consts:
        return None

    rewritten: list[Expr] = []
    changed = False
    for conj in conjuncts:
        if _is_defining_equality(conj, consts):
            rewritten.append(conj)  # keep `col = literal` so it still filters
            continue
        new_conj = substitute_columns(conj, consts)
        changed = changed or new_conj.to_ir() != conj.to_ir()
        rewritten.append(new_conj)
    if not changed:
        return None
    return Filter(node.input, combine_conjuncts(rewritten))


def _is_defining_equality(conj: Expr, consts: dict[str, Expr]) -> bool:
    """Whether `conj` is exactly the `col = literal` that established `consts[col]`
    (same column *and* same literal) — those are left unsubstituted so the filter
    keeps applying them; a conflicting `col = other` is substituted (and folds away)."""
    if not (isinstance(conj, Binary) and conj.op == "eq"):
        return False
    cs = comparison_col_side(conj)
    return cs is not None and cs[0] in consts and Lit(cs[1]).to_ir() == consts[cs[0]].to_ir()


@rule(name="prune_true_filter", phase=Phase.NORMALIZE, matches=(Filter,))
def prune_true_filter(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Filter(x, TRUE)` → `x`. A predicate that constant-folds to the boolean
    literal true keeps every row, so the filter is dead. Runs in NORMALIZE so it
    fires right after constant folding turns a tautology (e.g. `1 = 1`) into TRUE."""
    p = node.predicate
    if isinstance(p, Lit) and p.value is True:
        return node.input
    return None


@rule(name="push_filter_into_union", phase=Phase.PUSHDOWN, matches=(Filter,))
def push_filter_into_union(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Filter(Union(a, b, …), p)` → `Union(Filter(a, p), Filter(b, p), …)`.

    Filtering distributes over union: the rows of the union satisfying `p` are
    exactly the union of each input's rows satisfying `p`. Union inputs share an
    identical schema (enforced by `Union`), so the same predicate applies to each
    input unchanged. Holds for UNION ALL and DISTINCT alike (filter commutes with
    dedup). Pushing the filter into each branch shrinks inputs before the (possibly
    parallel/distributed) union and lets predicate pushdown continue into each
    branch independently.
    """
    inner = node.input
    if isinstance(inner, Union):
        filtered = tuple(Filter(i, node.predicate) for i in inner.inputs)
        return Union(filtered, inner.distinct)
    return None


@rule(name="merge_adjacent_filters", phase=Phase.PUSHDOWN, matches=(Filter,))
def merge_adjacent_filters(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Filter(Filter(x, a), b)` → `Filter(x, a AND b)`. One predicate evaluation
    instead of two, and it hands predicate pushdown a single conjunction to split
    across a join below."""
    inner = node.input
    if isinstance(inner, Filter):
        return Filter(inner.input, Binary("and", inner.predicate, node.predicate))
    return None


@rule(name="remove_redundant_distinct", phase=Phase.REWRITE, matches=(Distinct,))
def remove_redundant_distinct(node: Distinct, ctx: OptimizerContext) -> LogicalPlan | None:
    """Drop a `Distinct` whose input is already duplicate-free:

    - `Distinct(Distinct(x))`        → `Distinct(x)`        (idempotent)
    - `Distinct(Union(..., distinct=True))` → the union     (already dedupes)
    - `Distinct(Aggregate(...))`     → the aggregate         (one row per group key,
       and the group keys are in the output, so rows are already distinct)
    - `Distinct(x)` where `x` provably has ≤ 1 row — a 0/1-row relation cannot hold
       a duplicate, so the dedup is pure overhead (e.g. `DISTINCT` over a scalar
       aggregate). Gated on an EXACT row count so an estimate can never wrongly drop it.
    """
    inner = node.input
    if isinstance(inner, (Distinct, Aggregate)):
        return inner
    if isinstance(inner, Union) and inner.distinct:
        return inner
    if ctx is not None:
        stats = ctx.estimator.estimate(inner)
        if stats.rows <= 1 and stats.provenance.is_exact:
            return inner
    return None


@rule(
    name="combine_limits",
    phase=Phase.FUSION,
    matches=(Limit,),
    category=RuleCategory.REWRITE,
)
def combine_limits(node: Limit, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Limit(Limit(x, n_in, o_in), n_out, o_out)` → one `Limit`.

    The inner limit yields rows `[o_in : o_in + n_in)`; the outer then takes
    `[o_out : o_out + n_out)` of *those*, i.e. original rows
    `[o_in + o_out : …)` for at most `min(n_out, n_in - o_out)` rows.
    """
    inner = node.input
    if isinstance(inner, Limit):
        new_offset = inner.offset + node.offset
        new_n = max(0, min(node.n, inner.n - node.offset))
        return Limit(inner.input, new_n, new_offset)
    return None


@rule(name="push_limit_through_project", phase=Phase.PUSHDOWN, matches=(Limit,))
def push_limit_through_project(node: Limit, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Limit(Project(x))` → `Project(Limit(x))`. Projection is row- and order-
    preserving, so the top-N of a projection is the projection of the top-N — and
    limiting first means fewer rows flow through the (possibly expensive)
    projection."""
    inner = node.input
    if isinstance(inner, Project):
        return Project(Limit(inner.input, node.n, node.offset), inner.items)
    return None


@rule(name="push_limit_into_union", phase=Phase.PUSHDOWN, matches=(Limit,))
def push_limit_into_union(node: Limit, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Limit(UNION ALL(a, b, …), n)` → `Limit(UNION ALL(Limit(a, n), …), n)`.

    UNION ALL concatenates its inputs, so the top-N of the union never needs more
    than N rows from any single input — cap each, keep the outer limit to take the
    true top-N. Restricted to non-distinct unions (dedup changes counts) at offset 0
    (an offset spans the concatenation). The guard against already-capped inputs
    keeps the rule idempotent (it fires once, then leaves the plan at a fixpoint).
    """
    inner = node.input
    if (
        isinstance(inner, Union)
        and not inner.distinct
        and node.offset == 0
        and not any(isinstance(i, Limit) for i in inner.inputs)
    ):
        capped = tuple(Limit(i, node.n, 0) for i in inner.inputs)
        return Limit(Union(capped, distinct=False), node.n, 0)
    return None
