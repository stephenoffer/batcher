"""Algebraic relational identities — small, local, always-correct simplifications.

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

import dataclasses

from batcher._internal.mathx import clamp
from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase, RuleCategory
from batcher.kyber.stats.selectivity import comparison_col_side
from batcher.plan.expr_ir import Binary, Expr, Lit
from batcher.plan.expr_rewrite import (
    combine_conjuncts,
    split_conjuncts,
    substitute_columns,
)
from batcher.plan.logical import (
    Aggregate,
    Distinct,
    Filter,
    Limit,
    LogicalPlan,
    Project,
    Sample,
    Sort,
    Union,
)
from batcher.plan.stats import Provenance

__all__ = [
    "collapse_full_key_distinct",
    "combine_limits",
    "constant_propagation",
    "eliminate_sort_before_aggregate",
    "merge_adjacent_filters",
    "prune_true_filter",
    "push_distinct_into_union",
    "push_filter_into_union",
    "push_limit_into_union",
    "push_limit_through_project",
    "remove_redundant_distinct",
]


@rule(name="push_distinct_into_union", phase=Phase.REWRITE, matches=(Distinct,))
def push_distinct_into_union(node: Distinct, ctx: OptimizerContext) -> LogicalPlan | None:
    """`Distinct(Union-all(a, b, …))` → `Distinct(Union-all(…, Distinct(branch), …))`.

    Deduplicating a branch *before* the union shrinks what the still-required outer
    `Distinct` (and a distributed union shuffle) must carry, with no change to the result:
    dedup of a concatenation equals dedup of the per-branch-deduped concatenation. A branch
    is deduplicated early only when learned statistics show it genuinely has duplicates
    (its distinct-row estimate is ≥10% below its row count), so the rule never adds a
    speculative breaker — branches already producing distinct rows (`Distinct`/`Aggregate`)
    and low-duplication branches are left untouched, and it does nothing if none qualify.
    Only for `UNION ALL` (a `UNION`-distinct already deduplicates).
    """
    inner = node.input
    if not isinstance(inner, Union) or inner.distinct:
        return None
    new_inputs = []
    changed = False
    for branch in inner.inputs:
        # Deliberately *without* `node.limit`: the outer dedup's row cap is a property of the
        # concatenation, and applying it per branch would discard rows the union still needs.
        early = Distinct(branch, node.keys, node.order)
        if not isinstance(branch, Distinct | Aggregate) and _dedup_shrinks(ctx, branch, early):
            new_inputs.append(early)
            changed = True
        else:
            new_inputs.append(branch)
    if not changed:
        return None
    # `replace`, so the *outer* dedup keeps every field it had. The positional rebuild dropped
    # `limit`, and `fuse_limit_into_distinct` is in this same phase -- so a `DISTINCT ... LIMIT`
    # over a `UNION ALL` could have its cap fused in and then silently discarded here within one
    # fixpoint. The result stayed correct (that rule leaves the `Limit` node above), but the
    # early exit it exists to enable was gone, which is exactly the kind of loss no test sees.
    return dataclasses.replace(node, input=Union(tuple(new_inputs), distinct=False))


def _dedup_shrinks(ctx: OptimizerContext, branch: LogicalPlan, deduped_branch: Distinct) -> bool:
    """Whether learned statistics show `branch` has enough duplicate rows that
    deduplicating it early is worthwhile (≥10% fewer rows) — so `push_distinct_into_union`
    only fires on real evidence, never a guess."""
    rows = ctx.estimator.estimate(branch).rows
    deduped = ctx.estimator.estimate(deduped_branch)
    return deduped.provenance == Provenance.LEARNED and deduped.rows <= rows * 0.9


# Aggregates whose result is provably independent of the order their rows arrive in.
# Grouping itself is order-independent, but the aggregate functions are not all so:
#
#   * `list_agg` collects its values *in arrival order* — dropping a sort below it
#     silently returns a differently-ordered list (a wrong result, not a slower plan).
#   * `arg_min`/`arg_max` (and the `first`/`last` that lower to them) and `mode` break
#     ties by arrival order, so a sort decides which of several equal candidates wins.
#   * `approx_quantile` feeds a KLL sketch whose sampling depends on insertion order, so
#     the approximation itself changes.
#
# Anything not listed is treated as order-sensitive: a new aggregate must opt *in*, so
# adding one can never silently turn this rewrite into a wrong-result bug.
_ORDER_INSENSITIVE_AGGS = frozenset(
    {
        "sum",
        "product",
        "count",
        "count_distinct",
        "approx_count_distinct",
        "min",
        "max",
        "mean",
        "median",
        "quantile",
        "var",
        "stddev",
        "skewness",
        "kurtosis",
        "bit_and",
        "bit_or",
        "bit_xor",
        "bool_and",
        "bool_or",
        "histogram",
    }
)


def _sort_under_order_indifferent(input_: LogicalPlan) -> LogicalPlan | None:
    """`input_` with a dead `Sort` removed, or None if there is no removable sort.

    The aggregate above this is the only consumer, and its output order is unspecified, so
    a sort anywhere below is dead — **provided** nothing between the two can see the order.
    An intervening `Sample` cannot: a row is kept by a stable seeded hash of its *values*,
    so the sampled multiset is identical whatever order the rows arrive in. That is the
    argument the deleted `eliminate_sort_before_sample` made, and it is sound here (where
    the order is genuinely unobservable) and unsound there (where it was the user's
    output).

    A `Sort` carrying a `limit` is a top-N and stays: it selects *which* rows survive, so
    removing it changes the multiset the aggregate sees.

    Args:
        input_: The aggregate's input.

    Returns:
        The rewritten input, or None when nothing can be removed.
    """
    if isinstance(input_, Sort):
        return None if input_.limit is not None else input_.input
    if isinstance(input_, Sample):
        rewritten = _sort_under_order_indifferent(input_.input)
        return None if rewritten is None else dataclasses.replace(input_, input=rewritten)
    return None


@rule(name="eliminate_sort_before_aggregate", phase=Phase.NORMALIZE, matches=(Aggregate,))
def eliminate_sort_before_aggregate(node: Aggregate, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Aggregate(Sort(x))` → `Aggregate(x)` when every aggregate is order-independent.

    A sort feeding into a group-by is wasted work — but only for aggregates whose value
    cannot see the row order (`_ORDER_INSENSITIVE_AGGS`). `list_agg` collects in arrival
    order, and `arg_min`/`arg_max`/`mode` break ties by it, so for those the sort is
    load-bearing and must survive.

    The sort need not be *directly* beneath: an intervening `Sample` selects rows by a hash
    of their values and so cannot observe the order either, and `Aggregate(Sample(Sort(x)))`
    is the shape a sampled aggregation actually takes.

    Skipped when the sort carries a `limit` (a top-N changes *which* rows are aggregated),
    and when any aggregate carries a secondary `input2` order key.
    """
    if any(
        spec.agg.func not in _ORDER_INSENSITIVE_AGGS or spec.agg.input2 is not None
        for spec in node.aggregates
    ):
        return None
    rewritten = _sort_under_order_indifferent(node.input)
    if rewritten is None:
        return None
    return dataclasses.replace(node, input=rewritten)


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
    across a join below.

    The merged predicate is built by flattening both sides' conjuncts and recombining
    them into a *balanced* `AND` tree (`combine_conjuncts`). A naive `a AND b` would nest
    one level deeper on every merge, so a long filter chain (hundreds of stacked
    `.filter(...)` calls — common in generated/programmatic pipelines) would build a
    predicate deep enough to exceed the data plane's IR-deserialization recursion limit.
    Balancing keeps the depth at O(log n) so the chain collapses safely out of the box."""
    inner = node.input
    if isinstance(inner, Filter):
        conjuncts = split_conjuncts(inner.predicate) + split_conjuncts(node.predicate)
        return Filter(inner.input, combine_conjuncts(conjuncts))
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
    - `Distinct(x, keys=K)` where `K` is provably a **key** of `x` — unique and never
       null — so no two rows share a `K` and the dedup removes nothing. This is the one
       case that can delete the operator outright rather than merge it, and it is the
       common shape of a defensive `distinct(["id"])` written over a table that already
       has `id` as its primary key. Gated on `is_key`, which needs an exact distinct
       count *and* an exact null count: a unique-but-nullable key still collapses its
       null rows, and an estimated ndv could drop a dedup that was doing work.

    Only the last case applies to a keyed dedup. The first three would be wrong for one:
    the inner relation being duplicate-*free* says nothing about it being free of duplicate
    **keys**, so `distinct(["k"])` over an aggregate grouped by `(k, other)` genuinely has
    work to do. That is a wrong answer rather than a slower plan, and nothing about the
    plan's shape reveals it.
    """
    inner = node.input
    if not node.keys:
        if isinstance(inner, (Distinct, Aggregate)):
            return inner
        if isinstance(inner, Union) and inner.distinct:
            return inner
    # An identical keyed dedup is still idempotent: same key, same ordering, same survivor.
    if (
        node.keys
        and isinstance(inner, Distinct)
        and inner.keys == node.keys
        and inner.order == node.order
    ):
        return inner
    if ctx is not None:
        stats = ctx.estimator.estimate(inner)
        if stats.rows <= 1 and stats.provenance.is_exact:
            return inner
        if node.keys and _a_key_column_is_unique(stats, node.keys):
            return inner
    return None


def _a_key_column_is_unique(stats, keys: tuple[str, ...]) -> bool:
    """Whether some column of `keys` is provably a key of the relation `stats` describes.

    One suffices: if `k` alone is unique and never null then no two rows agree on `k`, so
    they cannot agree on a *superset* of it either. Asking about one column at a time is also
    the only question the statistics can answer exactly — a per-column ndv says nothing
    exact about a composite's distinct count, which is why `combine_ndv` exists and damps.
    """
    from batcher.kyber.shortcuts.distinct import is_key
    from batcher.kyber.shortcuts.facts import facts_from_relstats

    facts = facts_from_relstats(stats)
    return any(is_key(facts, k) for k in keys)


@rule(name="collapse_full_key_distinct", phase=Phase.REWRITE, matches=(Distinct,))
def collapse_full_key_distinct(node: Distinct, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Distinct(x, keys=every column)` → `Distinct(x)` — the whole-row form of itself.

    When the key covers every column there is no payload, so nothing distinguishes the rows
    that collapse and no ordering can pick between them: the two forms return the same
    relation. The whole-row form is the cheaper one to execute. It reaches a presence-bitmap
    pass over a dense integer column and a single-pass hash-partitioned dedup, and it emits
    the group representatives the hash table already holds instead of gathering whole rows by
    index. `unique()` with an explicit column list over a narrow frame is how this shape
    arrives, and `select` pruning can create it too.
    """
    if not node.keys or set(node.keys) != set(node.input.available_columns()):
        return None
    return Distinct(node.input)


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
        new_n = clamp(inner.n - node.offset, 0, node.n)
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
