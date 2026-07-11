"""Cost-based filter splitting — pay an expensive predicate only on surviving rows.

The data plane's `and` evaluates **both** operands over **every** row: `bc-expr`'s
`Binary{op: And}` computes `left.eval(batch)` and `right.eval(batch)` in full and then
combines them with `and_kleene`. There is no short-circuit and no selection vector. So
for `WHERE x > 5 AND regexp_matches(s, '…')`, the regex runs on every row — including
the ~2/3 that `x > 5` would have discarded.

Stacked `Filter`s do not have that problem: `filter_batch` applies its mask with Arrow's
`filter_record_batch`, which **compacts** the batch, so a second `Filter` above it sees
only the survivors. Splitting a conjunction therefore converts

    Filter(x > 5 AND regexp(s))        cost ≈ n · (c_cheap + c_expensive)
    Filter(regexp(s), Filter(x > 5))   cost ≈ n · (c_cheap + m) + n · sel · c_expensive

and the second is far cheaper whenever the cheap conjunct is selective and the expensive
one is genuinely expensive. `m` is the price of materializing one extra compacted batch
(`filter_split_materialize_cost`), which is what stops this rule from firing on a
conjunction of two cheap comparisons — there, fusing into one vectorized pass wins, and
`merge_adjacent_filters` (PUSHDOWN) has already done it.

This is the rule that turns `expr_cost`'s JIT knowledge into a plan change: the "cheap"
group is precisely the one Cranelift compiles to a vector compare, and the "expensive"
group is what the Arrow-kernel interpreter must walk per row.

Ordering follows the Krishnamurthy-Boral-Zaniolo rank rule (`cost / (1 - selectivity)`
ascending), which is the optimal evaluation order for independent predicates; the split
point is then chosen by costing every prefix.

Correctness. Splitting an `AND` is exact under Kleene logic: `and_kleene(a, b)` is TRUE
exactly when both `a` and `b` are TRUE, and `filter` keeps only TRUE (a NULL mask entry
drops the row), so `Filter(b, Filter(a, r))` keeps precisely the rows `Filter(a AND b, r)`
keeps. Conjunct order is likewise free — `AND` is commutative. The one observable
difference is *fewer* evaluation errors: a conjunct like `10 / x > 1` guarded by `x <> 0`
now sees only rows where the guard held, so a query that used to raise may now succeed.
That matches DuckDB, and it never turns a succeeding query into a failing one.

Runs in SELECTION (a run-once, cost-based phase) rather than a fixpoint phase, so it can
never ping-pong against `merge_adjacent_filters`, which fuses stacked filters back
together in the earlier PUSHDOWN phase.
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase, RuleCategory
from batcher.plan.expr_ir import Expr
from batcher.plan.expr_rewrite import combine_conjuncts, split_conjuncts
from batcher.plan.logical import Filter, LogicalPlan

__all__ = ["split_expensive_filter"]

# A predicate that keeps every row can never pay for a split; guard the rank divisor
# against division by zero without special-casing it.
_MIN_DISCARD = 1e-6


def _rank(cost: float, selectivity: float) -> float:
    """The rank of a conjunct: expected cost per row it discards.

    Evaluating independent predicates in ascending rank order minimizes total work
    (Krishnamurthy-Boral-Zaniolo). A cheap, highly selective predicate ranks lowest and
    is evaluated first.
    """
    return cost / max(_MIN_DISCARD, 1.0 - selectivity)


def _best_split(
    conjuncts: list[Expr], costs: list[float], ctx: OptimizerContext
) -> tuple[int, float] | None:
    """The prefix length minimizing the split cost, and that cost.

    Costs every split point `k` (evaluate `conjuncts[:k]` first, then `conjuncts[k:]`
    only on the rows that survive). Returns `None` if there is no interior split point.
    """
    n = len(conjuncts)
    if n < 2:
        return None
    materialize = ctx.config.optimizer.filter_split_materialize_cost
    suffix_cost = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_cost[i] = suffix_cost[i + 1] + costs[i]
    best: tuple[int, float] | None = None
    prefix_cost = 0.0
    for k in range(1, n):
        prefix_cost += costs[k - 1]
        # The prefix's *joint* selectivity, not the product of its parts: the estimator
        # applies exponential backoff across conjuncts, which is what keeps a correlated
        # prefix from looking impossibly selective.
        sel = ctx.estimator.expr_selectivity(combine_conjuncts(conjuncts[:k]))
        cost = prefix_cost + materialize + sel * suffix_cost[k]
        if best is None or cost < best[1]:
            best = (k, cost)
    return best


@rule(
    name="split_expensive_filter",
    phase=Phase.SELECTION,
    matches=(Filter,),
    category=RuleCategory.SELECTION,
)
def split_expensive_filter(node: Filter, ctx: OptimizerContext) -> LogicalPlan | None:
    """Split a conjunctive filter so an expensive predicate runs only on survivors.

    Orders the conjuncts by rank, costs every split point against the fused alternative,
    and rewrites into stacked `Filter`s when the winning split beats the fused predicate
    by `filter_split_min_gain`. Returns `None` (no change) otherwise — including for the
    overwhelmingly common case of a conjunction of cheap comparisons, where one fused
    vectorized pass is correctly the cheaper plan.
    """
    conjuncts = split_conjuncts(node.predicate)
    if len(conjuncts) < 2:
        return None

    # Price through the cost model so conjuncts are costed at the *measured* JIT
    # speedup, exactly as `op_cost` prices the filter this rule is rewriting.
    cost_of = ctx.costs().expr_cost
    costs = [cost_of(c) for c in conjuncts]
    fused = sum(costs)
    # Rank order: cheap and selective first. Ties keep the author's original order, so
    # the rewrite is deterministic.
    order = sorted(
        range(len(conjuncts)),
        key=lambda i: (_rank(costs[i], ctx.estimator.expr_selectivity(conjuncts[i])), i),
    )
    ranked = [conjuncts[i] for i in order]
    ranked_costs = [costs[i] for i in order]

    best = _best_split(ranked, ranked_costs, ctx)
    if best is None:
        return None
    k, split_cost = best
    if fused < ctx.config.optimizer.filter_split_min_gain * split_cost:
        return None  # fusing into one vectorized pass is cheaper; leave it alone

    inner = Filter(node.input, combine_conjuncts(ranked[:k]))
    return Filter(inner, combine_conjuncts(ranked[k:]))
