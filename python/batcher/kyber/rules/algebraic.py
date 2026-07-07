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

import datetime as _dt

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase, RuleCategory
from batcher.kyber.stats.selectivity import comparison_col_side
from batcher.plan.expr_ir import Binary, Col, Expr, InList, Lit
from batcher.plan.expr_rewrite import (
    combine_conjuncts,
    combine_disjuncts,
    split_conjuncts,
    split_disjuncts,
    substitute_columns,
)
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
from batcher.plan.stats import Provenance

__all__ = [
    "combine_limits",
    "constant_propagation",
    "eliminate_sort_before_aggregate",
    "factor_common_conjuncts",
    "fold_in_list",
    "merge_adjacent_filters",
    "prune_true_filter",
    "push_distinct_into_union",
    "push_filter_into_union",
    "push_limit_into_union",
    "push_limit_through_project",
    "remove_redundant_distinct",
]


@rule(name="factor_common_conjuncts", phase=Phase.NORMALIZE, matches=(Filter,))
def factor_common_conjuncts(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Factor a conjunct common to every branch of an `OR` out of the disjunction.

    `(A AND X) OR (A AND Y) OR (A AND Z)` → `A AND (X OR Y OR Z)`. Boolean algebra,
    always semantics-preserving. The payoff is structural: the factored `A` becomes a
    top-level conjunct, so an equi-join condition hidden inside a disjunction (TPC-H
    Q19's `(p=l AND ...) OR (p=l AND ...) OR ...`) is exposed where predicate pushdown
    and join-key derivation can see it — without it the join degrades to a cartesian
    product. A branch whose conjuncts are *all* common contributes a `TRUE` disjunct,
    collapsing the residual `OR` to nothing (the factored conjuncts alone).

    Fires only when at least one conjunct is shared by all branches and the predicate
    is a genuine top-level `OR` (a non-OR predicate is left to the other rules).
    """
    disjuncts = split_disjuncts(node.predicate)
    if len(disjuncts) < 2:
        return None  # not a disjunction

    per_branch = [split_conjuncts(d) for d in disjuncts]
    # Conjuncts present (by structural identity) in the first branch and every other.
    branch_key_sets = [{_ir_key(c) for c in br} for br in per_branch]
    common: list[Expr] = []
    common_keys: set = set()
    for conj in per_branch[0]:
        k = _ir_key(conj)
        if k in common_keys:
            continue
        if all(k in s for s in branch_key_sets):
            common.append(conj)
            common_keys.add(k)
    if not common:
        return None

    # Each branch's residual = its conjuncts minus the common ones. An empty residual
    # means the branch is implied by `common` alone → the OR is satisfied there, i.e.
    # a TRUE disjunct, which makes the whole residual OR vanish.
    residuals: list[Expr] = []
    any_empty = False
    for br in per_branch:
        rest = [c for c in br if _ir_key(c) not in common_keys]
        if not rest:
            any_empty = True
            break
        residuals.append(combine_conjuncts(rest))

    factored = list(common)
    if not any_empty:
        factored.append(combine_disjuncts(residuals))
    return Filter(node.input, combine_conjuncts(factored))


def _ir_key(expr: Expr):
    """A hashable structural identity for an expression (its IR rendered hashable)."""
    import json

    return json.dumps(expr.to_ir(), sort_keys=True)


# Fold an OR-of-equals chain into `IN` once it has at least this many branches — below
# it the chain is cheap (and JIT-compilable), above it the hash-set membership wins.
_IN_LIST_MIN = 5


@rule(name="fold_in_list", phase=Phase.NORMALIZE, matches=(Filter,))
def fold_in_list(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Fold `(x = a) OR (x = b) OR …` into `x IN (a, b, …)` (hash-set membership).

    The SQL front end lowers an `IN (literal, …)` list to a chain of equality
    disjuncts; for a long list that is an O(rows · k) scan of `k` comparisons per row.
    This collapses each single-column run of equality disjuncts into one `InList`
    (an O(rows) hash-set lookup, the `eval_in_list` kernel), which is also the form a
    runtime join filter pushes the build side's key set as. Semantics are identical —
    `x = a OR x = b` and `x IN (a, b)` share the same Kleene null behavior. Only
    folds runs of ≥ `_IN_LIST_MIN` over int / string / date literals (the kernel's
    supported types); shorter runs and other types stay as comparisons.
    """
    new_pred = _fold_or_chains(node.predicate)
    if new_pred is node.predicate:
        return None
    return Filter(node.input, new_pred)


def _fold_or_chains(expr: Expr) -> Expr:
    """Recurse through `AND`/`OR`, folding each disjunction's single-column equality runs."""
    if isinstance(expr, Binary) and expr.op == "and":
        left, right = _fold_or_chains(expr.left), _fold_or_chains(expr.right)
        return expr if left is expr.left and right is expr.right else Binary("and", left, right)
    if isinstance(expr, Binary) and expr.op == "or":
        return _fold_disjunction(expr)
    return expr


def _fold_disjunction(expr: Expr) -> Expr:
    # Group the disjuncts that are `col = <supported literal>` by column (preserving a
    # consistent literal type per column); fold any group of ≥ threshold into an InList.
    by_col: dict[str, list] = {}
    others: list[Expr] = []
    order: list[str] = []
    for d in split_disjuncts(expr):
        pair = _eq_col_literal(d)
        if pair is None:
            others.append(_fold_or_chains(d))
            continue
        col, value = pair
        if col not in by_col:
            by_col[col] = []
            order.append(col)
        by_col[col].append(value)

    folded: list[Expr] = []
    changed = False
    for col in order:
        values = by_col[col]
        if len(values) >= _IN_LIST_MIN and _one_supported_type(values):
            folded.append(InList(Col(col), tuple(values)))
            changed = True
        else:
            folded.extend(Binary("eq", Col(col), Lit(v)) for v in values)
    if not changed:
        return expr
    return combine_disjuncts(folded + others)


def _eq_col_literal(expr: Expr) -> tuple[str, object] | None:
    """`(col == lit)` or `(lit == col)` over a foldable literal → `(col_name, value)`."""
    if not (isinstance(expr, Binary) and expr.op == "eq"):
        return None
    left, right = expr.left, expr.right
    if isinstance(left, Col) and isinstance(right, Lit) and _foldable(right.value):
        return left.name, right.value
    if isinstance(right, Col) and isinstance(left, Lit) and _foldable(left.value):
        return right.name, left.value
    return None


def _foldable(value: object) -> bool:
    """Whether a literal is a type the `InList` kernel supports — Int64 / Utf8 / Date32.

    Excludes bool (an `int` subclass), float (NaN / precision make a set unsafe), and
    `datetime` (a `date` subclass that lowers to Timestamp, not Date32)."""
    if isinstance(value, (bool, _dt.datetime)):
        return False
    return isinstance(value, (int, str, _dt.date))


def _one_supported_type(values: list) -> bool:
    """All values share one supported kind (so the engine builds one typed set)."""

    def kind(v: object) -> str:
        if isinstance(v, _dt.date):
            return "date"
        return "int" if isinstance(v, int) else "str"

    return len({kind(v) for v in values}) == 1


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
        if not isinstance(branch, Distinct | Aggregate) and _dedup_shrinks(ctx, branch):
            new_inputs.append(Distinct(branch))
            changed = True
        else:
            new_inputs.append(branch)
    if not changed:
        return None
    return Distinct(Union(tuple(new_inputs), distinct=False))


def _dedup_shrinks(ctx: OptimizerContext, branch: LogicalPlan) -> bool:
    """Whether learned statistics show `branch` has enough duplicate rows that
    deduplicating it early is worthwhile (≥10% fewer rows) — so `push_distinct_into_union`
    only fires on real evidence, never a guess."""
    rows = ctx.estimator.estimate(branch).rows
    deduped = ctx.estimator.estimate(Distinct(branch))
    return deduped.provenance == Provenance.LEARNED and deduped.rows <= rows * 0.9


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
