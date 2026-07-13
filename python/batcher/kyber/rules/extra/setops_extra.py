"""Set-operation rewrites that `setops.py` leaves on the table — bag vs set, precisely.

`setops.py` covers the structural union simplifications (flatten, singleton, empty branch,
duplicate branch, branch-level `Distinct`/`Sort`). This module adds the rewrites that need
to look *inside* the branches — at the predicates they filter by and the relation they read
— plus the two places a union's dedup is provably redundant because the operator *above* it
already collapses duplicates.

**Every rule here is gated on `Union.distinct`.** A distinct union produces a set: an outer
dedup dominates everything below it, which is what licenses merging branches, absorbing a
subsumed branch, and rewriting a two-branch partition back into its base relation. None of
that is valid for `UNION ALL`, where duplicates are the result — and where branch *order* is
observable besides (`setops.py` treats it as such; so does this module). A rule that could
change a multiset or a row order under `UNION ALL` simply refuses to fire.

Evaluated and **deliberately not implemented**:

- `push_limit_through_distinct_union` — **unsound**. `Limit(n)` does not distribute into the
  branches of a distinct union: the branch's first `n` rows may all be duplicates of each
  other. With `a = [1, 1, 1, 2, 3]` and `n = 3`, the distinct union yields `{1, 2, 3}` → 3
  rows, while capping the branch at 3 rows first yields `[1, 1, 1]` → dedup → `{1}` → **1
  row**. (Pushing a limit into a `UNION ALL` *is* sound and is already `push_limit_into_union`
  / `push_offset_limit_into_union`.)
- `merge_union_of_identical_branches` — already shipped: `dedup_distinct_union_branches`
  drops the repeated branch of a DISTINCT union and `simplify_singleton_union` turns what is
  left into `Distinct(branch)`. And `UNION ALL` of two identical branches is **not
  reducible** — the duplicate rows it emits are the answer, not redundancy.
- `drop_union_branch_that_is_provably_empty` — already shipped as `prune_empty_union_branch`
  (plus `propagate_empty_relation` in SELECTION).
- `union_of_one_branch` — already shipped as `simplify_singleton_union`.
- `push_distinct_through_union_all_branches` — already shipped as `push_distinct_into_union`
  (a `Distinct` over a `UNION ALL`, gated on learned stats showing the branch has duplicates).
  Its analogue *inside* a distinct union — `Union(distinct)(a, b)` →
  `Union(distinct)(Distinct(a), Distinct(b))` — is **refused on purpose**: it is the exact
  inverse of `drop_distinct_in_distinct_union`, and both live in REWRITE, so the pair would
  oscillate until the fixpoint cap and the plan a query gets would depend on
  `fixpoint_iterations`. (Worth noting for whoever owns that trade: because
  `fold_distinct_union_all` rewrites `Distinct(UnionAll(...))` into a distinct `Union`,
  `drop_distinct_in_distinct_union` then *strips the very branch dedups*
  `push_distinct_into_union` just added — the partial-dedup win is lost today. Resolving that
  belongs in those two rules, not in a third rule fighting them.)
- `hoist_common_filter_out_of_union_branches` / `hoist_common_projection_out_of_union_branches`
  — sound, but **counterproductive here**: they are the exact inverses of the shipped
  `push_filter_into_union` and `push_project_through_union` (PUSHDOWN), which run *after*
  REWRITE and would immediately push the hoisted operator back into the branches. Net effect:
  none, at the cost of two more traversals — and in the same phase they would oscillate. The
  engine's chosen direction is *down*; do not fight it.
- `reorder_union_branches_by_estimated_size` — a `UNION ALL`'s branch order is observable in
  the output (this codebase treats it as such), so reordering is a result change, and a union
  has no build/probe asymmetry for the reordering to exploit anyway. No win, real risk.
"""

from __future__ import annotations

import dataclasses

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase

# Sibling families' helpers, imported rather than re-implemented: `_key`/`_NEGATED_COMPARISON`
# (structural identity of an expression; the exact Kleene negation of each comparison),
# `_DUP_INSENSITIVE` (the aggregates a duplicate row cannot change), `_relation_key` (identity
# of a *relation* — the IR with `source_id`s blanked plus the bound source objects, because two
# branches reading the same table get different `source_id`s), and the nullability analysis (a
# predicate is only *total* — TRUE or FALSE, never NULL — when its operands can never be NULL).
from batcher.kyber.rules.extra.agg_extra import _DUP_INSENSITIVE
from batcher.kyber.rules.extra.boolean_algebra import _NEGATED_COMPARISON, _key
from batcher.kyber.rules.extra.join_elim import _relation_key
from batcher.kyber.rules.extra.nullability import _never_null, _non_null_cols
from batcher.plan.expr_ir import Binary, Expr, IsNotNull, IsNull, Not
from batcher.plan.expr_rewrite import combine_disjuncts
from batcher.plan.logical import (
    Aggregate,
    Distinct,
    Filter,
    Join,
    Limit,
    LogicalPlan,
    Sample,
    Sort,
    Union,
)

__all__ = [
    "absorb_subsumed_branch_in_distinct_union",
    "drop_union_dedup_on_semi_join_build",
    "drop_union_dedup_under_aggregate",
    "merge_distinct_union_of_complementary_filters",
    "merge_distinct_union_of_filters_on_same_input",
]

# Unary operators whose output rows are a subset of their input's rows (they select,
# truncate, sample, dedup, or reorder — none *invents* a row and none changes a row's
# values). `Project` is deliberately absent: it rewrites the columns, so its rows are not
# its input's rows.
_ROW_SUBSET_OPS = (Filter, Limit, Sample, Sort, Distinct)


def _complementary(p: Expr, q: Expr, non_null: frozenset[str]) -> bool:
    """Whether `p` and `q` partition every row — exactly one of them is TRUE, always.

    Under three-valued logic that needs `p` to be **total** (never NULL): for a nullable
    `a`, `a > 5` and `a <= 5` are *both* NULL on a null row, which lands it in neither
    branch — so the pair does not partition and the caller must not fire. `IS NULL` /
    `IS NOT NULL` are total by construction; an ordinary predicate is total only when its
    operands are provably non-null (`_never_null`). Three spellings of the complement are
    recognized: an explicit `NOT`, a negated comparison (`>` vs `<=` on the same operands —
    the form NORMALIZE's `fold_not_comparison` leaves behind), and the `IS NULL`/`IS NOT
    NULL` pair.
    """
    if isinstance(p, IsNull) and isinstance(q, IsNotNull) and _key(p.input) == _key(q.input):
        return True
    if isinstance(p, IsNotNull) and isinstance(q, IsNull) and _key(p.input) == _key(q.input):
        return True
    if not _never_null(p, non_null):
        return False
    if isinstance(q, Not) and _key(q.input) == _key(p):
        return True
    if isinstance(p, Not) and _key(p.input) == _key(q):
        return True
    return (
        isinstance(p, Binary)
        and isinstance(q, Binary)
        and _NEGATED_COMPARISON.get(p.op) == q.op
        and _key(p.left) == _key(q.left)
        and _key(p.right) == _key(q.right)
    )


@rule(name="merge_distinct_union_of_complementary_filters", phase=Phase.REWRITE, matches=(Union,))
def merge_distinct_union_of_complementary_filters(
    node: Union, ctx: OptimizerContext
) -> LogicalPlan | None:
    """`filter(x, p) UNION filter(x, NOT p)` → `Distinct(x)` — a partition, put back together.

    When `p` is *total* (never NULL) every row of `x` satisfies exactly one of `p` and its
    complement, so the two branches partition `x`'s rows and their distinct union is exactly
    `Distinct(x)`. The payoff is large: one scan instead of two, and both predicates gone.
    The classic shape is `WHERE c IS NULL` union `WHERE c IS NOT NULL`, which is total for *any*
    `c`; an ordinary predicate qualifies only over provably non-null operands, because a NULL
    row would otherwise fall into neither branch and be lost.

    Restricted to a DISTINCT union: `UNION ALL` would keep `x`'s duplicate rows twice over
    (the partition is by row, not by multiplicity — a row appearing twice in `x` appears
    twice in its branch) and, being a concatenation, it also reorders the output relative to
    `x`. Registered ahead of `merge_distinct_union_of_filters_on_same_input`, which would
    otherwise reduce this same shape only as far as `Distinct(filter(x, p OR NOT p))` — an
    always-true filter that NORMALIZE (already run) would no longer be there to fold away.
    """
    if ctx is None or not node.distinct or len(node.inputs) != 2:
        return None
    left, right = node.inputs
    if not (isinstance(left, Filter) and isinstance(right, Filter)):
        return None
    base_key = _relation_key(left.input, ctx)
    if base_key is None or base_key != _relation_key(right.input, ctx):
        return None
    if not _complementary(left.predicate, right.predicate, _non_null_cols(left.input)):
        return None
    return Distinct(left.input)


@rule(name="merge_distinct_union_of_filters_on_same_input", phase=Phase.REWRITE, matches=(Union,))
def merge_distinct_union_of_filters_on_same_input(
    node: Union, ctx: OptimizerContext
) -> LogicalPlan | None:
    """`filter(x, p) UNION filter(x, q)` → `Distinct(filter(x, p OR q))` — one scan, one filter.

    Every branch filters the *same* relation (structurally identical IR), so the set of rows
    the union produces is the set of `x`'s rows satisfying `p` or `q` — which is precisely
    what one filter with the disjoined predicate selects. Exact under three-valued logic: a
    filter keeps a row only where its predicate is TRUE, and `p OR q` is TRUE exactly where
    `p` is TRUE or `q` is TRUE (Kleene `OR` — a NULL disjunct cannot manufacture a TRUE).
    The outer `Distinct` is required and kept: `x` may itself hold duplicate rows, and the
    union it replaces would have collapsed them.

    Restricted to a DISTINCT union — under `UNION ALL` a row satisfying *both* predicates
    appears twice, and the merged filter would emit it once. Identical predicates are
    de-duplicated before disjoining (so the merged form never carries `p OR p`), and the
    disjunction is balanced.
    """
    if ctx is None or not node.distinct or len(node.inputs) < 2:
        return None
    if not all(isinstance(branch, Filter) for branch in node.inputs):
        return None
    branches: tuple[Filter, ...] = node.inputs  # type: ignore[assignment]
    base = branches[0].input
    base_key = _relation_key(base, ctx)
    if base_key is None or any(_relation_key(b.input, ctx) != base_key for b in branches[1:]):
        return None
    preds: list[Expr] = []
    seen: set[str] = set()
    for b in branches:
        k = _key(b.predicate)
        if k not in seen:
            seen.add(k)
            preds.append(b.predicate)
    return Distinct(Filter(base, combine_disjuncts(preds)))


def _row_subset_base(node: LogicalPlan) -> LogicalPlan:
    """The relation a branch's rows are drawn from, stripping the row-subsetting operators."""
    while isinstance(node, _ROW_SUBSET_OPS):
        node = node.input
    return node


@rule(name="absorb_subsumed_branch_in_distinct_union", phase=Phase.REWRITE, matches=(Union,))
def absorb_subsumed_branch_in_distinct_union(
    node: Union, ctx: OptimizerContext
) -> LogicalPlan | None:
    """Drop a DISTINCT-union branch whose rows are a subset of another branch's.

    Set absorption: the union of `A` with any sub-relation of `A` is `A`.

    `Filter`/`Limit`/`Sample`/`Sort`/`Distinct` each emit a sub-*set* of their input's rows
    (they select, truncate, sample, reorder or dedup — none invents a row and none alters
    one), so a branch that strips down through them to another branch of the same union
    contributes nothing that branch does not already contribute. In a *set* union that makes
    it dead weight — a whole branch, and the pipeline that feeds it, deleted.

    Only for a DISTINCT union: under `UNION ALL` the subsumed branch's rows are genuinely
    *additional* copies and dropping it would change the multiset. The branch a subsumed
    branch reduces to is itself never dropped (its own base *is* itself), so at least one
    branch always survives; a branch equal to another (nothing stripped) is left to
    `dedup_distinct_union_branches`. Relation identity is by bound source, not raw IR — the
    two branches of `a.union(b)` get different `source_id`s even when `a` and `b` read the
    same table — and a branch whose scan is unbound never matches.
    """
    if ctx is None or not node.distinct or len(node.inputs) < 2:
        return None
    keys = [_relation_key(branch, ctx) for branch in node.inputs]
    kept: list[LogicalPlan] = []
    for branch, key in zip(node.inputs, keys, strict=True):
        base_key = _relation_key(_row_subset_base(branch), ctx)
        if base_key is not None and base_key != key and base_key in keys:
            continue  # its rows already arrive via the branch it reduces to
        kept.append(branch)
    if len(kept) == len(node.inputs):
        return None
    if len(kept) == 1:
        return Distinct(kept[0])
    return Union(tuple(kept), True)


@rule(name="drop_union_dedup_under_aggregate", phase=Phase.REWRITE, matches=(Aggregate,))
def drop_union_dedup_under_aggregate(node: Aggregate, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Aggregate(UNION(a, b))` → `Aggregate(UNION ALL(a, b))` when the aggregates cannot see
    a duplicate row.

    A distinct union's dedup is a full pipeline breaker (and, distributed, a full shuffle) —
    and it is pure waste when the operator above collapses duplicates anyway. A whole-row
    `Distinct` removes only rows identical in *every* column, so the values reaching each
    group are unchanged (a removed row had an identical twin): min/max, boolean AND/OR and the
    distinct counts are the same either way, and the set of groups is the same. Exactly the
    argument `drop_distinct_before_agg` makes for a `Distinct` child — this is the shape it
    cannot see, where the dedup lives *inside* the union node.

    Refuses on any duplicate-sensitive aggregate (`count`, `sum`, `mean`, `mode`,
    `arg_min`/`arg_max` — they all depend on multiplicity). A group-only aggregate qualifies
    vacuously: it dedups by definition.
    """
    inner = node.input
    if not (isinstance(inner, Union) and inner.distinct):
        return None
    if any(
        spec.agg.func not in _DUP_INSENSITIVE or spec.agg.input2 is not None
        for spec in node.aggregates
    ):
        return None
    relaxed = Union(inner.inputs, distinct=False)
    return Aggregate(relaxed, node.group_keys, node.aggregates, node.watermark)


@rule(name="drop_union_dedup_on_semi_join_build", phase=Phase.REWRITE, matches=(Join,))
def drop_union_dedup_on_semi_join_build(node: Join, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`Semi/Anti Join(L, UNION(a, b))` → `Semi/Anti Join(L, UNION ALL(a, b))`.

    A semi/anti join asks only whether a left row's key has *any* match on the right; a
    duplicate right row cannot change that membership, so the union's dedup — a breaker, and
    a shuffle when distributed — buys nothing and is removed. The same argument
    `drop_redundant_distinct_build` makes for an explicit `Distinct` on the build side; this
    is the shape where the dedup is folded into the union node instead, which that rule (and
    `fold_distinct_union_all`, which *creates* this shape) leaves behind.

    Only the right/build side, and only for `semi`/`anti` — every other join type emits the
    matching right rows (or one output row per match), so right-side duplicates are load
    bearing there.
    """
    if node.join_type not in ("semi", "anti"):
        return None
    right = node.right
    if not (isinstance(right, Union) and right.distinct):
        return None
    return dataclasses.replace(node, right=Union(right.inputs, distinct=False))
