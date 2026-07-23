"""Scan-level data skipping — decide a predicate's *parts* from the column's metadata.

`zonemap_prune_filter` decides a predicate *as a whole*: one undecidable conjunct and the whole
conjunction is undecidable, so an always-true sibling is still evaluated on every row of the
scan and an unreachable `IN`-list member is still probed. These rules close that gap, deciding a
conjunct, a disjunct, and an individual `IN` member against the same shared oracle
(`_predicate_status`) and the same absence proofs (`evidence.py`) — never a second copy of them.

Everything here runs in **PUSHDOWN** (see `evidence.SIP`): a rewrite that lands after SELECTION
changes the plan signature the learning loop was keyed on. These rules are safe in a fixpoint
phase because each is *monotone* — it only ever removes a conjunct, a disjunct, or an `IN` member,
so re-running it finds nothing left to remove.

The joins' companion: a key range or value set proven impossible does not merely shrink a
filter, it settles the join. An inner/semi join requires a match on *every* key, so one key that
cannot be equal on any pair of rows makes the whole join empty — from an all-NULL key, a
bloom-refuted value set, or two disjoint value sets. Anti and outer joins are excluded
throughout: for them "nothing matches" means the preserved side survives *whole*
(`no_match_join_to_preserved_side`), and emptying an input would delete the answer.
"""

from __future__ import annotations

from collections.abc import Callable

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.kyber.rules.extra.runtime_filters.evidence import (
    _all_null_key,
    _all_refuted,
    _bloom_refutes,
    _candidate_key_values,
    _empty_marker,
    _no_match_candidate,
    _out_of_range,
    _real_key_pairs,
    _value_set,
)
from batcher.kyber.rules.zonemap_pruning import _predicate_status
from batcher.plan.expr_ir import Col, Expr, InList
from batcher.plan.expr_rewrite import (
    combine_conjuncts,
    combine_disjuncts,
    split_conjuncts,
    split_disjuncts,
)
from batcher.plan.logical import Filter, Join, Limit, LogicalPlan
from batcher.plan.stats import ColumnStat, RelStats

__all__ = [
    "drop_filter_conjunct_implied_by_zonemap",
    "drop_filter_disjunct_refuted_by_zonemap",
    "empty_join_from_all_null_key",
    "empty_join_from_bloom_absent_key",
    "empty_join_from_disjoint_key_values",
    "prune_in_list_by_bloom",
    "prune_in_list_by_zonemap",
]


@rule(name="drop_filter_conjunct_implied_by_zonemap", phase=Phase.PUSHDOWN, matches=(Filter,))
def drop_filter_conjunct_implied_by_zonemap(
    node: Filter, ctx: OptimizerContext
) -> LogicalPlan | None:
    """Drop a conjunct the column's bounds prove every row already satisfies.

    `WHERE ship_date >= '1992-01-01' AND price > :p` over a column whose EXACT minimum is later
    than that date evaluates a tautology on every row of the scan, because the conjunction as a
    whole is undecidable (the second conjunct is) and `zonemap_prune_filter` therefore declines.
    Deciding each conjunct on its own recovers exactly that.

    `_predicate_status` is the shared oracle, and it is the one that requires a known-zero null
    count before calling anything *always true* — a filter drops a NULL row, so dropping the
    filter over a nullable column would keep it. Fires only on a real conjunction (the
    single-predicate case belongs to `zonemap_prune_filter`, and racing it would be duplication);
    an always-*false* conjunct is left in place for that rule to fold into an empty relation.
    """
    conjuncts = split_conjuncts(node.predicate)
    if len(conjuncts) < 2:
        return None
    stats = ctx.estimator.estimate(node.input)
    kept = [c for c in conjuncts if _predicate_status(c, stats) is not True]
    if len(kept) == len(conjuncts):
        return None
    return node.input if not kept else Filter(node.input, combine_conjuncts(kept))


@rule(name="drop_filter_disjunct_refuted_by_zonemap", phase=Phase.PUSHDOWN, matches=(Filter,))
def drop_filter_disjunct_refuted_by_zonemap(
    node: Filter, ctx: OptimizerContext
) -> LogicalPlan | None:
    """Drop a disjunct the column's bounds prove no row can satisfy.

    The `OR` counterpart, and sound under Kleene logic for the mirror reason: `FALSE OR x ≡ x` at
    every truth value (`FALSE OR NULL` is NULL, which the filter drops either way). So `WHERE
    region = 'EU' OR ship_date < '1990'` over a table whose dates all postdate 1990 collapses to
    the bare equality — which is sargable, bloom-probeable and pushable to the source, none of
    which a disjunction is. Every disjunct refuted → the conjunct is false on every row → the
    relation is empty.

    Bounds refute at any provenance (a row-shrinking operator only narrows the true range, so a
    value outside the recorded `[min, max]` is absent regardless). The *always-true* direction,
    which would need `Provenance.EXACT` and a zero null count, stays with `zonemap_prune_filter`.
    """
    stats = ctx.estimator.estimate(node.input)
    out: list[Expr] = []
    changed = False
    for conj in split_conjuncts(node.predicate):
        disjuncts = split_disjuncts(conj)
        if len(disjuncts) < 2:
            out.append(conj)
            continue
        kept = [d for d in disjuncts if _predicate_status(d, stats) is not False]
        if len(kept) == len(disjuncts):
            out.append(conj)
            continue
        changed = True
        if not kept:
            return Limit(node.input, 0)  # false on every row → the filter keeps nothing
        out.append(combine_disjuncts(kept))
    return Filter(node.input, combine_conjuncts(out)) if changed else None


def _rewrite_in_lists(
    node: Filter, refutes: Callable[[ColumnStat, object], bool], stats: RelStats
) -> LogicalPlan | None:
    """`node` with every `col IN (…)` narrowed to the members `refutes` cannot rule out."""
    out: list[Expr] = []
    changed = False
    for conj in split_conjuncts(node.predicate):
        if not (isinstance(conj, InList) and isinstance(conj.input, Col)):
            out.append(conj)
            continue
        stat = stats.column(conj.input.name)
        survivors = tuple(v for v in conj.values if not refutes(stat, v))
        if len(survivors) == len(conj.values):
            out.append(conj)
            continue
        changed = True
        if not survivors:
            return Limit(node.input, 0)  # no member can match → the filter keeps nothing
        out.append(InList(conj.input, survivors))
    return Filter(node.input, combine_conjuncts(out)) if changed else None


@rule(name="prune_in_list_by_zonemap", phase=Phase.PUSHDOWN, matches=(Filter,))
def prune_in_list_by_zonemap(node: Filter, ctx: OptimizerContext) -> LogicalPlan | None:
    """Drop `IN`-list members outside the column's `[min, max]` — they match no row.

    `WHERE id IN (5, 500, 5000000)` over a column bounded at 10,000 is `id IN (5, 500)`: the third
    member is provably absent, so testing every row against it is wasted work, and the narrower
    list is a narrower predicate for the source to push down. An emptied list proves the filter
    keeps nothing. The existing `IN` refinements (`refine_in_list_by_*`, `intersect_in_lists`)
    reason from *sibling conjuncts*; this is the first to reason from the column's own bounds.
    """
    return _rewrite_in_lists(node, _out_of_range, ctx.estimator.estimate(node.input))


@rule(name="prune_in_list_by_bloom", phase=Phase.PUSHDOWN, matches=(Filter,))
def prune_in_list_by_bloom(node: Filter, ctx: OptimizerContext) -> LogicalPlan | None:
    """Drop `IN`-list members the column's membership bloom proves absent.

    Where `prune_in_list_by_zonemap` removes the members outside the range, this removes the ones
    *inside* it that the column simply does not hold — the point-lookup case min/max is blind to
    (`id IN (9700123, 9700124)` over a 10M-row id column). A bloom has no false negatives, so
    `contains() -> False` is a proof, not a guess; every probe is domain-guarded, because a
    cross-domain probe reports a definitive absence for a value that *is* present and would delete
    rows. An all-refuted list proves the filter empty.
    """
    return _rewrite_in_lists(node, _bloom_refutes, ctx.estimator.estimate(node.input))


@rule(name="empty_join_from_all_null_key", phase=Phase.PUSHDOWN, matches=(Join,))
def empty_join_from_all_null_key(node: Join, ctx: OptimizerContext) -> LogicalPlan | None:
    """An inner/semi join whose key is *all NULL* on one side → empty.

    `NULL = NULL` is NULL, so a side whose key holds nothing but nulls has no row that can ever
    match, and an inner or semi join (both of which require a match) emits nothing.
    `join_disjoint_keys_to_empty` proves the same conclusion from disjoint `[min, max]` ranges and
    cannot see this case at all — an all-null column has no bounds to be disjoint from.

    EXACT-gated on both the null count and the row count: an *estimated* all-null column that
    holds one real value would delete the entire answer. Anti and outer joins are excluded — for
    them, a side that matches nothing means the preserved side survives whole.
    """
    if not _no_match_candidate(node):
        return None
    if _all_null_key(node.left, node.left_keys, ctx) or _all_null_key(
        node.right, node.right_keys, ctx
    ):
        return _empty_marker(node)
    return None


@rule(name="empty_join_from_bloom_absent_key", phase=Phase.PUSHDOWN, matches=(Join,))
def empty_join_from_bloom_absent_key(node: Join, ctx: OptimizerContext) -> LogicalPlan | None:
    """An inner/semi join whose candidate key values are *all* refuted by the other side's bloom.

    When one side pins its key to a finite value set — an `IN`/`=` constraint, or a column whose
    EXACT bounds collapse to a single value — and the other side's key bloom proves every one of
    those values absent, no pair of rows can satisfy the equi-condition and the join is empty.
    This is the point-lookup emptiness `join_disjoint_keys_to_empty` cannot reach: the values lie
    *inside* the other side's `[min, max]`; they are simply not there.

    A bloom answers only absence and is domain-guarded, so the proof never over-claims, and the
    candidate set is an *upper* bound on the side's key values, so refuting all of it refutes the
    join. Inner/semi only — a preserved side's unmatched rows survive a no-match join.
    """
    if not _no_match_candidate(node):
        return None
    left_stats = ctx.estimator.estimate(node.left)
    right_stats = ctx.estimator.estimate(node.right)
    for lk, rk in _real_key_pairs(node):
        left_vals = _candidate_key_values(node.left, lk, left_stats)
        right_vals = _candidate_key_values(node.right, rk, right_stats)
        if _all_refuted(left_vals, right_stats.column(rk)) or _all_refuted(
            right_vals, left_stats.column(lk)
        ):
            return _empty_marker(node)
    return None


@rule(name="empty_join_from_disjoint_key_values", phase=Phase.PUSHDOWN, matches=(Join,))
def empty_join_from_disjoint_key_values(node: Join, _ctx: OptimizerContext) -> LogicalPlan | None:
    """An inner/semi join whose two sides pin one key to *disjoint* value sets → empty.

    `WHERE a.k IN (1, 2) AND b.k IN (3, 4)` over `a ⋈ b ON a.k = b.k` can never match: each side's
    `IN`/`=` constraints bound its key to a finite set (both forms are null-rejecting, so a
    surviving row's key is exactly one of the literals), and disjoint upper bounds admit no equal
    pair. `join_disjoint_keys_to_empty` proves the same from EXACT `[min, max]` ranges — which a
    *filtered* side no longer has (a filter downgrades provenance away from EXACT), so it cannot
    see the case this rule is for. Purely syntactic: no statistic is consulted, so nothing is
    estimated and nothing can be over-claimed.
    """
    if not _no_match_candidate(node):
        return None
    for lk, rk in _real_key_pairs(node):
        left_vals = _value_set(node.left, lk)
        right_vals = _value_set(node.right, rk)
        if left_vals is not None and right_vals is not None and not (left_vals & right_vals):
            return _empty_marker(node)
    return None
