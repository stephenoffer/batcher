"""The join-elimination rewrites — outer, self, cartesian, inner-reduction, disjoint-key.

One module per *responsibility*, not per rule: this file holds every rewrite in the family,
and `evidence` holds the proofs they consult. The family contract — why a plain inner join
is **not** eliminable here, and what evidence each shape needs — is in the package
docstring (`__init__`), which is the thing to read before adding a rule to this file.
"""

from __future__ import annotations

import dataclasses
import operator

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase

# Reused rather than re-derived: copy-pasting a *soundness* proof is how two rules drift
# apart and one of them starts deleting rows. `_right_unique_on_keys` is the engine's one
# uniqueness proof; `_is_empty`/`_output_side`/`_passthrough` are its one empty-marker and
# single-side-projection vocabulary; `_exact_rows` its one EXACT-cardinality gate; and the
# `evidence` module is where this family's own proofs live.
from batcher.kyber.rules.extra.adaptive_meta import _exact_rows
from batcher.kyber.rules.extra.join_elim.evidence import (
    _cartesian_keys,
    _disjoint_keys,
    _fold,
    _keys_non_null,
    _same_relation,
)
from batcher.kyber.rules.extra.join_extra import _is_empty, _output_side, _passthrough
from batcher.kyber.rules.joins import _right_unique_on_keys
from batcher.plan.expr_ir import IsNotNull, IsNull, referenced_columns, remap_columns
from batcher.plan.logical import Distinct, Filter, Join, Limit, LogicalPlan, Project, Projection

__all__ = [
    "anti_join_of_nonempty_cartesian_to_empty",
    "eliminate_cross_join_of_single_row",
    "eliminate_left_join_under_distinct",
    "inner_join_to_semi_when_right_unique",
    "join_disjoint_keys_to_empty",
    "no_match_join_to_preserved_side",
    "self_anti_join_to_null_keys",
    "self_join_elimination",
    "self_semi_join_to_filter",
    "semi_join_of_nonempty_cartesian",
]

# The side whose rows a join *preserves* even without a match — the only side a
# no-match join can degenerate to. A `full` join preserves both, so with no matches it
# emits |L| + |R| rows (each null-extended on the other side) and is NOT a passthrough
# of either side; it is deliberately absent.
_PRESERVED_SIDE = {"left": "left", "right": "right", "anti": "left"}


# --- outer-join elimination --------------------------------------------------


@rule(name="eliminate_left_join_under_distinct", phase=Phase.REWRITE, matches=(Distinct,))
def eliminate_left_join_under_distinct(
    node: Distinct, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`Distinct(Project(LEFT JOIN, <preserved-side columns only>))` → drop the join.

    A left join emits every left row **at least once** (once per match, or once
    null-extended when there is none) and invents no new left-column value. So the *set*
    of left-column tuples it produces is exactly the set the left input holds — the join
    can only change multiplicities, and the enclosing `Distinct` erases those. The join
    is therefore pure cost.

    This is the sibling of `join_to_semijoin` (which reduces the *inner* case to a semi
    join) and the uniqueness-free sibling of `eliminate_left_join`: no key proof is
    needed at all, because fan-out is exactly what `Distinct` undoes. Mirrored for a
    `right` join (its preserved side is the right). A `full` join is excluded: it also
    null-extends *right*-only rows, which would add an all-null left tuple that the left
    input does not contain. Fires only when the projection reads no column of the
    null-supplying side; idempotent (the result holds no `Join`).
    """
    proj = node.input
    if not isinstance(proj, Project):
        return None
    join = proj.input
    if not isinstance(join, Join) or join.join_type not in ("left", "right"):
        return None
    kept = join.join_type  # "left"/"right" join → the like-named side is preserved
    sources = {o.alias: o.name for o in join.output if o.side == kept}
    used: set[str] = set()
    for item in proj.items:
        used |= referenced_columns(item.expr)
    if not used <= set(sources):  # reads a null-supplied column → the join is needed
        return None
    src = join.left if kept == "left" else join.right
    items = tuple(Projection(it.alias, remap_columns(it.expr, sources)) for it in proj.items)
    return Distinct(Project(src, items))


# --- self-join elimination ---------------------------------------------------


@rule(name="self_join_elimination", phase=Phase.PUSHDOWN, matches=(Join,))
def self_join_elimination(node: Join, ctx: OptimizerContext) -> LogicalPlan | None:
    """A relation inner/full-joined to **itself** on a unique, non-null key → that relation.

    Three proofs, all required. The two subtrees compute the same relation (structurally
    identical IR over the *same bound sources* — `_relation_key`), the join key is proven
    unique on it, and the key is proven to hold no null. Then row `r` matches exactly one
    row — itself: some row (itself) has equal keys, so nothing is filtered (the one case
    where an inner join's filtering half is provable *without* referential integrity — the
    relation trivially references itself), and uniqueness means no other row shares its key,
    so nothing is duplicated. A `full` join adds nothing either: every row being matched,
    neither side has an unmatched row to null-extend.

    The non-null proof is what makes it safe: a null-keyed row does **not** match itself
    (`null = null` is null), so an inner self-join *drops* it and a full self-join emits
    it *twice* (once from each side). Both would be silent corruption, so an unknown null
    count means no rewrite. Requires a single-sided output (post-pruning), and re-projects
    that side under the join's exact output aliases, so the schema is unchanged.
    """
    if ctx is None or node.join_type not in ("inner", "full"):
        return None
    if node.left_keys != node.right_keys:  # a self-join is `k = k`, not `k = j`
        return None
    side = _output_side(node.output)
    if side is None or not _same_relation(node.left, node.right, ctx):
        return None
    if not _right_unique_on_keys(node, ctx) or not _keys_non_null(node.right, node.right_keys, ctx):
        return None
    return _passthrough(node.left if side == "left" else node.right, node.output)


@rule(name="self_semi_join_to_filter", phase=Phase.PUSHDOWN, matches=(Join,))
def self_semi_join_to_filter(node: Join, ctx: OptimizerContext) -> LogicalPlan | None:
    """`Semi Join(L, L)` on the same key → `L` filtered to its non-null keys.

    A semi join asks only "does *some* row on the right share my key?". When the right
    side **is** the left side, row `r` is its own witness — provided its keys are non-null,
    since `null = null` never matches. So the answer is a pure per-row predicate:
    ``k1 IS NOT NULL AND k2 IS NOT NULL AND …``. No uniqueness is needed (a semi join
    cannot duplicate), which makes this the one join-elimination rule that needs no
    statistics at all — the shape alone proves it. `x IN (SELECT x FROM t)` over the same
    table, and self-`EXISTS` correlation, lower to exactly this.

    When the keys are additionally *proven* non-null the predicate is a tautology and the
    filter is skipped entirely. Runs in PUSHDOWN — where the semi join may only just have
    been *created*, by `join_to_semijoin` or `inner_join_to_semi_when_right_unique` over a
    self-join — and the `IS NOT NULL` it leaves behind is pushed into the scan by the same
    phase. Idempotent (the result is a `Project`, not a `Join`).
    """
    if ctx is None or node.join_type != "semi" or node.left_keys != node.right_keys:
        return None
    if not _same_relation(node.left, node.right, ctx):
        return None
    if _keys_non_null(node.left, node.left_keys, ctx):
        return _passthrough(node.left, node.output)
    matched = Filter(node.left, _fold(operator.and_, IsNotNull, node.left_keys))
    return _passthrough(matched, node.output)


@rule(name="self_anti_join_to_null_keys", phase=Phase.PUSHDOWN, matches=(Join,))
def self_anti_join_to_null_keys(node: Join, ctx: OptimizerContext) -> LogicalPlan | None:
    """`Anti Join(L, L)` on the same key → `L` filtered to its **null** keys (usually empty).

    The exact complement of `self_semi_join_to_filter`, and the same proof read backwards:
    a row of `L` has *no* match in `L` precisely when it fails to match itself, which
    happens only when one of its keys is null. So the anti join is the per-row predicate
    ``k1 IS NULL OR k2 IS NULL OR …``. When the keys are *proven* non-null, no row can
    survive and the result is the canonical empty marker (`Limit(…, 0)`), carrying the
    join's exact output schema. Needs no uniqueness (an anti join emits each left row at
    most once). Idempotent (the result holds no `Join`).
    """
    if ctx is None or node.join_type != "anti" or node.left_keys != node.right_keys:
        return None
    if not _same_relation(node.left, node.right, ctx):
        return None
    if _keys_non_null(node.left, node.left_keys, ctx):
        projected = _passthrough(node.left, node.output)
        return None if projected is None else Limit(projected, 0)
    unmatched = Filter(node.left, _fold(operator.or_, IsNull, node.left_keys))
    return _passthrough(unmatched, node.output)


# --- cartesian (constant-key) joins ------------------------------------------


@rule(name="eliminate_cross_join_of_single_row", phase=Phase.PUSHDOWN, matches=(Join,))
def eliminate_cross_join_of_single_row(node: Join, ctx: OptimizerContext) -> LogicalPlan | None:
    """A cross join to a **one-row** relation whose columns are unread → drop the join.

    The one inner-join shape that *is* removable, and removable precisely because the
    missing referential-integrity guarantee is not missing here: a cartesian key (`1 = 1`)
    matches unconditionally, so the join cannot filter — no FK needed. An EXACT one-row
    other side then kills duplication too (exactly one match per row). Both halves proven,
    the join is an identity on the surviving side, so it becomes a projection of it under
    the join's own output aliases.

    Fires for `inner`/`left` when the *right* is the one-row side (a scalar subquery — a
    global aggregate is EXACT one row — cross-joined in and then not used), and for
    `inner`/`right` when the *left* is. The count must be EXACT: an *estimated* single row
    that is really two would halve the output. The symmetric-looking "`right` join, one-row
    right side" is **not** here — with an empty left it must still emit that row
    null-extended, which is a projection of neither input.
    """
    if ctx is None or not _cartesian_keys(node):
        return None
    side = _output_side(node.output)
    if side is None:
        return None
    if side == "left" and node.join_type in ("inner", "left"):
        other = node.right
    elif side == "right" and node.join_type in ("inner", "right"):
        other = node.left
    else:
        return None
    if _exact_rows(other, ctx) != 1:
        return None
    return _passthrough(node.left if side == "left" else node.right, node.output)


@rule(name="semi_join_of_nonempty_cartesian", phase=Phase.PUSHDOWN, matches=(Join,))
def semi_join_of_nonempty_cartesian(node: Join, ctx: OptimizerContext) -> LogicalPlan | None:
    """`Semi Join(L, R)` on a cartesian key with `R` provably non-empty → `L`.

    The decorrelated form of an *uncorrelated* ``WHERE EXISTS (SELECT … FROM r)``: with a
    constant key every left row matches every right row, so the existence test is a single
    global question — "does `R` have a row?" — and a proven-non-empty `R` answers yes for
    every left row at once. The semi join then keeps all of `L` unchanged (a semi join
    never duplicates), so it is a projection of `L` under the join's output aliases (which
    are left-only by definition of a semi join). The row count must be EXACT and ≥ 1; an
    estimated non-empty relation that is actually empty would turn an empty result into
    all of `L`. Runs in PUSHDOWN, where a *cross* join to a one-row relation also arrives
    in this shape (`inner_join_to_semi_when_right_unique` reduces it there — a one-row
    relation is trivially key-unique). Idempotent (the result is a `Project`).
    """
    if ctx is None or node.join_type != "semi" or not _cartesian_keys(node):
        return None
    rows = _exact_rows(node.right, ctx)
    if rows is None or rows < 1:
        return None
    return _passthrough(node.left, node.output)


@rule(name="anti_join_of_nonempty_cartesian_to_empty", phase=Phase.PUSHDOWN, matches=(Join,))
def anti_join_of_nonempty_cartesian_to_empty(
    node: Join, ctx: OptimizerContext
) -> LogicalPlan | None:
    """`Anti Join(L, R)` on a cartesian key with `R` provably non-empty → empty.

    The complement of `semi_join_of_nonempty_cartesian`, and the decorrelated form of an
    uncorrelated ``WHERE NOT EXISTS (SELECT … FROM r)``. Every left row matches (the
    constant key matches unconditionally, and `R` provably has a row to match), so *no*
    left row survives an anti join and the result is empty — the canonical `Limit(…, 0)`
    marker over a projection of `L`, which carries the join's exact output schema and
    lets `count()`/`is_empty()` answer from metadata without executing. EXACT row count
    required, for the same reason as the semi case. Idempotent (the result holds no `Join`).
    """
    if ctx is None or node.join_type != "anti" or not _cartesian_keys(node):
        return None
    rows = _exact_rows(node.right, ctx)
    if rows is None or rows < 1:
        return None
    projected = _passthrough(node.left, node.output)
    return None if projected is None else Limit(projected, 0)


# --- inner-join reduction ----------------------------------------------------


@rule(name="inner_join_to_semi_when_right_unique", phase=Phase.PUSHDOWN, matches=(Join,))
def inner_join_to_semi_when_right_unique(node: Join, ctx: OptimizerContext) -> LogicalPlan | None:
    """`Inner Join(L, R)` reading only L's columns, R unique on the key → a **semi** join.

    This is as far as an inner join can legally be reduced. It cannot be *removed*: with no
    referential-integrity guarantee (see the module docstring) it still drops every L row
    whose key is absent from R — the filtering half of a join, which no statistic proves
    away. A proven-unique R key *does* prove the other half: each L row matches at most one
    R row, so the join never duplicates. An inner join that neither duplicates nor exports
    an R column is exactly a semi join, and the win is real (the build side keeps keys only,
    no payload; the probe stops fanning out).

    It also unlocks the semi-join family — `drop_redundant_distinct_build`,
    `push_semijoin_through_join`. Uniqueness must be *proven* (structural `GROUP BY` /
    `DISTINCT` on the key, or an EXACT ndv reaching an EXACT row count); an estimate would
    silently collapse a legitimate fan-out. Runs in PUSHDOWN so column pruning has already
    reduced the output to one side, and is registered **after** the self-join and cartesian
    rules: they can delete such a join outright, and a rule that only matches `inner` must
    not get to rewrite it into a semi join first. Idempotent (semi is not inner).
    """
    if ctx is None or node.join_type != "inner":
        return None
    if not node.output or any(o.side == "right" for o in node.output):
        return None
    if not _right_unique_on_keys(node, ctx):
        return None
    return dataclasses.replace(node, join_type="semi")


# --- provably-disjoint keys --------------------------------------------------


@rule(name="join_disjoint_keys_to_empty", phase=Phase.REWRITE, matches=(Join,))
def join_disjoint_keys_to_empty(node: Join, ctx: OptimizerContext) -> LogicalPlan | None:
    """An inner/semi join whose key ranges are provably disjoint → an empty probe side.

    When one key's two sides have non-overlapping EXACT `[min, max]` ranges, no pair of
    rows can satisfy the equi-condition, so an inner or semi join emits nothing (both
    require a match). The rewrite marks the *left* input empty (`Limit(L, 0)`, the
    canonical empty marker) rather than fabricating a zero-row relation with the join's
    two-sided output schema — which the IR cannot express. The existing empty machinery
    then takes it from there: the estimator reports an EXACT zero row count (so `count()`
    answers from metadata) and `inner_join_empty_to_empty` /
    `semi_anti_join_empty_left` collapse the join once its output is single-sided.

    `anti` and the outer joins are excluded and handled by `no_match_join_to_preserved_side`
    — for them "nothing matches" means the preserved side survives *whole*, so emptying an
    input would delete the answer. Guarded on the left not already being empty, so it fires
    once (idempotent).
    """
    if ctx is None or node.join_type not in ("inner", "semi") or _is_empty(node.left):
        return None
    if not _disjoint_keys(node, ctx):
        return None
    return dataclasses.replace(node, left=Limit(node.left, 0))


@rule(name="no_match_join_to_preserved_side", phase=Phase.PUSHDOWN, matches=(Join,))
def no_match_join_to_preserved_side(node: Join, ctx: OptimizerContext) -> LogicalPlan | None:
    """A left/right/anti join whose key ranges are provably disjoint → its preserved side.

    With EXACT disjoint key ranges nothing matches, and a join that matches nothing does
    exactly one thing to its preserved side: pass it through. A `left` join emits every
    left row once (null-extended); an `anti` join emits every left row (none has a match);
    a `right` join emits every right row once. Each is its preserved side, unchanged — so
    the join is replaced by a projection of that side under the join's output aliases.

    Requires the output to be entirely that preserved side: a null-supplied column would
    have to be materialized as an all-null column the IR has no literal for. `full` is
    absent by construction (`_PRESERVED_SIDE`) — it preserves *both* sides, so with no
    matches it emits |L| + |R| rows and is a passthrough of neither. Runs in PUSHDOWN,
    after column pruning has had a chance to make the output single-sided; idempotent.
    """
    if ctx is None:
        return None
    preserved = _PRESERVED_SIDE.get(node.join_type)
    if preserved is None or _output_side(node.output) != preserved:
        return None
    if not _disjoint_keys(node, ctx):
        return None
    return _passthrough(node.left if preserved == "left" else node.right, node.output)
