"""The proofs a join elimination must clear before it may delete or degenerate a join.

Every helper here answers one question about *evidence*, never about a rewrite: are these
two key ranges provably disjoint, is this key provably non-null, is this a cartesian
(constant) key, are these two subtrees provably the same relation? They are separated from
the rewrites (`rules`) because that is the seam that matters in this family — a rule is only
as sound as the proof it consults, and a proof that lives once cannot drift between the
rules that share it.

Two invariants bind all of it. Evidence must be `Provenance.EXACT`: a sketched or learned
statistic is a guess, and a guess can never justify deleting rows. And a float bound is
refused whenever it is ambiguous (`-0.0`/NaN), because the engine's key equality is
canonicalized while these comparisons are Python's — the two can disagree about which
values are "the same", and that disagreement once deleted a row a join would have matched.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import reduce

from batcher.kyber.pass_base import OptimizerContext
from batcher.plan.expr_ir import Col, Expr
from batcher.plan.logical import Join, LogicalPlan, Scan, is_cartesian_key_pair
from batcher.plan.logical.transforms import constant_column_value
from batcher.plan.stats import ColumnStat, Provenance
from batcher.plan.visitor import walk

__all__: list[str] = []


def _exact_range(stat: ColumnStat) -> bool:
    """Whether `stat` carries an EXACT, fully-populated ``[min, max]`` range."""
    return stat.provenance is Provenance.EXACT and stat.min is not None and stat.max is not None


def _disjoint_keys(node: Join, ctx: OptimizerContext) -> bool:
    """Whether some key pair's EXACT value ranges prove **no** row can ever match.

    An equi-join emits a pair only when *every* key is equal, so one key whose two sides
    have non-overlapping `[min, max]` ranges is enough to prove the join matches nothing.
    Both ranges must be `Provenance.EXACT` (a filtered/estimated bound is only a guess).
    Nulls need no care: `min`/`max` ignore them and a null key matches nothing anyway.
    Incomparable bound types (a `TypeError` on the comparison) are simply undecidable.
    """
    from batcher.plan.stats import ambiguous_float_bound

    left = ctx.estimator.estimate(node.left)
    right = ctx.estimator.estimate(node.right)
    for lk, rk in zip(node.left_keys, node.right_keys, strict=True):
        a, b = left.column(lk), right.column(rk)
        if not (_exact_range(a) and _exact_range(b)):
            continue
        # A NaN or zero float bound proves no disjointness: the engine's key equality is
        # *canonicalized* (`-0.0` folds into `0.0`, every NaN into one value) while this
        # comparison is Python's, so the two can disagree about which values are "the same".
        if any(ambiguous_float_bound(v) for v in (a.min, a.max, b.min, b.max)):
            continue
        try:
            if a.max < b.min or b.max < a.min:
                return True
        except TypeError:
            continue  # incomparable bounds → undecidable, not disjoint
    return False


def _keys_non_null(plan: LogicalPlan, keys: tuple[str, ...], ctx: OptimizerContext) -> bool:
    """Whether every one of `keys` is *proven* to hold no null on `plan`.

    An equi-join never matches a null key (`null = null` is null, not true), so a rule that
    claims "every row matches itself" needs this proof, not just uniqueness. Only an EXACT
    null count proves it; unknown (`None`) is not zero.
    """
    stats = ctx.estimator.estimate(plan)
    for key in keys:
        stat = stats.column(key)
        if stat.provenance is not Provenance.EXACT or stat.null_count != 0:
            return False
    return True


def _cartesian_keys(node: Join) -> bool:
    """Whether every key pair is the same **non-null** constant on both sides.

    That is the `__cross_key` a comma/cross join lowers to: the condition is `1 = 1`, so it
    connects nothing and every left row matches every right row. The non-null check is
    load-bearing, not pedantic — a constant *null* key on both sides matches *nothing*
    (`null = null` is null), the exact opposite conclusion.
    """
    if not node.left_keys:
        return False
    for lk, rk in zip(node.left_keys, node.right_keys, strict=True):
        if not is_cartesian_key_pair(node.left, lk, node.right, rk):
            return False
        if constant_column_value(node.left, lk) is None:
            return False
    return True


def _relation_key(plan: LogicalPlan, ctx: OptimizerContext) -> tuple | None:
    """A structural identity for `plan` in which a `Scan` names the *source it is bound to*.

    Two sides of a self-join are not IR-equal: `ds.join(ds, ...)` binds the same source
    object twice and the right `Scan` gets a fresh `source_id`. So the IR is normalized
    (every `source_id` blanked) and the bound source objects are carried alongside, in walk
    order. Returns `None` — never a match — when a scan is unbound (a plan-shape-only
    optimize) or there is no scan: an identity we cannot resolve to data is not one we may
    act on.
    """
    identities: list[int] = []
    for node in walk(plan):
        if isinstance(node, Scan):
            if node.source_id >= len(ctx.sources):
                return None
            identities.append(id(ctx.sources[node.source_id]))
    if not identities:
        return None
    return (_blank_source_ids(plan.to_ir()), tuple(identities))


def _blank_source_ids(ir: object) -> object:
    """`ir` with every `source_id` zeroed, so two scans differ only by what they read."""
    if isinstance(ir, dict):
        return {k: (0 if k == "source_id" else _blank_source_ids(v)) for k, v in ir.items()}
    if isinstance(ir, list):
        return [_blank_source_ids(v) for v in ir]
    return ir


def _same_relation(left: LogicalPlan, right: LogicalPlan, ctx: OptimizerContext) -> bool:
    """Whether two subtrees provably compute the same relation (a self-join's two sides)."""
    key = _relation_key(left, ctx)
    return key is not None and key == _relation_key(right, ctx)


def _fold(op: Callable[[Expr, Expr], Expr], test: type[Expr], keys: tuple[str, ...]) -> Expr:
    """Fold a per-key null test over the join keys with `op` (AND for semi, OR for anti)."""
    return reduce(op, [test(Col(k)) for k in keys])
