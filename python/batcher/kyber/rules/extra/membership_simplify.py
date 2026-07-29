"""`IN`-list and `coalesce` simplifications.

Split out of `extra/boolean_algebra`, which is the Kleene-logic family proper (annihilators,
idempotence, absorption, complement, De Morgan). These three are simplifications of two
*n-ary* constructs rather than of the boolean connectives, and they reason about argument
lists: a one-element `IN` is an equality, a repeated member is redundant, and `coalesce`
drops everything after its first argument that can never be null.

Registration order is run order, so this module is imported from `extra/__init__` directly
after `boolean_algebra` -- the position these rules held when the two were one file.
"""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.kyber.rules.extra.boolean_algebra import _key, _keys, _rewrite_node, _safe
from batcher.plan.expr_ir import Binary, Coalesce, Expr, InList, Lit
from batcher.plan.logical import Filter, LogicalPlan, Project

__all__ = ["coalesce_simplify", "dedup_in_list", "single_in_list"]


def _single_in_list(expr: Expr) -> Expr:
    if isinstance(expr, InList) and len(expr.values) == 1:
        return Binary("eq", expr.input, Lit(expr.values[0]))
    return expr


@rule(
    name="single_in_list",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_single_in_list,
    expr_matches=(InList,),
)
def single_in_list(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`x IN (v) → x = v`. A one-element membership test is exactly an equality, with
    identical null behavior (`NULL IN (v)` and `NULL = v` are both null). Turns the
    opaque hash-set probe into a `col = literal` shape that constant propagation and
    pruning understand."""
    return _rewrite_node(node, _single_in_list)


def _dedup_in_list(expr: Expr) -> Expr:
    if isinstance(expr, InList) and len(expr.values) > 1:
        unique = tuple(dict.fromkeys(expr.values))
        if len(unique) < len(expr.values):
            return InList(expr.input, unique)
    return expr


@rule(
    name="dedup_in_list",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_dedup_in_list,
    expr_matches=(InList,),
)
def dedup_in_list(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`x IN (a, b, a) → x IN (a, b)`. Set membership is unchanged by duplicate values,
    so de-duplicating (first occurrence kept) shrinks the probe set with no change to
    the result. May expose a single-element list for `single_in_list` to fold to an
    equality."""
    return _rewrite_node(node, _dedup_in_list)


# --- COALESCE flattening ----------------------------------------------------


def _coalesce_simplify(expr: Expr) -> Expr:
    if not isinstance(expr, Coalesce):
        return expr
    flat: list[Expr] = []
    for arg in expr.inputs:
        flat.extend(arg.inputs if isinstance(arg, Coalesce) else [arg])
    out: list[Expr] = []
    seen: set[str] = set()
    for arg in flat:
        if _safe(arg):
            k = _key(arg)
            if k in seen:
                continue  # an earlier identical (deterministic) arg already covers it
            seen.add(k)
        out.append(arg)
    # NB: truncating the tail after the first non-null *literal* is deliberately NOT done
    # here. It is sound only when the dropped tail's type is already carried by a kept arm
    # — otherwise it narrows the result type (a `COALESCE`'s type is the *join* of its
    # arms', so dropping `CAST(-1 AS DOUBLE)` from `coalesce(5, CAST(-1 AS DOUBLE))` turns
    # a DOUBLE `5.0` into an INT `5`). That type-guarded truncation lives in
    # `coalesce_drop_nulls_after_first_non_null` (`_droppable`); doing an unguarded version
    # here silently changed the output dtype and value. Flatten + dedup below only ever
    # drop an arm structurally identical to a kept one, so they cannot move the type.
    if len(out) == 1:
        return out[0]
    if _keys(out) == _keys(expr.inputs):
        return expr
    return Coalesce(out)


@rule(
    name="coalesce_simplify",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_coalesce_simplify,
    expr_matches=(Coalesce,),
)
def coalesce_simplify(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Flatten and shrink a `COALESCE`: inline a nested `COALESCE`, drop a later
    duplicate of an earlier `_safe` argument, and unwrap `COALESCE(x)` to `x`. Each step
    preserves "first non-null argument" *and* the result type: a repeated argument is only
    ever reached when it is null and is structurally identical to a kept one, so removing
    it moves neither the value nor the type. Truncating the tail after the first non-null
    *literal* is left to `coalesce_drop_nulls_after_first_non_null`, which guards it with a
    type check (`_droppable`) — dropping a differently-typed tail here would narrow the
    `COALESCE`'s join type (turning DOUBLE `5.0` into INT `5`)."""
    return _rewrite_node(node, _coalesce_simplify)
