"""GREATEST / LEAST rewrites."""

from __future__ import annotations

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase

# `_key` (structural identity), `_rewrite_node` (leaf Expr rule → rebuilt node, or None) and
# `_safe` (deterministic + non-erroring) are the sibling family's helpers, imported rather than
# re-implemented — copy-paste is the one wrong way to share.
from batcher.kyber.rules.extra.boolean_algebra import _key, _rewrite_node
from batcher.kyber.rules.extra.conditional.shared import (
    _FOLDABLE_LIT_CLASSES,
    _lit_class,
    _Node,
    _pure,
)
from batcher.plan.expr_ir import (
    Expr,
    Greatest,
    Least,
    Lit,
)
from batcher.plan.logical import Filter, LogicalPlan, Project


def _greatest_least_single(expr: Expr) -> Expr:
    if isinstance(expr, (Greatest, Least)) and len(expr.inputs) == 1:
        return expr.inputs[0]
    return expr


@rule(
    name="greatest_least_single_arg",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_greatest_least_single,
    expr_matches=(Greatest, Least),
)
def greatest_least_single_arg(node: _Node, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`greatest(x)` → `x`, `least(x)` → `x`. The extremum of one value is that value — including a
    null one: both ignore nulls, and a lone null argument leaves nothing to return, i.e. null. The
    type is the single argument's type either way."""
    return _rewrite_node(node, _greatest_least_single)


def _greatest_least_flatten(expr: Expr) -> Expr:
    if not isinstance(expr, (Greatest, Least)):
        return expr
    kind = type(expr)
    if not any(isinstance(a, kind) for a in expr.inputs):
        return expr
    flat: list[Expr] = []
    for arg in expr.inputs:
        flat.extend(arg.inputs if isinstance(arg, kind) else [arg])
    return kind(flat)


@rule(
    name="greatest_least_flatten_nested",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_greatest_least_flatten,
    expr_matches=(Greatest, Least),
)
def greatest_least_flatten_nested(node: _Node, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`greatest(a, greatest(b, c))` → `greatest(a, b, c)`, and the `least` dual. The extremum over
    the non-null arguments is associative, and the null case agrees: the inner call is null only if
    `b` and `c` are both null, and the outer call ignores that null exactly as the flat form ignores
    the two. Only a *same-kind* nesting is spliced (a `least` inside a `greatest` is a real
    sub-computation); no argument moves, so the type join stands."""
    return _rewrite_node(node, _greatest_least_flatten)


def _greatest_least_dedup(expr: Expr) -> Expr:
    if not isinstance(expr, (Greatest, Least)) or len(expr.inputs) < 2:
        return expr
    kept: list[Expr] = []
    seen: set[str] = set()
    for arg in expr.inputs:
        if _key(arg) in seen and _pure(arg):
            continue
        seen.add(_key(arg))
        kept.append(arg)
    if len(kept) == len(expr.inputs):
        return expr
    return type(expr)(kept)


@rule(
    name="greatest_least_dedup_args",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_greatest_least_dedup,
    expr_matches=(Greatest, Least),
)
def greatest_least_dedup_args(node: _Node, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`greatest(a, b, a)` → `greatest(a, b)`, and the `least` dual. The extremum is idempotent
    (`max(v, v) = v`) and null-ignoring, so a repeated argument can change neither which value wins
    nor whether the result is null. Identity is structural and the repeat must be pure, so the two
    occurrences provably carry the same value; the surviving copy keeps the type."""
    return _rewrite_node(node, _greatest_least_dedup)


def _greatest_least_fold(expr: Expr) -> Expr:
    if not isinstance(expr, (Greatest, Least)) or len(expr.inputs) < 2:
        return expr
    if not all(isinstance(a, Lit) for a in expr.inputs):
        return expr
    values = [a.value for a in expr.inputs]
    classes = {_lit_class(v) for v in values}
    if len(classes) != 1 or classes.pop() not in _FOLDABLE_LIT_CLASSES:
        return expr
    return Lit(max(values) if isinstance(expr, Greatest) else min(values))


@rule(
    name="greatest_least_fold_literals",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_greatest_least_fold,
    expr_matches=(Greatest, Least, Lit),
)
def greatest_least_fold_literals(node: _Node, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`greatest(2, 5, 3)` → `5`, and the `least` dual, when *every* argument is a literal of one
    comparable type. The extremum of constants is a constant, and one type class keeps the ordering
    and the result type exact — a mixed `greatest(1, 2.5)` takes the *join* type, and folding it to
    an INT literal would narrow it. Floats and booleans are excluded: NaN's ordering is
    engine-specific, and `-0.0 == 0.0` makes it observable *which* equal zero survives."""
    return _rewrite_node(node, _greatest_least_fold)
