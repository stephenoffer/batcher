"""Conditional algebra: moving work across a `CASE`, and pruning `GREATEST`/`LEAST`.

`extra/conditional` already drops unreachable branches, de-duplicates conditions and
arguments, folds an all-literal `GREATEST`, and turns a two-branch null test into
`COALESCE`. What it does not do is *move* work across a conditional, and that is
where the remaining wins are.

The centrepiece is `push_foldable_into_case_branches`, Spark's `PushFoldableIntoBranches`.
A `CASE ... END = 5` evaluates the comparison once per row over a materialized
intermediate column; pushed inside, every branch result is a literal, so the whole
comparison constant-folds and the `CASE` collapses to a boolean pick. It is also an
enabler rather than just a saving: a `CASE` is opaque to zone-map pruning and to
predicate pushdown, and the folded form is a plain `col OP literal` that both can act
on. `flatten_nested_case_in_else` does the structural half of the same job, turning
the right-leaning chain a SQL `CASE` ladder or a chain of `if_else` calls produces
into one flat multi-branch node the engine evaluates in a single pass.

The rest unwraps conditionals that are something simpler in disguise: a `CASE` whose
branches are just `TRUE`/`FALSE` is a predicate written the long way, a `NOT` over a
literal-branch `CASE` is a `CASE` over negated literals, and a `GREATEST` carrying
several literals only needs the largest of them.

One shape is deliberately absent. There is no null *literal* in this IR -- `Lit`
rejects `None`, so `NULLIF(x, NULL)` and `greatest(x, NULL)` cannot be built at all.
Rules matching those forms would be unreachable code, and a rule *producing* a null
literal would emit a plan that fails to serialize, so neither exists here.
"""

from __future__ import annotations

import dataclasses

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import DEFAULT_REGISTRY, rule
from batcher.kyber.rule import Phase, node_rule
from batcher.kyber.rules.leaf_rewrite import rewrite_node, safe_expr
from batcher.plan.expr_ir import Binary, Case, Expr, Greatest, Least, Lit, Not
from batcher.plan.expr_rewrite import combine_conjuncts, expr_key, split_conjuncts
from batcher.plan.logical import Filter, LogicalPlan, Project

__all__ = [
    "PUSH_FOLDABLE_INTO_CASE_RULES",
    "case_boolean_branches_to_predicate",
    "drop_case_branch_matching_else",
    "flatten_nested_case_in_else",
    "prune_dominated_literal_in_greatest_least",
    "push_not_into_case_branches",
]

#: Comparison and wrapping-arithmetic operators a literal may be pushed through into
#: every branch of a `CASE`. Each is total on the literal branch results the rule
#: requires, so the pushed form raises exactly when the original did: never.
_PUSHABLE_OPS = frozenset({"eq", "ne", "lt", "le", "gt", "ge", "add", "sub", "mul"})


def _is_literal(expr: Expr) -> bool:
    return isinstance(expr, Lit)


def _all_literal_results(case: Case) -> bool:
    """Whether every value a `CASE` can produce is a literal."""
    return all(_is_literal(v) for _, v in case.branches) and _is_literal(case.otherwise)


def _distribute_over_case(op: str, case: Case, other: Expr, *, case_on_left: bool) -> Case:
    """Rebuild `case` with `op` applied to each branch result against `other`."""

    def combine(value: Expr) -> Expr:
        return Binary(op, value, other) if case_on_left else Binary(op, other, value)

    return Case(
        [(cond, combine(value)) for cond, value in case.branches],
        combine(case.otherwise),
    )


def _push_foldable(op: str):
    """Build the leaf rewrite distributing one operator into a literal-branch `CASE`."""

    def leaf(expr: Expr) -> Expr:
        if not (isinstance(expr, Binary) and expr.op == op):
            return expr
        for case_side, other, case_on_left in (
            (expr.left, expr.right, True),
            (expr.right, expr.left, False),
        ):
            if (
                isinstance(case_side, Case)
                and _is_literal(other)
                and _all_literal_results(case_side)
            ):
                return _distribute_over_case(op, case_side, other, case_on_left=case_on_left)
        return expr

    return leaf


def _make_push_rule(op: str):
    leaf = _push_foldable(op)

    def apply(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
        return rewrite_node(node, leaf)

    return apply


# One rule per pushable operator, over a shared body.
#
# `(CASE WHEN c THEN 1 ELSE 2 END) = 1 -> CASE WHEN c THEN TRUE ELSE FALSE END`. When
# every branch produces a literal and the other operand is one too, moving the operation
# inside makes each branch a constant the folder evaluates away, and the boolean-branch
# rule then collapses the survivor to a bare predicate.
#
# That chain is worth more than the arithmetic it saves: a `CASE` is opaque to zone-map
# pruning, predicate pushdown, and the sargable normalizers, so a predicate wearing one
# is stuck where it was written. What comes out is a plain `col OP literal`, which all
# three can act on. Requiring *literal* branch results (not merely foldable ones) is what
# keeps this unable to raise or to duplicate per-row work.
PUSH_FOLDABLE_INTO_CASE_RULES = [
    DEFAULT_REGISTRY.add(
        node_rule(
            f"push_{op}_into_case_branches",
            Phase.NORMALIZE,
            _make_push_rule(op),
            matches=(Filter, Project),
            expr_fn=_push_foldable(op),
            expr_matches=(Binary,),
            expr_ops=(op,),
        )
    )
    for op in sorted(_PUSHABLE_OPS)
]


def _flatten_case_else(expr: Expr) -> Expr:
    if isinstance(expr, Case) and isinstance(expr.otherwise, Case):
        inner = expr.otherwise
        return Case(list(expr.branches) + list(inner.branches), inner.otherwise)
    return expr


@rule(
    name="flatten_nested_case_in_else",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_flatten_case_else,
    expr_matches=(Case,),
)
def flatten_nested_case_in_else(
    node: Filter | Project, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """Fold a `CASE` nested in another's `ELSE` into one flat multi-branch `CASE`.

    `CASE WHEN a THEN x ELSE (CASE WHEN b THEN y ELSE z END) END` and
    `CASE WHEN a THEN x WHEN b THEN y ELSE z END` are the same function: branches are
    tested in order and the first true one wins, so appending the inner branches after
    the outer ones preserves the priority exactly.

    A chain of `if_else` calls, and a SQL `CASE` ladder lowered one branch at a time,
    both arrive right-leaning. Flattening lets the engine evaluate the whole ladder in
    a single pass over the batch instead of nesting a select per level, and it exposes
    the branch set to the duplicate-condition and unreachable-branch rules, which only
    look one level deep."""
    return rewrite_node(node, _flatten_case_else)


def _case_boolean_predicate(expr: Expr) -> Expr:
    if (
        isinstance(expr, Case)
        and len(expr.branches) == 1
        and isinstance(expr.otherwise, Lit)
        and isinstance(expr.branches[0][1], Lit)
    ):
        condition, then = expr.branches[0]
        otherwise = expr.otherwise
        if then.value is True and otherwise.value is False:
            return condition
        # The mirror -- `THEN FALSE ELSE TRUE` -> `NOT c` -- is NOT sound here, even at
        # the top of a filter, and a differential test caught it. On a row where `c` is
        # `NULL` the `CASE` falls to its `ELSE` and yields `TRUE`, keeping the row,
        # while `NOT c` is `NULL` and drops it. The safe direction is only the one
        # where the `CASE` produces `FALSE` and the predicate produces `NULL`, since a
        # filter discards both.
    return expr


def _rewrite_conjuncts(node: Filter, leaf) -> LogicalPlan | None:
    """Apply `leaf` to each *top-level* conjunct of a filter predicate only.

    The distinction matters for any rewrite that trades `NULL` for `FALSE`. A filter
    keeps a row only when the whole predicate is true, so at the top level the two are
    interchangeable -- but one `NOT` or `OR` deeper they are not, and a plain
    bottom-up walk would happily rewrite there too.
    """
    conjuncts = split_conjuncts(node.predicate)
    rewritten = [leaf(c) for c in conjuncts]
    if all(new is old for new, old in zip(rewritten, conjuncts, strict=True)):
        return None
    combined = combine_conjuncts(rewritten)
    return dataclasses.replace(node, predicate=combined)


@rule(name="case_boolean_branches_to_predicate", phase=Phase.NORMALIZE, matches=(Filter,))
def case_boolean_branches_to_predicate(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`CASE WHEN c THEN TRUE ELSE FALSE END -> c`, inside a filter.

    This is a predicate written the long way, and unwrapping it exposes `c` to every
    rule that matches on comparison shape -- pushdown, zone-map pruning, join-key
    inference -- none of which can see through a `CASE`.

    Two restrictions carry the soundness argument, and neither is a scoping choice.

    **Filter only, top-level conjuncts only.** The two forms are not equal as *values*:
    when `c` is `NULL` the `CASE` takes its `ELSE` and yields `FALSE`, while `c` itself
    stays `NULL`. A filter keeps a row only when the predicate is true, and neither
    `FALSE` nor `NULL` is true, so the same rows survive -- but only at the top level.
    Under a `NOT` or an `OR`, or inside a `Project`, the difference is observable, which
    is why the leaf is applied to `split_conjuncts` of the predicate rather than walked
    over the whole tree.

    **One direction only.** The apparent mirror, `THEN FALSE ELSE TRUE -> NOT c`, is
    wrong even at the top level: a `NULL` `c` makes the `CASE` yield `TRUE` and keep the
    row, while `NOT c` is `NULL` and drops it. A differential test against DuckDB caught
    exactly that, and the rewrite is now refused."""
    return _rewrite_conjuncts(node, _case_boolean_predicate)


def _drop_branch_matching_else(expr: Expr) -> Expr:
    if isinstance(expr, Case) and expr.branches:
        last_condition, last_value = expr.branches[-1]
        if safe_expr(last_condition) and expr_key(last_value) == expr_key(expr.otherwise):
            branches = list(expr.branches[:-1])
            return Case(branches, expr.otherwise) if branches else expr.otherwise
    return expr


@rule(
    name="drop_case_branch_matching_else",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_drop_branch_matching_else,
    expr_matches=(Case,),
)
def drop_case_branch_matching_else(
    node: Filter | Project, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """Drop a `CASE`'s last branch when its result already equals the `ELSE`.

    A row reaching the last branch produces the same value whether the branch fires or
    falls through, so the branch changes nothing and its condition need not be
    evaluated at all. Only the *last* branch qualifies: an earlier one shadows the
    branches after it, so removing it would let a later condition win where it
    previously could not.

    The dropped condition must be `safe_expr`, since it stops being evaluated."""
    return rewrite_node(node, _drop_branch_matching_else)


def _push_not_into_case(expr: Expr) -> Expr:
    if isinstance(expr, Not) and isinstance(expr.input, Case) and _all_literal_results(expr.input):
        case = expr.input
        return Case(
            [(cond, Not(value)) for cond, value in case.branches],
            Not(case.otherwise),
        )
    return expr


@rule(
    name="push_not_into_case_branches",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_push_not_into_case,
    expr_matches=(Case, Not),
)
def push_not_into_case_branches(
    node: Filter | Project, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`NOT (CASE WHEN c THEN TRUE ELSE FALSE END) -> CASE WHEN c THEN FALSE ELSE TRUE END`.

    De Morgan for the conditional: negating the whole expression and negating each
    branch result agree, because exactly one branch supplies the value on any row and
    `NOT` is applied to that value either way. Null needs no special case -- a branch
    yielding `NULL` yields `NULL` under `NOT` on both sides.

    Restricted to literal branch results, which is what makes it a win rather than a
    shuffle: each negation is then a constant the folder evaluates away, leaving a
    `CASE` with no `NOT` above it for the boolean-branch rule to unwrap."""
    return rewrite_node(node, _push_not_into_case)


def _prune_dominated_literal(expr: Expr) -> Expr:
    if not isinstance(expr, (Greatest, Least)) or len(expr.inputs) < 2:
        return expr
    literals = [e for e in expr.inputs if isinstance(e, Lit)]
    if len(literals) < 2:
        return expr
    kinds = {type(e.value) for e in literals}
    if kinds - {int} and kinds - {str}:  # one exact, totally ordered type class only
        return expr
    keep = (max if isinstance(expr, Greatest) else min)(e.value for e in literals)
    seen = False
    kept: list[Expr] = []
    for e in expr.inputs:
        if isinstance(e, Lit):
            if e.value == keep and not seen:
                seen = True
                kept.append(e)
        else:
            kept.append(e)
    return type(expr)(kept)


@rule(
    name="prune_dominated_literal_in_greatest_least",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_prune_dominated_literal,
    expr_matches=(Greatest, Least, Lit),
)
def prune_dominated_literal_in_greatest_least(
    node: Filter | Project, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`GREATEST(x, 1, 5) -> GREATEST(x, 5)`, and the `LEAST` dual keeping the smallest.

    Among several constant arguments only the extreme one can ever win, whatever the
    columns hold, so the rest are dead weight -- each is materialized as a full-length
    array and compared against on every row. The existing all-literal fold cannot help
    here, because it requires *every* argument to be a literal; this is the mixed case.

    Only integer or string literals qualify, and only when they are all of one class.
    That excludes floats and booleans for the same reasons the all-literal fold does:
    NaN's position in the order is engine-specific, and `-0.0 == 0.0` makes it
    observable which of two equal zeros survived."""
    return rewrite_node(node, _prune_dominated_literal)
