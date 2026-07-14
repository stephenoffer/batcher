"""NORMALIZE-phase boolean / CASE / COALESCE / NULL simplifications.

A family of small, node-local rewrites that clean up the boolean and null-handling
shapes a SQL front end (and the earlier normalize rules) leave behind: annihilators
(`x AND FALSE`), idempotence (`x AND x`), absorption (`x AND (x OR y)`),
complementation on total predicates (`is_null(c) AND NOT is_null(c)`), pushing `NOT`
through a comparison (`NOT (a < b) → a >= b`), collapsing a boolean equality against a
literal (`x = TRUE → x`), single-value / duplicate `IN` lists, and `COALESCE`
flattening. Each shrinks the expression the data plane evaluates and exposes cleaner
`col OP literal` shapes for pushdown and zone-map pruning.

Every rule here is proven under the engine's **three-valued (Kleene) logic** — the
`and_kleene`/`or_kleene` kernels give `NULL AND FALSE = FALSE`, `NULL OR TRUE = TRUE`
— and under its **total-order** float comparisons (so negating an ordered comparison
is exact, NaN included). Rewrites that would only hold for non-null operands are
guarded (`complement_total_bool` fires only on `is_null`/`is_not_null`, which never
yield null); rewrites that drop or de-duplicate a sub-expression require it to be
`_safe` (deterministic and non-erroring) so the query's value **and** its error
behavior are preserved. Anything that cannot be proven under those rules is left
untouched (return `None`).

These do not overlap the existing NORMALIZE rules (`constant_folding`,
`expr_simplification`'s identity elements, `constant_propagation`,
`factor_common_conjuncts`, `fold_in_list`, `like_prefix_to_range`,
`or_to_in_and_range`, `date_trunc_to_range`): those fold constants and drop identity
elements; these handle annihilators, absorption, complementation, comparison
negation, and CASE/COALESCE/IN structure.
"""

from __future__ import annotations

import json

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.plan.expr_ir import (
    Binary,
    Coalesce,
    Col,
    Expr,
    InList,
    IsNotNull,
    IsNull,
    Lit,
    Not,
)
from batcher.plan.expr_ir.core import IsInf, IsNan
from batcher.plan.expr_rewrite import map_node_expressions, transform_expr_up
from batcher.plan.logical import Filter, LogicalPlan, Project

__all__ = [
    "and_absorption",
    "and_false_annihilator",
    "and_idempotent",
    "bool_eq_literal",
    "coalesce_simplify",
    "complement_total_bool",
    "dedup_in_list",
    "fold_not_comparison",
    "or_absorption",
    "or_idempotent",
    "or_true_annihilator",
    "single_in_list",
]

# Binary ops that are deterministic and cannot raise (no div/mod, whose zero divisor
# aborts; no cast, whose strict form errors on a bad value). Wrapping add/sub/mul and
# the comparison / boolean ops are total.
_SAFE_BINARY_OPS = frozenset({"and", "or", "eq", "ne", "lt", "le", "gt", "ge", "add", "sub", "mul"})
# Exact Kleene/total-order negation of each comparison: null-in-null-out on both
# sides, and trichotomy holds under the engine's total float order (NaN included).
_NEGATED_COMPARISON = {"eq": "ne", "ne": "eq", "lt": "ge", "le": "gt", "gt": "le", "ge": "lt"}


def _key(expr: Expr) -> str:
    """A hashable structural identity for an expression (its IR rendered stable)."""
    return json.dumps(expr.to_ir(), sort_keys=True)


def _keys(exprs: list[Expr]) -> list[str]:
    return [_key(e) for e in exprs]


def _safe(expr: Expr) -> bool:
    """Whether `expr` is deterministic and cannot raise, so removing or de-duplicating
    it preserves the query's value *and* its error behavior.

    A conservative whitelist: columns, literals, comparison/boolean/wrapping-arith
    binaries, `NOT`, and the null/NaN/inf predicates and `IN` membership. It excludes
    division/modulo (a zero divisor aborts), casts (strict casts error), and every
    opaque function call (potentially non-total)."""
    if isinstance(expr, (Lit, Col)):
        return True
    if isinstance(expr, Binary):
        return expr.op in _SAFE_BINARY_OPS and _safe(expr.left) and _safe(expr.right)
    if isinstance(expr, (Not, IsNull, IsNotNull, IsNan, IsInf)):
        return _safe(expr.input)
    if isinstance(expr, InList):
        return _safe(expr.input)
    return False


def _total_bool(expr: Expr) -> bool:
    """Whether `expr` is a boolean that never yields null — `is_null`/`is_not_null`
    map a null input to true/false, so complementation collapses to a real constant."""
    return isinstance(expr, (IsNull, IsNotNull))


def _bool_valued(expr: Expr) -> bool:
    """Whether `expr` provably evaluates to a boolean (so `= TRUE` / `= FALSE` is a
    boolean-vs-boolean comparison). A bare `Col` is excluded — its type is unknown."""
    if isinstance(expr, Binary):
        return expr.op in _NEGATED_COMPARISON or expr.op in ("and", "or")
    return isinstance(expr, (Not, IsNull, IsNotNull, IsNan, IsInf, InList))


def _is_true(expr: Expr) -> bool:
    return isinstance(expr, Lit) and expr.value is True


def _is_false(expr: Expr) -> bool:
    return isinstance(expr, Lit) and expr.value is False


def _rewrite_node(node: LogicalPlan, leaf) -> LogicalPlan | None:
    """Apply a leaf `Expr → Expr` rewrite to every expression in `node`, returning the
    rebuilt node, or `None` when nothing changed (so the driver reaches a fixpoint).

    **Identity first.** `map_node_expressions` and `transform_expr_up` share structure: when
    a rule touches nothing — the overwhelming case, since each of the hundred-odd expression
    rules matches a handful of shapes and passes over the rest — the *same* node object comes
    back. That is an O(1) "no change", and it is the answer almost every time this is called.

    Falling straight through to `to_ir() != to_ir()` instead meant serializing the node's
    whole expression tree to JSON **twice, per rule, per node, per fixpoint iteration** just
    to conclude nothing had happened. That serialization — not the rewriting — was what made
    the rule set expensive to plan with: it is quadratic in (rules x expression size), and it
    is pure waste. The IR comparison is still needed on the path where the object *did*
    change, because a rule may rebuild an equal-but-new tree (`Lit(False)` over an already
    `Lit(False)`), and treating that as a change would spin the fixpoint forever."""
    new = map_node_expressions(node, lambda e: transform_expr_up(e, leaf))
    if new is node:
        return None  # structural sharing proved the rewrite was a no-op
    return new if new.to_ir() != node.to_ir() else None


# --- annihilators -----------------------------------------------------------


def _and_false(expr: Expr) -> Expr:
    if isinstance(expr, Binary) and expr.op == "and":
        if _is_false(expr.right) and _safe(expr.left):
            return Lit(False)
        if _is_false(expr.left) and _safe(expr.right):
            return Lit(False)
    return expr


@rule(
    name="and_false_annihilator",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_and_false,
)
def and_false_annihilator(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`x AND FALSE → FALSE`. Under the engine's Kleene AND, `NULL AND FALSE` and
    `TRUE AND FALSE` are both FALSE, so a `FALSE` operand forces the whole conjunction
    to FALSE regardless of the other side. Fires only when the dropped operand is
    `_safe` (deterministic, non-erroring), so removing it changes neither the result
    nor whether the query errors."""
    return _rewrite_node(node, _and_false)


def _or_true(expr: Expr) -> Expr:
    if isinstance(expr, Binary) and expr.op == "or":
        if _is_true(expr.right) and _safe(expr.left):
            return Lit(True)
        if _is_true(expr.left) and _safe(expr.right):
            return Lit(True)
    return expr


@rule(
    name="or_true_annihilator",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_or_true,
)
def or_true_annihilator(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`x OR TRUE → TRUE`. Under the engine's Kleene OR, `NULL OR TRUE` and
    `FALSE OR TRUE` are both TRUE, so a `TRUE` operand forces the disjunction to TRUE.
    Fires only when the dropped operand is `_safe`."""
    return _rewrite_node(node, _or_true)


# --- idempotence ------------------------------------------------------------


def _and_idem(expr: Expr) -> Expr:
    if (
        isinstance(expr, Binary)
        and expr.op == "and"
        and _safe(expr.left)
        and _key(expr.left) == _key(expr.right)
    ):
        return expr.left
    return expr


@rule(
    name="and_idempotent",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_and_idem,
)
def and_idempotent(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`x AND x → x`. In Kleene logic `v AND v = v` for every value (`T,F,N`), so a
    self-conjunction is redundant. Restricted to a `_safe` `x` so collapsing two
    evaluations to one cannot change a value or drop an error."""
    return _rewrite_node(node, _and_idem)


def _or_idem(expr: Expr) -> Expr:
    if (
        isinstance(expr, Binary)
        and expr.op == "or"
        and _safe(expr.left)
        and _key(expr.left) == _key(expr.right)
    ):
        return expr.left
    return expr


@rule(
    name="or_idempotent",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_or_idem,
)
def or_idempotent(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`x OR x → x`. In Kleene logic `v OR v = v` for every value, so a
    self-disjunction is redundant. Restricted to a `_safe` `x`."""
    return _rewrite_node(node, _or_idem)


# --- absorption -------------------------------------------------------------


def _absorbs(x: Expr, compound: Expr, inner_op: str) -> bool:
    """Whether `compound` is `Binary(inner_op, …)` with `x` as one operand, and both
    are `_safe` (so dropping the compound's other operand and de-duplicating `x` is
    sound)."""
    return (
        isinstance(compound, Binary)
        and compound.op == inner_op
        and _safe(x)
        and _safe(compound)
        and (_key(x) == _key(compound.left) or _key(x) == _key(compound.right))
    )


def _and_absorb(expr: Expr) -> Expr:
    if isinstance(expr, Binary) and expr.op == "and":
        if _absorbs(expr.left, expr.right, "or"):
            return expr.left
        if _absorbs(expr.right, expr.left, "or"):
            return expr.right
    return expr


@rule(
    name="and_absorption",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_and_absorb,
)
def and_absorption(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`x AND (x OR y) → x`. The Kleene absorption law holds for all three values
    (`x=T`: `T`; `x=F`: `F`; `x=N`: `N`), so the disjunction is redundant. Both
    operands must be `_safe` so dropping `y` and the duplicate `x` preserves value and
    error behavior."""
    return _rewrite_node(node, _and_absorb)


def _or_absorb(expr: Expr) -> Expr:
    if isinstance(expr, Binary) and expr.op == "or":
        if _absorbs(expr.left, expr.right, "and"):
            return expr.left
        if _absorbs(expr.right, expr.left, "and"):
            return expr.right
    return expr


@rule(
    name="or_absorption",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_or_absorb,
)
def or_absorption(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`x OR (x AND y) → x`. The dual Kleene absorption law, valid for all three
    values. Both operands must be `_safe`."""
    return _rewrite_node(node, _or_absorb)


# --- complementation (guarded to never-null predicates) ---------------------


def _is_negation(x: Expr, other: Expr) -> bool:
    """Whether `other` is `NOT x` for a *total* boolean `x` (never null)."""
    return isinstance(other, Not) and _total_bool(x) and _key(other.input) == _key(x)


def _complement(expr: Expr) -> Expr:
    if (
        isinstance(expr, Binary)
        and expr.op in ("and", "or")
        and (_is_negation(expr.left, expr.right) or _is_negation(expr.right, expr.left))
    ):
        return Lit(False) if expr.op == "and" else Lit(True)
    return expr


@rule(
    name="complement_total_bool",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_complement,
)
def complement_total_bool(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`x AND NOT x → FALSE` and `x OR NOT x → TRUE`, but **only** when `x` never
    yields null (`is_null`/`is_not_null`). For a nullable `x`, `x AND NOT x` is NULL
    (not FALSE) when `x` is null under Kleene logic, so the collapse would be wrong;
    restricting to the total predicates keeps it exact."""
    return _rewrite_node(node, _complement)


# --- NOT over a comparison --------------------------------------------------


def _fold_not_comparison(expr: Expr) -> Expr:
    if (
        isinstance(expr, Not)
        and isinstance(expr.input, Binary)
        and expr.input.op in _NEGATED_COMPARISON
    ):
        inner = expr.input
        return Binary(_NEGATED_COMPARISON[inner.op], inner.left, inner.right)
    return expr


@rule(
    name="fold_not_comparison",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_fold_not_comparison,
)
def fold_not_comparison(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`NOT (a = b) → a <> b`, `NOT (a < b) → a >= b`, … — push `NOT` into a
    comparison. Exact under three-valued logic: a comparison and its complement are
    both null in exactly the null-operand rows, and the engine's total float order
    gives trichotomy (so `NOT (a < b)` really is `a >= b`, NaN included). Exposes a
    bare `col OP literal` that pushdown and zone-map pruning can use, where the opaque
    `NOT` could not."""
    return _rewrite_node(node, _fold_not_comparison)


# --- boolean equality against a literal -------------------------------------


def _bool_and_literal(a: Expr, b: Expr) -> tuple[Expr, bool] | None:
    if isinstance(b, Lit) and isinstance(b.value, bool) and _bool_valued(a):
        return a, b.value
    if isinstance(a, Lit) and isinstance(a.value, bool) and _bool_valued(b):
        return b, a.value
    return None


def _bool_eq_literal(expr: Expr) -> Expr:
    if isinstance(expr, Binary) and expr.op in ("eq", "ne"):
        found = _bool_and_literal(expr.left, expr.right)
        if found is not None:
            x, lit_true = found
            # eq&TRUE → x, eq&FALSE → NOT x, ne&TRUE → NOT x, ne&FALSE → x.
            keep = (expr.op == "eq") == lit_true
            return x if keep else Not(x)
    return expr


@rule(
    name="bool_eq_literal",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_bool_eq_literal,
)
def bool_eq_literal(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`x = TRUE → x`, `x = FALSE → NOT x` (and the `<>` duals) for a boolean `x`.
    Comparing a boolean to a boolean literal is the identity or the negation in all
    three values (`x = TRUE` is `x`; `x = FALSE` is `NOT x`, since `N = F` is `N` and
    `NOT N` is `N`). Fires only when `x` is provably boolean, never on a bare column
    (whose type is unknown)."""
    return _rewrite_node(node, _bool_eq_literal)


# --- IN-list cleanup --------------------------------------------------------


def _single_in_list(expr: Expr) -> Expr:
    if isinstance(expr, InList) and len(expr.values) == 1:
        return Binary("eq", expr.input, Lit(expr.values[0]))
    return expr


@rule(
    name="single_in_list",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=_single_in_list,
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
    # A non-null literal makes every later argument unreachable — drop them when the
    # dropped tail cannot error (so removing it changes neither value nor errors).
    for i, arg in enumerate(out):
        if isinstance(arg, Lit):
            if all(_safe(rest) for rest in out[i + 1 :]):
                out = out[: i + 1]
            break
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
)
def coalesce_simplify(node: Filter | Project, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Flatten and shrink a `COALESCE`: inline a nested `COALESCE`, drop a later
    duplicate of an earlier `_safe` argument, truncate everything after the first
    non-null literal (when the dropped tail cannot error), and unwrap `COALESCE(x)` to
    `x`. Each step preserves "first non-null argument" exactly — a repeated or
    post-literal argument is only ever reached when it is null (or never), so removing
    it is sound."""
    return _rewrite_node(node, _coalesce_simplify)
