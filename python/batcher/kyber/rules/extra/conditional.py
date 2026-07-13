"""NORMALIZE-phase rewrites for the conditional family — CASE / NULLIF / COALESCE / GREATEST /
LEAST.

A SQL front end (and a `when(...).then(...)` builder chain) emits conditional shapes that are
frequently dead on arrival: a branch guarded by a constant, a `CASE` whose arms all return the
same value, a `COALESCE` nested in a `COALESCE`, a `GREATEST` over constants. Each rule here
removes one such shape — shrinking the tree the data plane evaluates and, where a `CASE` collapses
to a `COALESCE` or a bare column, exposing a simpler expression to pushdown and constant folding.

**Branch selection under three-valued logic.** A `CASE` branch fires only where its condition is
*TRUE*: a NULL condition selects nothing, exactly like FALSE. That is what makes it sound to *drop*
a branch whose condition is constant-null, and why `case_first_true_branch_wins` may fire only on a
literal TRUE (a FALSE/NULL condition proves nothing about the branches below it). In this IR a NULL
condition is not even expressible as a literal — `Lit` cannot hold `None`, and the engine's
typed-NULL idiom is `NULLIF(lit(v), lit(v))` — so the constant-condition rules key on
`Lit(False)`/`Lit(True)`, and `_is_null_lit` recognizes that idiom.

**Two guards every dropping rule passes.** *Purity* (`_pure`): a removed sub-expression must be
deterministic and unable to raise, so deleting it changes neither the value nor whether the query
errors. *Type preservation* (`_droppable`): CASE/COALESCE/GREATEST/LEAST take their result type from
the *join* of their arms' types, so dropping an arm can move the output type —
`CASE WHEN FALSE THEN 1 ELSE 2.5 END` is a DOUBLE, and naively deleting the dead branch would make
it an INT. An arm is dropped only when a *kept* arm provably carries the same type.

Float edges are refused, not reasoned about: no rule folds a float extremum or a NaN comparison.

`nullif_same_operands` (`NULLIF(x, x)` → NULL) is deliberately **not** implemented: there is no NULL
literal in this IR to rewrite it to, and `NULLIF(lit(v), lit(v))` *is* the engine's canonical typed
NULL — rewriting it would destroy the type it carries.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase

# `_key` (structural identity), `_rewrite_node` (leaf Expr rule → rebuilt node, or None) and
# `_safe` (deterministic + non-erroring) are the sibling family's helpers, imported rather than
# re-implemented — copy-paste is the one wrong way to share.
from batcher.kyber.rules.extra.boolean_algebra import _SAFE_BINARY_OPS, _key, _rewrite_node, _safe
from batcher.plan.expr_ir import (
    Binary,
    Case,
    Cast,
    Coalesce,
    Expr,
    Greatest,
    InList,
    IsNotNull,
    IsNull,
    Least,
    Lit,
    Not,
    NullIf,
)
from batcher.plan.expr_ir.core import IsInf, IsNan
from batcher.plan.logical import Filter, LogicalPlan, Project

# The nodes these rules rewrite: `_rewrite_node` walks every expression a Filter/Project carries.
_Node = Filter | Project

__all__ = [
    "case_all_branches_same_result",
    "case_drop_duplicate_conditions",
    "case_drop_unreachable_branches",
    "case_first_true_branch_wins",
    "case_no_branches_to_else",
    "case_to_coalesce",
    "coalesce_dedup_args",
    "coalesce_drop_nulls_after_first_non_null",
    "coalesce_flatten_nested",
    "coalesce_single_arg",
    "greatest_least_dedup_args",
    "greatest_least_flatten_nested",
    "greatest_least_fold_literals",
    "greatest_least_single_arg",
    "nullif_distinct_literals",
]

# Nodes whose result is BOOLEAN whatever their input is.
_BOOL_NODES = (Not, IsNull, IsNotNull, IsNan, IsInf, InList)
# Binary operators whose result is BOOLEAN (comparisons + the Kleene connectives).
_BOOL_BINARY_OPS = frozenset({"eq", "ne", "lt", "le", "gt", "ge", "and", "or"})
# A literal's type class, most specific first (bool subclasses int; datetime subclasses date).
_LIT_CLASSES = ((bool, "bool"), (int, "int"), (float, "float"), (str, "str"))
_DATE_CLASSES = ((dt.datetime, "timestamp"), (dt.date, "date"))
# Literal classes whose GREATEST/LEAST fold is exact. Floats and booleans are excluded.
_FOLDABLE_LIT_CLASSES = frozenset({"int", "str", "date", "timestamp"})


def _pure(expr: Expr) -> bool:
    """Whether `expr` is deterministic and cannot raise — so removing it preserves the query's value
    *and* its error behavior. `boolean_algebra._safe` answers this for the boolean/arithmetic
    vocabulary but stops at the conditional nodes; this extends it over them, and delegates the rest
    (division, strict casts, opaque calls — all rejected) to `_safe`."""
    if isinstance(expr, (Coalesce, Greatest, Least)):
        return all(_pure(arg) for arg in expr.inputs)
    if isinstance(expr, NullIf):
        return _pure(expr.left) and _pure(expr.right)
    if isinstance(expr, Case):
        return all(_pure(c) and _pure(t) for c, t in expr.branches) and _pure(expr.otherwise)
    if isinstance(expr, Binary):
        return expr.op in _SAFE_BINARY_OPS and _pure(expr.left) and _pure(expr.right)
    if isinstance(expr, _BOOL_NODES):
        return _pure(expr.input)
    return _safe(expr)


def _lit_class(value: object) -> str | None:
    """The coarse type class of a literal's Python value, or `None` for one we can't name."""
    for cls, name in (*_LIT_CLASSES, *_DATE_CLASSES):
        if isinstance(value, cls):
            return name
    return None


def _type_tag(expr: Expr) -> str | None:
    """A coarse but *provable* output-type tag, or `None` when unknown. Only schema-free shapes get
    one: a literal (its Python class), a cast (its dtype), a boolean-valued node, and a conditional
    node all of whose arms share a tag. A bare `Col` — or arithmetic over one — is unknown."""
    if isinstance(expr, Lit):
        return _lit_class(expr.value)
    if isinstance(expr, Cast):
        return f"cast:{expr.dtype}"
    if isinstance(expr, _BOOL_NODES):
        return "bool"
    if isinstance(expr, Binary):
        return "bool" if expr.op in _BOOL_BINARY_OPS else None
    if isinstance(expr, (Coalesce, Greatest, Least)):
        return _uniform_tag(expr.inputs)
    if isinstance(expr, NullIf):
        return _uniform_tag([expr.left, expr.right])
    if isinstance(expr, Case):
        return _uniform_tag([t for _, t in expr.branches] + [expr.otherwise])
    return None


def _uniform_tag(exprs: Sequence[Expr]) -> str | None:
    """The one tag shared by every expression, or `None` if they differ (or it is unknown)."""
    tags = {_type_tag(e) for e in exprs}
    return tags.pop() if len(tags) == 1 else None


def _droppable(dropped: Sequence[Expr], kept: Sequence[Expr]) -> bool:
    """The guard for deleting arms of a type-joining node: each dropped arm is pure, and the output
    type survives. That type is the *join* of the arms' types, and a join is monotone + idempotent —
    so if a *kept* arm already contributes the dropped arm's type (proven by structural identity, or
    by a shared `_type_tag`), the join over `kept` alone equals the join over all of them."""
    if not kept or not all(_pure(arm) for arm in dropped):
        return False
    kept_keys = {_key(e) for e in kept}
    kept_tags = {tag for tag in (_type_tag(e) for e in kept) if tag is not None}
    for arm in dropped:
        if _key(arm) not in kept_keys and _type_tag(arm) not in kept_tags:  # None ⇒ unknown ⇒ keep
            return False
    return True


def _is_true_lit(expr: Expr) -> bool:
    return isinstance(expr, Lit) and expr.value is True


def _is_false_lit(expr: Expr) -> bool:
    return isinstance(expr, Lit) and expr.value is False


def _is_null_lit(expr: Expr) -> bool:
    """Whether `expr` is the engine's typed-NULL idiom `NULLIF(lit(v), lit(v))` — NULL on every row
    (`v = v` holds, so NULLIF nulls it out) while carrying `v`'s type. NaN is refused: it is the one
    value whose self-equality is not textual."""
    if not (
        isinstance(expr, NullIf) and isinstance(expr.left, Lit) and isinstance(expr.right, Lit)
    ):
        return False
    value = expr.left.value
    if isinstance(value, float) and math.isnan(value):
        return False
    return _key(expr.left) == _key(expr.right)


def _drop_unreachable(expr: Expr) -> Expr:
    if not isinstance(expr, Case):
        return expr
    kept = [b for b in expr.branches if not _is_false_lit(b[0])]
    dropped = [t for c, t in expr.branches if _is_false_lit(c)]
    if not dropped or not _droppable(dropped, [t for _, t in kept] + [expr.otherwise]):
        return expr
    return Case(kept, expr.otherwise)


@rule(
    name="case_drop_unreachable_branches",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=lambda e: _drop_unreachable(e),
)
def case_drop_unreachable_branches(node: _Node, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Delete a `WHEN` whose condition is the literal FALSE: a branch fires only where its condition
    is TRUE, so it fires on no row and contributes nothing but its type. (A constant-NULL condition
    is just as dead — NULL selects no rows either — but no NULL *literal* exists here.) The branch
    goes only when its `then` is pure and its type is already carried by a surviving arm, so neither
    the error behavior nor the CASE's result type moves."""
    return _rewrite_node(node, _drop_unreachable)


def _first_true(expr: Expr) -> Expr:
    if not isinstance(expr, Case):
        return expr
    i = next((i for i, (c, _) in enumerate(expr.branches) if _is_true_lit(c)), None)
    if i is None:
        return expr
    head, winner, tail = expr.branches[:i], expr.branches[i][1], expr.branches[i + 1 :]
    dropped = [t for _, t in tail] + [expr.otherwise]
    kept = [t for _, t in head] + [winner]
    if not all(_pure(c) for c, _ in tail) or not _droppable(dropped, kept):
        return expr
    return Case(head, winner)


@rule(
    name="case_first_true_branch_wins",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=lambda e: _first_true(e),
)
def case_first_true_branch_wins(node: _Node, _ctx: OptimizerContext) -> LogicalPlan | None:
    """A `WHEN` whose condition is the literal TRUE becomes the `ELSE`, and every branch after it
    (plus the old `ELSE`) is deleted: it fires on every row that got past the earlier branches, so
    its result *is* the default and nothing below it is reachable. Only a literal TRUE qualifies —
    FALSE or NULL says nothing about the branches beneath. Dropped conditions and results must be
    pure, and the result type must survive."""
    return _rewrite_node(node, _first_true)


def _all_same_result(expr: Expr) -> Expr:
    if not isinstance(expr, Case) or not expr.branches:
        return expr
    results = [t for _, t in expr.branches] + [expr.otherwise]
    if len({_key(r) for r in results}) != 1 or not all(_pure(c) for c, _ in expr.branches):
        return expr
    return results[0]


@rule(
    name="case_all_branches_same_result",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=lambda e: _all_same_result(e),
)
def case_all_branches_same_result(node: _Node, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Every arm (each `then` *and* the `otherwise`) is the same expression → that expression.
    Whichever branch a row selects it yields the same value, so the conditional *is* that value; and
    since every arm has the identical type, the join that types the CASE is that type too. The
    conditions are dropped, so they must be pure."""
    return _rewrite_node(node, _all_same_result)


def _no_branches(expr: Expr) -> Expr:
    if isinstance(expr, Case) and not expr.branches:
        return expr.otherwise
    return expr


@rule(
    name="case_no_branches_to_else",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=lambda e: _no_branches(e),
)
def case_no_branches_to_else(node: _Node, _ctx: OptimizerContext) -> LogicalPlan | None:
    """A `CASE` with no `WHEN` branches left is its `ELSE`: with nothing to select, every row falls
    through to the default, and the type join over one arm is its own type. This is the collapse the
    other CASE rules feed — once they delete the last unreachable branch, the husk disappears."""
    return _rewrite_node(node, _no_branches)


def _dedup_conditions(expr: Expr) -> Expr:
    if not isinstance(expr, Case) or len(expr.branches) < 2:
        return expr
    kept: list[tuple[Expr, Expr]] = []
    dropped: list[Expr] = []
    seen: set[str] = set()
    for cond, then in expr.branches:
        if _key(cond) in seen and _pure(cond):
            dropped.append(then)
            continue
        seen.add(_key(cond))
        kept.append((cond, then))
    if not dropped or not _droppable(dropped, [t for _, t in kept] + [expr.otherwise]):
        return expr
    return Case(kept, expr.otherwise)


@rule(
    name="case_drop_duplicate_conditions",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=lambda e: _dedup_conditions(e),
)
def case_drop_duplicate_conditions(node: _Node, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Delete a `WHEN` whose condition repeats an earlier branch's. Wherever the later condition is
    TRUE the earlier (structurally identical, and required to be pure, hence equal-valued) one was
    TRUE too — and first-true-wins already fired it — so the repeat is unreachable. Its result is
    removed under the usual purity + type guard."""
    return _rewrite_node(node, _dedup_conditions)


def _case_to_coalesce(expr: Expr) -> Expr:
    if not isinstance(expr, Case) or len(expr.branches) != 1:
        return expr
    cond, then = expr.branches[0]
    if isinstance(cond, IsNotNull) and _pure(cond.input) and _key(cond.input) == _key(then):
        return Coalesce([then, expr.otherwise])  # WHEN x IS NOT NULL THEN x ELSE y
    if isinstance(cond, IsNull) and _pure(cond.input) and _key(cond.input) == _key(expr.otherwise):
        return Coalesce([expr.otherwise, then])  # WHEN x IS NULL THEN y ELSE x
    return expr


@rule(
    name="case_to_coalesce",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=lambda e: _case_to_coalesce(e),
)
def case_to_coalesce(node: _Node, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`CASE WHEN x IS NOT NULL THEN x ELSE y END` → `coalesce(x, y)`, and the mirrored
    `CASE WHEN x IS NULL THEN y ELSE x END`. Both compute "x unless it is null, then y":
    `IS NOT NULL` is total (never NULL), so the branch fires on exactly the non-null rows — exactly
    where COALESCE takes `x`. Both arms survive verbatim, so the type join is unchanged; `x` appears
    twice in the CASE, so it must be pure for the two occurrences to agree."""
    return _rewrite_node(node, _case_to_coalesce)


def _nullif_distinct_literals(expr: Expr) -> Expr:
    if not (
        isinstance(expr, NullIf) and isinstance(expr.left, Lit) and isinstance(expr.right, Lit)
    ):
        return expr
    left, right = expr.left, expr.right
    cls = _lit_class(left.value)
    if cls is None or cls != _lit_class(right.value) or left.value == right.value:
        return expr
    if cls == "float" and (math.isnan(left.value) or math.isnan(right.value)):
        return expr
    return left


@rule(
    name="nullif_distinct_literals",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=lambda e: _nullif_distinct_literals(e),
)
def nullif_distinct_literals(node: _Node, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`NULLIF(a, b)` over two *distinct* literals of one type → `a`. NULLIF nulls its left operand
    exactly where `left = right`; two unequal constants are never equal, so the result is `a` on
    every row. Guarded three ways: one type class for both (so the result type stays `a`'s, not the
    join `NULLIF(1, 2.5)` would take); NaN is refused (`NaN = NaN` is TRUE in SQL but False in
    Python, so a NaN pair is not "distinct"); and `-0.0 == 0.0` in Python, so a signed-zero pair is
    correctly *not* distinct and does not fire."""
    return _rewrite_node(node, _nullif_distinct_literals)


def _coalesce_flatten(expr: Expr) -> Expr:
    if not isinstance(expr, Coalesce) or not any(isinstance(a, Coalesce) for a in expr.inputs):
        return expr
    flat: list[Expr] = []
    for arg in expr.inputs:
        flat.extend(arg.inputs if isinstance(arg, Coalesce) else [arg])
    return Coalesce(flat)


@rule(
    name="coalesce_flatten_nested",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=lambda e: _coalesce_flatten(e),
)
def coalesce_flatten_nested(node: _Node, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`coalesce(a, coalesce(b, c))` → `coalesce(a, b, c)`. "First non-null" is associative: the
    nested call is null exactly when `b` and `c` are both null, which is exactly when the flat form
    moves past them. No argument is added or removed, so the type join and the error behavior are
    identical and no purity guard is needed; bottom-up, one pass splices every level."""
    return _rewrite_node(node, _coalesce_flatten)


def _coalesce_drop_unreachable(expr: Expr) -> Expr:
    if not isinstance(expr, Coalesce):
        return expr
    kept = [a for a in expr.inputs if not _is_null_lit(a)]
    # A `Lit` is never null (no NULL literal here), so it wins once reached; the rest is dead code.
    first_lit = next((i for i, a in enumerate(kept) if isinstance(a, Lit)), None)
    if first_lit is not None:
        kept = kept[: first_lit + 1]
    # Nothing to do — or every argument was a typed NULL, whose type *is* the value: leave it.
    if len(kept) == len(expr.inputs) or not kept:
        return expr
    kept_keys = {_key(a) for a in kept}
    if not _droppable([a for a in expr.inputs if _key(a) not in kept_keys], kept):
        return expr
    return Coalesce(kept)


@rule(
    name="coalesce_drop_nulls_after_first_non_null",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=lambda e: _coalesce_drop_unreachable(e),
)
def coalesce_drop_nulls_after_first_non_null(
    node: _Node, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """Delete a `COALESCE` argument that is a constant NULL, and truncate everything after the first
    constant non-NULL. A provably-null argument is skipped on every row and can never be the answer;
    a `Lit` is never null, so it is *always* the answer once reached and all behind it is dead code.
    Dropped arguments must be pure and must not be the sole carrier of the result type: dropping the
    `NULL::double` from `coalesce(int_col, NULL::double)` would narrow DOUBLE to INT."""
    return _rewrite_node(node, _coalesce_drop_unreachable)


def _coalesce_single(expr: Expr) -> Expr:
    if isinstance(expr, Coalesce) and len(expr.inputs) == 1:
        return expr.inputs[0]
    return expr


@rule(
    name="coalesce_single_arg",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=lambda e: _coalesce_single(e),
)
def coalesce_single_arg(node: _Node, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`coalesce(x)` → `x`. The first non-null of one argument is that argument (a null `x` yields
    null either way), and the type join over one arm is that arm's type — the collapse the other
    COALESCE rules feed once they have removed the redundant arguments."""
    return _rewrite_node(node, _coalesce_single)


def _coalesce_dedup(expr: Expr) -> Expr:
    if not isinstance(expr, Coalesce) or len(expr.inputs) < 2:
        return expr
    kept: list[Expr] = [expr.inputs[0]]
    for arg in expr.inputs[1:]:
        if not (_key(arg) == _key(kept[-1]) and _pure(arg)):
            kept.append(arg)
    if len(kept) == len(expr.inputs):
        return expr
    return Coalesce(kept)


@rule(
    name="coalesce_dedup_args",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=lambda e: _coalesce_dedup(e),
)
def coalesce_dedup_args(node: _Node, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`coalesce(a, a, b)` → `coalesce(a, b)` — drop an argument identical to the one before it.
    COALESCE only advances past an argument that evaluated to null, so the repeat is reached only in
    the rows where it is *itself* null: it can never be the answer. The surviving twin has the same
    type, so the join is untouched; the dropped copy must be pure, so the two provably agree."""
    return _rewrite_node(node, _coalesce_dedup)


def _greatest_least_single(expr: Expr) -> Expr:
    if isinstance(expr, (Greatest, Least)) and len(expr.inputs) == 1:
        return expr.inputs[0]
    return expr


@rule(
    name="greatest_least_single_arg",
    phase=Phase.NORMALIZE,
    matches=(Filter, Project),
    expr=lambda e: _greatest_least_single(e),
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
    expr=lambda e: _greatest_least_flatten(e),
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
    expr=lambda e: _greatest_least_dedup(e),
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
    expr=lambda e: _greatest_least_fold(e),
)
def greatest_least_fold_literals(node: _Node, _ctx: OptimizerContext) -> LogicalPlan | None:
    """`greatest(2, 5, 3)` → `5`, and the `least` dual, when *every* argument is a literal of one
    comparable type. The extremum of constants is a constant, and one type class keeps the ordering
    and the result type exact — a mixed `greatest(1, 2.5)` takes the *join* type, and folding it to
    an INT literal would narrow it. Floats and booleans are excluded: NaN's ordering is
    engine-specific, and `-0.0 == 0.0` makes it observable *which* equal zero survives."""
    return _rewrite_node(node, _greatest_least_fold)
