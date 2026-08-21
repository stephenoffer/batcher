"""Rewrites of a disjunction — factoring an `OR`, and folding one into `IN`.

Both rules here take a `Filter` whose predicate is a top-level `OR` and rewrite the
disjunction itself, which is why they live together: `factor_common_conjuncts` pulls
the conjuncts every branch shares out in front, and `fold_in_list` collapses what is
left of a run of equalities into a single `InList`. Order matters between them and is
the order they are defined in — factoring first exposes the bare equality chain that
folding then recognizes, so `(x = 1 AND p) OR (x = 2 AND p)` reaches `p AND x IN (1, 2)`
in one pass rather than not at all.

Both are unconditionally semantics-preserving: they do not consult cardinality or cost.
"""

from __future__ import annotations

import datetime as _dt

from batcher.kyber.expr_cost import jit_compilable
from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.plan.expr_ir import Binary, Col, Expr, InList, Lit
from batcher.plan.expr_rewrite import (
    combine_conjuncts,
    combine_disjuncts,
    expr_key,
    split_conjuncts,
    split_disjuncts,
)
from batcher.plan.logical import Filter, LogicalPlan

__all__ = [
    "factor_common_conjuncts",
    "fold_in_list",
]


@rule(name="factor_common_conjuncts", phase=Phase.NORMALIZE, matches=(Filter,))
def factor_common_conjuncts(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Factor a conjunct common to every branch of an `OR` out of the disjunction.

    `(A AND X) OR (A AND Y) OR (A AND Z)` → `A AND (X OR Y OR Z)`. Boolean algebra,
    always semantics-preserving. The payoff is structural: the factored `A` becomes a
    top-level conjunct, so an equi-join condition hidden inside a disjunction (TPC-H
    Q19's `(p=l AND ...) OR (p=l AND ...) OR ...`) is exposed where predicate pushdown
    and join-key derivation can see it — without it the join degrades to a cartesian
    product.

    Every conjunct of the predicate is factored, not just a predicate that is *itself* a
    top-level `OR`. That distinction is the whole rule in practice: TPC-H Q19's `WHERE` is
    one bare disjunction, so matching the top level was enough for it — but TPC-DS q13 and
    q48 write `join-preds AND (…OR…OR…) AND (…OR…OR…)`, where the *only* mention of the
    equi-key for two of the dimension tables is inside those disjunctions. Matching only the
    top level left both keys buried and the six-way join planned as a chain of cartesian
    products: 1.2e16 estimated rows on q13, which does not fail with an error, it gets the
    process killed. (q85 is the milder form of the same shape — the equalities it shares
    across branches are between two aliases of `customer_demographics`.)
    """
    conjuncts = split_conjuncts(node.predicate)
    factored: list[Expr] = []
    changed = False
    for conjunct in conjuncts:
        pulled = _factor_disjunction(conjunct)
        if pulled is None:
            factored.append(conjunct)
        else:
            factored.extend(pulled)
            changed = True
    if not changed:
        return None
    return Filter(node.input, combine_conjuncts(factored))


def _factor_disjunction(predicate: Expr) -> list[Expr] | None:
    """The conjuncts `predicate` factors into, or `None` if it is not a factorable `OR`.

    A branch whose conjuncts are *all* common contributes a `TRUE` disjunct, collapsing
    the residual `OR` to nothing — the factored conjuncts alone imply it.

    Args:
        predicate: One conjunct of a filter's predicate.

    Returns:
        The replacement conjuncts, or `None` when no conjunct is shared by every branch
        (including when `predicate` is not a disjunction at all).
    """
    disjuncts = split_disjuncts(predicate)
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

    out = list(common)
    if not any_empty:
        out.append(combine_disjuncts(residuals))
    return out


def _ir_key(expr: Expr) -> str:
    """A hashable structural identity for an expression: its canonical IR.

    Both rules here need to know when two sub-expressions are *the same expression* —
    `factor_common_conjuncts` to find the conjuncts every branch shares, `fold_in_list` to
    group disjuncts by the target they compare. An `Expr` is deliberately unhashable and its
    `==` builds a predicate rather than a bool, so identity is taken over the canonical IR:
    the same wire form the engine is handed, so two keys match exactly when the engine would
    evaluate the same thing.
    """
    return expr_key(expr)


# Fold an OR-of-equals chain into `IN` once it has at least this many branches — below
# it the chain is cheap (and JIT-compilable), above it the hash-set membership wins.
_IN_LIST_MIN = 5


# A *computed* target (anything but a bare column read) is re-evaluated once per
# disjunct, so folding removes k-1 evaluations of it — a win from the second member on,
# long before the hash-vs-compares tradeoff `_IN_LIST_MIN` is about.
_IN_LIST_MIN_COMPUTED = 2


@rule(name="fold_in_list", phase=Phase.NORMALIZE, matches=(Filter,))
def fold_in_list(node: Filter, _ctx: OptimizerContext) -> LogicalPlan | None:
    """Fold `(x = a) OR (x = b) OR …` into `x IN (a, b, …)` (hash-set membership).

    The SQL front end lowers an `IN (literal, …)` list to a chain of equality
    disjuncts; for a long list that is an O(rows · k) scan of `k` comparisons per row.
    This collapses each run of equality disjuncts over one target into a single `InList`
    (an O(rows) hash-set lookup, the `eval_in_list` kernel), which is also the form a
    runtime join filter pushes the build side's key set as. Semantics are identical —
    `x = a OR x = b` and `x IN (a, b)` share the same Kleene null behavior — and only
    int / string / date literals fold, all of one kind.

    The target need not be a bare column. `substring(c_phone, 1, 2) IN (…)` (TPC-H q22)
    expands to `k` disjuncts that each recompute the substring over the whole column, so
    the fold saves `k - 1` *evaluations of the target*, not just `k - 1` comparisons.
    That is worth doing from the second member on, which is why a computed target folds
    at `_IN_LIST_MIN_COMPUTED` rather than `_IN_LIST_MIN`. A target the Cranelift tier
    compiles is excluded from the lower threshold: `InList` is not in the JIT's subset,
    so folding `(a + b) = 1 OR (a + b) = 2` would trade a compiled predicate for an
    interpreted one. Its re-evaluation is nearly free anyway.
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
    # Group the disjuncts that are `<target> = <supported literal>` by target (preserving a
    # consistent literal type per target); fold any group of ≥ threshold into an InList.
    by_target: dict[str, list] = {}
    targets: dict[str, Expr] = {}
    others: list[Expr] = []
    order: list[str] = []
    for d in split_disjuncts(expr):
        pair = _eq_target_literal(d)
        if pair is None:
            others.append(_fold_or_chains(d))
            continue
        target, value = pair
        key = _ir_key(target)
        if key not in by_target:
            by_target[key] = []
            targets[key] = target
            order.append(key)
        by_target[key].append(value)

    folded: list[Expr] = []
    changed = False
    for key in order:
        values = by_target[key]
        target = targets[key]
        if len(values) >= _fold_threshold(target) and _one_supported_type(values):
            folded.append(InList(target, tuple(values)))
            changed = True
        else:
            folded.extend(Binary("eq", target, Lit(v)) for v in values)
    if not changed:
        return expr
    return combine_disjuncts(folded + others)


def _fold_threshold(target: Expr) -> int:
    """How many members it takes to fold a run over `target` into one `InList`."""
    if isinstance(target, Col) or jit_compilable(target):
        return _IN_LIST_MIN
    return _IN_LIST_MIN_COMPUTED


def _eq_target_literal(expr: Expr) -> tuple[Expr, object] | None:
    """`(t == lit)` or `(lit == t)` over a foldable literal → `(target_expr, value)`."""
    if not (isinstance(expr, Binary) and expr.op == "eq"):
        return None
    left, right = expr.left, expr.right
    if isinstance(right, Lit) and not isinstance(left, Lit) and _foldable(right.value):
        return left, right.value
    if isinstance(left, Lit) and not isinstance(right, Lit) and _foldable(left.value):
        return right, left.value
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
