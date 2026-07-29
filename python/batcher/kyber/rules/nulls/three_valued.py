"""Three-valued logic: collapsing the null predicates that other rules generate.

`strictness` pushes null tests down onto columns, and doing so multiplies them. A
`greatest(a, b) IS NOT NULL` becomes a disjunction, a `CASE` gets a null test per
branch, and a predicate that already carried `x IS NOT NULL` ends up next to a second
copy. This module is the other half of that pass: it recognizes the shapes those pushes
produce and folds them away.

The unifying concept is *never-null*: an expression whose value is non-null on every
row, whatever its inputs are. `x IS NULL` is the canonical example — it answers `true`
or `false` for a null `x` rather than propagating the null, which is precisely what
makes SQL's null logic three-valued rather than two. Once an expression is known to be
never-null, four rewrites follow immediately, and they are how the null tests that
survive pushdown disappear: the test on it folds to a constant, and a `COALESCE` that
reaches it can drop every argument behind it.

The `greatest`/`least` rules encode a semantics that is easy to get backwards.
`greatest` **skips** nulls rather than propagating them (verified against the engine and
matching DuckDB): `greatest(NULL, 5)` is `5`, so the result is null only when *every*
argument is null. The null test therefore distributes as a conjunction, not the
disjunction the strict binary operators take -- and the `IS NOT NULL` test as a
disjunction, not a conjunction. Getting that pair the wrong way round produces a rule
that is correct on single-argument calls and silently wrong on the rest.
"""

from __future__ import annotations

from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase

# `_replaceable` is the conditional family's type-preservation guard and `schema_rule` the
# schema-threading lift -- imported rather than re-implemented, since copy-paste is the one
# wrong way to share. Both modules are imported by `rules/__init__` *before* this one, so
# these are re-imports of already-loaded modules and cannot move any rule's registration.
from batcher.kyber.rules.exprs.guards import schema_rule
from batcher.kyber.rules.extra.nullability import _replaceable
from batcher.kyber.rules.leaf_rewrite import rewrite_node, safe_expr
from batcher.plan.expr_ir import (
    Array,
    Case,
    Coalesce,
    Expr,
    Greatest,
    IsNotNull,
    IsNull,
    Least,
    Lit,
    Not,
)
from batcher.plan.expr_ir.core import Binary
from batcher.plan.expr_ir.nodes import HashRows, MakeStruct
from batcher.plan.expr_rewrite import combine_conjuncts, combine_disjuncts, expr_key
from batcher.plan.logical import Aggregate, Filter, LogicalPlan, Project, Sort, Window

__all__ = [
    "drop_coalesce_args_after_never_null",
    "drop_coalesce_of_never_null_first_arg",
    "is_not_null_of_greatest_to_any_not_null",
    "is_not_null_of_least_to_any_not_null",
    "is_not_null_of_never_null_to_true",
    "is_not_null_through_case_branches",
    "is_null_of_greatest_to_all_null",
    "is_null_of_least_to_all_null",
    "is_null_of_never_null_to_false",
    "is_null_through_case_branches",
    "never_null",
    "null_check_contradiction_to_false",
    "null_check_tautology_to_true",
]

_NODES = (Filter, Project, Aggregate, Sort, Window)


def never_null(expr: Expr) -> bool:
    """Whether `expr` provably evaluates to a non-null value on every row.

    Conservative: everything it accepts provably never yields null, but an expression it
    rejects may still be non-null (a column the schema marks non-nullable, for instance,
    which `nullability.py` answers from the schema instead).

    Args:
        expr: The expression to classify.

    Returns:
        ``True`` when the expression cannot evaluate to null.
    """
    if isinstance(expr, Lit):
        return expr.value is not None
    if isinstance(expr, (IsNull, IsNotNull)):
        # The whole point of the null predicates: they answer for a null input.
        return True
    if isinstance(expr, Not):
        return never_null(expr.input)
    if isinstance(expr, (Array, MakeStruct)):
        # Constructing a list or a struct never produces a null *container*, only one
        # holding nulls — verified against the engine for an all-null argument list.
        return True
    if isinstance(expr, HashRows):
        # A row hash is defined for every input, nulls included.
        return True
    if isinstance(expr, (Coalesce, Greatest, Least)):
        # All three answer with the first/extreme non-null argument, so one never-null
        # argument anywhere in the list is enough.
        return any(never_null(e) for e in expr.inputs)
    if isinstance(expr, Case):
        return all(never_null(v) for _, v in expr.branches) and never_null(expr.otherwise)
    if isinstance(expr, Binary) and expr.op in ("and", "or"):
        # Kleene `AND`/`OR` are null only when a null operand is not already decided by
        # the other one, so two never-null operands give a never-null result.
        return never_null(expr.left) and never_null(expr.right)
    return False


def _is_null_never_null(expr: Expr) -> Expr:
    if isinstance(expr, IsNull) and never_null(expr.input) and safe_expr(expr.input):
        return Lit(False)
    return expr


@rule(
    name="is_null_of_never_null_to_false",
    phase=Phase.NORMALIZE,
    matches=_NODES,
    expr=_is_null_never_null,
    expr_matches=(IsNull,),
)
def is_null_of_never_null_to_false(node: LogicalPlan, _ctx) -> LogicalPlan | None:
    """`e IS NULL -> false` when `e` provably never evaluates to null.

    The `safe_expr` guard is what makes discarding `e` sound rather than merely
    value-preserving: a never-null expression that can still *raise* must keep being
    evaluated, or the query stops reporting an error it is entitled to.
    """
    return rewrite_node(node, _is_null_never_null)


def _is_not_null_never_null(expr: Expr) -> Expr:
    if isinstance(expr, IsNotNull) and never_null(expr.input) and safe_expr(expr.input):
        return Lit(True)
    return expr


@rule(
    name="is_not_null_of_never_null_to_true",
    phase=Phase.NORMALIZE,
    matches=_NODES,
    expr=_is_not_null_never_null,
    expr_matches=(IsNotNull,),
)
def is_not_null_of_never_null_to_true(node: LogicalPlan, _ctx) -> LogicalPlan | None:
    """`e IS NOT NULL -> true` when `e` provably never evaluates to null.

    The dual of `is_null_of_never_null_to_false`, and the one that fires in practice:
    pushing an `IS NOT NULL` down through a `COALESCE` with a literal default lands
    exactly here, and the whole predicate folds out of the plan.
    """
    return rewrite_node(node, _is_not_null_never_null)


def _coalesce_never_null(expr: Expr, schema) -> Expr:
    if (
        isinstance(expr, Coalesce)
        and len(expr.inputs) > 1
        and never_null(expr.inputs[0])
        # The trailing arguments are *dropped*, so they need the same totality guard the
        # truncating sibling applies: the engine evaluates every branch of a `COALESCE`
        # columnwise, so discarding one that could raise removes an error the query would
        # have hit.
        and all(safe_expr(rest) for rest in expr.inputs[1:])
        # And the same *type* guard. `COALESCE` takes its type from the join of its
        # arguments, so dropping one can move the output type: `coalesce(5, double_col)` is
        # a DOUBLE, and returning the bare `5` narrows the column to BIGINT. That is a
        # visible schema change, not an optimization -- caught by
        # `tests/differential/test_diff_kyber3_coalesce_type.py`.
        and _replaceable(expr, expr.inputs[0], list(expr.inputs[1:]), schema)
    ):
        return expr.inputs[0]
    return expr


@rule(
    name="drop_coalesce_of_never_null_first_arg",
    phase=Phase.NORMALIZE,
    matches=_NODES,
    expr_schema=_coalesce_never_null,
    expr_matches=(Coalesce,),
)
def drop_coalesce_of_never_null_first_arg(node: LogicalPlan, _ctx) -> LogicalPlan | None:
    """`coalesce(e, …) -> e` when `e` provably never evaluates to null.

    The schema-driven sibling in `extra/nullability.py` answers this for a column the
    schema marks non-nullable. This one answers it for an *expression* — a literal
    default, a null predicate, a nested `coalesce` that already ends in one — and so
    fires where no schema is available at all.
    """
    return schema_rule(node, _coalesce_never_null, carries=(Coalesce,))


def _coalesce_truncate(expr: Expr, schema) -> Expr:
    if isinstance(expr, Coalesce) and len(expr.inputs) > 1:
        for i, arg in enumerate(expr.inputs[:-1]):
            kept = list(expr.inputs[: i + 1])
            dropped = list(expr.inputs[i + 1 :])
            if (
                never_null(arg)
                and all(safe_expr(rest) for rest in dropped)
                # Truncating drops arguments from the type join exactly as the sibling
                # above does, so it needs the same type-preservation check.
                and _replaceable(expr, Coalesce(kept), dropped, schema)
            ):
                return Coalesce(kept)
    return expr


@rule(
    name="drop_coalesce_args_after_never_null",
    phase=Phase.NORMALIZE,
    matches=_NODES,
    expr_schema=_coalesce_truncate,
    expr_matches=(Coalesce,),
)
def drop_coalesce_args_after_never_null(node: LogicalPlan, _ctx) -> LogicalPlan | None:
    """`coalesce(a, e, z) -> coalesce(a, e)` when `e` provably never evaluates to null.

    Every argument behind the first never-null one is unreachable. They are dropped only
    when all of them are `safe_expr`, since an unreachable argument that can raise is
    still not evaluated — but one that is *not* provably total is left in place rather
    than reasoned about.
    """
    return schema_rule(node, _coalesce_truncate, carries=(Coalesce,))


def _complementary_pair(left: Expr, right: Expr) -> Expr | None:
    """The operand `x` when `{left, right}` is exactly `{x IS NULL, x IS NOT NULL}`."""
    for a, b in ((left, right), (right, left)):
        if (
            isinstance(a, IsNull)
            and isinstance(b, IsNotNull)
            and expr_key(a.input) == expr_key(b.input)
        ):
            return a.input
    return None


def _null_tautology(expr: Expr) -> Expr:
    if isinstance(expr, Binary) and expr.op == "or":
        operand = _complementary_pair(expr.left, expr.right)
        if operand is not None and safe_expr(operand):
            return Lit(True)
    return expr


@rule(
    name="null_check_tautology_to_true",
    phase=Phase.NORMALIZE,
    matches=_NODES,
    expr=_null_tautology,
    expr_matches=(Binary,),
    expr_ops=("or",),
)
def null_check_tautology_to_true(node: LogicalPlan, _ctx) -> LogicalPlan | None:
    """`x IS NULL OR x IS NOT NULL -> true`.

    A row is one or the other, so the disjunction holds unconditionally — and unlike
    most tautologies over a nullable operand, this one is `true` rather than `NULL`,
    because neither null predicate propagates a null. The shape arises when two branches
    of a rewritten `CASE`, or two sides of a union of filters, are folded together.
    """
    return rewrite_node(node, _null_tautology)


def _null_contradiction(expr: Expr) -> Expr:
    if isinstance(expr, Binary) and expr.op == "and":
        operand = _complementary_pair(expr.left, expr.right)
        if operand is not None and safe_expr(operand):
            return Lit(False)
    return expr


@rule(
    name="null_check_contradiction_to_false",
    phase=Phase.NORMALIZE,
    matches=_NODES,
    expr=_null_contradiction,
    expr_matches=(Binary,),
    expr_ops=("and",),
)
def null_check_contradiction_to_false(node: LogicalPlan, _ctx) -> LogicalPlan | None:
    """`x IS NULL AND x IS NOT NULL -> false`.

    The contradiction dual of the tautology, and worth more: a `false` conjunct turns
    the whole predicate into `false`, which `filter_false_to_empty` then turns into an
    empty relation — the plan stops reading the input at all.
    """
    return rewrite_node(node, _null_contradiction)


def _case_null_test(expr: Expr, *, positive: bool) -> Expr:
    check: type = IsNull if positive else IsNotNull
    if isinstance(expr, check) and isinstance(expr.input, Case):
        case = expr.input
        return Case(
            [(cond, check(value)) for cond, value in case.branches],
            check(case.otherwise),
        )
    return expr


def _case_is_null(expr: Expr) -> Expr:
    return _case_null_test(expr, positive=True)


def _case_is_not_null(expr: Expr) -> Expr:
    return _case_null_test(expr, positive=False)


@rule(
    name="is_null_through_case_branches",
    phase=Phase.NORMALIZE,
    matches=_NODES,
    expr=_case_is_null,
    expr_matches=(IsNull,),
)
def is_null_through_case_branches(node: LogicalPlan, _ctx) -> LogicalPlan | None:
    """`(CASE WHEN c THEN a ELSE b END) IS NULL -> CASE WHEN c THEN a IS NULL ELSE b IS NULL END`.

    The `CASE` decides *which* value is produced, and the null test asks a question about
    that value; moving the test inside changes neither decision. The conditions are
    evaluated exactly once on both sides, so this is not a duplication.

    It pays off because the branch values are so often literals: `CASE WHEN c THEN NULL
    ELSE 0 END IS NULL` becomes `CASE WHEN c THEN true ELSE false END`, which the boolean
    normalizer then collapses to `c` itself.
    """
    return rewrite_node(node, _case_is_null)


@rule(
    name="is_not_null_through_case_branches",
    phase=Phase.NORMALIZE,
    matches=_NODES,
    expr=_case_is_not_null,
    expr_matches=(IsNotNull,),
)
def is_not_null_through_case_branches(node: LogicalPlan, _ctx) -> LogicalPlan | None:
    """The `IS NOT NULL` dual of `is_null_through_case_branches`, folding to the negated
    branch constants and, through them, to the branch condition itself."""
    return rewrite_node(node, _case_is_not_null)


def _extreme_null_test(expr: Expr, kind: type, *, positive: bool) -> Expr:
    """Distribute one null test over a null-skipping `greatest`/`least` call.

    `kind` is the call node the test must sit on; `positive` selects `IS NULL` (which
    distributes as a conjunction) or `IS NOT NULL` (a disjunction).
    """
    check: type = IsNull if positive else IsNotNull
    if isinstance(expr, check) and isinstance(expr.input, kind):
        args = list(expr.input.inputs)
        if len(args) > 1:
            tests = [check(a) for a in args]
            return combine_conjuncts(tests) if positive else combine_disjuncts(tests)
    return expr


def _greatest_is_null(expr: Expr) -> Expr:
    return _extreme_null_test(expr, Greatest, positive=True)


def _greatest_is_not_null(expr: Expr) -> Expr:
    return _extreme_null_test(expr, Greatest, positive=False)


def _least_is_null(expr: Expr) -> Expr:
    return _extreme_null_test(expr, Least, positive=True)


def _least_is_not_null(expr: Expr) -> Expr:
    return _extreme_null_test(expr, Least, positive=False)


@rule(
    name="is_null_of_greatest_to_all_null",
    phase=Phase.NORMALIZE,
    matches=_NODES,
    expr=_greatest_is_null,
    expr_matches=(IsNull,),
)
def is_null_of_greatest_to_all_null(node: LogicalPlan, _ctx) -> LogicalPlan | None:
    """`greatest(a, b, …) IS NULL -> a IS NULL AND b IS NULL AND …`.

    `greatest` skips nulls instead of propagating them, so its result is null only when
    it had nothing to choose from. The conjunction is the exact restatement of that, and
    it is the pushable form: each conjunct names one column, so predicate pushdown can
    send each to the side of the join that owns it.
    """
    return rewrite_node(node, _greatest_is_null)


@rule(
    name="is_not_null_of_greatest_to_any_not_null",
    phase=Phase.NORMALIZE,
    matches=_NODES,
    expr=_greatest_is_not_null,
    expr_matches=(IsNotNull,),
)
def is_not_null_of_greatest_to_any_not_null(node: LogicalPlan, _ctx) -> LogicalPlan | None:
    """`greatest(a, b, …) IS NOT NULL -> a IS NOT NULL OR b IS NOT NULL OR …` — the
    De Morgan dual of `is_null_of_greatest_to_all_null`, over the same null-skipping
    semantics."""
    return rewrite_node(node, _greatest_is_not_null)


@rule(
    name="is_null_of_least_to_all_null",
    phase=Phase.NORMALIZE,
    matches=_NODES,
    expr=_least_is_null,
    expr_matches=(IsNull,),
)
def is_null_of_least_to_all_null(node: LogicalPlan, _ctx) -> LogicalPlan | None:
    """`least(a, b, …) IS NULL -> a IS NULL AND b IS NULL AND …`. `least` skips nulls on
    exactly the terms `greatest` does, so it carries the same null algebra."""
    return rewrite_node(node, _least_is_null)


@rule(
    name="is_not_null_of_least_to_any_not_null",
    phase=Phase.NORMALIZE,
    matches=_NODES,
    expr=_least_is_not_null,
    expr_matches=(IsNotNull,),
)
def is_not_null_of_least_to_any_not_null(node: LogicalPlan, _ctx) -> LogicalPlan | None:
    """`least(a, b, …) IS NOT NULL -> a IS NOT NULL OR b IS NOT NULL OR …`."""
    return rewrite_node(node, _least_is_not_null)
