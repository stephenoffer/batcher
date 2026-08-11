"""Schema-driven NULL reasoning — rewrites proved by *declared* nullability.

The engine knows per column whether it may hold a NULL — a source schema's Arrow fields carry
`nullable`, and `Scan` keeps it — and almost nothing consumed it. This family does: it derives
the provably non-null columns at a node's input, lifts that to a *never-null* judgement over
whole expressions, and folds the null-handling shapes a SQL front end emits (`IS NULL`,
`COALESCE`, `fill_null`, `eq_missing`, `COUNT(col)`, `NULLS FIRST`) into something cheaper —
or into nothing.

**Declared nullability is a contract, not a statistic.** A `nullable=False` field carrying a
NULL is invalid input (pyarrow does not enforce it; DuckDB's `NOT NULL` does), and every engine
that optimizes on it — Calcite, DuckDB, Spark — reads it as a promise. These rules are exact
*given* that promise. Where nullability is unknown the analysis answers "may be null", and
every rule returns `None`.

**Three-valued logic is respected literally.** `IS NULL` / `IS NOT NULL` are *total* — TRUE or
FALSE on every row, never NULL — which is what makes folding them to a boolean literal sound
in **any** context: inside a `Filter` (NULL and FALSE both drop a row) and equally inside a
`Project` (where they are different *values*). Nothing here relies on a filter's
NULL-drops-the-row semantics. The never-null judgement is conservative the other way: `AND`/`OR`
are Kleene, so a null operand could still yield a non-null result — never exploited.

Deliberately **not** implemented: `filter_is_null_on_non_nullable_to_empty` (already shipped as
`empty_on_impossible_null_check`); `is_null_of_literal` and its dual (subsumed — a `Lit` cannot
hold NULL, so it *is* the degenerate never-null expression); `drop_nullif_when_operand_non_nullable`
(**unsound** — `NULLIF` *introduces* NULL wherever its operands are equal, which non-nullability
says nothing about); and `propagate_non_nullability_through_project` (that is the *analysis*,
`_non_null_cols`, not a rewrite).
"""

from __future__ import annotations

from collections.abc import Callable

from batcher.kyber.pass_base import OptimizerContext
from batcher.kyber.registry import rule
from batcher.kyber.rule import Phase
from batcher.kyber.rules.exprs.guards import SchemaNode

# The sibling families' helpers, imported rather than re-implemented (copy-paste is the one
# wrong way to share): `_key` (structural identity), `_rewrite_node` (leaf Expr rule → rebuilt
# node, or None), `SAFE_BINARY_OPS` (the null-propagating, non-erroring binaries), `_pure`
# (deterministic and non-erroring, so deleting it changes neither value nor errors), and
# `_non_nullable_cols` (the columns a scan's source schema declares NOT NULL).
from batcher.kyber.rules.extra.boolean_algebra import _key, _rewrite_node
from batcher.kyber.rules.extra.conditional import _pure
from batcher.kyber.rules.extra.projection_scan import _non_nullable_cols
from batcher.plan.expr_ir import (
    Binary,
    Cast,
    Coalesce,
    Col,
    Expr,
    IsNotNull,
    IsNull,
    Lit,
    Not,
)
from batcher.plan.expr_ir.core import IsInf, IsNan
from batcher.plan.ir_tags import SAFE_BINARY_OPS
from batcher.plan.logical import (
    Distinct,
    Filter,
    Limit,
    LogicalPlan,
    Project,
    Sample,
    Sort,
)
from batcher.plan.schema import SchemaRef
from batcher.plan.types import infer_type

__all__ = [
    "drop_coalesce_args_after_non_nullable",
    "drop_coalesce_of_non_nullable_first_arg",
    "drop_is_not_null_on_non_nullable_column",
    "drop_is_null_on_non_nullable_column",
    "simplify_null_safe_comparison_on_non_nullable",
]

# The unary operators that neither rename a column nor null-extend one, so declared
# nullability passes through them unchanged on the way down to the originating scan.
_ROW_PRESERVING = (Filter, Sort, Limit, Sample, Distinct)

#: The node types whose expressions this family rewrites (a filter's predicate, a projection's
#: items), and the leaf `Expr → Expr` rewrite — specialized to what is known at the node's
#: input: the columns declared NOT NULL, and the schema (`None` when not inferable) the
#: arm-deleting rules need to prove the type survives.
_EXPR_NODES = (Filter, Project)
_Leaf = Callable[[Expr], Expr]
_LeafFactory = Callable[[frozenset[str], SchemaRef | None], _Leaf]


def _non_null_cols(node: LogicalPlan) -> frozenset[str]:
    """The columns provably non-NULL in `node`'s output — declared nullability only.

    Descends the row/column-preserving unary operators to the originating `Scan` (whose *raw*
    `schema` carries the `nullable` flags — `Scan.available_schema()` widens types and drops
    them), and extends that through a `Project`: a derived column is non-nullable exactly when
    its expression is `_never_null` over the projection's own input. Every other node (`Join`,
    `Aggregate`, `Union`, `Unnest`, …) stops the descent and yields the empty set — an outer
    join null-extends, an aggregate emits NULL over an empty group, and a union's branches may
    disagree. Unknown is always answered as *may be null*, never guessed.
    """
    while isinstance(node, _ROW_PRESERVING):
        node = node.input
    if isinstance(node, Project):
        base = _non_null_cols(node.input)
        return frozenset(item.alias for item in node.items if _never_null(item.expr, base))
    return _non_nullable_cols(node)


def _never_null(expr: Expr, non_null: frozenset[str]) -> bool:
    """Whether `expr` provably evaluates to a non-NULL value on **every** row.

    A conservative whitelist, given the columns known non-nullable at the input: a `Lit` (the
    IR has no NULL literal — its typed NULL is `NULLIF(l, l)`); a NOT NULL `Col`; `IS NULL` /
    `IS NOT NULL` (total by construction); `NOT` / `IS NAN` / `IS INF` over a never-null input;
    a **strict** `Cast` of one (an unconvertible value *errors* rather than becoming NULL —
    `try_cast`, which does manufacture one, is excluded); a null-propagating binary from
    `SAFE_BINARY_OPS` with both operands never-null (`div`/`mod` and the bit/shift ops are
    not in it); and a `COALESCE` with a never-null argument. Everything else answers False —
    erring that way prevents a rewrite, never licenses one.
    """
    if isinstance(expr, Lit):
        return True
    if isinstance(expr, Col):
        return expr.name in non_null
    if isinstance(expr, (IsNull, IsNotNull)):
        return True
    if isinstance(expr, (Not, IsNan, IsInf)):
        return _never_null(expr.input, non_null)
    if isinstance(expr, Cast):
        return not expr.try_cast and _never_null(expr.input, non_null)
    if isinstance(expr, Binary):
        return (
            expr.op in SAFE_BINARY_OPS
            and _never_null(expr.left, non_null)
            and _never_null(expr.right, non_null)
        )
    if isinstance(expr, Coalesce):
        return any(_never_null(arg, non_null) for arg in expr.inputs)
    return False


def _rewrite_with_nullability(node: SchemaNode, make_leaf: _LeafFactory) -> LogicalPlan | None:
    """Apply a nullability-parameterized leaf rewrite to every expression in `node`.

    The judgement is made against the node's *input* — which is where the node's expressions
    are evaluated — both for nullability and (for the rules that delete a sub-expression) for
    the types. Returns `None` when nothing changed (the driver's fixpoint signal), so every
    rule built on this is idempotent.
    """
    return _rewrite_node(node, make_leaf(_non_null_cols(node.input), node.input.available_schema()))


def _replaceable(
    original: Expr, replacement: Expr, dropped: list[Expr], schema: SchemaRef | None
) -> bool:
    """Whether `original` may be replaced by `replacement`, deleting `dropped`.

    Two guards, the ones every arm-deleting rule in the conditional family passes. *Purity*: a
    deleted sub-expression must be deterministic and unable to raise, so removing it changes
    neither the value nor whether the query errors. *Type preservation*: COALESCE takes its
    type from the **join** of its arms, so deleting one can move the output type —
    `coalesce(int_col, 1.5)` is a DOUBLE, and reducing it to `int_col` would narrow it to
    BIGINT. Rather than re-derive the join, this asks the engine's own inference (`infer_type`)
    whether the two forms agree, and refuses whenever either is unknown.
    """
    if schema is None or not all(_pure(arm) for arm in dropped):
        return False
    original_type = infer_type(original, schema)
    return original_type is not None and original_type == infer_type(replacement, schema)


# --- IS NULL / IS NOT NULL over a provably non-null expression ---------------


def _drop_is_null(non_null: frozenset[str], _schema: SchemaRef | None) -> _Leaf:
    def leaf(expr: Expr) -> Expr:
        if isinstance(expr, IsNull) and _never_null(expr.input, non_null):
            return Lit(False)
        return expr

    return leaf


@rule(
    name="drop_is_null_on_non_nullable_column",
    phase=Phase.NORMALIZE,
    matches=_EXPR_NODES,
    expr_matches=(IsNull,),
)
def drop_is_null_on_non_nullable_column(
    node: SchemaNode, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`x IS NULL` → `FALSE` when `x` provably never yields NULL (a NOT NULL column, a
    literal, or anything built from them by a null-propagating operator).

    `IS NULL` is *total* — TRUE or FALSE on every row, never NULL — so replacing it with a
    boolean literal is exact **anywhere**: under a `Filter` (where the FALSE kills the
    conjunct, or the whole filter via `empty_on_impossible_null_check`) and equally inside a
    `Project`, where FALSE and NULL would be different values. Declared nullability only: a
    proof, not an estimate.
    """
    return _rewrite_with_nullability(node, _drop_is_null)


def _drop_is_not_null(non_null: frozenset[str], _schema: SchemaRef | None) -> _Leaf:
    def leaf(expr: Expr) -> Expr:
        if isinstance(expr, IsNotNull) and _never_null(expr.input, non_null):
            return Lit(True)
        return expr

    return leaf


@rule(
    name="drop_is_not_null_on_non_nullable_column",
    phase=Phase.NORMALIZE,
    matches=_EXPR_NODES,
    expr_matches=(IsNotNull,),
)
def drop_is_not_null_on_non_nullable_column(
    node: SchemaNode, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`x IS NOT NULL` → `TRUE` when `x` provably never yields NULL — the dual of
    `drop_is_null_on_non_nullable_column`, sound in a `Project` for the same reason.

    Complements `drop_always_true_null_check`, which strips such a check only as a *top-level
    conjunct of a filter*: this folds it wherever it sits — nested under an `OR`, inside a
    `CASE` condition, or as a projected boolean column — feeding the identity rules a `TRUE`.
    """
    return _rewrite_with_nullability(node, _drop_is_not_null)


# --- COALESCE truncated by a provably non-null argument ----------------------


def _coalesce_first_non_null(non_null: frozenset[str], schema: SchemaRef | None) -> _Leaf:
    def leaf(expr: Expr) -> Expr:
        if not isinstance(expr, Coalesce) or len(expr.inputs) < 2:
            return expr
        head = expr.inputs[0]
        if not _never_null(head, non_null):
            return expr
        rest = list(expr.inputs[1:])
        return head if _replaceable(expr, head, rest, schema) else expr

    return leaf


@rule(
    name="drop_coalesce_of_non_nullable_first_arg",
    phase=Phase.NORMALIZE,
    matches=_EXPR_NODES,
    expr_matches=(Coalesce,),
)
def drop_coalesce_of_non_nullable_first_arg(
    node: SchemaNode, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`coalesce(x, …)` → `x` when `x` provably never yields NULL.

    COALESCE returns its first non-null argument; if that argument is never null it is
    *always* the answer and the rest is dead code. This is also the `fill_null` of a NOT NULL
    column (it lowers to `Coalesce([x, v])`), a shape ETL code emits by habit. Guarded by
    `_replaceable`: the dropped tail must be pure, and the survivor must carry the COALESCE's
    own inferred type — else `coalesce(int_not_null, 1.5)`, a DOUBLE, would narrow to BIGINT.
    """
    return _rewrite_with_nullability(node, _coalesce_first_non_null)


def _coalesce_truncate(non_null: frozenset[str], schema: SchemaRef | None) -> _Leaf:
    def leaf(expr: Expr) -> Expr:
        if not isinstance(expr, Coalesce) or len(expr.inputs) < 3:
            return expr
        for i in range(1, len(expr.inputs) - 1):  # a later arg exists to drop
            if not _never_null(expr.inputs[i], non_null):
                continue
            kept, dropped = list(expr.inputs[: i + 1]), list(expr.inputs[i + 1 :])
            truncated = Coalesce(kept)
            return truncated if _replaceable(expr, truncated, dropped, schema) else expr
        return expr

    return leaf


@rule(
    name="drop_coalesce_args_after_non_nullable",
    phase=Phase.NORMALIZE,
    matches=_EXPR_NODES,
    expr_matches=(Coalesce,),
)
def drop_coalesce_args_after_non_nullable(
    node: SchemaNode, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`coalesce(a, b, c)` → `coalesce(a, b)` when `b` provably never yields NULL.

    COALESCE only advances past an argument that evaluated to NULL, so once a never-null
    argument is reached it is the answer and everything behind it is unreachable. The literal
    case is `coalesce_drop_nulls_after_first_non_null`'s; this generalizes it to a NOT NULL
    *column* (or any never-null expression), under the same `_replaceable` type/purity guard.
    """
    return _rewrite_with_nullability(node, _coalesce_truncate)


# --- the null-safe comparison idiom (`eq_missing`) --------------------------


def _eq_missing_equality(expr: Binary) -> Binary | None:
    """The plain `a = b` inside the `eq_missing` shape `coalesce(a = b, FALSE) OR (a IS NULL
    AND b IS NULL)`, or None if `expr` is not that shape."""
    left, right = expr.left, expr.right
    if not (isinstance(left, Coalesce) and len(left.inputs) == 2):
        return None
    eq, fallback = left.inputs
    if not (isinstance(eq, Binary) and eq.op == "eq"):
        return None
    if not (isinstance(fallback, Lit) and fallback.value is False):
        return None
    if not (isinstance(right, Binary) and right.op == "and"):
        return None
    a_null, b_null = right.left, right.right
    if not (isinstance(a_null, IsNull) and isinstance(b_null, IsNull)):
        return None
    if _key(a_null.input) != _key(eq.left) or _key(b_null.input) != _key(eq.right):
        return None
    return eq


def _null_safe_comparison(non_null: frozenset[str], _schema: SchemaRef | None) -> _Leaf:
    def leaf(expr: Expr) -> Expr:
        if not (isinstance(expr, Binary) and expr.op == "or"):
            return expr
        eq = _eq_missing_equality(expr)
        if eq is None:
            return expr
        if _never_null(eq.left, non_null) and _never_null(eq.right, non_null):
            return eq
        return expr

    return leaf


@rule(
    name="simplify_null_safe_comparison_on_non_nullable",
    phase=Phase.NORMALIZE,
    matches=_EXPR_NODES,
    expr_matches=(Binary,),
    expr_ops=("or",),
)
def simplify_null_safe_comparison_on_non_nullable(
    node: SchemaNode, _ctx: OptimizerContext
) -> LogicalPlan | None:
    """`a IS NOT DISTINCT FROM b` → `a = b` when both operands provably never yield NULL.

    `eq_missing` lowers to `coalesce(a = b, FALSE) OR (a IS NULL AND b IS NULL)` — three extra
    kernels guarding a NULL case that, over NOT NULL operands, cannot happen: the `IS NULL`s
    are FALSE, so the right disjunct is FALSE; `a = b` is never NULL, so the COALESCE is the
    equality itself; and `x OR FALSE` is `x` for a never-null `x`. What is left is the bare
    equality, which pushdown and zone-map pruning can use where the OR/COALESCE wrapper
    blocked them. Fires only when *both* operands are provably non-null.
    """
    return _rewrite_with_nullability(node, _null_safe_comparison)


# --- pushing a null check through COALESCE ----------------------------------
