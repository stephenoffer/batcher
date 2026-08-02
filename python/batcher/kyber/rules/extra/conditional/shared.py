"""Shared guards for the conditional family: purity, type tags, and droppability.

The delicate one is `_droppable`. `CASE WHEN FALSE THEN 1 ELSE 2.5 END` is a DOUBLE, so
removing an arm can silently *narrow* the result type; an arm is dropped only when a kept
arm provably carries the same type. `_pure` stops a drop from erasing an error.

There are two independent ways to name an arm's type here, and `_droppable` accepts either
as proof. `_type_tag` is schema-free and so bottoms out on a bare column: it can name a
literal, a cast, and anything boolean-valued, but `CASE WHEN TRUE THEN a ELSE b END` over two
`int` columns leaves it with `None` on both arms — which read as "unknown", so the whole
family declined on it. That is the shape a SQL front end, a governance rewrite, or a
partially-folded predicate produces constantly, and declining meant the fold never happened
on a real query.

`_arrow_tag` closes that: given the node's schema it asks `infer_type` for the arm's *exact*
Arrow type, which is both sharper than the coarse classes and available for expressions
`_type_tag` cannot see into at all. The schema-free path stays, because a node whose schema
cannot be inferred still gets the literal-only conclusions it always did — so every rule in
the family declares both leaves and the driver runs whichever it can.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence

from batcher._internal.mathx import is_nan

# `_key` (structural identity), `_rewrite_node` (leaf Expr rule → rebuilt node, or None) and
# `_safe` (deterministic + non-erroring) are the sibling family's helpers, imported rather than
# re-implemented — copy-paste is the one wrong way to share.
from batcher.kyber.rules.extra.boolean_algebra import (
    SAFE_BINARY_OPS,
    _key,
    _rewrite_node,
    _safe,
)
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
from batcher.plan.logical import Filter, Project
from batcher.plan.schema import SchemaRef
from batcher.plan.types import infer_type

# The nodes these rules rewrite: `_rewrite_node` walks every expression a Filter/Project carries.
_Node = Filter | Project

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
        return expr.op in SAFE_BINARY_OPS and _pure(expr.left) and _pure(expr.right)
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


def _arrow_tag(expr: Expr, schema: SchemaRef | None) -> str | None:
    """The expression's exact Arrow type as a tag, or ``None`` when it cannot be inferred.

    The schema-aware counterpart to `_type_tag`, and strictly sharper where it applies: two
    expressions carry the same tag exactly when they have the same Arrow type, which is the
    relation droppability needs. `infer_type` raising (an expression the inference does not
    cover, a column the schema lacks) reads as "unknown", the same as a `None` tag.
    """
    if schema is None:
        return None
    try:
        return str(infer_type(expr, schema))
    except Exception:
        return None


def _droppable(
    dropped: Sequence[Expr], kept: Sequence[Expr], schema: SchemaRef | None = None
) -> bool:
    """The guard for deleting arms of a type-joining node: each dropped arm is pure, and the output
    type survives. That type is the *join* of the arms' types, and a join is monotone + idempotent —
    so if a *kept* arm already contributes the dropped arm's type, the join over `kept` alone equals
    the join over all of them.

    Three independent proofs that a kept arm carries a dropped arm's type, and any one suffices:
    structural identity, a shared schema-free `_type_tag`, or — when `schema` is given — a shared
    exact Arrow type. The third is what lets the fold happen over bare columns, where the first two
    are silent."""
    if not kept or not all(_pure(arm) for arm in dropped):
        return False
    kept_keys = {_key(e) for e in kept}
    kept_tags = {tag for tag in (_type_tag(e) for e in kept) if tag is not None}
    kept_arrow = {tag for tag in (_arrow_tag(e, schema) for e in kept) if tag is not None}
    for arm in dropped:
        if _key(arm) in kept_keys:
            continue
        if _type_tag(arm) in kept_tags:  # None ⇒ unknown ⇒ falls through to the Arrow tag
            continue
        arrow = _arrow_tag(arm, schema)
        if arrow is not None and arrow in kept_arrow:
            continue
        return False
    return True


def _rewrite_typed(
    node: _Node,
    leaf: Callable[..., Expr],
    *,
    carries: tuple[type, ...],
) -> _Node | None:
    """The node-local form of a conditional leaf, run schema-free and then schema-aware.

    Every leaf in this family takes an optional `schema`, so this applies the *same* function
    twice: once with no type information and once with the node's input schema. That mirrors
    exactly what the driver does for a rule declaring both an `expr` and an `expr_schema` leaf,
    which is what keeps the standalone form equivalent to the fused one.

    Args:
        node: The Filter/Project whose expressions should be rewritten.
        leaf: The leaf rewrite, callable as `leaf(expr)` and as `leaf(expr, schema)`.
        carries: The expression node types the leaf can act on, so the schema pass declines
            without resolving a schema when the node carries none of them.

    Returns:
        The rebuilt node, or ``None`` when neither pass changed anything.
    """
    from batcher.kyber.rules.exprs.guards import schema_rule

    plain = _rewrite_node(node, leaf)
    current = node if plain is None else plain
    typed = schema_rule(current, leaf, carries=carries)
    result = current if typed is None else typed
    return None if result is node else result


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
    if is_nan(value):
        return False
    return _key(expr.left) == _key(expr.right)
