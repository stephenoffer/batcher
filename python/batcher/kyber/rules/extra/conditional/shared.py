"""Shared guards for the conditional family: purity, type tags, and droppability.

The delicate one is `_droppable`. `CASE WHEN FALSE THEN 1 ELSE 2.5 END` is a DOUBLE, so
removing an arm can silently *narrow* the result type; an arm is dropped only when a kept
arm provably carries the same type. `_pure` stops a drop from erasing an error.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

from batcher._internal.mathx import is_nan

# `_key` (structural identity), `_rewrite_node` (leaf Expr rule → rebuilt node, or None) and
# `_safe` (deterministic + non-erroring) are the sibling family's helpers, imported rather than
# re-implemented — copy-paste is the one wrong way to share.
from batcher.kyber.rules.extra.boolean_algebra import _SAFE_BINARY_OPS, _key, _safe
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
    if is_nan(value):
        return False
    return _key(expr.left) == _key(expr.right)
