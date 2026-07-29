"""Declarative base for the scalar `Expr` IR nodes — kills the `to_ir()` boilerplate.

Every concrete IR node used to hand-write the same three things: a ``__slots__``
tuple, an ``__init__`` that copies args to attributes, and a ``to_ir()`` that emits
``{"e": <tag>, ...}`` while recursing into children, lifting literals, and omitting
absent optionals. That is mechanical and identical across ~40 nodes, so it lives
here once.

A node now declares its shape as data: subclass `IRNode`, set the class-level
``tag`` (from `ir_tags.ExprTag`), and annotate each field with one of the field
factories below — `child` (recurse `to_ir`), `children` (a list of them), `scalar`
(emit as-is), or `literal` (wrap a Python constant through `Lit`). The
``@expr_node`` decorator (a thin alias for ``dataclass(eq=False)``) generates the
constructor; `IRNode.to_ir` reads the field metadata and assembles the wire dict.

``eq=False`` is mandatory: `Expr` overloads ``__eq__`` to *build* an expression
(``col("x") == 1`` is a predicate, not a bool), so a dataclass-generated ``__eq__``
would silently break expression building. Nodes inherit `Expr`'s ``__hash__ = None``
and stay unhashable, exactly as before. The emitted IR is byte-identical to the
hand-written ``to_ir`` it replaces — locked by ``tests/unit/test_ir_snapshot.py``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, ClassVar, TypeVar

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.core import Expr

__all__ = ["IRNode", "child", "child_fields", "children", "expr_node", "literal", "scalar"]

_T = TypeVar("_T")

# Metadata key under which a field stashes its wire spec (dataclass field metadata
# is an arbitrary read-only mapping; we namespace ours to avoid collisions).
_META: str = "batcher_ir"
_NODEFAULT: Any = object()


class _Kind(Enum):
    """How a field's Python value becomes its JSON value."""

    CHILD = "child"  # a sub-`Expr` → value.to_ir()
    CHILDREN = "children"  # a list of sub-`Expr` → [e.to_ir() for e in value]
    SCALAR = "scalar"  # a str/int/bool/float → emitted as-is
    LITERAL = "literal"  # a Python constant → lifted through Lit(value)


class _Omit(Enum):
    """When a field is dropped from the wire dict entirely."""

    NEVER = "never"
    IF_NONE = "if_none"  # absent optional (value is None)
    IF_FALSY = "if_falsy"  # zero/empty component (serde defaults it)


@dataclasses.dataclass(frozen=True)
class _FieldSpec:
    kind: _Kind
    ir_key: str | None = None  # JSON key when it differs from the attribute name
    omit: _Omit = _Omit.NEVER


def _make_field(spec: _FieldSpec, default: Any) -> Any:
    meta = {_META: spec}
    if default is _NODEFAULT:
        return dataclasses.field(metadata=meta)
    return dataclasses.field(default=default, metadata=meta)


def child(*, key: str | None = None, omit_none: bool = False, default: Any = _NODEFAULT) -> Any:
    """A sub-expression field — serialized by recursing into ``value.to_ir()``."""
    omit = _Omit.IF_NONE if omit_none else _Omit.NEVER
    return _make_field(_FieldSpec(_Kind.CHILD, key, omit), default)


def children(*, key: str | None = None, default: Any = _NODEFAULT) -> Any:
    """A list-of-sub-expressions field — serialized to ``[e.to_ir() for e in value]``."""
    return _make_field(_FieldSpec(_Kind.CHILDREN, key), default)


def scalar(
    *,
    key: str | None = None,
    omit_none: bool = False,
    omit_falsy: bool = False,
    default: Any = _NODEFAULT,
) -> Any:
    """A plain JSON scalar field (string tag, int, bool, float) emitted as-is.

    ``omit_falsy`` drops zero/empty values (the engine's serde defaults them);
    ``omit_none`` drops only ``None``.
    """
    omit = _Omit.IF_FALSY if omit_falsy else (_Omit.IF_NONE if omit_none else _Omit.NEVER)
    return _make_field(_FieldSpec(_Kind.SCALAR, key, omit), default)


def literal(*, key: str | None = None, omit_none: bool = False, default: Any = _NODEFAULT) -> Any:
    """A Python constant lifted through `Lit` to its tagged wire value (``{"int": 5}``)."""
    omit = _Omit.IF_NONE if omit_none else _Omit.NEVER
    return _make_field(_FieldSpec(_Kind.LITERAL, key, omit), default)


def _encode_child(value: Any) -> Any:
    return value.to_ir()


def _encode_children(value: Any) -> Any:
    return [e.to_ir() for e in value]


# `core` is this module's own dependency, so `Lit` cannot come in at module level — and it
# was re-imported for every literal-valued field of every node lowered.
_LIT: type | None = None


def _encode_literal(value: Any) -> Any:
    global _LIT
    if _LIT is None:
        from batcher.plan.expr_ir.core import Lit

        _LIT = Lit
    return _LIT(value).to_ir()["value"]


# Per-kind encoder, resolved once when a class's wire plan is built. `SCALAR` maps to
# `None`, the "emit the attribute unchanged" sentinel, so the overwhelmingly common
# field kind costs a truth test rather than a function call.
_ENCODERS: dict[_Kind, Any] = {
    _Kind.CHILD: _encode_child,
    _Kind.CHILDREN: _encode_children,
    _Kind.LITERAL: _encode_literal,
    _Kind.SCALAR: None,
}

# Class attributes holding a node class's precomputed shape: its serialization plan
# (`_wire_plan`) and its sub-expression fields (`child_fields`).
_PLAN_ATTR = "_ir_wire_plan"
_CHILDREN_ATTR = "_ir_child_fields"


def _wire_plan(cls: type) -> tuple[tuple[str, str, Any, bool, bool], ...]:
    """`cls`'s serialization plan: ``(attr, ir_key, encoder, omit_none, omit_falsy)`` per field.

    `to_ir` used to re-derive this on every node it serialized: `dataclasses.fields`
    materializes a fresh tuple per call, each field's metadata mapping is then probed for
    the wire spec, and the encoder is chosen by a chain of enum comparisons — all of it a
    pure function of the *class*, recomputed per *instance*. Expression trees are built and
    lowered constantly (every `select`, every optimizer re-lowering), so this is one of the
    hottest loops in the control plane.

    Resolving it once per class turns the per-node work into a walk over a flat tuple of
    pre-resolved values. Stored on the class (not a module dict) so a class that is
    garbage-collected takes its plan with it, and looked up through `cls.__dict__` so a
    subclass never inherits its parent's plan.
    """
    plan = cls.__dict__.get(_PLAN_ATTR)
    if plan is None:
        plan = tuple(
            (
                f.name,
                spec.ir_key or f.name,
                _ENCODERS[spec.kind],
                spec.omit is _Omit.IF_NONE,
                spec.omit is _Omit.IF_FALSY,
            )
            for f in fields(cls)
            if (spec := f.metadata.get(_META)) is not None
        )
        setattr(cls, _PLAN_ATTR, plan)
    return plan


class IRNode(Expr):
    """Base for declarative `Expr` IR nodes — a generic, metadata-driven `to_ir`.

    Subclasses are ``@expr_node`` dataclasses that set ``tag`` and declare fields via
    `child`/`children`/`scalar`/`literal`. Irregular nodes (`Lit`, `Case`, …) may
    subclass this and override `to_ir`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.col("x").to_ir()
            {'e': 'col', 'name': 'x'}
    """

    tag: ClassVar[str]
    # When set, the node's ``fn`` field is validated against this vocabulary at
    # construction, so an unknown function name fails early with a clear error
    # rather than as an opaque engine error. See `fn_names`.
    vocab: ClassVar[frozenset[str] | None] = None

    def __post_init__(self) -> None:
        if self.vocab is not None and self.fn not in self.vocab:  # type: ignore[attr-defined]
            raise PlanError(
                f"unknown {type(self).__name__} function "
                f"{self.fn!r}; "  # type: ignore[attr-defined]
                "add it to the family vocabulary in plan/expr_ir/fn_names.py"
            )

    def to_ir(self) -> dict[str, Any]:
        # `to_ir` is a pure function of an immutable node, but the optimizer calls it
        # heavily — canonical keys for CSE/dedup, plus recursive re-lowering as rules
        # rewrite ancestors — and each call otherwise re-walks the whole subtree (a
        # superlinear cost on large plans). Memoize the result on the node: `Expr` sets
        # no `__slots__`, so every node has a `__dict__` to cache in, and the node is
        # immutable after construction. Callers treat the IR as read-only (verified: no
        # code mutates a `to_ir()` dict in place), so sharing the cached dict is safe.
        cached = self.__dict__.get("_ir_cache")
        if cached is not None:
            return cached
        out: dict[str, Any] = {"e": self.tag}
        for name, key, encode, omit_none, omit_falsy in _wire_plan(type(self)):
            value = getattr(self, name)
            if omit_none and value is None:
                continue
            if omit_falsy and not value:
                continue
            out[key] = value if encode is None else encode(value)
        self.__dict__["_ir_cache"] = out
        return out


def child_fields(node: IRNode) -> tuple[tuple[str, bool], ...]:
    """The ``(field_name, is_list)`` of each sub-expression field of an `IRNode`.

    A generic view of a node's shape drawn from the same field metadata `to_ir` uses:
    ``CHILD`` fields yield ``(name, False)``, ``CHILDREN`` fields ``(name, True)``.
    It lets a caller recurse into and rebuild an arbitrary node (via
    ``dataclasses.replace``) without a hand-written per-node visitor — used by the
    aggregate-expression splitter to swap aggregate leaves for column references.

    Like `to_ir`'s wire plan, the shape is a property of the *class*, so it is resolved
    once and cached on it. The generic walks in `expr_ir.walk` — column collection,
    column remapping, the aggregate splitter — call this on every node of every
    expression they visit, and each call otherwise rebuilt the field tuple and re-probed
    every field's metadata to rediscover a fixed answer.
    """
    cls = type(node)
    out = cls.__dict__.get(_CHILDREN_ATTR)
    if out is None:
        out = tuple(
            (f.name, spec.kind is _Kind.CHILDREN)
            for f in fields(cls)
            if (spec := f.metadata.get(_META)) is not None
            and spec.kind in (_Kind.CHILD, _Kind.CHILDREN)
        )
        setattr(cls, _CHILDREN_ATTR, out)
    return out


def expr_node(cls: type[_T]) -> type[_T]:
    """Class decorator turning an `IRNode` subclass into its constructor.

    A thin alias for ``dataclass(eq=False, repr=False)`` — ``eq=False`` preserves
    `Expr`'s expression-building ``__eq__``/``__ne__`` and its ``__hash__ = None``, and
    ``repr=False`` keeps `Expr`'s source-like ``__repr__`` instead of the dataclass's
    field dump. Named for intent so node definitions read as declarations.
    """
    return dataclass(eq=False, repr=False)(cls)
