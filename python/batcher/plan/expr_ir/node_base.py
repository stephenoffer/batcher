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

__all__ = ["IRNode", "child", "children", "expr_node", "literal", "scalar"]

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


def _encode(kind: _Kind, value: Any) -> Any:
    if kind is _Kind.CHILD:
        return value.to_ir()
    if kind is _Kind.CHILDREN:
        return [e.to_ir() for e in value]
    if kind is _Kind.LITERAL:
        from batcher.plan.expr_ir.core import Lit

        return Lit(value).to_ir()["value"]
    return value  # SCALAR


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
        out: dict[str, Any] = {"e": self.tag}
        for f in fields(self):
            spec = f.metadata.get(_META)
            if spec is None:
                continue  # constructor-only field, not part of the wire shape
            value = getattr(self, f.name)
            if spec.omit is _Omit.IF_NONE and value is None:
                continue
            if spec.omit is _Omit.IF_FALSY and not value:
                continue
            out[spec.ir_key or f.name] = _encode(spec.kind, value)
        return out


def expr_node(cls: type[_T]) -> type[_T]:
    """Class decorator turning an `IRNode` subclass into its constructor.

    A thin alias for ``dataclass(eq=False)`` — ``eq=False`` preserves `Expr`'s
    expression-building ``__eq__``/``__ne__`` and its ``__hash__ = None``. Named for
    intent so node definitions read as declarations, not "dataclasses".
    """
    return dataclass(eq=False)(cls)
