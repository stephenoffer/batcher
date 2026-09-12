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

__all__ = [
    "IRNode",
    "child",
    "child_fields",
    "child_fields_of",
    "children",
    "expr_node",
    "literal",
    "scalar",
    "scalar_fields_of",
]

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


# The JSON scalars a `scalar` field may carry. Anything else -- a Python type, an
# `Expr`, an arbitrary object -- cannot cross the wire, and until this check existed it
# reached `json.dumps` and surfaced as "Object of type X is not JSON serializable":
# an error naming the serializer rather than the argument the user got wrong.
_JSON_SCALARS: tuple[type, ...] = (str, int, float, bool)


def _json_safe(value: Any) -> bool:
    """Whether `value` is a JSON scalar, or a list/tuple of them (nested allowed)."""
    if value is None or isinstance(value, _JSON_SCALARS):
        return True
    if isinstance(value, (list, tuple)):
        return all(_json_safe(v) for v in value)
    return False


def _where(node: Any) -> str:
    """``str.contains`` for a family node that carries its function, else the tag."""
    tag = getattr(node.tag, "value", node.tag)
    fn = getattr(node, "fn", None)
    return f"{tag}.{fn}" if isinstance(fn, str) else str(tag)


def _reject(node: Any, key: str, value: Any, expected: str) -> None:
    """Raise the one wrong-argument message the whole expression surface shares."""
    raise PlanError(
        f"{_where(node)}(): {key}={value!r} is not valid for this argument - expected "
        f"{expected}, got {type(value).__name__}. Most functions take their "
        f"pattern/format/key as a plain Python value known when the plan is built, "
        f"not a column or an object."
    )


def _check_scalar(node: Any, key: str, value: Any, types: tuple[type, ...] | None = None) -> None:
    """Reject a `scalar` field value that cannot cross the JSON wire.

    Names the *function* rather than the node tag where the node carries one, so a
    family node (`StrFunc`, tag ``str``) reports ``str.contains()`` and not ``str()``.
    """
    if types is not None and value is not None:
        if not isinstance(value, types):
            _reject(node, key, value, " or ".join(sorted({t.__name__ for t in types})))
        return
    if _json_safe(value):
        return
    tag = getattr(node.tag, "value", node.tag)
    fn = getattr(node, "fn", None)
    where = f"{tag}.{fn}" if isinstance(fn, str) else str(tag)
    raise PlanError(
        f"{where}(): {key}={value!r} is not a valid value for this argument - it must "
        f"be a string, number, boolean, or a list of those, not a "
        f"{type(value).__name__}. Most functions take their pattern/format/key as a "
        f"plain Python value known when the plan is built, not a column or an object."
    )


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
# Scalar field annotations are simple unions of builtins (``str``, ``int | None``), so the
# declared type is itself the spec for what the field may carry -- no second table to keep
# in sync. A token this does not recognise (``list[str]``, ``object``) yields no constraint,
# so an unusual field is left exactly as permissive as it was.
_ANNOTATION_TYPES: dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
}


def _scalar_types(annotation: Any) -> tuple[type, ...] | None:
    """The types a `scalar` field may hold, read off its annotation, or None for 'any'."""
    text = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "")
    tokens = [t.strip() for t in text.split("|")]
    allowed: list[type] = []
    for token in tokens:
        if token == "None":
            continue
        mapped = _ANNOTATION_TYPES.get(token)
        if mapped is None:
            return None  # not a plain builtin union -- impose no constraint
        allowed.append(mapped)
    if not allowed:
        return None
    # `bool` is a subclass of `int`, and an int is a fine stand-in for a float, so widen
    # rather than reject a value the engine already accepts.
    if int in allowed and bool not in allowed:
        allowed.append(bool)
    if float in allowed and int not in allowed:
        allowed.append(int)
        allowed.append(bool)
    return tuple(allowed)


# (`_wire_plan`) and its sub-expression fields (`child_fields`).
_PLAN_ATTR = "_ir_wire_plan"
_CHILDREN_ATTR = "_ir_child_fields"


def _wire_plan(cls: type) -> tuple[tuple[str, str, Any, bool, bool, Any], ...]:
    """`cls`'s plan: ``(attr, ir_key, encoder, omit_none, omit_falsy, types)`` per field.

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
                _scalar_types(f.type) if spec.kind is _Kind.SCALAR else None,
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
        for name, key, encode, omit_none, omit_falsy, types in _wire_plan(type(self)):
            value = getattr(self, name)
            if omit_none and value is None:
                continue
            if omit_falsy and not value:
                continue
            if encode is None:
                _check_scalar(self, key, value, types)
                out[key] = value
            else:
                out[key] = encode(value)
        self.__dict__["_ir_cache"] = out
        return out


def child_fields_of(cls: type) -> tuple[tuple[str, bool], ...]:
    """The ``(field_name, is_list)`` of each sub-expression field of an `IRNode` *class*.

    The class-level form of `child_fields`, for callers that have the type rather than an
    instance — the expression-rewrite tables build their per-class traversal plans from
    this at import time, before any node of that type exists.

    Args:
        cls: An `IRNode` subclass.

    Returns:
        One ``(name, is_list)`` pair per `child`/`children` field, in declaration order.
        Empty for a node that predates the declarative base (`InList`, `Aliased`), which
        declares no field metadata to read and so must be handled explicitly by callers.
    """
    out = cls.__dict__.get(_CHILDREN_ATTR)
    if out is None:
        if not dataclasses.is_dataclass(cls):
            return ()
        out = tuple(
            (f.name, spec.kind is _Kind.CHILDREN)
            for f in fields(cls)
            if (spec := f.metadata.get(_META)) is not None
            and spec.kind in (_Kind.CHILD, _Kind.CHILDREN)
        )
        setattr(cls, _CHILDREN_ATTR, out)
    return out


def scalar_fields_of(cls: type) -> tuple[str, ...]:
    """The names of `cls`'s non-sub-expression fields — its `scalar`/`literal` parameters.

    The complement of `child_fields_of`. A rewrite that rebuilds a node from new children
    must carry every one of these across unchanged; naming them here is what lets that be
    derived from the node's own declaration rather than restated per node type.

    Args:
        cls: An `IRNode` subclass.

    Returns:
        The parameter field names, in declaration order. Empty for a node that predates
        the declarative base, for the reason given on `child_fields_of`.
    """
    if not dataclasses.is_dataclass(cls):
        return ()
    kids = {name for name, _ in child_fields_of(cls)}
    return tuple(f.name for f in fields(cls) if f.name not in kids)


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
    return child_fields_of(type(node))


def expr_node(cls: type[_T]) -> type[_T]:
    """Class decorator turning an `IRNode` subclass into its constructor.

    A thin alias for ``dataclass(eq=False, repr=False)`` — ``eq=False`` preserves
    `Expr`'s expression-building ``__eq__``/``__ne__`` and its ``__hash__ = None``, and
    ``repr=False`` keeps `Expr`'s source-like ``__repr__`` instead of the dataclass's
    field dump. Named for intent so node definitions read as declarations.
    """
    return dataclass(eq=False, repr=False)(cls)
