"""The structural ladder for scalar `Expr` trees — child access and rebuilding.

One table says what an expression node's sub-expressions are (`_EXPR_KIDS`) and one says
how to rebuild it from new ones (`_EXPR_REBUILD`). Every traversal in the package is built
on this pair, so a new `Expr` node type is taught to the whole optimizer by registering it
here rather than by extending an `isinstance` ladder in each rule.

Registering it is naming it in `_REGULAR`: both entries are then *derived* from the
node's own `child`/`scalar` declarations, the same declarations `expr_ir.walk` reads.
That is deliberate. When the pair was hand-written per node type, the hand-written
version drifted from the declaration it was copying, and a rebuild that quietly dropped
a node's own parameters is invisible to every gate — see the note above `_derived`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from operator import attrgetter
from typing import Any

from batcher.plan.expr_ir import (
    Array,
    Binary,
    Case,
    Cast,
    Coalesce,
    DateFunc,
    DateTrunc,
    Expr,
    Greatest,
    InList,
    IsNan,
    IsNotNull,
    IsNull,
    Least,
    ListContains,
    ListFunc,
    ListGet,
    ListGetDyn,
    ListJoin,
    ListSimhash,
    ListSlice,
    Math2Expr,
    MathExpr,
    Not,
    NullIf,
    StrFunc,
    StrFuncDyn,
    StructField,
)
from batcher.plan.expr_ir.audio import AudioFunc
from batcher.plan.expr_ir.core import Aliased, IsInf
from batcher.plan.expr_ir.func_nodes import (
    ConvertTimezone,
    DateOffset,
    GeoFunc,
    ListBinary,
    ListFilter,
    ListPosition,
    ListSet,
    ListTransform,
    ListZip,
    MakeTemporal,
    MapFunc,
    SpatialFunc,
    Strftime,
    Strptime,
    WindowBuckets,
    WindowStart,
)
from batcher.plan.expr_ir.image import ImageCrop, ImageFunc
from batcher.plan.expr_ir.namespaces.sequence import SeqFunc
from batcher.plan.expr_ir.node_base import child_fields_of
from batcher.plan.expr_ir.nodes import HashRows, MakeStruct, Sequence
from batcher.plan.expr_ir.video import VideoFunc

__all__ = ["ExprRule", "transform_expr_up"]

ExprRule = Callable[[Expr], Expr]


def _case_kids(e: Case) -> tuple[Expr, ...]:
    """A `Case`'s children, flattened: cond/then per branch, then `otherwise`."""
    return (*(x for branch in e.branches for x in branch), e.otherwise)


def _case_rebuild(_e: Case, kids: tuple[Expr, ...]) -> Expr:
    pairs = [(kids[i], kids[i + 1]) for i in range(0, len(kids) - 1, 2)]
    return Case(pairs, kids[-1])


def _make_struct_rebuild(e: MakeStruct, kids: tuple[Expr, ...]) -> Expr:
    """Rebuild a `MakeStruct` from rewritten field values, keeping each field's name."""
    return MakeStruct([(name, kid) for (name, _value), kid in zip(e.fields, kids, strict=True)])


# --- The traversal plan is derived, not restated --------------------------------------
#
# Every node already declares which of its fields hold sub-expressions (`child`/
# `children`) and which hold parameters (`scalar`/`literal`), and `expr_ir.walk` reads
# that declaration rather than repeating it. These two tables used to repeat it, as a
# hand-written lambda pair per node type -- and the repetition drifted. Three of the
# multimodal nodes rebuilt themselves without their own parameters, so a rewrite turned
# `.image.encode("jpeg")` into an `encode` with no format (a hard engine error),
# stripped `.image.to_tensor_f32`'s `mean`/`std` normalization and
# `.audio.mel_spectrogram`'s filterbank sizes (right shapes, silently wrong numbers),
# and hid `.video.frame_at`'s per-row timestamp from column pruning. An ordinary
# two-step projection was enough to trigger all of it, and the three tests guarding
# these tables all passed throughout, because each checks that a node type is *present*
# and none checked that a rebuilt node still carries its own fields.
#
# So the plan is derived from the same field metadata. A rebuild that drops a parameter
# is now unrepresentable rather than merely tested for.


def _optional_children(cls: type, names: tuple[str, ...]) -> frozenset[str]:
    """Those of `names` declared with a ``None`` default -- children a node may not have."""
    defaults = {f.name: f.default for f in dataclasses.fields(cls)}
    return frozenset(n for n in names if defaults.get(n) is None)


def _derived(
    cls: type, *, exclude: frozenset[str] = frozenset()
) -> tuple[Callable[[Any], tuple[Expr, ...]], Callable[[Any, tuple[Expr, ...]], Expr]]:
    """`cls`'s ``(children, rebuild)`` pair, read off its own field declarations.

    `exclude` drops a declared sub-expression from the traversal, for the nodes whose
    body is scoped to something other than the enclosing relation.

    Args:
        cls: An `IRNode` subclass whose sub-expression fields are plain fields.
        exclude: Field names to treat as opaque rather than descend into.

    Returns:
        The children accessor and the rebuild function for `cls`.
    """
    spec = tuple((n, is_list) for n, is_list in child_fields_of(cls) if n not in exclude)
    names = tuple(n for n, _ in spec)
    list_names = frozenset(n for n, is_list in spec if is_list)
    optional = _optional_children(cls, names)

    def rebuild(e: Any, kids: tuple[Expr, ...]) -> Expr:
        # `replace` copies every field it is not given, so the node's parameters ride
        # along by construction -- that is the whole point of deriving this.
        supplied = iter(kids)
        updates: dict[str, Any] = {}
        for name, is_list in spec:
            current = getattr(e, name)
            if current is None and name in optional:
                continue
            if is_list:
                updates[name] = [next(supplied) for _ in current]
            else:
                updates[name] = next(supplied)
        return dataclasses.replace(e, **updates)

    if list_names or optional:

        def kids_of(e: Any) -> tuple[Expr, ...]:
            out: list[Expr] = []
            for name, is_list in spec:
                value = getattr(e, name)
                if value is None and name in optional:
                    continue
                if is_list:
                    out.extend(value)
                else:
                    out.append(value)
            return tuple(out)

        return kids_of, rebuild

    # `attrgetter` with two or more names returns the tuple directly, so the multi-child
    # nodes now reach their children with no Python frame at all -- `transform_expr_up`
    # is the optimizer's hottest function and this is its first call.
    if len(names) == 1:
        get_one = attrgetter(names[0])
        return (lambda e: (get_one(e),)), rebuild
    return attrgetter(*names), rebuild


# Node types whose traversal is exactly their declaration: every `child`/`children`
# field is a sub-expression over the enclosing relation, and every other field is a
# parameter to carry across a rebuild unchanged.
_REGULAR: tuple[type, ...] = (
    Array, AudioFunc, Binary, Cast, Coalesce, ConvertTimezone, DateFunc, DateOffset,
    DateTrunc, GeoFunc, Greatest, HashRows, ImageCrop, ImageFunc, IsInf, IsNan,
    IsNotNull, IsNull, Least, ListBinary, ListContains, ListFunc, ListGet, ListGetDyn,
    ListJoin, ListPosition, ListSet, ListSimhash, ListSlice, ListZip, MakeTemporal,
    Math2Expr, MathExpr, MapFunc, Not, NullIf, SeqFunc, Sequence, SpatialFunc, Strftime,
    Strptime, StrFunc, StrFuncDyn, StructField, VideoFunc, WindowBuckets, WindowStart,
)  # fmt: skip

# `func`/`pred` are element-scoped sub-expressions -- they close over `element()`, not
# over the outer relation's columns -- so a rewrite of the outer projection must not
# descend into them. `walk.referenced_columns` and `walk.remap_columns` draw the line in
# the same place. Everything else about these two nodes derives as usual.
_SCOPED_OUT: dict[type, frozenset[str]] = {
    ListTransform: frozenset({"func"}),
    ListFilter: frozenset({"pred"}),
}

# Exact-type -> (children, rebuild) dispatch for `transform_expr_up`. `_EXPR_KIDS` yields
# a node's direct sub-expressions; `_EXPR_REBUILD` reconstructs the node from rewritten
# children (called only when a child actually changed). Leaves (Col, Lit, AggExpr, ...)
# are intentionally absent from both.
_EXPR_KIDS: dict[type, Callable[[Any], tuple[Expr, ...]]] = {}
_EXPR_REBUILD: dict[type, Callable[[Any, tuple[Expr, ...]], Expr]] = {}

for _cls in _REGULAR:
    _EXPR_KIDS[_cls], _EXPR_REBUILD[_cls] = _derived(_cls)
for _cls, _excluded in _SCOPED_OUT.items():
    _EXPR_KIDS[_cls], _EXPR_REBUILD[_cls] = _derived(_cls, exclude=_excluded)
del _cls, _excluded

# The four that cannot be derived. `InList` and `Aliased` predate the declarative base
# and carry their own `to_ir`, so they have no field metadata to read; `Case` and
# `MakeStruct` nest their sub-expressions inside tuples, a shape the per-field
# declaration cannot describe.
_EXPR_KIDS.update(
    {
        InList: lambda e: (e.input,),
        Aliased: lambda e: (e.inner,),
        MakeStruct: lambda e: tuple(value for _name, value in e.fields),
        Case: _case_kids,
    }
)
_EXPR_REBUILD.update(
    {
        InList: lambda e, k: InList(k[0], e.values),
        Aliased: lambda e, k: Aliased(k[0], e.name),
        MakeStruct: _make_struct_rebuild,
        Case: _case_rebuild,
    }
)


def transform_expr_up(expr: Expr, rule: ExprRule) -> Expr:
    """Bottom-up expression rewrite: rebuild children first, then apply `rule` to
    the rebuilt node. A `rule` only has to handle one node given already-rewritten
    children — the structural recursion lives here, once.

    Dispatch is an O(1) exact-type lookup (`_EXPR_KIDS`) rather than a long
    ``isinstance`` ladder: this runs for every node of every rule of every fixpoint
    iteration, and leaves (`Col`/`Lit` — the bulk of nodes) previously paid the full
    ladder before falling through. A node type absent from the table is a leaf with no
    sub-expressions (these concrete IR node types are never subclassed, so exact-type
    dispatch matches the old ``isinstance`` semantics).

    **Structural sharing:** a node whose rewritten children are all the *same objects*
    (`is`) is not rebuilt — `expr` itself is passed to `rule`. Most rules match a
    handful of nodes and leave the rest alone, so without this every rule reallocated
    the entire expression tree (discarding each node's memoized `to_ir`, and
    re-running the enclosing plan node's column validation) on every fixpoint pass.
    Nodes are immutable and value-typed, so reusing one is indistinguishable from
    rebuilding an equal copy — except that the optimizer's `is`-based change detection
    and per-node memo caches now hit."""
    kids_of = _EXPR_KIDS.get(type(expr))
    if kids_of is None:
        return rule(expr)  # leaf: no sub-expressions
    kids = kids_of(expr)
    # The one- and two-child cases are spelled out because they are almost every node in
    # almost every expression (unary functions and `Binary`), and this is the hottest
    # function in the optimizer — ~900 calls per plan per pass, across 300+ rules. The
    # general path below builds a generator, a tuple, a `zip`, and an `all` generator per
    # node; at this size those frames cost more than the work they wrap. Semantics are
    # identical, including the structural sharing: a node whose children all come back
    # `is`-identical is passed through rather than rebuilt.
    n = len(kids)
    if n == 2:
        left, right = kids
        new_left = transform_expr_up(left, rule)
        new_right = transform_expr_up(right, rule)
        if new_left is left and new_right is right:
            return rule(expr)
        return rule(_EXPR_REBUILD[type(expr)](expr, (new_left, new_right)))
    if n == 1:
        only = kids[0]
        new_only = transform_expr_up(only, rule)
        if new_only is only:
            return rule(expr)
        return rule(_EXPR_REBUILD[type(expr)](expr, (new_only,)))
    new = tuple(transform_expr_up(k, rule) for k in kids)
    rebuilt = (
        expr
        if all(a is b for a, b in zip(new, kids, strict=True))
        else _EXPR_REBUILD[type(expr)](expr, new)
    )
    return rule(rebuilt)
