"""The structural ladder for scalar `Expr` trees — child access and rebuilding.

One table says what an expression node's sub-expressions are (`_EXPR_KIDS`) and one says
how to rebuild it from new ones (`_EXPR_REBUILD`). Every traversal in the package is built
on this pair, so a new `Expr` node type is taught to the whole optimizer by adding two
entries here rather than by extending an `isinstance` ladder in each rule.
"""

from __future__ import annotations

from collections.abc import Callable
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
    ListJoin,
    ListSimhash,
    ListSlice,
    Math2Expr,
    MathExpr,
    Not,
    NullIf,
    StrFunc,
    StructField,
)
from batcher.plan.expr_ir.audio import AudioFunc
from batcher.plan.expr_ir.core import Aliased, IsInf
from batcher.plan.expr_ir.func_nodes import (
    ConvertTimezone,
    DateOffset,
    ListBinary,
    ListFilter,
    ListPosition,
    ListSet,
    ListTransform,
    MapFunc,
    Strftime,
    Strptime,
    WindowBuckets,
    WindowStart,
)
from batcher.plan.expr_ir.image import ImageFunc
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


# Exact-type → (children, rebuild) dispatch for `transform_expr_up`. `_EXPR_KIDS` yields
# a node's direct sub-expressions; `_EXPR_REBUILD` reconstructs the node from rewritten
# children (called only when a child actually changed). Leaves (Col, Lit, AggExpr, …)
# are intentionally absent from both.
_EXPR_KIDS: dict[type, Callable[[Any], tuple[Expr, ...]]] = {
    Binary: lambda e: (e.left, e.right),
    Not: lambda e: (e.input,),
    Cast: lambda e: (e.input,),
    IsNull: lambda e: (e.input,),
    IsNotNull: lambda e: (e.input,),
    IsNan: lambda e: (e.input,),
    IsInf: lambda e: (e.input,),
    InList: lambda e: (e.input,),
    MathExpr: lambda e: (e.input,),
    DateFunc: lambda e: (e.input,),
    DateTrunc: lambda e: (e.input,),
    ListFunc: lambda e: (e.input,),
    ListGet: lambda e: (e.input,),
    ListSimhash: lambda e: (e.input,),
    ListContains: lambda e: (e.input,),
    ListSlice: lambda e: (e.input,),
    StructField: lambda e: (e.input,),
    ListJoin: lambda e: (e.input,),
    StrFunc: lambda e: (e.input,),
    ImageFunc: lambda e: (e.input,),
    Coalesce: lambda e: tuple(e.inputs),
    Greatest: lambda e: tuple(e.inputs),
    HashRows: lambda e: tuple(e.inputs),
    Least: lambda e: tuple(e.inputs),
    Array: lambda e: tuple(e.elements),
    NullIf: lambda e: (e.left, e.right),
    Math2Expr: lambda e: (e.left, e.right),
    ListBinary: lambda e: (e.left, e.right),
    ListSet: lambda e: (e.left, e.right),
    ListPosition: lambda e: (e.input,),
    MapFunc: lambda e: (e.input,),
    ConvertTimezone: lambda e: (e.input,),
    DateOffset: lambda e: (e.input,),
    Strftime: lambda e: (e.input,),
    Strptime: lambda e: (e.input,),
    WindowStart: lambda e: (e.input,),
    WindowBuckets: lambda e: (e.input,),
    # `func`/`pred` are element-scoped sub-expressions (they close over `element()`,
    # not over the outer relation's columns), so a rewrite of the outer projection
    # must not descend into them. `walk.remap_columns` draws the line in the same place.
    ListTransform: lambda e: (e.input,),
    ListFilter: lambda e: (e.input,),
    Aliased: lambda e: (e.inner,),
    AudioFunc: lambda e: (e.input,),
    VideoFunc: lambda e: (e.input,),
    MakeStruct: lambda e: tuple(value for _name, value in e.fields),
    Sequence: lambda e: (e.start, e.stop, e.step),
    Case: _case_kids,
}

_EXPR_REBUILD: dict[type, Callable[[Any, tuple[Expr, ...]], Expr]] = {
    Binary: lambda e, k: Binary(e.op, k[0], k[1]),
    Not: lambda _e, k: Not(k[0]),
    Cast: lambda e, k: Cast(k[0], e.dtype, try_cast=e.try_cast),
    IsNull: lambda _e, k: IsNull(k[0]),
    IsNotNull: lambda _e, k: IsNotNull(k[0]),
    IsNan: lambda _e, k: IsNan(k[0]),
    IsInf: lambda _e, k: IsInf(k[0]),
    InList: lambda e, k: InList(k[0], e.values),
    MathExpr: lambda e, k: MathExpr(e.fn, k[0]),
    DateFunc: lambda e, k: DateFunc(e.fn, k[0]),
    DateTrunc: lambda e, k: DateTrunc(k[0], e.unit),
    ListFunc: lambda e, k: ListFunc(e.fn, k[0]),
    ListGet: lambda e, k: ListGet(k[0], e.index),
    ListSimhash: lambda e, k: ListSimhash(k[0], e.num_bits, e.seed),
    ListContains: lambda e, k: ListContains(k[0], e.value),
    ListSlice: lambda e, k: ListSlice(k[0], e.offset, e.length),
    StructField: lambda e, k: StructField(k[0], e.field),
    ListJoin: lambda e, k: ListJoin(k[0], e.separator),
    StrFunc: lambda e, k: StrFunc(
        e.fn,
        k[0],
        pattern=e.pattern,
        replacement=e.replacement,
        start=e.start,
        length=e.length,
    ),
    ImageFunc: lambda e, k: ImageFunc(e.fn, k[0], width=e.width, height=e.height),
    Coalesce: lambda _e, k: Coalesce(list(k)),
    Greatest: lambda _e, k: Greatest(list(k)),
    HashRows: lambda e, k: HashRows(list(k), e.seed),
    Least: lambda _e, k: Least(list(k)),
    Array: lambda _e, k: Array(list(k)),
    NullIf: lambda _e, k: NullIf(k[0], k[1]),
    Math2Expr: lambda e, k: Math2Expr(e.fn, k[0], k[1]),
    ListBinary: lambda e, k: ListBinary(e.fn, k[0], k[1]),
    ListSet: lambda e, k: ListSet(e.fn, k[0], k[1]),
    ListPosition: lambda e, k: ListPosition(k[0], e.value),
    MapFunc: lambda e, k: MapFunc(e.fn, k[0], e.key),
    ConvertTimezone: lambda e, k: ConvertTimezone(k[0], e.from_tz, e.to_tz),
    DateOffset: lambda e, k: DateOffset(k[0], e.months, e.days, e.micros),
    Strftime: lambda e, k: Strftime(k[0], e.format),
    Strptime: lambda e, k: Strptime(k[0], e.format),
    WindowStart: lambda e, k: WindowStart(k[0], e.width_micros, e.origin_micros),
    WindowBuckets: lambda e, k: WindowBuckets(k[0], e.width_micros, e.slide_micros),
    ListTransform: lambda e, k: ListTransform(k[0], e.func),
    ListFilter: lambda e, k: ListFilter(k[0], e.pred),
    Aliased: lambda e, k: Aliased(k[0], e.name),
    AudioFunc: lambda e, k: AudioFunc(e.fn, k[0], e.rate),
    VideoFunc: lambda e, k: VideoFunc(e.fn, k[0]),
    MakeStruct: _make_struct_rebuild,
    Sequence: lambda _e, k: Sequence(k[0], k[1], k[2]),
    Case: _case_rebuild,
}


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
    new = tuple(transform_expr_up(k, rule) for k in kids)
    rebuilt = (
        expr
        if all(a is b for a, b in zip(new, kids, strict=True))
        else _EXPR_REBUILD[type(expr)](expr, new)
    )
    return rule(rebuilt)
