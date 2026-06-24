"""Structural traversals over the expression tree.

`referenced_columns` collects the input column names an expression reads;
`remap_columns` returns a copy with column names rewritten (used to push a
predicate through a join). Both walk every node kind, so they import the node
classes from `core` and `namespaces`.
"""

from __future__ import annotations

from batcher.plan.expr_ir.audio import AudioFunc
from batcher.plan.expr_ir.core import (
    Binary,
    Cast,
    Coalesce,
    Expr,
    IsInf,
    IsNan,
    IsNotNull,
    IsNull,
    Math2Expr,
    MathExpr,
    Not,
)
from batcher.plan.expr_ir.func_nodes import WindowBuckets, WindowStart
from batcher.plan.expr_ir.image import ImageFunc
from batcher.plan.expr_ir.namespaces import (
    ConvertTimezone,
    DateFunc,
    DateOffset,
    DateTrunc,
    ListBinary,
    ListContains,
    ListFilter,
    ListFunc,
    ListGet,
    ListPosition,
    ListSet,
    ListSlice,
    ListTransform,
    MapFunc,
    Strftime,
    StrFunc,
    Strptime,
    StructField,
)
from batcher.plan.expr_ir.nodes import (
    Array,
    Case,
    Col,
    Greatest,
    Least,
    ListJoin,
    MakeStruct,
    NullIf,
    Sequence,
)
from batcher.plan.expr_ir.video import VideoFunc


def referenced_columns(expr: Expr) -> set[str]:
    """The set of input column names an expression reads."""
    if isinstance(expr, Col):
        return {expr.name}
    if isinstance(expr, Binary):
        return referenced_columns(expr.left) | referenced_columns(expr.right)
    if isinstance(
        expr,
        (
            Not,
            Cast,
            IsNull,
            IsNotNull,
            IsNan,
            StrFunc,
            Strftime,
            ConvertTimezone,
            DateFunc,
            DateOffset,
            DateTrunc,
            ImageFunc,
            AudioFunc,
            VideoFunc,
            MathExpr,
            ListFunc,
            ListGet,
            ListContains,
            ListPosition,
            ListTransform,
            ListFilter,
            ListSlice,
            ListJoin,
            StructField,
            MapFunc,
            WindowStart,
            WindowBuckets,
        ),
    ):
        return referenced_columns(expr.input)
    if isinstance(expr, (Coalesce, Greatest, Least)):
        cols: set[str] = set()
        for e in expr.inputs:
            cols |= referenced_columns(e)
        return cols
    if isinstance(expr, Array):
        out: set[str] = set()
        for e in expr.elements:
            out |= referenced_columns(e)
        return out
    if isinstance(expr, MakeStruct):
        cols: set[str] = set()
        for _name, value in expr.fields:
            cols |= referenced_columns(value)
        return cols
    if isinstance(expr, Sequence):
        return (
            referenced_columns(expr.start)
            | referenced_columns(expr.stop)
            | referenced_columns(expr.step)
        )
    if isinstance(expr, (NullIf, Math2Expr, ListBinary, ListSet)):
        return referenced_columns(expr.left) | referenced_columns(expr.right)
    if isinstance(expr, Case):
        cols = referenced_columns(expr.otherwise)
        for cond, then in expr.branches:
            cols |= referenced_columns(cond) | referenced_columns(then)
        return cols
    return set()  # Lit and other leaves reference nothing


def remap_columns(expr: Expr, mapping: dict[str, str]) -> Expr:
    """Return a copy of `expr` with column names rewritten via `mapping`.

    Used to push a predicate through a join: a conjunct phrased in the join's
    output names is rewritten into one side's source names before being attached
    below the join.
    """
    if isinstance(expr, Col):
        return Col(mapping.get(expr.name, expr.name))
    if isinstance(expr, Binary):
        return Binary(
            expr.op, remap_columns(expr.left, mapping), remap_columns(expr.right, mapping)
        )
    if isinstance(expr, Not):
        return Not(remap_columns(expr.input, mapping))
    if isinstance(expr, Cast):
        return Cast(remap_columns(expr.input, mapping), expr.dtype, try_cast=expr.try_cast)
    if isinstance(expr, IsNull):
        return IsNull(remap_columns(expr.input, mapping))
    if isinstance(expr, IsNotNull):
        return IsNotNull(remap_columns(expr.input, mapping))
    if isinstance(expr, IsNan):
        return IsNan(remap_columns(expr.input, mapping))
    if isinstance(expr, IsInf):
        return IsInf(remap_columns(expr.input, mapping))
    if isinstance(expr, StrFunc):
        return StrFunc(
            expr.fn,
            remap_columns(expr.input, mapping),
            pattern=expr.pattern,
            replacement=expr.replacement,
            start=expr.start,
            length=expr.length,
        )
    if isinstance(expr, DateFunc):
        return DateFunc(expr.fn, remap_columns(expr.input, mapping))
    if isinstance(expr, ImageFunc):
        return ImageFunc(
            expr.fn, remap_columns(expr.input, mapping), width=expr.width, height=expr.height
        )
    if isinstance(expr, AudioFunc):
        return AudioFunc(expr.fn, remap_columns(expr.input, mapping))
    if isinstance(expr, VideoFunc):
        return VideoFunc(expr.fn, remap_columns(expr.input, mapping))
    if isinstance(expr, DateTrunc):
        return DateTrunc(remap_columns(expr.input, mapping), expr.unit)
    if isinstance(expr, Strftime):
        return Strftime(remap_columns(expr.input, mapping), expr.format)
    if isinstance(expr, Strptime):
        return Strptime(remap_columns(expr.input, mapping), expr.format)
    if isinstance(expr, ConvertTimezone):
        return ConvertTimezone(remap_columns(expr.input, mapping), expr.from_tz, expr.to_tz)
    if isinstance(expr, DateOffset):
        return DateOffset(remap_columns(expr.input, mapping), expr.months, expr.days, expr.micros)
    if isinstance(expr, WindowStart):
        return WindowStart(
            remap_columns(expr.input, mapping), expr.width_micros, expr.origin_micros
        )
    if isinstance(expr, WindowBuckets):
        return WindowBuckets(
            remap_columns(expr.input, mapping), expr.width_micros, expr.slide_micros
        )
    if isinstance(expr, MathExpr):
        return MathExpr(expr.fn, remap_columns(expr.input, mapping))
    if isinstance(expr, ListFunc):
        return ListFunc(expr.fn, remap_columns(expr.input, mapping))
    if isinstance(expr, ListGet):
        return ListGet(remap_columns(expr.input, mapping), expr.index)
    if isinstance(expr, ListContains):
        return ListContains(remap_columns(expr.input, mapping), expr.value)
    if isinstance(expr, ListPosition):
        return ListPosition(remap_columns(expr.input, mapping), expr.value)
    if isinstance(expr, ListTransform):
        return ListTransform(remap_columns(expr.input, mapping), expr.func)
    if isinstance(expr, ListFilter):
        return ListFilter(remap_columns(expr.input, mapping), expr.pred)
    if isinstance(expr, ListSlice):
        return ListSlice(remap_columns(expr.input, mapping), expr.offset, expr.length)
    if isinstance(expr, StructField):
        return StructField(remap_columns(expr.input, mapping), expr.field)
    if isinstance(expr, MapFunc):
        return MapFunc(expr.fn, remap_columns(expr.input, mapping), expr.key)
    if isinstance(expr, ListJoin):
        return ListJoin(remap_columns(expr.input, mapping), expr.separator)
    if isinstance(expr, ListBinary):
        return ListBinary(
            expr.fn, remap_columns(expr.left, mapping), remap_columns(expr.right, mapping)
        )
    if isinstance(expr, ListSet):
        return ListSet(
            expr.fn, remap_columns(expr.left, mapping), remap_columns(expr.right, mapping)
        )
    if isinstance(expr, Array):
        return Array([remap_columns(e, mapping) for e in expr.elements])
    if isinstance(expr, MakeStruct):
        return MakeStruct([(n, remap_columns(v, mapping)) for n, v in expr.fields])
    if isinstance(expr, Sequence):
        return Sequence(
            remap_columns(expr.start, mapping),
            remap_columns(expr.stop, mapping),
            remap_columns(expr.step, mapping),
        )
    if isinstance(expr, Coalesce):
        return Coalesce([remap_columns(e, mapping) for e in expr.inputs])
    if isinstance(expr, Greatest):
        return Greatest([remap_columns(e, mapping) for e in expr.inputs])
    if isinstance(expr, Least):
        return Least([remap_columns(e, mapping) for e in expr.inputs])
    if isinstance(expr, NullIf):
        return NullIf(remap_columns(expr.left, mapping), remap_columns(expr.right, mapping))
    if isinstance(expr, Math2Expr):
        return Math2Expr(
            expr.fn, remap_columns(expr.left, mapping), remap_columns(expr.right, mapping)
        )
    if isinstance(expr, Case):
        return Case(
            [(remap_columns(c, mapping), remap_columns(t, mapping)) for c, t in expr.branches],
            remap_columns(expr.otherwise, mapping),
        )
    return expr  # literals unchanged
