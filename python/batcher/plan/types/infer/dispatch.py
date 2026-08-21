"""The node-by-node dispatcher: which rule answers for which `Expr` class.

This module owns the recursion. Every family that needs an operand's type receives
`infer_type` itself (`arithmetic`), and every family that can answer from a resolved type
or a function name is called with that (`collections`, `scalars`, `media`, `sequence`), so
the dependency between the package's modules points one way and only this file is
self-referential.

The expression node classes are imported lazily inside the function because
`plan.expr_ir` imports this package (`CAST_DTYPES`) -- a top-level import would be a cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from batcher.plan.types.infer.arithmetic import binary_type, math2func_type, mathfunc_type
from batcher.plan.types.infer.collections import (
    as_list_type,
    list_element_type,
    list_operand,
    listfunc_type,
    mapfunc_type,
    struct_field_type,
)
from batcher.plan.types.infer.scalars import datefunc_type, make_temporal_type, strfunc_type
from batcher.plan.types.lattice import promote
from batcher.plan.types.media import audiofunc_type, imagefunc_type, videofunc_type
from batcher.plan.types.registry import resolve_dtype
from batcher.plan.types.sequence import seqfunc_type

if TYPE_CHECKING:
    from collections.abc import Iterable

    from batcher.plan.expr_ir import Expr
    from batcher.plan.schema import SchemaRef

__all__ = ["infer_type"]


def infer_type(expr: Expr, schema: SchemaRef) -> pa.DataType | None:
    """The Arrow type `expr` produces over `schema`, or ``None`` if not certain.

    ``None`` is always a sound answer — it means "fall back to executing a zero-row
    query for this column" — so a new or opaque expression never yields a wrong
    type. The schema passed in is the operator's *input* schema (already widened at
    the scan leaf), so a bare ``Col`` reports the engine's post-widening type.
    """
    from batcher.plan.expr_ir.audio import AudioFunc
    from batcher.plan.expr_ir.core import (
        Aliased,
        Binary,
        Cast,
        Coalesce,
        InList,
        IsInf,
        IsNan,
        IsNotNull,
        IsNull,
        Lit,
        Math2Expr,
        MathExpr,
        Not,
    )
    from batcher.plan.expr_ir.func_nodes import ListTransform, MakeTemporal
    from batcher.plan.expr_ir.image import ImageCrop, ImageFunc
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
        ListSimhash,
        ListSlice,
        MapFunc,
        Strftime,
        StrFunc,
        Strptime,
        StructField,
    )
    from batcher.plan.expr_ir.namespaces.sequence import SeqFunc
    from batcher.plan.expr_ir.nodes import (
        Case,
        Col,
        Greatest,
        HashRows,
        Least,
        MakeStruct,
        NullIf,
        Sequence,
    )
    from batcher.plan.expr_ir.video import VideoFunc

    if isinstance(expr, Col):
        return schema.field(expr.name).type if schema.has(expr.name) else None
    if isinstance(expr, Lit):
        return _lit_type(expr.value)
    if isinstance(expr, Aliased):
        return infer_type(expr.inner, schema)
    if isinstance(expr, Cast):
        return resolve_dtype(expr.dtype)
    if isinstance(expr, (Not, IsNull, IsNotNull, IsNan, IsInf)):
        return pa.bool_()
    if isinstance(expr, InList):
        # `x IN (…)` is Boolean whatever the members' type — three-valued in its *values*
        # (a null input yields null), which is a nullability question rather than a typing
        # one. Needed because `Expr.is_in` builds this node directly rather than the `OR`
        # chain the arms below would have covered: without it a projection carrying an `IN`
        # had no inferable schema at all, and an uninferable projection does not merely lose
        # this column — the zero-row fallback reports *every* column in it as `null`.
        return pa.bool_()
    if isinstance(expr, Binary):
        return binary_type(expr, schema, infer_type)
    if isinstance(expr, MathExpr):
        return mathfunc_type(expr, schema, infer_type)
    if isinstance(expr, Math2Expr):
        return math2func_type(expr, schema, infer_type)
    if isinstance(expr, HashRows):
        return pa.int64()  # a 64-bit digest, whatever the inputs' types
    if isinstance(expr, (Coalesce, Greatest, Least)):
        return _fold_promote(infer_type(e, schema) for e in expr.inputs)
    if isinstance(expr, NullIf):
        # `nullif(a, b)` is `a` with the matching rows nulled — the output type is
        # the left operand's type (verified: unaffected by the right operand).
        return infer_type(expr.left, schema)
    if isinstance(expr, Case):
        branch_thens = (infer_type(then, schema) for _cond, then in expr.branches)
        return _fold_promote([*branch_thens, infer_type(expr.otherwise, schema)])
    if isinstance(expr, ImageFunc):
        return imagefunc_type(expr)
    if isinstance(expr, ImageCrop):
        # A per-row window means rows genuinely differ in size, so the result is an
        # encoded still rather than a fixed-shape tensor.
        return pa.binary()
    if isinstance(expr, VideoFunc):
        return videofunc_type(expr)
    if isinstance(expr, AudioFunc):
        return audiofunc_type(expr)
    if isinstance(expr, SeqFunc):
        return seqfunc_type(expr.fn)
    if isinstance(expr, ListSimhash):
        return pa.list_(pa.int64())  # one Int64 bit per hyperplane
    if isinstance(expr, StrFunc):
        return strfunc_type(expr.fn)
    if isinstance(expr, DateFunc):
        return datefunc_type(expr.fn)
    if isinstance(expr, Strptime):
        return pa.timestamp("us")  # parses a string into a microsecond timestamp
    if isinstance(expr, Strftime):
        return pa.string()  # formats a Date/Timestamp into text
    if isinstance(expr, DateTrunc):
        # `date_trunc` returns a microsecond Timestamp for both date and timestamp
        # inputs (verified against the engine).
        return pa.timestamp("us")
    if isinstance(expr, (DateOffset, ConvertTimezone)):
        return infer_type(expr.input, schema)  # type-preserving (shift/tz-convert)
    if isinstance(expr, ListContains):
        return pa.bool_()
    if isinstance(expr, ListPosition):
        return pa.int64()  # 1-based index of the first match, 0 if absent
    if isinstance(expr, ListBinary):
        return pa.float64()  # pairwise reduction over two list columns
    if isinstance(expr, (ListSlice, ListSet)):
        # Sub-range / set-op of a list: the element type is unchanged.
        return as_list_type(infer_type(list_operand(expr), schema))
    if isinstance(expr, ListFilter):
        # A filter selects elements, it does not change them: the list type is the input's.
        return as_list_type(infer_type(expr.input, schema))
    if isinstance(expr, ListTransform):
        return _list_transform_type(expr, schema)
    if isinstance(expr, ListGet):
        return list_element_type(infer_type(expr.input, schema))
    if isinstance(expr, ListFunc):
        return listfunc_type(expr.fn, infer_type(expr.input, schema))
    if isinstance(expr, StructField):
        return struct_field_type(infer_type(expr.input, schema), expr.field)
    if isinstance(expr, MapFunc):
        return mapfunc_type(expr.fn, infer_type(expr.input, schema))
    if isinstance(expr, Sequence):
        return pa.list_(pa.int64())  # `sequence` always yields a List<Int64> series
    if isinstance(expr, MakeTemporal):
        return make_temporal_type(expr.fn)
    if isinstance(expr, MakeStruct):
        return _make_struct_type(expr.fields, schema)
    return None


def _make_struct_type(fields: list[tuple[str, Expr]], schema: SchemaRef) -> pa.DataType | None:
    """Struct type of a `MakeStruct`: one field per named sub-expression.

    Mirrors `eval_make_struct` (each field nullable). Uncertain in any field →
    ``None`` (the sound fallback), so a partially-known struct never mislabels a
    subfield's type.
    """
    arrow_fields: list[pa.Field] = []
    for name, value in fields:
        field_t = infer_type(value, schema)
        if field_t is None:
            return None
        arrow_fields.append(pa.field(name, field_t, nullable=True))
    return pa.struct(arrow_fields)


def _lit_type(value: object) -> pa.DataType:
    """The Arrow type of a Python literal, mirroring the engine's literal binding."""
    import datetime as _dt

    # bool before int (bool subclasses int); datetime before date.
    if isinstance(value, bool):
        return pa.bool_()
    if isinstance(value, int):
        return pa.int64()
    if isinstance(value, float):
        return pa.float64()
    if isinstance(value, str):
        return pa.string()
    if isinstance(value, _dt.datetime):
        return pa.timestamp("us")
    if isinstance(value, _dt.date):
        return pa.date32()
    return pa.null()


def _fold_promote(types: Iterable[pa.DataType | None]) -> pa.DataType | None:
    """Fold the lossless `promote` lattice over an iterable of (possibly None) types."""
    result: pa.DataType | None = None
    for t in types:
        if t is None:
            return None
        if result is None:
            result = t
        else:
            result = promote(result, t)
            if result is None:
                return None
    return result


def _list_transform_type(expr: Expr, schema: SchemaRef) -> pa.DataType | None:
    """`list.transform(body)` — a list of whatever `body` makes of one element.

    The body is written against `element()`, which is a `Col` under a reserved name with
    no binding in the operator's own schema, so inferring it needs that name bound to the
    input's *element* type first. Everything else follows from the ordinary recursion, and
    a body the recursion cannot type still yields `None` rather than a guess.
    """
    from batcher.plan.functions.collection import _ELEMENT_COL
    from batcher.plan.schema import SchemaRef as _SchemaRef

    element_t = list_element_type(infer_type(expr.input, schema))
    if element_t is None:
        return None
    bound = _SchemaRef.from_arrow(
        pa.schema([*schema.arrow, pa.field(_ELEMENT_COL, element_t, nullable=True)])
        if not schema.has(_ELEMENT_COL)
        else schema.arrow
    )
    body_t = infer_type(expr.func, bound)
    return pa.list_(body_t) if body_t is not None else None
