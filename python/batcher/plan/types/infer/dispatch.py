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

from batcher._internal.errors import PlanError
from batcher.plan.types.infer.arithmetic import binary_type, math2func_type, mathfunc_type
from batcher.plan.types.infer.collections import (
    as_list_type,
    list_element_type,
    list_operand,
    listfunc_type,
    mapfunc_type,
    struct_field_type,
    widened_element_out,
)
from batcher.plan.types.infer.geospatial import geofunc_type, spatialfunc_type
from batcher.plan.types.infer.scalars import datefunc_type, make_temporal_type, strfunc_type
from batcher.plan.types.lattice import promote
from batcher.plan.types.media import audiofunc_type, imagefunc_type, videofunc_type
from batcher.plan.types.registry import dtype_from_wire, resolve_dtype
from batcher.plan.types.sequence import seqfunc_type

if TYPE_CHECKING:
    from collections.abc import Iterable

    from batcher.plan.expr_ir import Expr
    from batcher.plan.schema import SchemaRef

__all__ = ["infer_type"]


# `infer_type` recurses over every expression the planner builds a schema for — 165,000 calls
# in the first planning of TPC-DS q64 — and it needs the node classes, which it cannot import
# at module level (see the module docstring). Importing them *inside* the function ran nine
# `from ... import` statements, some sixty names, on every call: 1.8 s of that planning under
# the profiler, more than the type inference itself. The laziness is kept and paid once: the
# first call binds the classes into this module's globals.
_NODES_BOUND = False


def _bind_nodes() -> None:
    """Import the expression node classes `infer_type` dispatches on into module globals."""
    global _NODES_BOUND
    global Aliased, Array, AudioFunc, Binary, Case, Cast, Coalesce, Col, ConvertTimezone
    global DateFunc, DateOffset, DateTrunc, GeoFunc, Greatest, HashRows, ImageCrop, ImageFunc
    global InList, IsInf, IsNan, IsNotNull, IsNull, Least, ListBinary, ListContains, ListFilter
    global ListFunc, ListGet, ListGetDyn, ListJoin, ListPosition, ListSet, ListSimhash
    global \
        ListSlice, \
        ListTransform, \
        ListZip, \
        ListZipStruct, \
        StructUpdate, \
        JsonDoc, \
        Lit, \
        MakeMap, \
        MakeStruct, \
        MakeTemporal, \
        MapFunc
    global Math2Expr, MathExpr, Not, NullIf, SeqFunc, Sequence, SpatialFunc, StrFunc
    global StrFuncDyn, Strftime, Strptime, StructField, VideoFunc, WindowBuckets, WindowStart
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
    from batcher.plan.expr_ir.func_nodes import (
        GeoFunc,
        JsonDoc,
        ListTransform,
        ListZipStruct,
        MakeTemporal,
        SpatialFunc,
        StructUpdate,
        WindowBuckets,
        WindowStart,
    )
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
        ListGetDyn,
        ListPosition,
        ListSet,
        ListSimhash,
        ListSlice,
        ListZip,
        MapFunc,
        Strftime,
        StrFunc,
        StrFuncDyn,
        Strptime,
        StructField,
    )
    from batcher.plan.expr_ir.namespaces.sequence import SeqFunc
    from batcher.plan.expr_ir.nodes import (
        Array,
        Case,
        Col,
        Greatest,
        HashRows,
        Least,
        ListJoin,
        MakeMap,
        MakeStruct,
        NullIf,
        Sequence,
    )
    from batcher.plan.expr_ir.video import VideoFunc

    _NODES_BOUND = True


def infer_type(expr: Expr, schema: SchemaRef) -> pa.DataType | None:
    """The Arrow type `expr` produces over `schema`, or ``None`` if not certain.

    ``None`` is always a sound answer — it means "fall back to executing a zero-row
    query for this column" — so a new or opaque expression never yields a wrong
    type. The schema passed in is the operator's *input* schema (already widened at
    the scan leaf), so a bare ``Col`` reports the engine's post-widening type.
    """
    if not _NODES_BOUND:
        _bind_nodes()

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
    if isinstance(expr, Array):
        # `array(a, b, ...)` gathers its arguments into one list per row, so the element
        # type is what they promote to -- the same fold `Coalesce`/`Greatest`/`Least` use
        # one line up, and reused rather than restated so the two cannot disagree about
        # what `array(int64, utf8)` is. Verified against the engine: int+float -> double,
        # int+string -> string, int+null -> int64.
        element = _fold_promote(infer_type(e, schema) for e in expr.elements)
        return None if element is None else pa.list_(element)
    if isinstance(expr, ListJoin):
        # `list.join(sep)` renders a whole list as one delimited string, whatever the
        # element type -- a `List<Int64>` joins to `"1,2"` exactly as a `List<Utf8>` joins
        # to `"a,b"`. The node had no arm at all, so every projection containing one
        # reported `null` for all of its columns.
        return pa.string()
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
    if isinstance(expr, StrFuncDyn):
        # The dynamic spelling (a per-row `pattern`/`start`/`length`, which the SQL parser
        # builds for `repeat(s, n)` and friends) computes the same function as `StrFunc`,
        # so it returns the same type. Only the arguments differ in when they are known.
        return strfunc_type(expr.fn)
    if isinstance(expr, DateFunc):
        return datefunc_type(expr.fn)
    if isinstance(expr, Strptime):
        return pa.timestamp("us")  # parses a string into a microsecond timestamp
    if isinstance(expr, Strftime):
        return pa.string()  # formats a Date/Timestamp into text
    if isinstance(expr, DateTrunc):
        # `date_trunc` returns a microsecond Timestamp for both date and timestamp
        # inputs (verified against the engine), unless `preserve_type` keeps a date a date.
        if expr.preserve_type:
            source = infer_type(expr.input, schema)
            if source is None:
                return None  # the engine answers the input's type, which is unknown here
            if pa.types.is_date32(source):
                return pa.date32()
        return pa.timestamp("us")
    if isinstance(expr, (DateOffset, ConvertTimezone)):
        return infer_type(expr.input, schema)  # type-preserving (shift/tz-convert)
    if isinstance(expr, ListContains):
        return pa.bool_()
    if isinstance(expr, ListPosition):
        return pa.int64()  # 1-based index of the first match, 0 if absent
    if isinstance(expr, GeoFunc):
        return geofunc_type(expr.fn)
    if isinstance(expr, SpatialFunc):
        return spatialfunc_type(expr.fn)
    if isinstance(expr, ListBinary):
        if expr.fn in _EQUAL_DIM_FNS:
            _check_fixed_dims(expr, schema)
        return pa.float64()  # pairwise reduction over two list columns
    if isinstance(expr, ListZip):
        # Element-wise `list_add`/`list_subtract`/`list_multiply` -- the embedding-math
        # primitive. The engine computes these in floating point whatever the operands'
        # element widths, exactly as `cum_sum`/`diff`/`softmax` do, so two Int64 lists add
        # to `List<Double>`. The node's own docstring has said so since it was written; the
        # rule was simply never implemented, and a projection carrying one declared `null`
        # for every column in it.
        _check_fixed_dims(expr, schema)
        left = list_element_type(infer_type(expr.left, schema))
        return pa.list_(pa.float64()) if left is not None else None
    if isinstance(expr, (ListSlice, ListSet)):
        # Sub-range / set-op of a list: the element type is unchanged.
        return as_list_type(infer_type(list_operand(expr), schema))
    if isinstance(expr, ListFilter):
        # A filter selects elements, it does not change them: the list type is the input's.
        return as_list_type(infer_type(expr.input, schema))
    if isinstance(expr, ListTransform):
        return _list_transform_type(expr, schema)
    if isinstance(expr, ListGet):
        # An element that becomes a column widens if it is a narrow integer — the boundary
        # stopped doing it, so the op that makes the column does. See `widened_element_out`.
        return widened_element_out(list_element_type(infer_type(expr.input, schema)))
    if isinstance(expr, ListGetDyn):
        # Same element type as `ListGet`; the index is per-row rather than constant, which
        # changes nothing about the type. SQL's `lst[i]` builds this one.
        return widened_element_out(list_element_type(infer_type(expr.input, schema)))
    if isinstance(expr, ListFunc):
        return listfunc_type(expr.fn, infer_type(expr.input, schema))
    if isinstance(expr, StructField):
        return struct_field_type(infer_type(expr.input, schema), expr.field)
    if isinstance(expr, MapFunc):
        return mapfunc_type(expr.fn, infer_type(expr.input, schema), expr.key)
    if isinstance(expr, Sequence):
        _check_sequence_operands(expr, schema)
        return pa.list_(pa.int64())  # `sequence` always yields a List<Int64> series
    if isinstance(expr, MakeTemporal):
        return make_temporal_type(expr.fn)
    if isinstance(expr, MakeStruct):
        return _make_struct_type(expr.fields, schema)
    if isinstance(expr, StructUpdate):
        return _struct_update_type(expr, schema)
    if isinstance(expr, ListZipStruct):
        left = list_element_type(infer_type(expr.left, schema))
        right = list_element_type(infer_type(expr.right, schema))
        if left is None or right is None:
            return None
        return pa.list_(pa.struct([pa.field("left", left), pa.field("right", right)]))
    if isinstance(expr, JsonDoc):
        return dtype_from_wire(expr.dtype) if expr.dtype is not None else pa.string()
    if isinstance(expr, MakeMap):
        return _make_map_type(expr.keys, expr.values, schema)
    if isinstance(expr, WindowStart):
        # A timestamp, whatever the input was -- NOT the input's own type. That reading is
        # the obvious one and it is false, which `_sql`'s bucket lowering proves: for a DATE
        # argument it builds `Cast(WindowStart(value, width), "date")`, and that cast is only
        # there because the window yields a timestamp. Declaring `date32` here made the cast
        # look redundant, it was eliminated, and `time_bucket(INTERVAL 1 DAY, DATE ...)` came
        # back `timestamp[us]` -- a wrong declared type turning into a wrong *result*.
        return pa.timestamp("us")
    if isinstance(expr, WindowBuckets):
        # The hopping form puts each instant in every overlapping bucket, so it is a list of
        # the same window starts.
        return pa.list_(pa.timestamp("us"))
    return None


#: The `ListBinary` reductions the engine refuses on two lists of different lengths (the
#: vector distances); `eval_list_binary`'s `dims_must_match` in `bc-expr` is the other half.
_EQUAL_DIM_FNS = frozenset({"dot", "cosine_similarity", "l2_distance", "l1_distance", "hamming"})


def _check_fixed_dims(expr: Expr, schema: SchemaRef) -> None:
    """Refuse two fixed-size lists whose declared sizes differ, before any row is read.

    Both sizes are in the schema, so the engine's per-row "list dimensions must be equal"
    is certain to fire on the first non-null pair; reporting it at plan build names the
    two columns instead of failing mid-query. Variable-length lists keep the runtime check.
    """
    left, right = infer_type(expr.left, schema), infer_type(expr.right, schema)
    if (
        left is not None
        and right is not None
        and pa.types.is_fixed_size_list(left)
        and pa.types.is_fixed_size_list(right)
        and left.list_size != right.list_size
    ):
        raise PlanError(
            f"list.{expr.fn} needs vectors of one dimension, but {expr.left!r} has "
            f"{left.list_size} elements and {expr.right!r} has {right.list_size}"
        )


def _struct_update_type(expr: Expr, schema: SchemaRef) -> pa.DataType | None:
    """The struct `StructUpdate` produces, refusing an edit that names a missing field.

    Mirrors `eval_struct_update`: drop, then rename, then set (replace in place or
    append). Naming a field the struct does not have is a `PlanError` here, at plan
    build, rather than the engine's error on the first batch.
    """
    source = infer_type(expr.input, schema)
    if source is None:
        return None
    if not pa.types.is_struct(source):
        raise PlanError(f"struct field editing needs a struct column, got {source}")
    fields = list(source)

    def find(name: str) -> int:
        names = [f.name for f in fields]
        if name not in names:
            raise PlanError(f"struct has no field {name!r}; its fields are: {', '.join(names)}")
        return names.index(name)

    for name in expr.drop:
        fields.pop(find(name))
    for old, new in expr.rename:
        at = find(old)
        fields[at] = fields[at].with_name(new)
    for name, value in zip(expr.names, expr.values, strict=True):
        value_t = infer_type(value, schema)
        if value_t is None:
            return None
        field = pa.field(name, value_t)
        at = next((i for i, f in enumerate(fields) if f.name == name), None)
        if at is None:
            fields.append(field)
        else:
            fields[at] = field
    if not fields:
        raise PlanError("a struct must keep at least one field")
    return pa.struct(fields)


def _check_sequence_operands(expr: Expr, schema: SchemaRef) -> None:
    """Refuse a `sequence` operand the engine would cast to Int64 into a wrong answer.

    The kernel casts every operand to Int64. A timestamp bound therefore became its epoch
    microseconds and a date its epoch days, so a "date range" came back as a list of
    integers in an unstated unit, and a text step such as ``'1 day'`` cast to null and
    nulled every row. Each was a result rather than an error.
    """
    for role, operand in (("start", expr.start), ("stop", expr.stop), ("step", expr.step)):
        t = infer_type(operand, schema)
        if t is None or pa.types.is_integer(t) or pa.types.is_null(t):
            continue
        if pa.types.is_floating(t) or pa.types.is_boolean(t) or pa.types.is_decimal(t):
            continue
        raise PlanError(
            f"sequence() builds an integer series, but its {role} is {t}. A date or "
            "timestamp range with an interval step is not supported; to step over days, "
            "build an integer series of offsets and convert it yourself"
        )


def _make_map_type(keys: Expr, values: Expr, schema: SchemaRef) -> pa.DataType | None:
    """Map type of a `MakeMap`: the *element* types of its key and value list arguments.

    `map_from_arrays` pairs a list of keys with a list of values, so the map's key and
    value types are those lists' element types, not the list types themselves. Without an
    arm here the cascade fell through to `None`, and `Dataset.schema` reported `null` for a
    column that comes back `map<string, int64>` -- the silent wrong answer a per-type
    cascade gives for any node nobody added an arm for, which is why the column walk in
    `expr_ir/walk.py` is declarative instead.

    Uncertain on either side → `None`, the sound fallback `_make_struct_type` also takes.
    """
    key_t = infer_type(keys, schema)
    value_t = infer_type(values, schema)
    if key_t is None or value_t is None:
        return None
    if not pa.types.is_list(key_t) or not pa.types.is_list(value_t):
        return None
    return pa.map_(key_t.value_type, value_t.value_type)


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

    The body is typed in the scope the engine evaluates it in: `element()` bound to the
    input's *element* type, `element_index()` to Int64, and each capture to the type its
    expression has over the enclosing schema. Binding `element` from the operator's own
    schema when it happened to hold a column of that name typed the body against the wrong
    column. A body the recursion cannot type still yields `None` rather than a guess.
    """
    from batcher.plan.expr_ir.func_nodes import ELEMENT_COL, ELEMENT_INDEX_COL
    from batcher.plan.schema import SchemaRef as _SchemaRef

    element_t = list_element_type(infer_type(expr.input, schema))
    if element_t is None:
        return None
    fields = [pa.field(ELEMENT_COL, element_t), pa.field(ELEMENT_INDEX_COL, pa.int64())]
    for name, capture in zip(expr.capture_names, expr.captures, strict=True):
        capture_t = infer_type(capture, schema)
        if capture_t is None:
            return None
        if name not in (ELEMENT_COL, ELEMENT_INDEX_COL):
            fields.append(pa.field(name, capture_t))
    body_t = infer_type(expr.func, _SchemaRef.from_arrow(pa.schema(fields)))
    return pa.list_(body_t) if body_t is not None else None
