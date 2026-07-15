"""Per-expression output-type inference — a column's Arrow type before the engine runs.

`infer_type(expr, schema)` computes the Arrow `DataType` an expression produces
given its input schema, mirroring the engine's actual behavior (post FFI
widening). It is **sound, not complete**: any node whose output type is not
certain returns ``None`` so the caller falls back to the proven zero-row execution
rather than ever reporting a wrong type. This is what lets `available_schema()`
answer `Dataset.schema` without scanning, and lets the plan validate types early.

Neutral layer. The expression node classes are imported lazily inside the function
because `plan.expr_ir` imports this package (`CAST_DTYPES`) — a top-level import
here would be a cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from batcher.plan.types.lattice import promote, widen
from batcher.plan.types.registry import DTYPE_REGISTRY

if TYPE_CHECKING:
    from batcher.plan.expr_ir import Expr
    from batcher.plan.schema import SchemaRef

__all__ = ["infer_type"]

# Binary operator → output-type category. Comparisons and logical ops yield bool;
# bit/shift ops yield int64; arithmetic promotes its operands (then widens to the
# engine's int64/float64 output). `div` is intentionally absent: true division's
# result type is not certain here, so it falls through to ``None`` (fallback).
_BINARY_BOOL = frozenset({"gt", "ge", "lt", "le", "eq", "ne", "and", "or"})
_BINARY_INT = frozenset({"bit_and", "bit_or", "bit_xor", "shift_left", "shift_right"})
_BINARY_ARITH = frozenset({"add", "sub", "mul", "mod"})

# `str` accessor functions whose output type is certain.
_STR_BOOL = frozenset(
    {"contains", "starts_with", "ends_with", "like", "ilike", "regexp_matches", "json_extract_bool"}
)
_STR_INT = frozenset(
    {
        "len",
        "position",
        "regexp_count",
        "levenshtein",
        "ascii",
        "bit_length",
        "octet_length",
        "crc32",
        "hash64",
        "xxhash64",
        "json_extract_int",
    }
)
_STR_FLOAT = frozenset({"json_extract_float"})

# `dt` accessor (`DateFunc`) output types. Every field-extraction fn yields Int64;
# these four are the exceptions. `last_day` yields a timestamp regardless of whether
# the input is a date or a timestamp (verified against the engine).
_DATE_STR = frozenset({"dayname", "monthname"})
_DATE_BOOL = frozenset({"is_leap_year"})
_DATE_TS = frozenset({"last_day"})
# `list` accessor (`ListFunc`) output types that do NOT depend on the element's
# numeric width. `len`/`n_unique`/`arg_max`/`arg_min` count or index → Int64;
# `reverse`/`sort`/`unique` return a list of the same element type. The numeric
# reductions (`sum`/`min`/`max`/`mean`/`std`/…) are intentionally omitted — their
# result width is decided in the engine, so they fall back to ``None``.
_LIST_INT = frozenset({"len", "n_unique", "arg_max", "arg_min"})
_LIST_SAME = frozenset({"reverse", "sort", "unique"})

_STR_STR = frozenset(
    {
        "strip_html",
        "upper",
        "lower",
        "trim",
        "l_trim",
        "r_trim",
        "lpad",
        "rpad",
        "substr",
        "repeat",
        "replace",
        "regexp_replace",
        "regexp_replace_all",
        "regexp_extract",
        "initcap",
        "hex",
        "base64",
        "from_base64",
        "soundex",
        "md5",
        "sha1",
        "sha256",
        "hmac_sha256",
        "aes_encrypt",
        "aes_decrypt",
        "mask",
        "right",
        "substring_index",
        "overlay",
        "split_part",
        "json_extract_string",
        "reverse",
        "translate",
        "unhex",
    }
)


def infer_type(expr: Expr, schema: SchemaRef) -> pa.DataType | None:
    """The Arrow type `expr` produces over `schema`, or ``None`` if not certain.

    ``None`` is always a sound answer — it means "fall back to executing a zero-row
    query for this column" — so a new or opaque expression never yields a wrong
    type. The schema passed in is the operator's *input* schema (already widened at
    the scan leaf), so a bare ``Col`` reports the engine's post-widening type.
    """
    from batcher.plan.expr_ir.core import (
        Aliased,
        Binary,
        Cast,
        Coalesce,
        IsInf,
        IsNan,
        IsNotNull,
        IsNull,
        Lit,
        Math2Expr,
        MathExpr,
        Not,
    )
    from batcher.plan.expr_ir.namespaces import (
        ConvertTimezone,
        DateFunc,
        DateOffset,
        DateTrunc,
        ListBinary,
        ListContains,
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
    from batcher.plan.expr_ir.nodes import Case, Col, Greatest, HashRows, Least, NullIf

    if isinstance(expr, Col):
        return schema.field(expr.name).type if schema.has(expr.name) else None
    if isinstance(expr, Lit):
        return _lit_type(expr.value)
    if isinstance(expr, Aliased):
        return infer_type(expr.inner, schema)
    if isinstance(expr, Cast):
        return DTYPE_REGISTRY.get(expr.dtype)
    if isinstance(expr, (Not, IsNull, IsNotNull, IsNan, IsInf)):
        return pa.bool_()
    if isinstance(expr, Binary):
        return _binary_type(expr, schema)
    if isinstance(expr, MathExpr):
        # `abs` preserves its (numeric) input type; every other unary math fn
        # (sqrt/ln/exp/floor/ceil/round/trunc/sign/trig/…) yields float64.
        return infer_type(expr.input, schema) if expr.fn == "abs" else pa.float64()
    if isinstance(expr, Math2Expr):
        return pa.float64()
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
    if isinstance(expr, ListSimhash):
        return pa.list_(pa.int64())  # one Int64 bit per hyperplane
    if isinstance(expr, StrFunc):
        return _strfunc_type(expr.fn)
    if isinstance(expr, DateFunc):
        return _datefunc_type(expr.fn)
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
        return _as_list_type(infer_type(_list_operand(expr), schema))
    if isinstance(expr, ListGet):
        return _list_element_type(infer_type(expr.input, schema))
    if isinstance(expr, ListFunc):
        return _listfunc_type(expr.fn, infer_type(expr.input, schema))
    if isinstance(expr, StructField):
        return _struct_field_type(infer_type(expr.input, schema), expr.field)
    if isinstance(expr, MapFunc):
        return _mapfunc_type(expr.fn, infer_type(expr.input, schema))
    return None


def _list_operand(expr: object) -> Expr:
    """The list-typed operand of a slice (`input`) or set-op (`left`) node."""
    inp = getattr(expr, "input", None)
    return inp if inp is not None else expr.left  # type: ignore[attr-defined]


def _as_list_type(t: pa.DataType | None) -> pa.DataType | None:
    """Return `t` only if it is a List type (an unchanged list output)."""
    return t if t is not None and pa.types.is_list(t) else None


def _list_element_type(t: pa.DataType | None) -> pa.DataType | None:
    """The element type of a List type, or ``None`` if `t` is not a list."""
    return t.value_type if t is not None and pa.types.is_list(t) else None


def _listfunc_type(fn: str, input_t: pa.DataType | None) -> pa.DataType | None:
    if fn in _LIST_INT:
        return pa.int64()
    if fn in _LIST_SAME:
        return _as_list_type(input_t)
    return None  # numeric reductions: engine decides the width → fall back


def _struct_field_type(struct_t: pa.DataType | None, field: str) -> pa.DataType | None:
    if struct_t is None or not pa.types.is_struct(struct_t):
        return None
    idx = struct_t.get_field_index(field)
    return struct_t.field(idx).type if idx >= 0 else None


def _mapfunc_type(fn: str, map_t: pa.DataType | None) -> pa.DataType | None:
    if map_t is None or not pa.types.is_map(map_t):
        return None
    if fn == "map_keys":
        return pa.list_(map_t.key_type)
    if fn == "map_values":
        return pa.list_(map_t.item_type)
    if fn == "element_at":
        return map_t.item_type
    return None


def _lit_type(value: object) -> pa.DataType:
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


def _binary_type(expr: object, schema: SchemaRef) -> pa.DataType | None:
    op = expr.op  # type: ignore[attr-defined]
    if op in _BINARY_BOOL:
        return pa.bool_()
    if op in _BINARY_INT:
        return pa.int64()
    if op in _BINARY_ARITH:
        left = infer_type(expr.left, schema)  # type: ignore[attr-defined]
        right = infer_type(expr.right, schema)  # type: ignore[attr-defined]
        if left is None or right is None:
            return None
        dec = _decimal_arith_type(op, left, right)
        if dec is not None:
            return dec
        common = promote(left, right)
        return widen(common) if common is not None else None
    if op == "div":
        # True division yields Float64 for int/float operands (the engine always
        # produces a double). It is only uncertain when a decimal operand is
        # involved (the result stays decimal), so fall back to ``None`` there.
        left = infer_type(expr.left, schema)  # type: ignore[attr-defined]
        right = infer_type(expr.right, schema)  # type: ignore[attr-defined]
        if left is None or right is None:
            return None
        numeric = pa.types.is_integer, pa.types.is_floating
        if all(any(p(t) for p in numeric) for t in (left, right)):
            return pa.float64()
        return None
    return None


# Arrow / DataFusion decimal128 arithmetic precision+scale rules (verified against the
# engine). `div` is intentionally excluded — its scale rule is not reproduced here, so it
# stays uncertain (→ ``None``) as the engine may return a decimal of an inferred scale.
_DECIMAL_MAX_PRECISION = 38


def _decimal_arith_type(op: str, left: pa.DataType, right: pa.DataType) -> pa.DataType | None:
    """The decimal128 result type of `add`/`sub`/`mul` over two decimal128 operands.

    Returns ``None`` (fall back) unless *both* operands are decimal128 and `op` is one of
    the three whose result precision/scale the engine derives deterministically. A decimal
    mixed with an int/float, ``mod``, or ``div`` is left uncertain on purpose.
    """
    if op not in ("add", "sub", "mul"):
        return None
    if not (pa.types.is_decimal128(left) and pa.types.is_decimal128(right)):
        return None
    p1, s1, p2, s2 = left.precision, left.scale, right.precision, right.scale
    if op == "mul":
        scale = s1 + s2
        precision = p1 + p2 + 1
    else:  # add / sub
        scale = max(s1, s2)
        precision = max(p1 - s1, p2 - s2) + scale + 1
    precision = min(precision, _DECIMAL_MAX_PRECISION)
    if scale > precision:  # cannot be represented → stay uncertain, don't guess
        return None
    return pa.decimal128(precision, scale)


def _strfunc_type(fn: str) -> pa.DataType | None:
    if fn == "minhash":
        return pa.list_(pa.int64())  # the signature: one value per permutation
    if fn == "chunk":
        return pa.list_(pa.string())
    if fn == "split":
        return pa.list_(pa.string())
    if fn == "regexp_extract_all":
        return pa.list_(pa.string())  # every match of the pattern
    if fn in _STR_BOOL:
        return pa.bool_()
    if fn in _STR_INT:
        return pa.int64()
    if fn in _STR_FLOAT:
        return pa.float64()
    if fn in _STR_STR:
        return pa.string()
    return None


def _datefunc_type(fn: str) -> pa.DataType | None:
    """The Arrow type a `dt` accessor function produces, or ``None`` if not certain."""
    if fn in _DATE_STR:
        return pa.string()
    if fn in _DATE_BOOL:
        return pa.bool_()
    if fn in _DATE_TS:
        return pa.timestamp("us")
    from batcher.plan.expr_ir.fn_names import DATE_FNS

    # Every remaining date field-extraction fn (year/month/day/hour/epoch/…) is Int64.
    if fn in DATE_FNS:
        return pa.int64()
    return None


def _fold_promote(types) -> pa.DataType | None:
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
