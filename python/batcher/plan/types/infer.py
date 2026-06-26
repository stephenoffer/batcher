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
_STR_STR = frozenset(
    {
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
        "right",
        "substring_index",
        "overlay",
        "split_part",
        "json_extract_string",
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
    from batcher.plan.expr_ir.namespaces import StrFunc
    from batcher.plan.expr_ir.nodes import Case, Col, Greatest, Least

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
    if isinstance(expr, (Coalesce, Greatest, Least)):
        return _fold_promote(infer_type(e, schema) for e in expr.inputs)
    if isinstance(expr, Case):
        branch_thens = (infer_type(then, schema) for _cond, then in expr.branches)
        return _fold_promote([*branch_thens, infer_type(expr.otherwise, schema)])
    if isinstance(expr, StrFunc):
        return _strfunc_type(expr.fn)
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
        common = promote(left, right)
        return widen(common) if common is not None else None
    return None


def _strfunc_type(fn: str) -> pa.DataType | None:
    if fn in _STR_BOOL:
        return pa.bool_()
    if fn in _STR_INT:
        return pa.int64()
    if fn in _STR_FLOAT:
        return pa.float64()
    if fn in _STR_STR:
        return pa.string()
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
