"""The scalar expression algebra.

`Expr` is the single expression representation in Batcher. The Python side builds
it (with operator overloading, so `col("x") > 2` is natural) and serializes it
via `to_ir()` to the exact JSON document the Rust `bc-expr` crate deserializes —
the same IR consumed by both the interpreter and (later) the JIT. The wire tags
here (`e`, `op`, literal kind) are a contract with the engine; keep them in sync.

This package re-exports the full expression surface from its submodules:
``core`` (the `Expr` base and the nodes its methods build), ``namespaces`` (the
`.str`/`.dt`/`.list`/`.struct`/`.json` accessors and their nodes), ``walk``
(structural traversals), and ``constructors`` (the free entry points).
"""

from __future__ import annotations

from batcher.plan.expr_ir.constructors import (
    array,
    atan2,
    coalesce,
    col,
    count,
    greatest,
    least,
    lit,
    nullif,
    when,
)
from batcher.plan.expr_ir.core import (
    AggExpr,
    Aliased,
    Binary,
    Cast,
    Coalesce,
    Expr,
    IsNan,
    IsNotNull,
    IsNull,
    Lit,
    Math2Expr,
    MathExpr,
    Not,
)
from batcher.plan.expr_ir.core import IntoExpr as IntoExpr
from batcher.plan.expr_ir.core import _wrap as _wrap
from batcher.plan.expr_ir.namespaces import (
    DateFunc,
    DateTrunc,
    ListContains,
    ListFunc,
    ListGet,
    ListSlice,
    StrFunc,
    StructField,
)
from batcher.plan.expr_ir.namespaces import _DtNamespace as _DtNamespace
from batcher.plan.expr_ir.namespaces import _JsonNamespace as _JsonNamespace
from batcher.plan.expr_ir.namespaces import _ListNamespace as _ListNamespace
from batcher.plan.expr_ir.namespaces import _StrNamespace as _StrNamespace
from batcher.plan.expr_ir.namespaces import _StructNamespace as _StructNamespace
from batcher.plan.expr_ir.nodes import (
    Array,
    Case,
    CaseBuilder,
    Col,
    Greatest,
    Least,
    ListJoin,
    NullIf,
)
from batcher.plan.expr_ir.walk import referenced_columns, remap_columns

__all__ = [
    "AggExpr",
    "Aliased",
    "Array",
    "Binary",
    "Case",
    "CaseBuilder",
    "Cast",
    "Coalesce",
    "Col",
    "DateFunc",
    "DateTrunc",
    "Expr",
    "Greatest",
    "IsNan",
    "IsNotNull",
    "IsNull",
    "Least",
    "ListContains",
    "ListFunc",
    "ListGet",
    "ListJoin",
    "ListSlice",
    "Lit",
    "Math2Expr",
    "MathExpr",
    "Not",
    "NullIf",
    "StrFunc",
    "StructField",
    "array",
    "atan2",
    "coalesce",
    "col",
    "count",
    "greatest",
    "least",
    "lit",
    "nullif",
    "referenced_columns",
    "remap_columns",
    "when",
]
