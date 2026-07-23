"""Accessor namespaces (`.str`/`.dt`/`.list`/`.struct`/`.json`) — package façade.

The namespace classes are returned by the corresponding `Expr` properties (via
deferred imports in `core.py`, to keep the `Expr` import acyclic). They are grouped
by family (`strings`, `temporal`, `collections`) and re-exported here, along with
the `func_nodes` IR classes callers historically imported from this module.
"""

from __future__ import annotations

from batcher.plan.expr_ir.compat.namespaces import bind_namespace_compat as _bind_namespace_compat
from batcher.plan.expr_ir.func_nodes import (
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
    ListTransform,
    ListZip,
    MapFunc,
    Strftime,
    StrFunc,
    Strptime,
    StructField,
)
from batcher.plan.expr_ir.namespaces.collections import (
    _JsonNamespace,
    _ListNamespace,
    _MapNamespace,
    _StructNamespace,
)
from batcher.plan.expr_ir.namespaces.strings import _StrNamespace
from batcher.plan.expr_ir.namespaces.temporal import _DtNamespace, parse_offset

__all__ = [
    "ConvertTimezone",
    "DateFunc",
    "DateOffset",
    "DateTrunc",
    "ListBinary",
    "ListContains",
    "ListFilter",
    "ListFunc",
    "ListGet",
    "ListPosition",
    "ListSet",
    "ListSimhash",
    "ListSlice",
    "ListTransform",
    "ListZip",
    "MapFunc",
    "StrFunc",
    "Strftime",
    "Strptime",
    "StructField",
    "_DtNamespace",
    "_JsonNamespace",
    "_ListNamespace",
    "_MapNamespace",
    "_StrNamespace",
    "_StructNamespace",
    "parse_offset",
]

# The ecosystem spellings (``str.isdigit``, ``dt.day_of_week``, ``list.element_at``)
# are attached once the accessor classes above exist. They live in `expr_ir.compat`
# so each namespace module stays the single home of its primary surface.
_bind_namespace_compat()
