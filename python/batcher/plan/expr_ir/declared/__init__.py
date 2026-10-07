"""Typing declarations for the methods bound onto expression classes at runtime.

The parameterless `.str`/`.dt`/`.list` accessors, `AggExpr`'s math methods and
`CaseBuilder`'s operators are attached with `setattr`, which a type checker cannot see.
`bound` declares them, generated from the live classes by `tools/gen_bound_decls.py`; each
runtime class inherits its declarations under ``TYPE_CHECKING`` only.
"""

from __future__ import annotations

from batcher.plan.expr_ir.declared.bound import (
    AggExprBound,
    CaseBuilderBound,
    DtBound,
    ListBound,
    StrBound,
)

__all__ = ["AggExprBound", "CaseBuilderBound", "DtBound", "ListBound", "StrBound"]
