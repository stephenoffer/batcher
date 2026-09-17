"""Migration-error guidance for `Expr` and its typed accessors.

Batcher keeps one spelling per capability. When a migrant types a name `Expr` does not
carry, `Expr.__getattr__` and the accessor hooks raise an `AttributeError` that names the
Batcher spelling: from the curated tables in `guidance` first, then from the migration
registry (`batcher._internal.migration`), which also knows every spelling Batcher removed.
"""

from __future__ import annotations

from batcher.plan.expr_ir.compat.guidance import (
    DT_UNSUPPORTED,
    LIST_UNSUPPORTED,
    STR_UNSUPPORTED,
    accessor_attribute_error,
    expr_attribute_error,
)

__all__ = [
    "DT_UNSUPPORTED",
    "LIST_UNSUPPORTED",
    "STR_UNSUPPORTED",
    "accessor_attribute_error",
    "expr_attribute_error",
]
