"""Ecosystem-compatible spellings bound onto `Expr`.

Batcher keeps one canonical name per capability (the SQL/Polars spelling). This
package adds the pandas names for the same operations, so a ported script runs
without a find-and-replace pass. Every alias is a thin delegation to the primary —
one implementation, one plan, no second semantics.

The aliases live here rather than in `core.py` so the fluent builder stays the
one-`Expr` hierarchy instead of carrying a parallel copy of its own surface;
`bind_compat_methods` attaches them at import time, the same way the typed
accessors are bound in `namespaces/_bind.py`.
"""

from __future__ import annotations

from batcher.plan.expr_ir.compat.binder import bind_compat_methods

__all__ = ["bind_compat_methods"]
