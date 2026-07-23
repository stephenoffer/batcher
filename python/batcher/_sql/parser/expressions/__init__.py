"""SQL scalar-expression translation — a sqlglot value node becomes an `Expr` (layer 6).

The scalar side of the SQL front-end: expression dispatch (`scalar`), named-function
builders (`functions`), literals/temporal/dtype tables (`literals`), and JSON path
extraction (`json`). Everything here lowers to the same `plan.expr_ir` expressions the
fluent API builds — there is no second expression representation.

This façade re-exports the two names the relational theme modules need, so `translator`
and `grouping` depend on the family, not on a module inside it.
"""

from __future__ import annotations

from batcher._sql.parser.expressions.literals import _AGG_FUNCS
from batcher._sql.parser.expressions.scalar import _scalar

__all__ = ["_AGG_FUNCS", "_scalar"]
