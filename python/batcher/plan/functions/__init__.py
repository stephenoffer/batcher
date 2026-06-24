"""The expression function library, grouped by family.

These are the broader SQL/PySpark/DuckDB-style free functions (string, conditional,
math, …), built by composing the core `Expr` nodes from `batcher.plan.expr_ir`.
They are kept separate from `expr_ir.constructors` (which holds the primitive
constructors tied directly to the node classes — `col`/`lit`/`when`/…) so this
package can grow by family without bloating the core module. Surfaced to users
through the `batcher.api.functions` façade.
"""

from __future__ import annotations

from batcher.plan.functions.aggregate import corr, count_if, covar_pop, covar_samp
from batcher.plan.functions.collection import element, named_struct, sequence, struct
from batcher.plan.functions.conditional import iff, ifnull, nanvl
from batcher.plan.functions.math import gcd, hypot, lcm, log, width_bucket
from batcher.plan.functions.string import concat, concat_ws, format_string
from batcher.plan.functions.temporal import (
    current_date,
    current_timestamp,
    date_add,
    date_part,
    date_sub,
    now,
    today,
    window,
)

__all__ = [
    "concat",
    "concat_ws",
    "corr",
    "count_if",
    "covar_pop",
    "covar_samp",
    "current_date",
    "current_timestamp",
    "date_add",
    "date_part",
    "date_sub",
    "element",
    "format_string",
    "gcd",
    "hypot",
    "iff",
    "ifnull",
    "lcm",
    "log",
    "named_struct",
    "nanvl",
    "now",
    "sequence",
    "struct",
    "today",
    "width_bucket",
    "window",
]
