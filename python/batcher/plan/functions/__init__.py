"""The expression function library, grouped by family.

These are the broader SQL/PySpark/DuckDB-style free functions (string, conditional,
math, …), built by composing the core `Expr` nodes from `batcher.plan.expr_ir`.
They are kept separate from `expr_ir.constructors` (which holds the primitive
constructors tied directly to the node classes — `col`/`lit`/`when`/…) so this
package can grow by family without bloating the core module. Surfaced to users
through the `batcher.api.functions` façade.
"""

from __future__ import annotations

from batcher.plan.functions.aggregate import (
    corr,
    count_if,
    covar_pop,
    covar_samp,
    max,
    mean,
    median,
    min,
    n_unique,
    std,
    sum,
    var,
)
from batcher.plan.functions.collection import element, named_struct, sequence, struct
from batcher.plan.functions.conditional import iff, nanvl
from batcher.plan.functions.horizontal import (
    all_horizontal,
    any_horizontal,
    count_horizontal,
    fold_horizontal,
    max_horizontal,
    mean_horizontal,
    min_horizontal,
    product_horizontal,
    reduce_horizontal,
    sum_horizontal,
)
from batcher.plan.functions.math import arctan2, gcd, hypot, lcm, log, width_bucket
from batcher.plan.functions.security import aes_decrypt, aes_encrypt, hmac_sha256, mask
from batcher.plan.functions.string import concat, concat_ws, format_string
from batcher.plan.functions.temporal import (
    current_date,
    current_timestamp,
    date_add,
    date_part,
    date_sub,
    window,
)

__all__ = [
    "aes_decrypt",
    "aes_encrypt",
    "all_horizontal",
    "any_horizontal",
    "arctan2",
    "concat",
    "concat_ws",
    "corr",
    "count_horizontal",
    "count_if",
    "covar_pop",
    "covar_samp",
    "current_date",
    "current_timestamp",
    "date_add",
    "date_part",
    "date_sub",
    "element",
    "fold_horizontal",
    "format_string",
    "gcd",
    "hmac_sha256",
    "hypot",
    "iff",
    "lcm",
    "log",
    "mask",
    "max",
    "max_horizontal",
    "mean",
    "mean_horizontal",
    "median",
    "min",
    "min_horizontal",
    "n_unique",
    "named_struct",
    "nanvl",
    "product_horizontal",
    "reduce_horizontal",
    "sequence",
    "std",
    "struct",
    "sum",
    "sum_horizontal",
    "var",
    "width_bucket",
    "window",
]
