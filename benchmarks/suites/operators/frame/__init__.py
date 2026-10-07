"""Operator-mix through the DataFrame API: Batcher's `Dataset`/`Expr` against Polars' expressions.

Every family one directory up runs Batcher through `bt.Session().sql(...)`, so the expression
namespaces a DataFrame user writes (`.dt`, `.str`, `.list`, the `cum_*`/`rolling_*`/`diff`
window expressions, `pivot`, `rollup`, `top_k`) were only ever timed by way of the SQL
front-end's lowering. These families time the API itself, against the competitor's own
spelling of the same operation: Polars' lazy expressions, DuckDB's SQL on its native storage,
and PyArrow compute where it has the function. Inputs are the real TPC-H tables, or a reshaping
of them built once outside the timed region, and every result is reduced to a few rows so the
correctness gate compares the whole answer before any timing is trusted.

Modules are auto-discovered like the parent package's.
"""

from __future__ import annotations

from discover import import_submodules

import_submodules(__name__)
