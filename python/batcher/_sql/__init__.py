"""SQL frontend — run standard SQL over Batcher datasets.

`sql(query, **tables)` parses the query (via sqlglot) and translates it into the
same `LogicalPlan` the fluent API builds, so SQL and the DataFrame API share one
optimizer and engine. A clean, well-tested subset is supported (SELECT / WHERE /
GROUP BY / HAVING / ORDER BY / LIMIT, simple INNER/LEFT joins, scalar + aggregate
expressions, CASE, CAST); unsupported constructs raise a clear error.
"""

from __future__ import annotations

from batcher._sql.parser import sql

__all__ = ["sql"]
