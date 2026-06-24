"""SQL frontend — run standard SQL over Batcher datasets.

`sql(query, **tables)` parses the query (via sqlglot, in a configurable dialect)
and translates it into the same `LogicalPlan` the fluent API builds, so SQL and the
DataFrame API share one optimizer and engine. `translate_ast` is the same entry for
an already-parsed statement (used by the session DDL path). A broad subset is
supported — SELECT / WHERE / GROUP BY / HAVING / ORDER BY / LIMIT, joins, set ops,
CTEs, subqueries, window functions, and registered Python functions; unsupported
constructs raise a clear error.
"""

from __future__ import annotations

from batcher._sql.parser import sql, translate_ast

__all__ = ["sql", "translate_ast"]
