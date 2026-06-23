"""Translate a SQL query (sqlglot AST) into a Batcher `Dataset`.

`sql(query, **tables)` parses the query (via sqlglot) and translates it into the
same `LogicalPlan` the fluent API builds. The translator class and its
theme-grouped method bodies live in the sibling modules; this package re-exports
the public `sql` entry point so `from batcher._sql.parser import sql` keeps
working.
"""

from __future__ import annotations

from batcher._sql.parser.translator import sql

__all__ = ["sql"]
