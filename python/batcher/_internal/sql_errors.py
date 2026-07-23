"""Turn a sqlglot parse failure into a Batcher `PlanError` with a plain-text message.

Two call sites parse SQL — the `Session` cache path and the stateless `_sql`
translator entry — and they sit in different layers (`api` and the `_sql` front end),
which may not import each other. Layer 0 is the only place both can see, so the
shared behaviour lives here rather than being pasted into each.

Two things are fixed on the way out:

* **The exception type.** sqlglot's `ParseError` is an implementation detail leaking
  through the public API. A user who wrote a typo should be able to catch
  `batcher.PlanError` like every other plan-time failure.
* **The message.** sqlglot underlines the offending token with ANSI escapes. That
  reads well on a terminal and badly everywhere a message actually ends up: a log
  aggregator, a CI transcript, a notebook cell, a test asserting on the text.
"""

from __future__ import annotations

import re
from typing import Any

from batcher._internal.errors import PlanError

__all__ = ["parse_sql"]

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def parse_sql(query: str, *, dialect: str) -> Any:
    """Parse `query` with sqlglot, raising `PlanError` on a syntax failure.

    Args:
        query: The SQL text to parse.
        dialect: The sqlglot dialect to read, e.g. ``"duckdb"``.

    Returns:
        The parsed sqlglot expression tree.

    Raises:
        PlanError: If `query` is not valid SQL in `dialect`.

    Examples:
        .. doctest::

            >>> from batcher._internal.sql_errors import parse_sql
            >>> type(parse_sql("SELECT 1", dialect="duckdb")).__name__
            'Select'
    """
    import sqlglot
    from sqlglot.errors import ParseError, TokenError

    try:
        return sqlglot.parse_one(query, read=dialect)
    except (ParseError, TokenError) as exc:
        detail = _ANSI_ESCAPE.sub("", str(exc)).strip()
        raise PlanError(f"could not parse SQL (dialect {dialect!r}): {detail}") from exc
