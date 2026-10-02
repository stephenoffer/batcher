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

One grammar difference is applied on the way in. sqlglot parses ``AI_CLASSIFY`` on
Snowflake's fixed three-argument grammar, ``(input, categories [, config])``, which rejects
the relational form Batcher gives every AI table function: the relation, the engine, and
named settings. Reading ``AI_CLASSIFY`` as an ordinary call lets the SQL front end give it
the same shape as ``AI_EXTRACT`` and ``AI_GENERATE``.

Three list functions get the same treatment, because sqlglot's typed node loses part of the
call. ``ARRAY_SLICE(l, begin, end)`` becomes the node Spark's ``slice(l, start, length)`` also
builds, so DuckDB's inclusive end was read as a length. ``LIST_REVERSE_SORT(l, nulls)`` and
``ARRAY_REVERSE_SORT`` drop their null-order argument. As ordinary calls they reach the SQL
front end with every argument intact. Spark's ``slice`` keeps its own node and meaning.
"""

from __future__ import annotations

import functools
import re
from typing import Any

from batcher._internal.errors import PlanError

__all__ = ["parse_sql"]

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

#: Functions sqlglot gives a fixed grammar that Batcher reads as an ordinary call instead.
_ORDINARY_CALLS = frozenset(
    {"AI_CLASSIFY", "ARRAY_SLICE", "LIST_REVERSE_SORT", "ARRAY_REVERSE_SORT"}
)


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
    from sqlglot import exp
    from sqlglot.dialects.dialect import Dialect
    from sqlglot.errors import ParseError, TokenError

    try:
        reader = Dialect.get_or_raise(dialect)
        parser = _parser_class(reader.parser_class)(dialect=reader)
        parsed = parser.parse(reader.tokenize(query), query)
        if not parsed or parsed[0] is None:
            raise ParseError(f"No expression was parsed from {query!r}")
        # What `sqlglot.parse_one` returns for a multi-statement script, which the callers
        # recognize and refuse with a message of their own.
        return exp.Block(expressions=parsed) if len(parsed) > 1 else parsed[0]
    except (ParseError, TokenError) as exc:
        detail = _ANSI_ESCAPE.sub("", str(exc)).strip()
        raise PlanError(f"could not parse SQL (dialect {dialect!r}): {detail}") from exc


@functools.cache
def _parser_class(base: type) -> type:
    """`base`, the dialect's sqlglot parser, with `_ORDINARY_CALLS` parsed as plain calls.

    Dropping a name from ``FUNCTIONS`` is what makes sqlglot build an ``exp.Anonymous`` for
    it, the node every untyped function call becomes. Cached per dialect parser, because
    sqlglot's parser metaclass does real work on each subclass.
    """
    functions = {k: v for k, v in base.FUNCTIONS.items() if k not in _ORDINARY_CALLS}
    return type(f"Batcher{base.__name__}", (base,), {"FUNCTIONS": functions})
