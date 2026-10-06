"""Turn a sqlglot parse failure into a Batcher `PlanError` with a plain-text message.

Two call sites parse SQL — the `Session` cache path and the stateless `_sql`
translator entry — and they sit in different layers (`api` and the `_sql` front end),
which may not import each other. Layer 0 is the only place both can see, so the
shared behaviour lives here rather than being pasted into each.

Two things are fixed on the way out:

* **The exception type.** sqlglot's `ParseError` is an implementation detail leaking
  through the public API. A typo raises `SQLSyntaxError`, a `PlanError`, carrying the
  line, column and character span sqlglot reported, so it is caught like every other
  plan-time failure.
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

Two things are recorded while parsing, because they cannot be recovered afterwards:

* **A named argument a typed function dropped.** sqlglot builds ``lpad(s, 3, 'x', z => 1)``
  into a `Pad` node with no slot for ``z``, so the argument vanished before any translator
  could see it and the call answered as if it had never been written. A dropped named
  argument is refused here, while the parser still holds it.
* **Where each parameter placeholder sits.** sqlglot keeps no position on a ``?``, and a
  tree walk does not visit nodes in the order they were written, so a positional binding
  needs the character offset the parser saw.
"""

from __future__ import annotations

import functools
import re
from typing import Any

from batcher._internal.errors import PlanError, SQLSyntaxError, SQLUnsupportedError

__all__ = ["PLACEHOLDER_OFFSET", "check_dialect", "parse_sql", "unsupported"]

#: The `meta` key holding a parameter placeholder's character offset in the query.
PLACEHOLDER_OFFSET = "bc_offset"

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

    reader = Dialect.get_or_raise(check_dialect(dialect))
    try:
        parser = _parser_class(reader.parser_class)(dialect=reader)
        parsed = [p for p in parser.parse(reader.tokenize(query), query) if p is not None]
        if not parsed:
            raise ParseError(f"No expression was parsed from {query!r}")
        # What `sqlglot.parse_one` returns for a multi-statement script, which the callers
        # recognize and refuse with a message of their own.
        return exp.Block(expressions=parsed) if len(parsed) > 1 else parsed[0]
    except (ParseError, TokenError) as exc:
        detail = _ANSI_ESCAPE.sub("", str(exc)).strip()
        raise SQLSyntaxError(
            f"could not parse SQL (dialect {dialect!r}): {detail}", **_error_span(exc, query)
        ) from exc


def check_dialect(dialect: str) -> str:
    """Return `dialect` when sqlglot can read it, else raise a `PlanError` naming the choices.

    Args:
        dialect: The sqlglot read dialect name, e.g. ``"duckdb"`` or ``"spark"``.

    Returns:
        `dialect`, unchanged.

    Raises:
        PlanError: If `dialect` is not a string sqlglot knows.

    Examples:
        .. doctest::

            >>> from batcher._internal.sql_errors import check_dialect
            >>> check_dialect("spark")
            'spark'
    """
    from sqlglot.dialects.dialect import Dialect

    if isinstance(dialect, str):
        try:
            Dialect.get_or_raise(dialect)
            return dialect
        except ValueError:
            pass
    names = sorted(d for d in Dialect.classes if d)
    raise PlanError(
        f"unknown SQL dialect {dialect!r}",
        available=names,
        available_label="Available dialects",
        hint="The dialect picks the SQL grammar only; Batcher's semantics do not change.",
    )


def unsupported(message: str, node: Any = None) -> SQLUnsupportedError:
    """A `SQLUnsupportedError` located at `node`, the expression being translated.

    The position is the first one sqlglot recorded at or beneath `node`: a function call and
    an identifier carry one, most other nodes do not. Without any, the fields stay None
    rather than pointing somewhere the refusal does not apply.

    Args:
        message: What is unsupported, and the rewrite that works where there is one.
        node: The sqlglot node being translated, or None.

    Returns:
        The error, ready to raise.

    Examples:
        .. doctest::

            >>> from batcher._internal.sql_errors import parse_sql, unsupported
            >>> call = parse_sql("SELECT nope(x)", dialect="duckdb").expressions[0]
            >>> err = unsupported("no such function", call)
            >>> (err.line, err.column, err.start, err.end)
            (1, 8, 7, 10)
    """
    meta: dict = {}
    if node is not None:
        meta = next((n.meta for n in node.walk() if "line" in n.meta and "start" in n.meta), {})
    if not meta:
        return SQLUnsupportedError(message)
    return _located(message, meta["line"], meta["col"], meta["start"], meta["end"])


def _located(message: str, line: int, last_col: int, start: int, end: int) -> SQLUnsupportedError:
    """A `SQLUnsupportedError` for the token sqlglot placed at `line`/`last_col`.

    sqlglot records the column of a token's *last* character; the error reports its first.
    """
    column = last_col - (end - start)
    return SQLUnsupportedError(
        f"{message} (line {line}, column {column})",
        line=line,
        column=column,
        start=start,
        end=end,
    )


def _error_span(exc: Exception, query: str) -> dict[str, int | None]:
    """The line, start column and character span of the token sqlglot stopped at.

    sqlglot reports the 1-based line and the column of the token's *last* character, plus
    the token text; the start column and the 0-based offsets follow from those and the query.
    A tokenizer failure reports none of it, and every field is then None.
    """
    errors = getattr(exc, "errors", None) or []
    first = errors[0] if errors else {}
    line, last_col = first.get("line"), first.get("col")
    if not isinstance(line, int) or not isinstance(last_col, int):
        return {"line": None, "column": None, "start": None, "end": None}
    width = max(len(first.get("highlight") or ""), 1)
    column = max(last_col - width + 1, 1)
    lines = query.split("\n")
    start = sum(len(text) + 1 for text in lines[: line - 1]) + column - 1
    return {"line": line, "column": column, "start": start, "end": start + width - 1}


@functools.cache
def _parser_class(base: type) -> type:
    """`base`, the dialect's sqlglot parser, with Batcher's three parse-time adjustments.

    Dropping a name from ``FUNCTIONS`` is what makes sqlglot build an ``exp.Anonymous`` for
    it, the node every untyped function call becomes. `validate_expression` refuses a named
    argument the typed builder dropped, and `_parse_placeholder` records each placeholder's
    offset. Cached per dialect parser, because sqlglot's parser metaclass does real work on
    each subclass.
    """
    functions = {k: v for k, v in base.FUNCTIONS.items() if k not in _ORDINARY_CALLS}

    def validate_expression(self: Any, expression: Any, args: list | None = None) -> Any:
        _refuse_dropped_kwargs(expression, args, _call_name_token(self))
        return base.validate_expression(self, expression, args)

    def _parse_placeholder(self: Any) -> Any:
        token = self._curr
        node = base._parse_placeholder(self)
        if node is not None and token is not None:
            node.meta.setdefault(PLACEHOLDER_OFFSET, token.start)
        return node

    return type(
        f"Batcher{base.__name__}",
        (base,),
        {
            "FUNCTIONS": functions,
            "validate_expression": validate_expression,
            "_parse_placeholder": _parse_placeholder,
        },
    )


def _call_name_token(parser: Any) -> Any:
    """The token naming the function call whose arguments the parser has just read.

    Walked back from the cursor to the opening parenthesis that balances it, so a nested
    call inside the arguments does not stop the walk early. None when no such token exists.
    """
    from sqlglot.tokens import TokenType

    depth = 0
    for index in range(parser._index - 1, 0, -1):
        kind = parser._tokens[index].token_type
        if kind is TokenType.R_PAREN:
            depth += 1
        elif kind is TokenType.L_PAREN:
            depth -= 1
            if depth < 0:
                return parser._tokens[index - 1]
    return None


def _refuse_dropped_kwargs(expression: Any, args: list | None, token: Any) -> None:
    """Raise when a ``name => value`` argument is absent from the node built from `args`.

    A typed builder keeps the arguments it has slots for, so one it has no slot for is lost
    without a trace: the call then answers as if it had never been written. The kept
    arguments are compared by name, since a builder may copy a node as it files it.
    """
    from sqlglot import exp

    named = [a for a in args or () if isinstance(a, exp.Kwarg)]
    if not named:
        return
    kept = {k.this.name.lower() for k in expression.find_all(exp.Kwarg)}
    dropped = [k.this.name for k in named if k.this.name.lower() not in kept]
    if dropped:
        message = f"takes no named argument {dropped[0]!r}; pass its arguments by position"
        if token is None:
            raise SQLUnsupportedError(f"{expression.key}() {message}")
        raise _located(f"{token.text}() {message}", token.line, token.col, token.start, token.end)
