"""A SQL mistake surfaces as a Batcher error with a plain-text message.

A typo in a query is the single most common thing a user does wrong, so it is the
error most worth getting right. Two properties are asserted here, on **both** parse
paths — the stateless `bt.sql` entry and the `Session` cache path, which parse in
different modules and had drifted:

1. The exception is `PlanError`, not sqlglot's `ParseError`. A user should catch one
   type for every plan-time failure rather than importing the parser to name its own.
2. The message is plain text. sqlglot underlines the offending token with ANSI escape
   codes, which read as line noise in a log file, a CI transcript, or a notebook.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher._internal.sql_errors import parse_sql

pytestmark = pytest.mark.unit

# Queries that fail in the tokenizer or the parser rather than the translator.
_MALFORMED = ["selct 1", "SELECT * FROM (", "SELECT FROM", "SELECT * FROM t WHERE"]


def _session() -> bt.Session:
    s = bt.Session()
    s.register("t", bt.from_pydict({"x": [1, 2, 3]}))
    return s


@pytest.mark.parametrize("query", _MALFORMED)
def test_bt_sql_raises_plan_error_on_a_syntax_error(query: str) -> None:
    with pytest.raises(PlanError, match="could not parse SQL"):
        bt.sql(query)


@pytest.mark.parametrize("query", _MALFORMED)
def test_session_sql_raises_plan_error_on_a_syntax_error(query: str) -> None:
    # The Session path parses in api/sql_session.py, not in the _sql translator; both
    # must route through the same helper or one of them regresses silently.
    with pytest.raises(PlanError, match="could not parse SQL"):
        _session().sql(query)


@pytest.mark.parametrize("query", _MALFORMED)
def test_the_message_carries_no_ansi_escapes(query: str) -> None:
    with pytest.raises(PlanError) as excinfo:
        bt.sql(query)
    assert "\x1b" not in str(excinfo.value)


def test_the_message_names_the_dialect() -> None:
    with pytest.raises(PlanError, match=r"dialect 'duckdb'"):
        bt.sql("selct 1")


def test_the_original_parse_error_is_kept_as_the_cause() -> None:
    with pytest.raises(PlanError) as excinfo:
        bt.sql("selct 1")
    assert excinfo.value.__cause__ is not None


def test_a_multi_statement_script_says_so_rather_than_naming_an_ast_node() -> None:
    # "got Block" told the user nothing about what they typed.
    with pytest.raises(PlanError, match="one statement"):
        bt.sql("SELECT 1; SELECT 2")


def test_dml_and_ddl_are_dispatched_before_the_query_translator() -> None:
    # These reach dedicated handlers, so the translator's "cannot translate" branch is
    # genuinely for unsupported statement *forms* — asserted so the error message,
    # which says exactly that, stays true.
    ds = bt.from_pydict({"x": [1]})
    for query in ("DELETE FROM t", "UPDATE t SET x = 1", "INSERT INTO t VALUES (2)"):
        assert bt.sql(query, t=ds) is not None


def test_valid_sql_is_untouched_by_the_wrapping() -> None:
    assert bt.sql("SELECT 1 AS a").to_pydict() == {"a": [1]}
    assert _session().sql("SELECT * FROM t WHERE x > 1").to_pydict() == {"x": [2, 3]}


def test_parse_sql_returns_the_ast_for_valid_sql() -> None:
    assert type(parse_sql("SELECT 1", dialect="duckdb")).__name__ == "Select"
