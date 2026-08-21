"""Generative AI from SQL — ``AI_GENERATE`` / ``ai_query`` / ``ai_complete`` / ``AI_EXTRACT``.

`ML_PREDICT` gives SQL the traditional model; these are the generative half, and they lower to
`Dataset.ml.generate` and `Dataset.ml.extract` rather than to a second inference path.

Every engine here is a plain ``list[str] -> list[str]`` closure, so the whole file runs with no
network, no API key and no GPU — which is the point of the engine contract being that narrow.

Two things these pin beyond the happy path. An engine is resolved from the session catalog and
never from query text, because writing one inline would mean putting an API key in a SQL
string. And the AI calls sqlglot parses but this does not translate are *claimed* rather than
left to fall through: without that they reach the table lookup and fail as ``unknown table ''``,
since those nodes carry no table name.
"""

from __future__ import annotations

import json

import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.unit


@pytest.fixture
def session() -> bt.Session:
    s = bt.Session()
    s.register(
        "reviews",
        bt.from_pydict({"id": [1, 2, 3], "body": ["love it", "broke fast", "it is fine"]}),
    )
    s.register_engine("shouty", lambda: lambda prompts: [p.upper() for p in prompts])
    s.register_engine(
        "grader",
        lambda: (
            lambda prompts: [
                json.dumps({"label": "positive" if "love" in p else "negative"}) for p in prompts
            ]
        ),
    )
    return s


# --- the catalog ---------------------------------------------------------------------


def test_engines_register_and_list(session: bt.Session) -> None:
    assert session.list_engines() == ["grader", "shouty"]


def test_an_engine_name_must_be_a_non_empty_string(session: bt.Session) -> None:
    with pytest.raises(PlanError, match="non-empty string"):
        session.register_engine("", lambda: lambda p: p)


def test_a_dialect_view_shares_the_engine_catalog(session: bt.Session) -> None:
    """`_with_dialect` shares every other catalog by reference; engines must not be the gap."""
    assert session._with_dialect("spark").list_engines() == ["grader", "shouty"]


# --- generation ----------------------------------------------------------------------


def test_ai_generate_appends_the_response_column(session: bt.Session) -> None:
    out = session.sql(
        "SELECT id, response FROM AI_GENERATE(reviews, shouty, prompt_column => 'body')"
    ).to_pydict()
    assert out["response"] == ["LOVE IT", "BROKE FAST", "IT IS FINE"]
    assert out["id"] == [1, 2, 3]


@pytest.mark.parametrize("spelling", ["AI_GENERATE", "ai_query", "ai_complete"])
def test_the_provider_aliases_all_mean_generate(session: bt.Session, spelling: str) -> None:
    """Databricks writes `ai_query` and Snowflake `AI_COMPLETE`; a ported query should run."""
    out = session.sql(
        f"SELECT response FROM {spelling}(reviews, shouty, prompt_column => 'body')"
    ).to_pydict()
    assert out["response"] == ["LOVE IT", "BROKE FAST", "IT IS FINE"]


def test_output_column_renames_the_result(session: bt.Session) -> None:
    out = session.sql(
        "SELECT summary FROM AI_GENERATE(reviews, shouty, prompt_column => 'body',"
        " output_column => 'summary')"
    ).to_pydict()
    assert out["summary"] == ["LOVE IT", "BROKE FAST", "IT IS FINE"]


def test_a_template_wraps_each_prompt(session: bt.Session) -> None:
    out = session.sql(
        "SELECT response FROM AI_GENERATE(reviews, shouty, prompt_column => 'body',"
        " template => 'Say: {body}')"
    ).to_pydict()
    assert out["response"] == ["SAY: LOVE IT", "SAY: BROKE FAST", "SAY: IT IS FINE"]


# --- extraction ----------------------------------------------------------------------


def test_ai_extract_types_each_declared_field(session: bt.Session) -> None:
    out = session.sql(
        "SELECT id, label FROM AI_EXTRACT(reviews, grader, prompt_column => 'body',"
        " schema => ['label string'])"
    ).to_pydict()
    assert out["label"] == ["positive", "negative", "negative"]


# --- it is an ordinary relation ------------------------------------------------------


def test_the_generated_column_is_visible_to_the_rest_of_the_statement(
    session: bt.Session,
) -> None:
    """The point of a table function: the result groups and filters without leaving SQL."""
    out = session.sql(
        "SELECT label, COUNT(*) AS n FROM AI_EXTRACT(reviews, grader,"
        " prompt_column => 'body', schema => ['label string'])"
        " GROUP BY label ORDER BY label"
    ).to_pydict()
    assert out == {"label": ["negative", "positive"], "n": [2, 1]}


def test_it_accepts_a_subquery_as_its_relation(session: bt.Session) -> None:
    out = session.sql(
        "SELECT response FROM AI_GENERATE((SELECT * FROM reviews WHERE id = 1), shouty,"
        " prompt_column => 'body')"
    ).to_pydict()
    assert out["response"] == ["LOVE IT"]


# --- the guards ----------------------------------------------------------------------


def test_an_unknown_engine_lists_the_registered_ones(session: bt.Session) -> None:
    with pytest.raises(PlanError, match=r"unknown engine 'nope'.*grader"):
        session.sql("SELECT * FROM AI_GENERATE(reviews, nope, prompt_column => 'body')")


def test_an_engine_cannot_be_written_inline_as_a_string(session: bt.Session) -> None:
    """The guard that keeps an endpoint and an API key out of query text."""
    with pytest.raises(PlanError, match="registered engine name, not the quoted string"):
        session.sql("SELECT * FROM AI_GENERATE(reviews, 'http://host/v1', prompt_column => 'b')")


def test_prompt_column_is_required(session: bt.Session) -> None:
    with pytest.raises(PlanError, match="needs 'prompt_column'"):
        session.sql("SELECT * FROM AI_GENERATE(reviews, shouty)")


def test_an_unknown_setting_lists_the_supported_ones(session: bt.Session) -> None:
    with pytest.raises(PlanError, match=r"unknown setting 'temperature'.*prompt_column"):
        session.sql(
            "SELECT * FROM AI_GENERATE(reviews, shouty, prompt_column => 'body',"
            " temperature => '1')"
        )


def test_the_relation_and_engine_are_both_required(session: bt.Session) -> None:
    with pytest.raises(PlanError, match="exactly two positional arguments"):
        session.sql("SELECT * FROM AI_GENERATE(reviews)")


def test_an_unknown_relation_is_named(session: bt.Session) -> None:
    with pytest.raises(PlanError, match="unknown table 'nosuch'"):
        session.sql("SELECT * FROM AI_GENERATE(nosuch, shouty, prompt_column => 'body')")


@pytest.mark.parametrize(
    ("schema_sql", "expected"),
    [
        ("schema => ['oops']", r"'<name> <type>'"),
        ("schema => 'label string'", r"must be a list"),
        ("schema => []", r"must not be empty"),
    ],
)
def test_a_malformed_extract_schema_says_how_to_write_it(
    session: bt.Session, schema_sql: str, expected: str
) -> None:
    with pytest.raises(PlanError, match=expected):
        session.sql(
            f"SELECT * FROM AI_EXTRACT(reviews, grader, prompt_column => 'body', {schema_sql})"
        )


@pytest.mark.parametrize(
    ("query", "points_at"),
    [
        ("SELECT * FROM AI_CLASSIFY(reviews, ['a', 'b'])", "AI_EXTRACT"),
        ("SELECT * FROM AI_EMBED(reviews, shouty)", "ds.ml.embed"),
        ("SELECT * FROM AI_SIMILARITY(reviews, shouty)", "ds.ml.similarity_to"),
    ],
)
def test_an_untranslated_ai_call_says_where_the_capability_is(
    session: bt.Session, query: str, points_at: str
) -> None:
    """Unclaimed, these reach the table lookup and fail as ``unknown table ''``."""
    with pytest.raises(PlanError, match=points_at):
        session.sql(query)


def test_the_untranslated_message_uses_the_sql_spelling(session: bt.Session) -> None:
    """Quoting sqlglot's ``AIClassify`` back sends the reader looking for another function."""
    with pytest.raises(PlanError, match=r"^AI_CLASSIFY is not translated"):
        session.sql("SELECT * FROM AI_CLASSIFY(reviews, ['a', 'b'])")
