"""JSON array-index paths and the SQL ``json_extract`` surface — vs DuckDB.

Two behaviours land here that the earlier JSON tests do not cover:

- **Array-index paths** (``$.tags[0]``, ``$.a[1].b``) in the ``.json`` accessor — the
  lazy path scanner descends array subscripts, which the old dotted-key parser could not.
- **The SQL surface** — ``json_extract_string`` / ``json_extract`` / ``->>`` lower to the
  same ``.json`` accessor, so a query written in SQL matches DuckDB's own JSON functions.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col


def _array_table() -> pa.Table:
    return pa.table(
        {
            "j": [
                '{"tags": ["mobile", "web"], "items": [{"id": 10}, {"id": 20}]}',
                '{"tags": ["promo"], "items": [{"id": 30}]}',
                '{"tags": [], "items": []}',  # empty arrays → out-of-range null
                None,
            ]
        }
    )


def test_json_array_index_string(duck):
    from conftest import assert_same

    t = _array_table()
    duck.register("j", t)
    out = (
        bt.from_arrow(t)
        .select(
            tag0=col("j").json.extract_string("$.tags[0]"),
            tag1=col("j").json.extract_string("$.tags[1]"),
        )
        .collect()
    )
    assert_same(
        out,
        duck.sql(
            "SELECT json_extract_string(j, '$.tags[0]') tag0, "
            "json_extract_string(j, '$.tags[1]') tag1 FROM j"
        ),
    )


def test_json_array_nested_index_int(duck):
    from conftest import assert_same

    t = _array_table()
    duck.register("j", t)
    out = bt.from_arrow(t).select(id1=col("j").json.extract_int("$.items[1].id")).collect()
    assert_same(out, duck.sql("SELECT CAST(json_extract(j, '$.items[1].id') AS BIGINT) id1 FROM j"))


def test_sql_json_extract_string(duck):
    from conftest import assert_same

    t = _array_table()
    duck.register("j", t)
    session = bt.Session()
    session.register("j", t)
    query = "SELECT json_extract_string(j, '$.tags[0]') AS t FROM j"
    assert_same(session.sql(query).collect(), duck.sql(query))


def test_sql_json_extract_cast_and_filter(duck):
    from conftest import assert_same

    t = pa.table(
        {
            "j": [
                '{"user": {"country": "US"}, "event": {"value": 12.5, "type": "purchase"}}',
                '{"user": {"country": "DE"}, "event": {"value": 0.0, "type": "view"}}',
                '{"user": {"country": "US"}, "event": {"value": 7.5, "type": "purchase"}}',
            ]
        }
    )
    duck.register("j", t)
    session = bt.Session()
    session.register("j", t)
    query = (
        "SELECT json_extract_string(j, '$.user.country') AS country, "
        "SUM(CAST(json_extract_string(j, '$.event.value') AS DOUBLE)) AS s, COUNT(*) AS n "
        "FROM j WHERE json_extract_string(j, '$.event.type') = 'purchase' GROUP BY 1"
    )
    assert_same(session.sql(query).collect(), duck.sql(query))


def test_sql_json_arrow_operator(duck):
    """The ``->>`` arrow operator lowers to the same extractor as the function form."""
    from conftest import assert_same

    t = pa.table({"j": ['{"a": {"b": "x"}}', '{"a": {"b": "y"}}']})
    duck.register("j", t)
    session = bt.Session()
    session.register("j", t)
    query = "SELECT j ->> '$.a.b' AS v FROM j"
    assert_same(session.sql(query).collect(), duck.sql(query))
